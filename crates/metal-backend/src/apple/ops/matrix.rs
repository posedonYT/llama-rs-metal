use super::*;
impl Batch<'_> {
    pub fn embedding(&mut self, w: &Weight, token: usize, out: &Buffer) -> Result<()> {
        if w.affine.is_some() {
            return self.embeddings(w, &[u(token)?], out);
        }
        if w.dims.len() != 2 || token >= w.dims[1] || out.len() != w.dims[0] {
            return Err(shape("embedding"));
        }
        self.linear(
            "embedding",
            &[&w.buffer, out],
            Params {
                n: u(out.len())?,
                a: u(token)?,
                t: w.dtype as u32,
                ..Params::default()
            },
        )
    }
    pub fn embeddings(&mut self, w: &Weight, tokens: &[u32], out: &Buffer) -> Result<()> {
        if w.dims.len() != 2
            || tokens.is_empty()
            || out.len() != tokens.len() * w.dims[0]
            || tokens.iter().any(|id| *id as usize >= w.dims[1])
        {
            return Err(shape("embeddings"));
        }
        let ids = self.gpu.upload(
            &tokens
                .iter()
                .map(|id| f32::from_bits(*id))
                .collect::<Vec<_>>(),
        )?;
        if let Some(q) = &w.affine {
            return self.linear(
                "affine_embedding",
                &[&w.buffer, &q.scales, &q.biases, &ids, out],
                Params {
                    n: u(out.len())?,
                    a: u(w.dims[0])?,
                    b: u(w.dims[1])?,
                    d: u(q.group_size)?,
                    e: q.bits,
                    h: q.group_size.trailing_zeros(),
                    t: q.scale_type as u32,
                    ..Params::default()
                },
            );
        }
        self.linear(
            "embedding_many",
            &[&w.buffer, &ids, out],
            Params {
                n: u(out.len())?,
                a: u(w.dims[0])?,
                b: u(w.dims[1])?,
                t: w.dtype as u32,
                ..Params::default()
            },
        )
    }
    pub fn matmul(&mut self, w: &Weight, x: &Buffer, y: &Buffer, tokens: usize) -> Result<()> {
        if w.dims.len() != 2
            || tokens == 0
            || x.len() != w.dims[0] * tokens
            || y.len() != w.dims[1] * tokens
        {
            return Err(shape(format!(
                "matmul {:?}, input {}, output {}, tokens {tokens}",
                w.dims,
                x.len(),
                y.len()
            )));
        }
        if let Some(q) = &w.affine {
            return self.dispatch(
                if tokens >= 4 {
                    "affine_mm"
                } else {
                    match q.bits {
                        2 => "affine_mv_2",
                        4 => "affine_mv_4",
                        8 => "affine_mv_8",
                        _ => unreachable!("validated affine width"),
                    }
                },
                &[&w.buffer, &q.scales, &q.biases, x, y],
                Params {
                    a: u(w.dims[0])?,
                    b: u(w.dims[1])?,
                    c: u(tokens)?,
                    d: u(q.group_size)?,
                    e: q.bits,
                    h: q.group_size.trailing_zeros(),
                    t: q.scale_type as u32,
                    ..Params::default()
                },
                if tokens >= 4 {
                    (w.dims[1].div_ceil(32), tokens.div_ceil(32))
                } else {
                    (w.dims[1].div_ceil(16), tokens)
                },
                4,
            );
        }
        if tokens >= 4 {
            let tile = if tokens >= 16 { 32 } else { 16 };
            return self.dispatch(
                if tile == 32 {
                    "gemm_simd_wide"
                } else {
                    "gemm_simd"
                },
                &[&w.buffer, x, y],
                Params {
                    a: u(w.dims[0])?,
                    b: u(w.dims[1])?,
                    c: u(tokens)?,
                    t: w.dtype as u32,
                    ..Params::default()
                },
                (w.dims[1].div_ceil(tile), tokens.div_ceil(tile)),
                4,
            );
        }
        let name = match w.dtype {
            DType::F32 => "gemv_f32",
            DType::F16 => "gemv_f16",
            DType::BF16 => "gemv_bf16",
            DType::Q4_0 => "gemv_q4",
            DType::Q8_0 => "gemv_q8",
            DType::Q4K => "packed_q4k",
            DType::Q5K => "packed_q5k",
            DType::Q6K => "packed_q6k",
        };
        self.dispatch(
            name,
            &[&w.buffer, x, y],
            Params {
                a: u(w.dims[0])?,
                b: u(w.dims[1])?,
                ..Params::default()
            },
            (w.dims[1].div_ceil(16), tokens),
            4,
        )
    }
    pub fn expert_matmul(
        &mut self,
        w: &Weight,
        x: &Buffer,
        ids: &Buffer,
        y: &Buffer,
        per_expert_input: bool,
    ) -> Result<()> {
        if w.dims.len() != 3
            || x.len() != w.dims[0] * if per_expert_input { ids.len() } else { 1 }
            || y.len() != w.dims[1] * ids.len()
        {
            return Err(shape("expert matmul"));
        }
        // IDs originate in route(), which clamps selection to the router's validated expert count.
        if w.dims[2] < ids.len() {
            return Err(shape("expert count"));
        }
        if let Some(q) = &w.affine {
            return self.dispatch(
                match q.bits {
                    2 => "affine_expert_2",
                    4 => "affine_expert_4",
                    8 => "affine_expert_8",
                    _ => unreachable!("validated affine width"),
                },
                &[&w.buffer, &q.scales, &q.biases, x, ids, y],
                Params {
                    a: u(w.dims[0])?,
                    b: u(w.dims[1])?,
                    c: per_expert_input as u32,
                    d: u(q.group_size)?,
                    e: q.bits,
                    h: q.group_size.trailing_zeros(),
                    g: u(w.dims[2])?,
                    t: q.scale_type as u32,
                    ..Params::default()
                },
                (w.dims[1].div_ceil(16), ids.len()),
                4,
            );
        }
        self.dispatch(
            "expert_mv",
            &[&w.buffer, x, ids, y],
            Params {
                a: u(w.dims[0])?,
                b: u(w.dims[1])?,
                c: per_expert_input as u32,
                d: u(w.dims[2])?,
                t: w.dtype as u32,
                ..Params::default()
            },
            (w.dims[1].div_ceil(4), ids.len()),
            4,
        )
    }
}
