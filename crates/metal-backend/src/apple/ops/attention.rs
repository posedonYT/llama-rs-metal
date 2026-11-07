use super::*;
impl Batch<'_> {
    pub fn norm(
        &mut self,
        x: &Buffer,
        w: &Weight,
        y: &Buffer,
        heads: usize,
        stride: usize,
        eps: f32,
    ) -> Result<()> {
        if w.affine.is_some() {
            return Err(shape("RMSNorm weights must be floating point"));
        }
        let n = w.elements();
        let needed_x = heads
            .checked_sub(1)
            .and_then(|h| h.checked_mul(stride))
            .and_then(|v| v.checked_add(n))
            .ok_or_else(|| shape("norm element count"))?;
        let needed_y = count(&[heads, n])?;
        if w.dims.len() != 1
            || heads == 0
            || stride < n
            || x.len() < needed_x
            || y.len() != needed_y
            || !eps.is_finite()
            || eps <= 0.
        {
            return Err(shape("norm"));
        }
        self.dispatch(
            "norm",
            &[x, &w.buffer, y],
            Params {
                n: u(n)?,
                a: u(stride)?,
                x: eps,
                t: w.dtype as u32,
                ..Params::default()
            },
            (heads, 1),
            1,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &mut self,
        x: &Buffer,
        y: &Buffer,
        dim: usize,
        rotary: usize,
        split: bool,
        pos: usize,
        base: f32,
        scale: f32,
        factors: &Buffer,
    ) -> Result<()> {
        self.rope_many(
            x,
            y,
            dim,
            rotary,
            split,
            pos,
            base,
            scale,
            factors,
            x.len() / dim.max(1),
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn rope_many(
        &mut self,
        x: &Buffer,
        y: &Buffer,
        dim: usize,
        rotary: usize,
        split: bool,
        pos: usize,
        base: f32,
        scale: f32,
        factors: &Buffer,
        heads: usize,
    ) -> Result<()> {
        let per_token = count(&[heads, dim])?;
        if per_token == 0
            || x.len() % per_token != 0
            || pos
                .checked_add(x.len() / per_token)
                .is_none_or(|n| n > u32::MAX as usize)
        {
            return Err(shape("RoPE positions"));
        }
        if x.len() != y.len()
            || dim == 0
            || x.len() % dim != 0
            || rotary > dim
            || rotary % 2 != 0
            || rotary == 0
            || base <= 0.
            || !base.is_finite()
            || !scale.is_finite()
            || factors.len() != rotary / 2
            || Rc::ptr_eq(&x.storage, &y.storage)
        {
            return Err(shape("rope needs distinct input/output"));
        }
        self.linear(
            "rope",
            &[x, y, factors],
            Params {
                n: u(x.len())?,
                a: u(dim)?,
                b: u(rotary)?,
                c: split as u32,
                d: u(pos)?,
                e: u(heads)?,
                x: base,
                y: scale,
                ..Params::default()
            },
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &mut self,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        out: &Buffer,
        heads: usize,
        kv_heads: usize,
        dim: usize,
        tokens: usize,
    ) -> Result<()> {
        if tokens == 0 {
            return Err(shape("empty attention"));
        }
        self.attention_many(q, k, v, out, heads, kv_heads, dim, tokens - 1, 1)
    }
    /// Parallel context partitions for one-token decode. `partial` is caller-owned
    /// reusable scratch, with [head, chunk, dimension + maximum + mass] layout.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode(
        &mut self,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        out: &Buffer,
        partial: &Buffer,
        heads: usize,
        kv_heads: usize,
        dim: usize,
        tokens: usize,
    ) -> Result<()> {
        let chunks = tokens.div_ceil(128).min(32);
        if dim == 0
            || dim > 512
            || heads == 0
            || kv_heads == 0
            || heads % kv_heads != 0
            || tokens == 0
            || q.len() != count(&[heads, dim])?
            || out.len() != q.len()
            || k.len() < count(&[tokens, kv_heads, dim])?
            || v.len() < count(&[tokens, kv_heads, dim])?
            || partial.len() != count(&[heads, chunks, dim + 2])?
        {
            return Err(shape("partitioned decode attention"));
        }
        let p = Params {
            a: u(dim)?,
            b: u(heads)?,
            c: u(kv_heads)?,
            d: u(tokens)?,
            e: u(chunks)?,
            f: u(tokens.div_ceil(chunks))?,
            x: (dim as f32).sqrt().recip(),
            ..Params::default()
        };
        self.dispatch(
            "attention_partial",
            &[q, k, v, partial],
            p,
            (heads, chunks),
            1,
        )?;
        self.dispatch("attention_merge", &[partial, out], p, (heads, 1), 1)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn attention_many(
        &mut self,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        out: &Buffer,
        heads: usize,
        kv_heads: usize,
        dim: usize,
        pos: usize,
        batch: usize,
    ) -> Result<()> {
        let tokens = pos
            .checked_add(batch)
            .ok_or_else(|| shape("attention overflow"))?;
        let needed_q = count(&[heads, dim, batch])?;
        let needed_kv = count(&[tokens, kv_heads, dim])?;
        if dim == 0
            || dim > 512
            || heads == 0
            || kv_heads == 0
            || heads % kv_heads != 0
            || tokens == 0
            || batch == 0
            || q.len() != needed_q
            || out.len() != q.len()
            || k.len() < needed_kv
            || v.len() < needed_kv
        {
            return Err(shape("attention"));
        }
        self.dispatch(
            "attention",
            &[q, k, v, out],
            Params {
                n: u(tokens)?,
                a: u(dim)?,
                b: u(heads)?,
                c: u(kv_heads)?,
                d: u(pos)?,
                x: (dim as f32).sqrt().recip(),
                ..Params::default()
            },
            (heads, batch),
            1,
        )
    }
}
