use super::*;
impl Batch<'_> {
    pub fn conv(&mut self, x: &Buffer, w: &Weight, state: &Buffer, out: &Buffer) -> Result<()> {
        if w.affine.is_some() {
            return Err(shape("convolution weights must be floating point"));
        }
        if w.dims.len() != 2
            || w.dims[0] == 0
            || w.dims[0] > 64
            || x.len() % w.dims[1] != 0
            || out.len() != x.len()
            || state.len() != w.dims[1] * (w.dims[0] - 1)
        {
            return Err(shape("conv"));
        }
        self.linear(
            "conv",
            &[x, &w.buffer, state, out],
            Params {
                n: u(w.dims[1])?,
                a: u(w.dims[0])?,
                f: u(x.len() / w.dims[1])?,
                t: w.dtype as u32,
                ..Params::default()
            },
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn delta(
        &mut self,
        qkv: &Buffer,
        alpha: &Buffer,
        beta: &Buffer,
        aw: &Buffer,
        dt: &Buffer,
        state: &Buffer,
        out: &Buffer,
        dk: usize,
        kh: usize,
        vh: usize,
        dv: usize,
        eps: f32,
        interleaved_heads: bool,
        add_epsilon: bool,
    ) -> Result<()> {
        let value_width = count(&[vh, dv])?;
        let key_width = count(&[kh, dk])?;
        let width = key_width
            .checked_mul(2)
            .and_then(|v| v.checked_add(value_width))
            .ok_or_else(|| shape("delta width"))?;
        let tokens = out.len() / value_width.max(1);
        let expected_qkv = count(&[width, tokens])?;
        let expected_heads = count(&[vh, tokens])?;
        let expected_state = count(&[value_width, dk])?;
        let expected_out = count(&[value_width, tokens])?;
        if tokens == 0
            || dk > 512
            || dk == 0
            || kh == 0
            || vh == 0
            || dv == 0
            || vh % kh != 0
            || qkv.len() != expected_qkv
            || alpha.len() != expected_heads
            || beta.len() != expected_heads
            || aw.len() != vh
            || dt.len() != vh
            || state.len() != expected_state
            || out.len() != expected_out
        {
            return Err(shape("delta"));
        }
        self.dispatch(
            "delta",
            &[qkv, alpha, beta, aw, dt, state, out],
            Params {
                a: u(dk)?,
                b: u(kh)?,
                c: u(dv)?,
                d: u(vh / kh)?,
                e: interleaved_heads as u32,
                f: u(tokens)?,
                g: u(vh)?,
                x: eps,
                y: if add_epsilon { 1. } else { 0. },
                ..Params::default()
            },
            (vh, dv),
            1,
        )
    }
}
