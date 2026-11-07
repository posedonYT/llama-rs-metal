use super::*;
impl Batch<'_> {
    pub fn argmax(&mut self, x: &Buffer, result: &Buffer) -> Result<()> {
        if x.is_empty() || result.len() != 1 {
            return Err(shape("argmax"));
        }
        self.dispatch(
            "argmax",
            &[x, result],
            Params {
                n: u(x.len())?,
                ..Params::default()
            },
            (1, 1),
            8,
        )
    }
}
impl Batch<'_> {
    /// Select experts entirely on the GPU. The output buffers contain IDs and probabilities.
    pub fn route(
        &mut self,
        logits: &Buffer,
        ids: &Buffer,
        weights: &Buffer,
        normalize: bool,
    ) -> Result<()> {
        if ids.len() != weights.len() || ids.is_empty() || ids.len() > logits.len() {
            return Err(shape("routing"));
        }
        self.dispatch(
            "route",
            &[logits, ids, weights],
            Params {
                n: u(logits.len())?,
                a: u(ids.len())?,
                b: normalize as u32,
                ..Params::default()
            },
            (1, 1),
            1,
        )
    }
}
