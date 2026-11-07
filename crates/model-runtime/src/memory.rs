use super::*;
impl Model {
    pub fn memory(&self, context: usize) -> Result<MemoryEstimate> {
        let c = &self.config;
        if context == 0 || context > c.context {
            return Err(Error::Context {
                capacity: c.context,
                requested: context,
            });
        }
        let mut kv = 0usize;
        let mut recurrent = 0usize;
        for l in &self.layers {
            match &l.mixer {
                Mixer::Attention { .. } => {
                    kv = kv
                        .checked_add(
                            context
                                .checked_mul(c.kv_heads * c.head_dim * 8)
                                .ok_or_else(|| bad("KV size overflow"))?,
                        )
                        .ok_or_else(|| bad("KV size overflow"))?
                }
                Mixer::Linear { .. } => {
                    recurrent = recurrent
                        .checked_add(
                            (c.linear_value_heads * c.linear_value_dim * c.linear_key_dim
                                + (2 * c.linear_key_heads * c.linear_key_dim
                                    + c.linear_value_heads * c.linear_value_dim)
                                    * (c.conv_kernel - 1))
                                * 4,
                        )
                        .ok_or_else(|| bad("state size overflow"))?
                }
            }
        }
        let scratch = Workspace::bytes(self.scratch_elements, c.vocab)?;
        let total = kv
            .checked_add(recurrent)
            .and_then(|n| n.checked_add(scratch))
            .ok_or_else(|| bad("session size overflow"))?;
        Ok(MemoryEstimate {
            weights: self.weight_bytes,
            kv,
            recurrent,
            scratch,
            session: total,
        })
    }
}
