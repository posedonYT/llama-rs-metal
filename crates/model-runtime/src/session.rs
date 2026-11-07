use super::*;
impl Session {
    pub fn new(gpu: &mut Gpu, model: Rc<Model>, context: usize, budget: usize) -> Result<Self> {
        let estimate = model.memory(context)?;
        if estimate.session > budget {
            return Err(Error::Memory {
                required: estimate.session,
                available: budget,
            });
        }
        let c = &model.config;
        let mut states = Vec::new();
        for l in &model.layers {
            states.push(match l.mixer {
                Mixer::Attention { .. } => State::Attention {
                    key: gpu.zeros(context * c.kv_heads * c.head_dim)?,
                    value: gpu.zeros(context * c.kv_heads * c.head_dim)?,
                },
                Mixer::Linear { .. } => State::Linear {
                    conv: gpu.zeros(
                        (2 * c.linear_key_heads * c.linear_key_dim
                            + c.linear_value_heads * c.linear_value_dim)
                            * (c.conv_kernel - 1),
                    )?,
                    state: gpu
                        .zeros(c.linear_value_heads * c.linear_value_dim * c.linear_key_dim)?,
                },
            })
        }
        let scratch = Workspace::new(gpu, model.scratch_elements, c.vocab)?;
        Ok(Self {
            model,
            states,
            scratch,
            context,
            tokens: Vec::new(),
            gpu_seconds: 0.,
            poisoned: false,
            last_logits: Vec::new(),
            greedy_token: None,
        })
    }
    /// Synchronization is guaranteed by the exclusive GPU borrow. This allocates
    /// host copies for debugging and is deliberately outside benchmark timing.
    pub fn state_snapshot(&self, gpu: &mut Gpu) -> Result<Vec<StateSnapshot>> {
        let c = &self.model.config;
        let mut result = Vec::new();
        for (layer, state) in self.states.iter().enumerate() {
            let mut save = |kind, shape: Vec<usize>, buffer: &Buffer| -> Result<()> {
                let count = shape.iter().product();
                let values = if count == 0 {
                    Vec::new()
                } else {
                    gpu.read(&buffer.view(0, count)?)?
                };
                result.push(StateSnapshot {
                    layer,
                    kind,
                    shape,
                    values,
                });
                Ok(())
            };
            match state {
                State::Attention { key, value } => {
                    let shape = vec![self.tokens.len(), c.kv_heads, c.head_dim];
                    save("key", shape.clone(), key)?;
                    save("value", shape, value)?;
                }
                State::Linear { conv, state } => {
                    let channels = 2 * c.linear_key_heads * c.linear_key_dim
                        + c.linear_value_heads * c.linear_value_dim;
                    save("convolution", vec![channels, c.conv_kernel - 1], conv)?;
                    save(
                        "recurrent",
                        vec![c.linear_value_heads, c.linear_value_dim, c.linear_key_dim],
                        state,
                    )?;
                }
            }
        }
        Ok(result)
    }
    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<()> {
        for state in &self.states {
            match state {
                State::Attention { key, value } => {
                    gpu.clear(key)?;
                    gpu.clear(value)?
                }
                State::Linear { conv, state } => {
                    gpu.clear(conv)?;
                    gpu.clear(state)?
                }
            }
        }
        self.tokens.clear();
        self.last_logits.clear();
        self.greedy_token = None;
        self.gpu_seconds = 0.;
        self.poisoned = false;
        Ok(())
    }
    /// Append tokens in chunks of up to 32, with layer-wise batched projections.
    pub fn prefill(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        cancel: &AtomicBool,
    ) -> Result<&[f32]> {
        if self.poisoned {
            return Err(Error::Poisoned);
        }
        if self
            .tokens
            .len()
            .checked_add(tokens.len())
            .is_none_or(|n| n > self.context)
        {
            return Err(Error::Context {
                capacity: self.context,
                requested: self.tokens.len().saturating_add(tokens.len()),
            });
        }
        if tokens
            .iter()
            .any(|t| *t as usize >= self.model.config.vocab)
        {
            return Err(bad("token outside vocabulary"));
        }
        for (index, chunk) in tokens.chunks(PREFILL_CHUNK).enumerate() {
            let final_chunk = (index + 1) * PREFILL_CHUNK >= tokens.len();
            if cancel.load(Ordering::Relaxed) {
                return Err(Error::Cancelled);
            }
            let base = self.tokens.len();
            let result = gpu
                .with_batch(|b| self.encode(b, chunk, base, final_chunk))
                .map(|(_, seconds)| seconds);
            match result {
                Ok(seconds) => {
                    self.gpu_seconds += seconds;
                    self.tokens.extend_from_slice(chunk);
                    if final_chunk {
                        self.last_logits =
                            gpu.read(&self.scratch.view(Slot::Logits, self.model.config.vocab)?)?;
                        self.greedy_token =
                            Some(gpu.read(&self.scratch.view(Slot::Greedy, 1)?)?[0].to_bits());
                    } else {
                        // Cancellation can leave a valid state prefix without a
                        // host prediction. Never expose logits for an older prefix.
                        self.last_logits.clear();
                        self.greedy_token = None;
                    }
                }
                Err(e) => {
                    self.poisoned = true;
                    return Err(e);
                }
            }
        }

        Ok(&self.last_logits)
    }
    pub fn decode(&mut self, gpu: &mut Gpu, token: u32, cancel: &AtomicBool) -> Result<&[f32]> {
        self.prefill(gpu, &[token], cancel)
    }
}
