use super::*;
impl Session {
    pub fn cancellation_handle(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }
    /// Copy persistent state for numerical diagnostics. This is not a checkpoint
    /// restore API and is not part of normal generation or timing.
    pub fn state_snapshot(&self) -> Result<Vec<StateSnapshot>> {
        Ok(self
            .runtime
            .state_snapshot(&mut self.model.inner.backend.gpu.borrow_mut())?)
    }
    pub fn reset(&mut self) -> Result<()> {
        self.runtime
            .reset(&mut self.model.inner.backend.gpu.borrow_mut())?;
        self.cancel.store(false, Ordering::Relaxed);
        Ok(())
    }
    pub fn context(&self) -> usize {
        self.runtime.context
    }
    pub fn tokens(&self) -> &[u32] {
        &self.runtime.tokens
    }
    pub fn logits(&self) -> &[f32] {
        &self.runtime.last_logits
    }
    pub fn gpu_seconds(&self) -> f64 {
        self.runtime.gpu_seconds
    }
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        self.runtime.prefill(
            &mut self.model.inner.backend.gpu.borrow_mut(),
            tokens,
            &self.cancel,
        )?;
        Ok(&self.runtime.last_logits)
    }
    pub fn decode(&mut self, token: u32) -> Result<&[f32]> {
        self.runtime.decode(
            &mut self.model.inner.backend.gpu.borrow_mut(),
            token,
            &self.cancel,
        )?;
        Ok(&self.runtime.last_logits)
    }
    pub fn synchronize_prompt(&mut self, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() {
            return Err(Error::Options(
                "prompt must contain at least one token".into(),
            ));
        }
        if tokens.len() > self.context() {
            return Err(model_runtime::Error::Context {
                capacity: self.context(),
                requested: tokens.len(),
            }
            .into());
        }
        if tokens
            .iter()
            .any(|id| (*id as usize) >= self.model.config().vocab)
        {
            return Err(Error::Options("token outside vocabulary".into()));
        }
        if !tokens.starts_with(&self.runtime.tokens) {
            self.reset()?
        }
        let n = self.runtime.tokens.len();
        self.prefill(&tokens[n..])?;
        if self.logits().is_empty() {
            return Err(Error::Options(
                "prompt must contain at least one token".into(),
            ));
        }
        Ok(())
    }
    pub fn generate<'a>(
        &'a mut self,
        prompt: &str,
        options: GenerationOptions,
    ) -> Result<Generation<'a>> {
        options.sampling.validate()?;
        if options.max_tokens == 0 {
            return Err(Error::Options("max_tokens must be positive".into()));
        }
        if self.cancel.load(Ordering::Relaxed) {
            return Err(model_runtime::Error::Cancelled.into());
        }
        // Chat templates may already contain the initial BOS marker.
        let explicit_bos = self
            .model
            .tokenizer()
            .bos
            .and_then(|id| self.model.tokenizer().token(id))
            .is_some_and(|bos| !bos.is_empty() && prompt.starts_with(bos));
        let tokens = self.model.tokenizer().encode(prompt, !explicit_bos, true)?;
        self.synchronize_prompt(&tokens)?;
        let seed = options.sampling.seed;
        Ok(Generation {
            session: self,
            options,
            count: 0,
            pending: None,
            decoder: TextDecoder::default(),
            rng: sampling::Rng::new(seed),
            done: false,
        })
    }
}
