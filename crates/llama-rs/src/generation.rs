use super::*;
impl Generation<'_> {
    fn finish(&mut self, reason: &'static str) -> Option<Result<GenerationEvent>> {
        self.done = true;
        Some(Ok(GenerationEvent::Finished {
            reason,
            tokens: self.count,
            text: self.decoder.finish(),
        }))
    }
}
impl Iterator for Generation<'_> {
    type Item = Result<GenerationEvent>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        if self.session.cancel.load(Ordering::Relaxed) {
            return self.finish("cancelled");
        }
        if self.count >= self.options.max_tokens {
            return self.finish("length");
        }
        if let Some(t) = self.pending.take() {
            if self.session.tokens().len() >= self.session.runtime.context {
                return self.finish("context");
            }
            if let Err(e) = self.session.decode(t) {
                self.done = true;
                return Some(Err(e));
            }
        }
        let sampled = if self.options.sampling.temperature == 0.
            && self.options.sampling.repeat_penalty == 1.
        {
            self.session
                .runtime
                .greedy_token
                .filter(|id| (*id as usize) < self.session.model.config().vocab)
                .ok_or_else(|| Error::Options("invalid GPU logits".into()))
        } else {
            self.options.sampling.sample(
                self.session.logits(),
                self.session.tokens(),
                &mut self.rng,
            )
        };
        let id = match sampled {
            Ok(id) => id,
            Err(e) => {
                self.done = true;
                return Some(Err(e));
            }
        };
        if self.session.model.tokenizer().eos.contains(&id) {
            return self.finish("eos");
        }
        let bytes = match self.session.model.tokenizer().decode_piece(id, false) {
            Ok(b) => b,
            Err(e) => {
                self.done = true;
                return Some(Err(e));
            }
        };
        let text = self.decoder.push(&bytes);
        self.pending = Some(id);
        self.count += 1;
        Some(Ok(GenerationEvent::Token { id, text }))
    }
}
