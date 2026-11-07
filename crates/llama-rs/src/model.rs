use super::*;
impl Model {
    pub fn checkpoint_format(&self) -> CheckpointFormat {
        self.inner.runtime.file.format
    }
    pub fn provenance(&self) -> &serde_json::Value {
        &self.inner.runtime.file.provenance
    }
    pub fn config(&self) -> &Config {
        &self.inner.runtime.config
    }
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.inner.tokenizer
    }
    pub fn memory(&self, context: usize) -> Result<MemoryEstimate> {
        Ok(self.inner.runtime.memory(context)?)
    }
    pub fn session(&self, options: SessionOptions) -> Result<Session> {
        let context = if options.context == 0 {
            8192.min(self.config().context)
        } else {
            options.context
        };
        let mem = self.memory(context)?;
        let lease = self.inner.backend.reserve(mem.session)?;
        let runtime = model_runtime::Session::new(
            &mut self.inner.backend.gpu.borrow_mut(),
            self.inner.runtime.clone(),
            context,
            mem.session,
        )?;
        Ok(Session {
            runtime,
            model: self.clone(),
            cancel: Arc::new(AtomicBool::new(false)),
            _lease: lease,
        })
    }
    pub fn render_chat(&self, messages: &[Message], add_generation_prompt: bool) -> Result<String> {
        for m in messages {
            if !["system", "user", "assistant", "tool"].contains(&m.role.as_str()) {
                return Err(Error::Template(format!("unknown role {}", m.role)));
            }
        }
        let fallback = if self.config().hybrid() {
            "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        } else if self
            .tokenizer()
            .token(self.tokenizer().bos.unwrap_or(u32::MAX))
            == Some("<|begin_of_text|>")
        {
            "{{ bos_token }}{% for message in messages %}<|start_header_id|>{{ message.role }}<|end_header_id|>\n\n{{ message.content }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
        } else if self.inner.template.is_none() {
            // Llama 2 dialogue formatting has role-dependent delimiters and system placement.
            return crate::chat::render_llama2(messages, add_generation_prompt);
        } else {
            ""
        };
        let source = self.inner.template.as_deref().unwrap_or(fallback);
        let bos = self
            .tokenizer()
            .bos
            .and_then(|id| self.tokenizer().token(id))
            .unwrap_or("");
        let eos = self
            .tokenizer()
            .eos
            .first()
            .and_then(|id| self.tokenizer().token(*id))
            .unwrap_or("");
        crate::chat::render(source, messages, add_generation_prompt, bos, eos)
    }
}
