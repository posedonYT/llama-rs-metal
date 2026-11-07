use super::*;
impl Drop for Lease {
    fn drop(&mut self) {
        self.backend
            .reserved
            .set(self.backend.reserved.get() - self.bytes)
    }
}
impl Backend {
    pub(super) fn reserve(self: &Rc<Self>, bytes: usize) -> Result<Lease> {
        let available = self.limit.saturating_sub(self.reserved.get());
        if bytes > available {
            return Err(model_runtime::Error::Memory {
                required: bytes,
                available,
            }
            .into());
        }
        self.reserved.set(self.reserved.get() + bytes);
        Ok(Lease {
            backend: self.clone(),
            bytes,
        })
    }
}
impl Engine {
    pub fn new(options: EngineOptions) -> Result<Self> {
        let gpu = Gpu::new()?;
        let default = gpu.recommended_working_set / 5 * 4;
        let limit = options
            .memory_limit
            .unwrap_or(default)
            .min(gpu.recommended_working_set);
        if limit == 0 {
            return Err(Error::Options("zero memory budget".into()));
        }
        Ok(Self {
            backend: Rc::new(Backend {
                gpu: RefCell::new(gpu),
                reserved: Cell::new(0),
                limit,
            }),
        })
    }
    pub fn device_name(&self) -> String {
        self.backend.gpu.borrow().name.clone()
    }
    pub fn allocated_bytes(&self) -> usize {
        self.backend.gpu.borrow().allocated_bytes()
    }
    pub fn memory_limit(&self) -> usize {
        self.backend.limit
    }
    /// Model files must remain unmodified while any model or session is alive.
    pub fn load(&self, path: impl AsRef<Path>) -> Result<Model> {
        self.load_with_options(path, LoadOptions::default())
    }
    pub fn load_with_options(&self, path: impl AsRef<Path>, options: LoadOptions) -> Result<Model> {
        let file = Rc::new(Checkpoint::open_with_float_storage(
            path,
            options.float_weights_f32,
        )?);
        let tokenizer = Tokenizer::from_checkpoint(&file)?;
        if tokenizer.vocab_size() > file.config.vocab {
            return Err(Error::Tokenizer(
                "tokenizer exceeds embedding vocabulary".into(),
            ));
        }
        let bytes =
            model_runtime::Model::weight_allocation_bound(&self.backend.gpu.borrow(), &file)?;
        let lease = self.backend.reserve(bytes)?;
        let template = file.chat_template.clone();
        let runtime = Rc::new(model_runtime::Model::load(
            &mut self.backend.gpu.borrow_mut(),
            file,
            bytes,
        )?);
        Ok(Model {
            inner: Rc::new(ModelData {
                runtime,
                tokenizer,
                template,
                backend: self.backend.clone(),
                _lease: lease,
            }),
        })
    }
}
