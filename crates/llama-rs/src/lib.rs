//! Standalone Apple Silicon inference. Handles are intentionally thread-confined:
//! each Engine owns one ordered Metal queue; sessions have independent persistent state.
mod chat;
mod sampling;
mod tokenizer;
use metal_backend::Gpu;
use model_format::{Checkpoint, CheckpointFormat};
pub use model_runtime::{Config, MemoryEstimate, StateSnapshot};
pub use sampling::Sampling;
use serde::Serialize;
use std::{
    cell::{Cell, RefCell},
    path::Path,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use thiserror::Error;
pub use tokenizer::{TextDecoder, Tokenizer};
#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Format(#[from] model_format::Error),
    #[error(transparent)]
    Runtime(#[from] model_runtime::Error),
    #[error(transparent)]
    Metal(#[from] metal_backend::Error),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("chat template: {0}")]
    Template(String),
    #[error("invalid generation options: {0}")]
    Options(String),
}
pub type Result<T> = std::result::Result<T, Error>;
#[derive(Clone, Default)]
pub struct EngineOptions {
    pub memory_limit: Option<usize>,
}
struct Backend {
    gpu: RefCell<Gpu>,
    reserved: Cell<usize>,
    limit: usize,
}
struct Lease {
    backend: Rc<Backend>,
    bytes: usize,
}

/// Loading preserves packed codes. F32 storage is an explicit benchmark option.
#[derive(Clone, Copy, Debug, Default)]
pub struct LoadOptions {
    pub float_weights_f32: bool,
}
pub struct Engine {
    backend: Rc<Backend>,
}

struct ModelData {
    runtime: Rc<model_runtime::Model>,
    tokenizer: Tokenizer,
    template: Option<String>,
    backend: Rc<Backend>,
    _lease: Lease,
}
#[derive(Clone)]
pub struct Model {
    inner: Rc<ModelData>,
}
#[derive(Clone, Debug, Default)]
pub struct SessionOptions {
    pub context: usize,
}
#[derive(Clone, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}
impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

pub struct Session {
    runtime: model_runtime::Session,
    model: Model,
    cancel: Arc<AtomicBool>,
    _lease: Lease,
}

#[derive(Clone, Debug)]
pub struct GenerationOptions {
    pub max_tokens: usize,
    pub sampling: Sampling,
}
impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            sampling: Sampling::default(),
        }
    }
}
#[derive(Debug, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum GenerationEvent {
    Token {
        id: u32,
        text: String,
    },
    Finished {
        reason: &'static str,
        tokens: usize,
        text: String,
    },
}
pub struct Generation<'a> {
    session: &'a mut Session,
    options: GenerationOptions,
    count: usize,
    pending: Option<u32>,
    decoder: TextDecoder,
    rng: sampling::Rng,
    done: bool,
}

mod engine;
mod generation;
mod model;
mod session;
