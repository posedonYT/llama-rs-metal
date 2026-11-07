//! Model-specific plans and bounded persistent session state. No llama.cpp runtime dependency.
use metal_backend::{Batch, Buffer, Gpu, Weight};
use model_format::{Checkpoint, L2Normalization, RecurrentHeadLayout, RotaryLayout};
use serde::Serialize;
use std::{
    collections::HashMap,
    rc::Rc,
    sync::atomic::{AtomicBool, Ordering},
};
use thiserror::Error;
#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Format(#[from] model_format::Error),
    #[error(transparent)]
    Metal(#[from] metal_backend::Error),
    #[error("unsupported model: {0}")]
    Unsupported(String),
    #[error("invalid model: {0}")]
    Invalid(String),
    #[error(
        "memory budget: need {required} bytes, available {available}; reduce context or use a smaller quantized model"
    )]
    Memory { required: usize, available: usize },
    #[error("context capacity {capacity} exceeded by {requested} tokens")]
    Context { capacity: usize, requested: usize },
    #[error("generation cancelled")]
    Cancelled,
    #[error("session must be reset after a GPU failure")]
    Poisoned,
}
pub type Result<T> = std::result::Result<T, Error>;
fn bad(s: impl Into<String>) -> Error {
    Error::Invalid(s.into())
}
pub use model_format::ModelConfig as Config;
#[derive(Clone)]
struct Dense {
    gate: Weight,
    up: Weight,
    down: Weight,
}
#[derive(Clone)]
// Plans are allocated once per layer; inline weights avoid decode-time indirection.
#[allow(clippy::large_enum_variant)]
enum FeedForward {
    Dense(Dense),
    Moe {
        router: Weight,
        gate: Weight,
        up: Weight,
        down: Weight,
        shared: Dense,
        shared_gate: Weight,
    },
}
#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
enum Mixer {
    Attention {
        q: Weight,
        k: Weight,
        v: Weight,
        out: Weight,
        qn: Option<Weight>,
        kn: Option<Weight>,
    },
    Linear {
        qkv: Option<Weight>,
        z: Option<Weight>,
        legacy: Option<Weight>,
        alpha: Option<Weight>,
        beta: Option<Weight>,
        ba: Option<Weight>,
        conv: Weight,
        norm: Weight,
        out: Weight,
        aw: Buffer,
        dt: Buffer,
    },
}
#[derive(Clone)]
struct Layer {
    norm: Weight,
    post: Weight,
    mixer: Mixer,
    ffn: FeedForward,
}
pub struct Model {
    pub config: Config,
    pub file: Rc<Checkpoint>,
    layers: Vec<Layer>,
    embedding: Weight,
    output: Weight,
    output_norm: Weight,
    factors: Buffer,
    pub weight_bytes: usize,
    pub scratch_elements: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct MemoryEstimate {
    pub weights: usize,
    pub kv: usize,
    pub recurrent: usize,
    pub scratch: usize,
    pub session: usize,
}
/// CPU diagnostic copy of persistent state. Not used by the execution hot path.
/// KV uses [token, head, dimension], convolution [channel, history], and
/// recurrent state [value_head, value_dimension, key_dimension].
#[derive(Debug, Serialize)]
pub struct StateSnapshot {
    pub layer: usize,
    pub kind: &'static str,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}
enum State {
    Attention { key: Buffer, value: Buffer },
    Linear { conv: Buffer, state: Buffer },
}
pub struct Session {
    model: Rc<Model>,
    states: Vec<State>,
    scratch: Workspace,
    pub context: usize,
    pub tokens: Vec<u32>,
    pub gpu_seconds: f64,
    poisoned: bool,
    pub last_logits: Vec<f32>,
    pub greedy_token: Option<u32>,
}

mod execute;
mod loader;
mod memory;
mod session;
mod workspace;
use workspace::{PREFILL_CHUNK, Slot, Workspace};
