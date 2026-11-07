//! Explicit scratch lifetimes. Mixers and FFNs execute sequentially and reuse
//! temporary slots; their semantic aliases document that reuse at call sites.
use super::*;
pub(crate) const PREFILL_CHUNK: usize = 32;
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum Slot {
    Hidden,
    Normalized,
    Branch,
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    FfnGate,
    FfnUp,
    Greedy,
    Logits,
}
impl Slot {
    pub(crate) const QUERY_PROJECTION: Self = Self::A;
    pub(crate) const QUERY_NORMALIZED: Self = Self::B;
    pub(crate) const QUERY_ROTATED: Self = Self::C;
    pub(crate) const KEY_PROJECTION: Self = Self::D;
    pub(crate) const KEY_NORMALIZED: Self = Self::E;
    pub(crate) const KEY_ROTATED: Self = Self::F;
    pub(crate) const VALUE_PROJECTION: Self = Self::G;
    pub(crate) const ATTENTION_OUTPUT: Self = Self::H;
    pub(crate) const ATTENTION_GATE: Self = Self::I;
    pub(crate) const LINEAR_QKV: Self = Self::A;
    pub(crate) const LINEAR_GATE: Self = Self::B;
    pub(crate) const LINEAR_ALPHA: Self = Self::C;
    pub(crate) const LINEAR_BETA: Self = Self::D;
    pub(crate) const LINEAR_CONV: Self = Self::E;
    pub(crate) const LINEAR_DELTA: Self = Self::F;
    pub(crate) const LINEAR_NORMALIZED: Self = Self::G;
    pub(crate) const LINEAR_MIXED: Self = Self::H;
    pub(crate) const ROUTER: Self = Self::A;
    pub(crate) const EXPERT_IDS: Self = Self::B;
    pub(crate) const EXPERT_PROBABILITIES: Self = Self::C;
    pub(crate) const EXPERT_GATE: Self = Self::D;
    pub(crate) const EXPERT_UP: Self = Self::E;
    pub(crate) const EXPERT_OUTPUT: Self = Self::F;
    pub(crate) const SHARED_OUTPUT: Self = Self::G;
    pub(crate) const SHARED_GATE: Self = Self::H;
}
pub(crate) struct Workspace {
    buffers: Vec<Buffer>,
}
impl Workspace {
    pub(crate) fn bytes(width: usize, vocab: usize) -> Result<usize> {
        width
            .checked_mul(14 * PREFILL_CHUNK)
            .and_then(|n| n.checked_add(vocab))
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| bad("scratch size overflow"))
    }
    pub(crate) fn new(gpu: &mut Gpu, width: usize, vocab: usize) -> Result<Self> {
        let mut buffers = Vec::new();
        for _ in 0..14 {
            buffers.push(gpu.zeros(width * PREFILL_CHUNK)?)
        }
        buffers.push(gpu.zeros(1)?);
        buffers.push(gpu.zeros(vocab)?);
        Ok(Self { buffers })
    }
    pub(crate) fn view(&self, slot: Slot, len: usize) -> metal_backend::Result<Buffer> {
        self.buffers[slot as usize].view(0, len)
    }
}
