use crate::{Error, Gguf, Result};
use serde::{Deserialize, Serialize};
fn bad(s: impl Into<String>) -> Error {
    Error::Invalid(s.into())
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RotaryLayout {
    Adjacent,
    SplitHalf,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecurrentHeadLayout {
    Tiled,
    Interleaved,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum L2Normalization {
    Clamp,
    AddEpsilon,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architecture: String,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary: usize,
    pub vocab: usize,
    /// MLX declares tying explicitly; GGUF convention infers it from an absent head.
    pub tied_embeddings: Option<bool>,
    pub context: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub rope_scale: f32,
    pub linear_key_dim: usize,
    pub linear_key_heads: usize,
    pub linear_value_heads: usize,
    pub linear_value_dim: usize,
    pub conv_kernel: usize,
    pub experts_used: usize,
    pub rotary_layout: RotaryLayout,
    pub recurrent_head_layout: RecurrentHeadLayout,
    pub l2_normalization: L2Normalization,
    pub linear_norm_eps: f32,
    pub normalize_expert_weights: bool,
}
impl ModelConfig {
    pub fn from_gguf(f: &Gguf) -> Result<Self> {
        let arch = f
            .str("general.architecture")
            .ok_or_else(|| bad("general.architecture"))?
            .to_string();
        if !["llama", "qwen35", "qwen35moe", "qwen3next"].contains(&arch.as_str()) {
            return Err(Error::Unsupported(arch));
        }
        let key = |s: &str| format!("{arch}.{s}");
        let req = |s: &str| -> Result<usize> {
            let n = f
                .uint(&key(s))
                .ok_or_else(|| bad(format!("missing {}", key(s))))?;
            if n == 0 || n > 1_000_000 {
                return Err(bad(format!("invalid {}", key(s))));
            }
            Ok(n)
        };
        let hidden = req("embedding_length")?;
        let layers = req("block_count")?;
        let heads = req("attention.head_count")?;
        if layers > 1024 || heads > 1024 {
            return Err(bad("layer/head limit"));
        }
        let kv_heads = req("attention.head_count_kv").unwrap_or(heads);
        let head_dim = f
            .uint(&key("attention.key_length"))
            .unwrap_or(hidden / heads);
        if head_dim == 0 || head_dim > 512 || kv_heads == 0 || heads % kv_heads != 0 {
            return Err(bad("attention dimensions"));
        }
        if f.uint(&key("attention.value_length")).unwrap_or(head_dim) != head_dim {
            return Err(Error::Unsupported("different K/V head dimensions".into()));
        }
        let rotary = f.uint(&key("rope.dimension_count")).unwrap_or(head_dim);
        if rotary == 0 || rotary > head_dim || rotary % 2 != 0 {
            return Err(bad("rotary dimension"));
        }
        let vocab = f
            .tensor("token_embd.weight")?
            .dims
            .get(1)
            .copied()
            .ok_or_else(|| bad("embedding dimensions"))?;
        let hybrid = arch != "llama";
        let dk = if hybrid { req("ssm.state_size")? } else { 0 };
        let kh = if hybrid { req("ssm.group_count")? } else { 0 };
        let vh = if hybrid {
            req("ssm.time_step_rank")?
        } else {
            0
        };
        let inner = if hybrid { req("ssm.inner_size")? } else { 0 };
        if hybrid && (vh % kh != 0 || inner % vh != 0 || dk > 512 || inner / vh > 512) {
            return Err(bad("linear attention dimensions"));
        }
        let eps = f
            .float(&key("attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let rope_base = f.float(&key("rope.freq_base")).unwrap_or(10000.);
        let scaling = f.str(&key("rope.scaling.type")).unwrap_or("none");
        if !["none", "linear", "llama3"].contains(&scaling) {
            return Err(Error::Unsupported(format!("RoPE scaling {scaling}")));
        }
        let factor = f.float(&key("rope.scaling.factor")).unwrap_or(1.);
        if !eps.is_finite()
            || eps <= 0.
            || !rope_base.is_finite()
            || rope_base <= 0.
            || !factor.is_finite()
            || factor <= 0.
        {
            return Err(bad("normalization/RoPE parameters"));
        }
        Ok(Self {
            rotary_layout: if hybrid {
                RotaryLayout::SplitHalf
            } else {
                RotaryLayout::Adjacent
            },
            recurrent_head_layout: if arch == "qwen3next" {
                RecurrentHeadLayout::Interleaved
            } else {
                RecurrentHeadLayout::Tiled
            },
            // Current GGUF Qwen graphs use rms_norm(eps / head_dim)
            // followed by 1/sqrt(head_dim): sqrt(sum(x*x) + eps).
            l2_normalization: if hybrid {
                L2Normalization::AddEpsilon
            } else {
                L2Normalization::Clamp
            },
            linear_norm_eps: eps,
            normalize_expert_weights: true,
            architecture: arch.clone(),
            hidden,
            layers,
            heads,
            kv_heads,
            head_dim,
            rotary,
            vocab,
            tied_embeddings: None,
            context: req("context_length")?,
            eps,
            rope_base,
            rope_scale: if scaling == "linear" {
                factor.recip()
            } else {
                1.
            },
            linear_key_dim: dk,
            linear_key_heads: kh,
            linear_value_heads: vh,
            linear_value_dim: if hybrid { inner / vh } else { 0 },
            conv_kernel: if hybrid { req("ssm.conv_kernel")? } else { 0 },
            experts_used: if hybrid {
                f.uint(&key("expert_used_count")).unwrap_or(0)
            } else {
                0
            },
        })
    }
    pub fn hybrid(&self) -> bool {
        self.architecture != "llama"
    }
}
