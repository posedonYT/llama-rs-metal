//! Hugging Face/MLX config -> execution semantics, independent of tensor storage.
use crate::{Error, L2Normalization, ModelConfig, RecurrentHeadLayout, Result, RotaryLayout};
use serde_json::Value;
fn bad(s: impl Into<String>) -> Error {
    Error::Invalid(s.into())
}
fn number(v: &Value, key: &str, default: Option<usize>) -> Result<usize> {
    let n = v
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .or(default)
        .ok_or_else(|| bad(format!("missing config.{key}")))?;
    if n > 1_000_000 {
        return Err(bad(format!("config.{key} exceeds supported limit")));
    }
    Ok(n)
}
fn real(v: &Value, key: &str, default: f32) -> Result<f32> {
    let n = v.get(key).and_then(Value::as_f64).unwrap_or(default as f64) as f32;
    if !n.is_finite() {
        return Err(bad(format!("nonfinite config.{key}")));
    }
    Ok(n)
}
pub(super) fn parse(root: &Value) -> Result<(ModelConfig, Vec<f32>)> {
    let text = root.get("text_config").unwrap_or(root);
    let kind = root
        .get("model_type")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("config.model_type"))?;
    let architecture = match kind {
        "llama" => "llama",
        "qwen3_5" | "qwen3_5_text" => {
            if number(text, "num_experts", Some(0))? > 0 {
                "qwen35moe"
            } else {
                "qwen35"
            }
        }
        "qwen3_5_moe" | "qwen3_5_moe_text" => "qwen35moe",
        "qwen3_next" => "qwen3next",
        _ => return Err(Error::Unsupported(format!("MLX model_type {kind}"))),
    };
    let hybrid = architecture != "llama";
    for flag in ["attention_bias", "mlp_bias"] {
        if text.get(flag).and_then(Value::as_bool) == Some(true) {
            return Err(Error::Unsupported(format!("{flag}=true")));
        }
    }
    if text
        .get("sliding_window")
        .and_then(Value::as_u64)
        .is_some_and(|v| v > 0)
    {
        return Err(Error::Unsupported("sliding-window attention".into()));
    }
    if text
        .get("hidden_act")
        .and_then(Value::as_str)
        .is_some_and(|s| s != "silu")
    {
        return Err(Error::Unsupported("activation other than SiLU".into()));
    }
    let hidden = number(text, "hidden_size", None)?;
    let heads = number(text, "num_attention_heads", None)?;
    if hidden == 0 || heads == 0 || heads > 1024 {
        return Err(bad("attention dimensions"));
    }
    let head_dim = number(text, "head_dim", Some(hidden / heads))?;
    let kv_heads = number(text, "num_key_value_heads", Some(heads))?;
    let layers = number(text, "num_hidden_layers", None)?;
    let vocab = number(text, "vocab_size", None)?;
    let rope = text
        .get("rope_parameters")
        .or_else(|| text.get("rope_scaling"))
        .filter(|v| !v.is_null());
    let rope_base = real(
        rope.unwrap_or(text),
        "rope_theta",
        real(text, "rope_theta", 10000.)?,
    )?;
    let rotary = (head_dim as f32
        * real(
            rope.unwrap_or(text),
            "partial_rotary_factor",
            real(
                text,
                "partial_rotary_factor",
                if hybrid { 0.25 } else { 1. },
            )?,
        )?) as usize;
    let eps = real(text, "rms_norm_eps", 1e-5)?;
    if layers == 0
        || layers > 1024
        || vocab == 0
        || head_dim == 0
        || head_dim > 512
        || kv_heads == 0
        || heads % kv_heads != 0
        || rotary == 0
        || rotary > head_dim
        || rotary % 2 != 0
        || eps <= 0.
        || rope_base <= 0.
    {
        return Err(bad("invalid model dimensions or normalization"));
    }
    if let Some(types) = text.get("layer_types").and_then(Value::as_array) {
        if types.len() != layers
            || types
                .iter()
                .any(|v| !matches!(v.as_str(), Some("linear_attention" | "full_attention")))
        {
            return Err(Error::Unsupported(
                "layer_types must describe full/linear attention for every layer".into(),
            ));
        }
    }
    let dk = if hybrid {
        number(text, "linear_key_head_dim", None)?
    } else {
        0
    };
    let kh = if hybrid {
        number(text, "linear_num_key_heads", None)?
    } else {
        0
    };
    let vh = if hybrid {
        number(text, "linear_num_value_heads", None)?
    } else {
        0
    };
    let dv = if hybrid {
        number(text, "linear_value_head_dim", None)?
    } else {
        0
    };
    let conv = if hybrid {
        number(text, "linear_conv_kernel_dim", None)?
    } else {
        0
    };
    if hybrid
        && (dk == 0
            || dk > 512
            || dv == 0
            || dv > 512
            || kh == 0
            || vh == 0
            || vh % kh != 0
            || !(2..=64).contains(&conv))
    {
        return Err(bad("linear attention dimensions"));
    }
    let mut factors = vec![1.; rotary / 2];
    let mut rope_scale = 1.;
    if let Some(rope) = rope {
        let kind = rope
            .get("rope_type")
            .or_else(|| rope.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("default");
        match kind {
            "default" | "none" => {}
            "linear" => {
                let factor = real(rope, "factor", 1.)?;
                if factor <= 0. {
                    return Err(bad("RoPE factor"));
                }
                rope_scale = 1. / factor;
            }
            "llama3" => {
                let factor = real(rope, "factor", 1.)?;
                let low = real(rope, "low_freq_factor", 1.)?;
                let high = real(rope, "high_freq_factor", 4.)?;
                let original = number(rope, "original_max_position_embeddings", None)? as f32;
                if factor <= 0. || low <= 0. || high <= low || original <= 0. {
                    return Err(bad("Llama3 RoPE parameters"));
                }
                for (i, f) in factors.iter_mut().enumerate() {
                    let wavelength =
                        2. * std::f32::consts::PI * rope_base.powf(2. * i as f32 / rotary as f32);
                    *f = if wavelength < original / high {
                        1.
                    } else if wavelength > original / low {
                        factor
                    } else {
                        let smooth = (original / wavelength - low) / (high - low);
                        ((1. - smooth) / factor + smooth).recip()
                    };
                }
            }
            _ => return Err(Error::Unsupported(format!("MLX RoPE {kind}"))),
        }
    }
    let config = ModelConfig {
        architecture: architecture.into(),
        hidden,
        layers,
        heads,
        kv_heads,
        head_dim,
        rotary,
        vocab,
        tied_embeddings: Some(
            text.get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        ),
        context: number(text, "max_position_embeddings", Some(8192))?,
        eps,
        rope_base,
        rope_scale,
        linear_key_dim: dk,
        linear_key_heads: kh,
        linear_value_heads: vh,
        linear_value_dim: dv,
        conv_kernel: conv,
        experts_used: number(text, "num_experts_per_tok", Some(0))?,
        rotary_layout: if text.get("rope_traditional").and_then(Value::as_bool) == Some(true) {
            RotaryLayout::Adjacent
        } else {
            RotaryLayout::SplitHalf
        },
        recurrent_head_layout: RecurrentHeadLayout::Interleaved,
        l2_normalization: L2Normalization::AddEpsilon,
        linear_norm_eps: 1e-6,
        normalize_expert_weights: text
            .get("norm_topk_prob")
            .and_then(Value::as_bool)
            .unwrap_or(architecture != "qwen3next"),
    };
    if config.context == 0 {
        return Err(bad("zero context capacity"));
    }
    Ok((config, factors))
}
