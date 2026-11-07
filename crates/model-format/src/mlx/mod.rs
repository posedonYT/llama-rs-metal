//! Native MLX checkpoint ingestion. No Python and no intermediate GGUF conversion.
mod config;
mod names;
mod safetensors;
use crate::{AffineQuantization, Checkpoint, DType, Error, Result, Tensor, WeightData};
use ::safetensors::tensor::Dtype;
use serde_json::Value;
use std::{collections::BTreeMap, path::Path};
fn bad(s: impl Into<String>) -> Error {
    Error::Invalid(s.into())
}
fn dtype(t: Dtype) -> Result<DType> {
    match t {
        Dtype::F32 => Ok(DType::F32),
        Dtype::F16 => Ok(DType::F16),
        Dtype::BF16 => Ok(DType::BF16),
        _ => Err(Error::Unsupported(format!("MLX tensor dtype {t:?}"))),
    }
}
fn quantization<'a>(root: &'a Value, name: &str) -> Result<&'a Value> {
    let q = root
        .get("quantization")
        .or_else(|| root.get("quantization_config"))
        .ok_or_else(|| bad("packed MLX weights require quantization metadata"))?;
    let module = name.strip_suffix(".weight").unwrap_or(name);
    Ok(q.get(module).filter(|v| v.is_object()).unwrap_or(q))
}
pub(super) fn open(directory: &Path) -> Result<Checkpoint> {
    let root = safetensors::read_json(&directory.join("config.json"))?;
    let (config, rope_factors) = config::parse(&root)?;
    let (raw, shards) = safetensors::load(directory)?;
    // MLX saves sanitized depthwise convolution [channels,kernel,1]. Original HF
    // files use [channels,1,kernel] and store Qwen3.5 RMSNorm weights as an offset.
    let unsanitized = config.architecture.starts_with("qwen35")
        && raw
            .iter()
            .any(|(n, t)| n.ends_with("conv1d.weight") && t.shape.last() != Some(&1));
    let mut tensors = BTreeMap::new();
    let mut source_names = BTreeMap::new();
    for (name, r) in &raw {
        let Some(canonical) = names::canonical(name, config.hybrid()) else {
            continue;
        };
        if canonical == "output.weight" && config.tied_embeddings == Some(true) {
            // Match the source framework's sanitize step: an incidental saved
            // head cannot override an explicitly tied embedding/output module.
            continue;
        }
        let module = name.strip_suffix(".weight").unwrap_or(name);
        let mut dims: Vec<usize> = r.shape.iter().rev().copied().collect();
        let mut tensor = if r.dtype == Dtype::U32 {
            let q = quantization(&root, name)?;
            let mode = q.get("mode").and_then(Value::as_str).unwrap_or("affine");
            if mode != "affine" {
                return Err(Error::Unsupported(format!("MLX quantization mode {mode}")));
            }
            let bits = q
                .get("bits")
                .and_then(Value::as_u64)
                .ok_or_else(|| bad(format!("{name}: missing quantization bits")))?
                as u32;
            let group_size = q
                .get("group_size")
                .and_then(Value::as_u64)
                .ok_or_else(|| bad(format!("{name}: missing group size")))?
                as usize;
            if ![2, 4, 8].contains(&bits) {
                return Err(Error::Unsupported(format!(
                    "MLX affine {bits}-bit (supported: 2/4/8)"
                )));
            }
            let scales = raw
                .get(&format!("{module}.scales"))
                .ok_or_else(|| bad(format!("{name}: missing scales")))?;
            let biases = raw
                .get(&format!("{module}.biases"))
                .ok_or_else(|| bad(format!("{name}: missing biases")))?;
            if dims.len() < 2
                || scales.dtype != biases.dtype
                || scales.shape != biases.shape
                || scales.shape.len() != r.shape.len()
                || scales.shape[..scales.shape.len() - 1] != r.shape[..r.shape.len() - 1]
            {
                return Err(bad(format!("{name}: affine metadata shape")));
            }
            dims[0] = dims[0]
                .checked_mul(32 / bits as usize)
                .ok_or_else(|| bad("packed dimension overflow"))?;
            if group_size == 0 || dims[0] / group_size != *scales.shape.last().unwrap() {
                return Err(bad(format!("{name}: affine group count")));
            }
            Tensor::new(
                dims,
                WeightData::Affine {
                    packed: r.bytes.clone(),
                    quantization: AffineQuantization {
                        bits,
                        group_size,
                        scale_type: dtype(scales.dtype)?,
                        scales: scales.bytes.clone(),
                        biases: biases.bytes.clone(),
                    },
                },
            )?
        } else {
            if canonical.ends_with("ssm_conv1d.weight") {
                if r.shape.len() != 3 || r.shape[0] == 0 || !(r.shape[1] == 1 || r.shape[2] == 1) {
                    return Err(bad(format!("{name}: depthwise convolution shape")));
                }
                dims = vec![r.shape[1].max(r.shape[2]), r.shape[0]];
            }
            if canonical.ends_with("ffn_gate_inp_shexp.weight") && dims.last() == Some(&1) {
                dims.pop();
            }
            Tensor::new(
                dims,
                WeightData::Ggml {
                    bytes: r.bytes.clone(),
                    dtype: dtype(r.dtype)?,
                },
            )?
        };
        if canonical.ends_with("ffn_gate_inp_shexp.weight")
            && tensor.dims().len() == 2
            && tensor.dims()[1] == 1
        {
            tensor = tensor.reshape(vec![tensor.dims()[0]])?
        }
        if canonical.ends_with("ssm_a") {
            let values: Vec<_> = tensor.decode()?.iter().map(|v| -v.exp()).collect();
            tensor = Tensor::from_f32(tensor.dims().to_vec(), &values)?;
        }
        if unsanitized
            && (canonical.ends_with("attn_norm.weight")
                || canonical.ends_with("post_attention_norm.weight")
                || canonical == "output_norm.weight"
                || canonical.ends_with("attn_q_norm.weight")
                || canonical.ends_with("attn_k_norm.weight"))
        {
            let values: Vec<_> = tensor.decode()?.iter().map(|v| v + 1.).collect();
            tensor = Tensor::from_f32(tensor.dims().to_vec(), &values)?;
        }
        if tensors.insert(canonical.clone(), tensor).is_some() {
            return Err(bad(format!("multiple source tensors map to {canonical}")));
        }
        source_names.insert(canonical, name.clone());
    }
    if tensors.is_empty() {
        return Err(bad("no supported language-model tensors"));
    }
    let tokenizer = safetensors::read_json(&directory.join("tokenizer_config.json"))?;
    let chat_template = if directory.join("chat_template.jinja").is_file() {
        Some(std::fs::read_to_string(
            directory.join("chat_template.jinja"),
        )?)
    } else {
        match tokenizer.get("chat_template") {
            Some(Value::String(s)) => Some(s.clone()),
            Some(Value::Array(a)) => a
                .iter()
                .find(|v| v["name"] == "default")
                .and_then(|v| v["template"].as_str())
                .map(str::to_string),
            None | Some(Value::Null) => None,
            _ => {
                return Err(Error::Unsupported(
                    "chat_template must be a string or named template list".into(),
                ));
            }
        }
    };
    Ok(Checkpoint::mlx(
        config,
        tensors,
        directory.to_path_buf(),
        chat_template,
        rope_factors,
        serde_json::json!({
            "format": "mlx",
            "shards": shards,
            "quantization": root.get("quantization").or_else(|| root.get("quantization_config")),
            "source_names": source_names,
            "sanitized_norm_offsets": unsanitized,
        }),
    ))
}
