//! Inspect file contracts without creating a Metal device or allocating GPU state.
use model_format::{Checkpoint, Tensor, WeightData};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};

fn tensor_info(tensor: &Tensor) -> Value {
    let (encoding, metadata_dtype) = match tensor.data() {
        WeightData::Ggml { dtype, .. } => (format!("{dtype:?}"), None),
        WeightData::Affine {
            quantization: q, ..
        } => (
            format!("mlx_affine_{}bit_g{}", q.bits, q.group_size),
            Some(q.scale_type),
        ),
    };
    json!({
        "dimensions": tensor.dims(),
        "storage_bytes": tensor.storage_bytes(),
        "encoding": encoding,
        "metadata_dtype": metadata_dtype,
    })
}
pub(super) fn run(path: &Path, show_tensors: bool) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = Checkpoint::open(path)?;
    let tensors = show_tensors.then(|| {
        checkpoint
            .tensors()
            .iter()
            .map(|(name, tensor)| (name, tensor_info(tensor)))
            .collect::<BTreeMap<_, _>>()
    });
    let report = json!({
        "format": checkpoint.format,
        "config": checkpoint.config,
        "tensor_count": checkpoint.tensors().len(),
        "weight_bytes": checkpoint.weight_bytes()?,
        "chat_template": checkpoint.chat_template.is_some(),
        "tensors": tensors,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
