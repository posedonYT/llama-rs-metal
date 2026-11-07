use crate::{Error, Result, TensorBytes};
use memmap2::Mmap;
use safetensors::{SafeTensors, tensor::Dtype};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::{Component, Path},
    sync::Arc,
};
#[derive(Clone)]
pub(super) struct RawTensor {
    pub bytes: TensorBytes,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}
fn bad(s: impl Into<String>) -> Error {
    Error::Invalid(s.into())
}
pub(super) fn read_json(path: &Path) -> Result<serde_json::Value> {
    let bytes = std::fs::read(path)?;
    if bytes.len() > 128 * 1024 * 1024 {
        return Err(bad("JSON metadata exceeds 128 MiB"));
    }
    serde_json::from_slice(&bytes).map_err(|e| bad(format!("{}: {e}", path.display())))
}
pub(super) fn load(directory: &Path) -> Result<(BTreeMap<String, RawTensor>, Vec<String>)> {
    let index_path = directory.join("model.safetensors.index.json");
    let index = if index_path.exists() {
        Some(read_json(&index_path)?)
    } else {
        None
    };
    let mut files = BTreeSet::new();
    if let Some(index) = &index {
        let map = index
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| bad("missing safetensors index weight_map"))?;
        for file in map.values() {
            let name = file.as_str().ok_or_else(|| bad("non-string shard name"))?;
            if !name.ends_with(".safetensors")
                || Path::new(name)
                    .components()
                    .any(|c| !matches!(c, Component::Normal(_)))
            {
                return Err(bad("unsafe safetensors shard name"));
            }
            files.insert(name.to_string());
        }
    } else if directory.join("model.safetensors").is_file() {
        files.insert("model.safetensors".into());
    } else {
        return Err(bad(
            "checkpoint needs model.safetensors or model.safetensors.index.json",
        ));
    }
    let mut tensors = BTreeMap::new();
    for name in &files {
        let file = File::open(directory.join(name))?;
        // SAFETY: checkpoint files are read-only and must remain unmodified while loaded.
        let map = Arc::new(unsafe { Mmap::map(&file)? });
        let safe = SafeTensors::deserialize(&map).map_err(|e| bad(format!("{name}: {e}")))?;
        for (key, view) in safe.tensors() {
            if let Some(index) = &index {
                if index["weight_map"]
                    .get(&key)
                    .and_then(serde_json::Value::as_str)
                    != Some(name)
                {
                    return Err(bad(format!("index does not assign {key} to {name}")));
                }
            }
            let offset = view.data().as_ptr() as usize - map.as_ptr() as usize;
            let raw = RawTensor {
                bytes: TensorBytes::mapped(map.clone(), offset, view.data().len())?,
                shape: view.shape().to_vec(),
                dtype: view.dtype(),
            };
            if tensors.insert(key.clone(), raw).is_some() {
                return Err(bad(format!("duplicate tensor {key}")));
            }
        }
    }
    if let Some(index) = &index {
        for key in index["weight_map"].as_object().unwrap().keys() {
            if !tensors.contains_key(key) {
                return Err(bad(format!("index tensor {key} is missing")));
            }
        }
    }
    Ok((tensors, files.into_iter().collect()))
}
