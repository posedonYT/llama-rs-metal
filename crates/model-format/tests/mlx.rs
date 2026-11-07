use model_format::{Checkpoint, CheckpointFormat, WeightData};
use safetensors::tensor::{Dtype, TensorView, serialize};
use serde_json::json;
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
static ID: AtomicU64 = AtomicU64::new(0);
type Raw = BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)>;
struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn fixture() -> Fixture {
    let dir = std::env::temp_dir().join(format!(
        "mlx-format-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&dir).unwrap();
    let config = json!({"model_type":"llama","hidden_size":64,"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,"rms_norm_eps":1e-5,"vocab_size":3,"max_position_embeddings":64,"quantization":{"bits":4,"group_size":32}});
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
    Fixture(dir)
}
fn save(path: PathBuf, tensors: &Raw) {
    let views: BTreeMap<_, _> = tensors
        .iter()
        .map(|(n, (ty, dims, data))| (n, TensorView::new(*ty, dims.clone(), data).unwrap()))
        .collect();
    std::fs::write(path, serialize(views, None).unwrap()).unwrap();
}
fn tensors() -> Raw {
    let mut t = Raw::new();
    t.insert(
        "model.embed_tokens.weight".into(),
        (Dtype::U32, vec![3, 8], vec![0x11; 96]),
    );
    t.insert(
        "model.embed_tokens.scales".into(),
        (
            Dtype::F32,
            vec![3, 2],
            [0.5f32; 6].iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
    );
    t.insert(
        "model.embed_tokens.biases".into(),
        (
            Dtype::F32,
            vec![3, 2],
            [-0.25f32; 6].iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
    );
    t
}
#[test]
fn loads_affine_shards_and_retains_mappings() {
    let f = fixture();
    let all = tensors();
    let a: Raw = all
        .iter()
        .take(1)
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let b: Raw = all
        .iter()
        .skip(1)
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    save(f.0.join("a.safetensors"), &a);
    save(f.0.join("b.safetensors"), &b);
    let map: BTreeMap<_, _> = all
        .keys()
        .map(|k| {
            (
                k,
                if a.contains_key(k) {
                    "a.safetensors"
                } else {
                    "b.safetensors"
                },
            )
        })
        .collect();
    std::fs::write(
        f.0.join("model.safetensors.index.json"),
        json!({"weight_map":map}).to_string(),
    )
    .unwrap();
    let checkpoint = Checkpoint::open(&f.0).unwrap();
    assert_eq!(checkpoint.format, CheckpointFormat::Mlx);
    let tensor = checkpoint.tensor("token_embd.weight").unwrap().clone();
    drop(checkpoint);
    assert_eq!(tensor.dims(), &[64, 3]);
    assert_eq!(tensor.decode().unwrap(), vec![0.25; 192]);
}
#[test]
fn rejects_missing_metadata_and_unsafe_shards() {
    let f = fixture();
    let mut all = tensors();
    all.remove("model.embed_tokens.biases");
    save(f.0.join("model.safetensors"), &all);
    assert!(Checkpoint::open(&f.0).is_err());
    std::fs::write(
        f.0.join("model.safetensors.index.json"),
        json!({"weight_map":{"model.embed_tokens.weight":"../outside.safetensors"}}).to_string(),
    )
    .unwrap();
    assert!(Checkpoint::open(&f.0).is_err());
}
#[test]
fn rejects_truncated_files_and_unsupported_quantization() {
    let f = fixture();
    save(f.0.join("model.safetensors"), &tensors());
    let path = f.0.join("model.safetensors");
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
    assert!(Checkpoint::open(&f.0).is_err());
    std::fs::write(&path, &bytes).unwrap();
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(f.0.join("config.json")).unwrap()).unwrap();
    config["quantization"]["mode"] = json!("mxfp4");
    std::fs::write(f.0.join("config.json"), config.to_string()).unwrap();
    assert!(Checkpoint::open(&f.0).unwrap_err_string().contains("mxfp4"));
}
trait ErrorText {
    fn unwrap_err_string(self) -> String;
}
impl<T> ErrorText for model_format::Result<T> {
    fn unwrap_err_string(self) -> String {
        match self {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected error"),
        }
    }
}
#[test]
fn mixed_quantization_override_is_per_module() {
    let f = fixture();
    let mut all = tensors();
    all.insert(
        "lm_head.weight".into(),
        (Dtype::U32, vec![3, 16], vec![7; 192]),
    );
    all.insert(
        "lm_head.scales".into(),
        (
            Dtype::F32,
            vec![3, 1],
            [0.5f32; 3].iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
    );
    all.insert(
        "lm_head.biases".into(),
        (Dtype::F32, vec![3, 1], vec![0; 12]),
    );
    save(f.0.join("model.safetensors"), &all);
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(f.0.join("config.json")).unwrap()).unwrap();
    config["quantization"]["lm_head"] = json!({"bits":8,"group_size":64});
    std::fs::write(f.0.join("config.json"), config.to_string()).unwrap();
    let c = Checkpoint::open(&f.0).unwrap();
    match c.tensor("output.weight").unwrap().data() {
        WeightData::Affine {
            quantization: q, ..
        } => assert_eq!(q.bits, 8),
        _ => panic!("expected affine"),
    }
}

#[test]
fn explicit_tying_ignores_an_incidental_saved_classifier() {
    let f = fixture();
    let mut raw = tensors();
    raw.insert(
        "lm_head.weight".into(),
        (
            Dtype::F32,
            vec![3, 64],
            [999f32; 192].iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
    );
    save(f.0.join("model.safetensors"), &raw);
    let path = f.0.join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["tie_word_embeddings"] = json!(false);
    std::fs::write(&path, config.to_string()).unwrap();
    assert_eq!(
        Checkpoint::open(&f.0)
            .unwrap()
            .decode("output.weight")
            .unwrap(),
        vec![999.; 192]
    );
    config["tie_word_embeddings"] = json!(true);
    std::fs::write(&path, config.to_string()).unwrap();
    let tied = Checkpoint::open(&f.0).unwrap();
    assert!(tied.tensor("output.weight").is_err());
    assert_eq!(tied.decode("token_embd.weight").unwrap(), vec![0.25; 192]);
}
