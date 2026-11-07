#[path = "../../llama-rs/tests/common/mod.rs"]
mod common;
use model_format::Gguf;
#[test]
fn rejects_truncation_and_bad_ranges() {
    let f = common::fixture("llama");
    let data = std::fs::read(&f.path).unwrap();
    let path = f.path.with_extension("bad.gguf");
    for n in [0, 4, 12, 23, 24, 32, data.len() - 1] {
        std::fs::write(&path, &data[..n]).unwrap();
        assert!(Gguf::open(&path).is_err(), "accepted prefix {n}")
    }
    let mut bad = data.clone();
    bad[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    std::fs::write(&path, &bad).unwrap();
    assert!(Gguf::open(&path).is_err());
    let name = b"output.weight";
    let start = data.windows(name.len()).position(|x| x == name).unwrap() + name.len();
    let rank = u32::from_le_bytes(data[start..start + 4].try_into().unwrap()) as usize;
    let offset = start + 4 + rank * 8 + 4;
    let mut bad = data.clone();
    bad[offset..offset + 8].copy_from_slice(&1u64.to_le_bytes());
    std::fs::write(&path, &bad).unwrap();
    assert!(Gguf::open(&path).is_err());
    bad[offset..offset + 8].copy_from_slice(&0u64.to_le_bytes());
    std::fs::write(&path, &bad).unwrap();
    assert!(Gguf::open(&path).is_err());
    std::fs::remove_file(path).unwrap();
}
