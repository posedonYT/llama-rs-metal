#[path = "../tests/common/mod.rs"]
mod common;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/fixtures".into());
    std::fs::create_dir_all(&dir)?;
    for arch in ["llama", "qwen35", "qwen35moe", "qwen3next"] {
        let f = common::fixture(arch);
        std::fs::copy(&f.path, format!("{dir}/{arch}.gguf"))?;
    }
    Ok(())
}
