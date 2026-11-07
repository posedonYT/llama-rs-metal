use llama_rs::*;
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if !(5..=6).contains(&a.len()) {
        return Err("usage: dump_logits MODEL PROMPT LOGITS TOKEN_IDS".into());
    }
    let engine = Engine::new(Default::default())?;
    let model = engine.load_with_options(
        &a[1],
        LoadOptions {
            float_weights_f32: a.get(5).is_some_and(|s| s == "--f32"),
        },
    )?;
    let tokens: Vec<u32> = if let Some(path) = a[2].strip_prefix("@") {
        std::fs::read_to_string(path)?
            .split_whitespace()
            .map(str::parse)
            .collect::<std::result::Result<_, _>>()?
    } else {
        model.tokenizer().encode(&a[2], true, true)?
    };
    std::fs::write(
        &a[4],
        tokens.iter().map(|t| format!("{t}\n")).collect::<String>(),
    )?;
    let mut session = model.session(SessionOptions {
        context: 512.min(model.config().context),
    })?;
    let logits = session.prefill(&tokens)?;
    let bytes: Vec<_> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&a[3], bytes)?;
    Ok(())
}
