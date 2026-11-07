//! Convenient single-engine smoke timing. Formal paired comparisons live in tools/benchmarks.
use super::{ModelArgs, load};
use std::{path::PathBuf, time::Instant};
pub(super) fn run(
    args: &ModelArgs,
    prompt_tokens: usize,
    decode_tokens: usize,
    repetitions: usize,
    output: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    if prompt_tokens == 0 || decode_tokens == 0 || repetitions == 0 {
        return Err("benchmark sizes must be positive".into());
    }
    if prompt_tokens
        .checked_add(decode_tokens)
        .is_none_or(|n| args.context > 0 && n > args.context)
    {
        return Err("benchmark exceeds context".into());
    }
    let (engine, model, mut session, load_seconds) = load(args)?;
    if prompt_tokens.saturating_add(decode_tokens) > session.context() {
        return Err("benchmark exceeds context".into());
    }
    let seed =
        model
            .tokenizer()
            .encode("Once upon a time, there was a small robot. ", true, false)?;
    if seed.is_empty() {
        return Err("empty benchmark prompt".into());
    }
    let prompt: Vec<_> = seed.iter().copied().cycle().take(prompt_tokens).collect();
    session.prefill(&prompt[..prompt.len().min(8)])?;
    session.decode(0)?;
    session.reset()?;
    let decode_inputs: Vec<_> = prompt.iter().copied().cycle().take(decode_tokens).collect();
    let mut runs = Vec::new();
    for _ in 0..repetitions {
        session.reset()?;
        let start = Instant::now();
        session.prefill(&prompt)?;
        let mut last_prediction = argmax(session.logits())?;
        let prefill = start.elapsed().as_secs_f64();
        let start = Instant::now();
        for &token in &decode_inputs {
            session.decode(token)?;
            last_prediction = argmax(session.logits())?;
        }
        let decode = start.elapsed().as_secs_f64();
        runs.push(serde_json::json!({
            "prefill_seconds": prefill,
            "ttft_seconds": prefill,
            "prefill_tokens_per_second": prompt_tokens as f64 / prefill,
            "decode_seconds": decode,
            "decode_tokens_per_second": decode_tokens as f64 / decode,
            "gpu_seconds": session.gpu_seconds(),
            "last_prediction": last_prediction,
        }));
    }
    let os = std::process::Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    let rss_bytes = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse::<usize>()
                .ok()
        })
        .map(|k| k * 1024);
    let json = serde_json::json!({
        "schema_version": 1,
        "engine": "llama-rs-metal",
        "version": env!("CARGO_PKG_VERSION"),
        "device": engine.device_name(),
        "os": os,
        "model": args.model,
        "architecture": model.config().architecture,
        "context": session.context(),
        "prompt_tokens": prompt,
        "decode_tokens": decode_tokens,
        "sampling": "greedy_teacher_forced",
        "decode_token_ids": decode_inputs,
        "kv_dtype": "f32",
        "recurrent_dtype": "f32",
        "load_seconds": load_seconds,
        "memory": model.memory(session.context())?,
        "metal_allocated_bytes": engine.allocated_bytes(),
        "rss_bytes": rss_bytes,
        "runs": runs,
    });
    let text = serde_json::to_string_pretty(&json)?;
    if let Some(path) = output {
        std::fs::write(path, &text)?
    }
    println!("{text}");
    Ok(())
}

fn argmax(logits: &[f32]) -> Result<u32, Box<dyn std::error::Error>> {
    if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
        return Err("invalid logits".into());
    }
    Ok(logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
        .unwrap()
        .0 as u32)
}
