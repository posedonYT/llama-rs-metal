//! Line-delimited JSON worker. Loading, timing boundaries and raw traces are
//! explicit so a coordinator can run balanced, isolated Rust/MLX/llama.cpp trials.
use llama_rs::{Engine, EngineOptions, LoadOptions, Session, SessionOptions};
use serde_json::{Value, json};
use std::{
    io::{self, BufRead, Write},
    path::Path,
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn ids(request: &Value, key: &str) -> Result<Vec<u32>> {
    request
        .get(key)
        .and_then(Value::as_array)
        .ok_or("missing token ID array")?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .ok_or_else(|| "invalid token ID".into())
        })
        .collect()
}
fn argmax(values: &[f32]) -> Result<u32> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite or empty logits".into());
    }
    Ok(values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
        .unwrap()
        .0 as u32)
}
fn save_logits(session: &Session, path: &Path) -> Result<Value> {
    let values = session.logits();
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes)?;
    Ok(json!({"path":path,"top1":argmax(values)?,"count":values.len()}))
}
fn one_run(session: &mut Session, prompt: &[u32], decode: &[u32]) -> Result<Value> {
    let reset = Instant::now();
    session.reset()?;
    let reset_seconds = reset.elapsed().as_secs_f64();
    let start = Instant::now();
    session.prefill(prompt)?;
    let mut last = argmax(session.logits())?;
    let prefill_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut latencies = Vec::with_capacity(decode.len());
    for &token in decode {
        let step = Instant::now();
        session.decode(token)?;
        last = argmax(session.logits())?;
        latencies.push(step.elapsed().as_secs_f64());
    }
    let decode_seconds = start.elapsed().as_secs_f64();
    Ok(
        json!({"reset_seconds":reset_seconds,"prefill_seconds":prefill_seconds,"request_ttft_seconds":reset_seconds+prefill_seconds,"decode_seconds":decode_seconds,"decode_step_seconds":latencies,"prefill_tokens_per_second":prompt.len()as f64/prefill_seconds,"decode_tokens_per_second":decode.len()as f64/decode_seconds,"gpu_seconds":session.gpu_seconds(),"last_prediction":last}),
    )
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let path = args.get(1).ok_or("model path required")?;
    let float32 = args.iter().any(|a| a == "--f32");
    let context = args
        .windows(2)
        .find(|a| a[0] == "--context")
        .map(|a| a[1].parse::<usize>())
        .transpose()?
        .unwrap_or(4096);
    let loading = Instant::now();
    let engine = Engine::new(EngineOptions::default())?;
    let model = engine.load_with_options(
        path,
        LoadOptions {
            float_weights_f32: float32,
        },
    )?;
    let mut session = model.session(SessionOptions { context })?;
    println!(
        "{}",
        json!({"type":"ready","protocol":1,"engine":"rust","version":env!("CARGO_PKG_VERSION"),"device":engine.device_name(),"format":model.checkpoint_format(),"config":model.config(),"context":context,"float_weights_f32":float32,"load_seconds":loading.elapsed().as_secs_f64(),"memory_estimate":model.memory(context)?,"metal_allocated_bytes":engine.allocated_bytes()})
    );
    io::stdout().flush()?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        let result = (|| -> Result<Value> {
            let request: Value = serde_json::from_str(&line)?;
            match request["action"].as_str().unwrap_or("") {
                "tokenize" => Ok(
                    json!({"ids":model.tokenizer().encode(request["text"].as_str().ok_or("missing text")?,request["special"].as_bool().unwrap_or(true),true)?}),
                ),
                "verify" => {
                    let prompt = ids(&request, "prompt_ids")?;
                    let decode = ids(&request, "decode_ids")?;
                    let dir = Path::new(
                        request["output_dir"]
                            .as_str()
                            .ok_or("missing output directory")?,
                    );
                    std::fs::create_dir_all(dir)?;
                    session.reset()?;
                    let chunk = request["chunk_size"]
                        .as_u64()
                        .unwrap_or(prompt.len() as u64) as usize;
                    if chunk == 0 {
                        return Err("zero chunk size".into());
                    }
                    for p in prompt.chunks(chunk) {
                        session.prefill(p)?;
                    }
                    let mut traces = vec![save_logits(&session, &dir.join("prefill.f32"))?];
                    for (i, &t) in decode.iter().enumerate() {
                        session.decode(t)?;
                        traces.push(save_logits(&session, &dir.join(format!("decode-{i}.f32")))?);
                    }
                    let mut snapshots = Vec::new();
                    if request["state"].as_bool().unwrap_or(false) {
                        for state in session.state_snapshot()? {
                            let path =
                                dir.join(format!("layer-{}-{}.f32", state.layer, state.kind));
                            let bytes: Vec<_> =
                                state.values.iter().flat_map(|v| v.to_le_bytes()).collect();
                            std::fs::write(&path, bytes)?;
                            snapshots.push(json!({"layer":state.layer,"kind":state.kind,"shape":state.shape,"path":path}));
                        }
                    }
                    Ok(json!({"traces":traces,"states":snapshots,"tokens":session.tokens()}))
                }
                "benchmark" => {
                    let prompt = ids(&request, "prompt_ids")?;
                    let decode = ids(&request, "decode_ids")?;
                    if prompt.is_empty() || decode.is_empty() {
                        return Err("empty benchmark workload".into());
                    }
                    let mut warmups = Vec::new();
                    for _ in 0..request["warmups"].as_u64().unwrap_or(2) {
                        warmups.push(one_run(&mut session, &prompt, &decode)?);
                    }
                    let mut runs = Vec::new();
                    for _ in 0..request["repetitions"].as_u64().unwrap_or(2) {
                        runs.push(one_run(&mut session, &prompt, &decode)?);
                    }
                    Ok(
                        json!({"runs":runs,"warmup_runs":warmups,"metal_allocated_bytes":engine.allocated_bytes(),"boundary":"full_logits_host_greedy","cache_policy":"preallocated_reset_outside_forward"}),
                    )
                }
                _ => Err("unknown worker action".into()),
            }
        })();
        let response = match result {
            Ok(v) => json!({"ok":true,"result":v}),
            Err(e) => json!({"ok":false,"error":e.to_string()}),
        };
        println!("{response}");
        io::stdout().flush()?;
    }
    Ok(())
}
