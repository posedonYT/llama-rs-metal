//! Opt-in real artifact test; no network or model download during `cargo test`.
use llama_rs::{Engine, EngineOptions, LoadOptions, SessionOptions};
use std::sync::atomic::Ordering;

fn near(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (index, (a, b)) in a.iter().zip(b).enumerate() {
        assert!(
            (a - b).abs() <= 0.01 + 0.001 * a.abs(),
            "logit {index}: {a} vs {b}"
        );
    }
}

#[test]
#[ignore = "set LLAMA_MLX_MODEL to a supported local checkpoint directory"]
fn native_mlx_state_and_bounded_allocations() {
    let path = std::env::var("LLAMA_MLX_MODEL").expect("LLAMA_MLX_MODEL required");
    let limited = Engine::new(EngineOptions {
        memory_limit: Some(1),
    })
    .unwrap();
    assert!(matches!(
        limited.load(&path),
        Err(llama_rs::Error::Runtime(
            model_runtime::Error::Memory { .. }
        ))
    ));
    assert_eq!(limited.allocated_bytes(), 0);
    let engine = Engine::new(Default::default()).unwrap();
    let model = engine
        .load_with_options(
            path,
            LoadOptions {
                float_weights_f32: true,
            },
        )
        .unwrap();
    let prompt = model
        .tokenizer()
        .encode(
            "A compiler preserves the meaning of a program. "
                .repeat(8)
                .as_str(),
            true,
            true,
        )
        .unwrap();
    let prompt = &prompt[..33];
    let mut a = model.session(SessionOptions { context: 256 }).unwrap();
    let mut b = model
        .clone()
        .session(SessionOptions { context: 256 })
        .unwrap();
    a.prefill(prompt).unwrap();
    for chunk in prompt.chunks(7) {
        b.prefill(chunk).unwrap();
    }
    near(a.logits(), b.logits());
    let expected = a.logits().to_vec();
    b.decode(prompt[0]).unwrap();
    near(a.logits(), &expected); // A's cache and host result are independent.
    b.reset().unwrap();
    b.prefill(prompt).unwrap();
    near(a.logits(), b.logits());
    let cancel = a.cancellation_handle();
    cancel.store(true, Ordering::Relaxed);
    assert!(a.decode(prompt[0]).is_err());
    assert_eq!(a.tokens(), prompt);
    a.reset().unwrap();
    a.prefill(prompt).unwrap();
    a.decode(prompt[0]).unwrap(); // Warm decode before checking retained bytes.
    let allocated = engine.allocated_bytes();
    for i in 0..128 {
        a.decode(prompt[i % prompt.len()]).unwrap();
    }
    assert_eq!(allocated, engine.allocated_bytes());
    assert_eq!(a.tokens().len(), 162);
    a.reset().unwrap();
    a.prefill(prompt).unwrap();
    near(a.logits(), &expected);
    assert_eq!(allocated, engine.allocated_bytes());
}
