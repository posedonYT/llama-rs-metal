mod common;
use llama_rs::*;
use std::sync::atomic::Ordering;
fn near(a: &[f32], b: &[f32]) {
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() < 2e-4, "logit {i}: {a} vs {b}")
    }
}
#[test]
fn prefill_decode_reset_and_independence() {
    for arch in ["llama", "qwen35", "qwen35moe", "qwen3next"] {
        eprintln!("testing {arch}");
        let f = common::fixture(arch);
        let e = Engine::new(Default::default()).unwrap();
        let m = e.load(&f.path).unwrap();
        let opts = SessionOptions { context: 64 };
        let mut a = m.session(opts.clone()).unwrap();
        let mut b = m.session(opts).unwrap();
        let prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let expected = a.prefill(&prompt).unwrap().to_vec();
        for t in prompt {
            b.decode(t).unwrap();
        }
        near(&expected, b.logits());
        a.decode(10).unwrap();
        b.decode(10).unwrap();
        near(a.logits(), b.logits());
        b.reset().unwrap();
        b.prefill(&prompt[..4]).unwrap();
        b.prefill(&prompt[4..]).unwrap();
        near(&expected, b.logits());
        b.reset().unwrap();
        assert!(b.tokens().is_empty());
        b.prefill(&prompt).unwrap();
        near(&expected, b.logits());
    }
}
#[test]
fn context_cancellation_and_memory() {
    let f = common::fixture("qwen35");
    let e = Engine::new(Default::default()).unwrap();
    let m = e.load(&f.path).unwrap();
    {
        let mut s = m.session(SessionOptions { context: 4 }).unwrap();
        s.prefill(&[1, 2, 3]).unwrap();
        assert!(s.prefill(&[4, 5]).is_err());
        assert_eq!(s.tokens(), &[1, 2, 3]);
        let cancel = s.cancellation_handle();
        cancel.store(true, Ordering::Relaxed);
        assert!(s.decode(4).is_err());
        assert_eq!(s.tokens(), &[1, 2, 3]);
        s.reset().unwrap();
        s.decode(4).unwrap();
    }
    assert!(m.session(SessionOptions { context: 65 }).is_err());
    let small = Engine::new(EngineOptions {
        memory_limit: Some(100),
    })
    .unwrap();
    assert!(small.load(&f.path).is_err());
    let _again = m.session(SessionOptions { context: 4 }).unwrap();
}

#[test]
fn long_decode_reuses_allocations() {
    let f = common::fixture("qwen35moe");
    let e = Engine::new(Default::default()).unwrap();
    let m = e.load(&f.path).unwrap();
    let mut s = m.session(SessionOptions { context: 64 }).unwrap();
    s.decode(1).unwrap();
    let allocated = e.allocated_bytes();
    for _ in 0..3 {
        s.reset().unwrap();
        for i in 0..64 {
            s.decode(i % 32).unwrap();
        }
        assert_eq!(
            e.allocated_bytes(),
            allocated,
            "GPU buffers grew during generation"
        );
    }
}

#[test]
fn prefix_edits_replay_recurrent_state() {
    let f = common::fixture("qwen35");
    let e = Engine::new(Default::default()).unwrap();
    let m = e.load(&f.path).unwrap();
    let mut s = m.session(SessionOptions { context: 8 }).unwrap();
    let mut reference = m.session(SessionOptions { context: 8 }).unwrap();
    s.synchronize_prompt(&[1, 2, 3]).unwrap();
    s.synchronize_prompt(&[1, 2, 3, 4]).unwrap();
    reference.prefill(&[1, 2, 3, 4]).unwrap();
    near(s.logits(), reference.logits());
    assert!(s.synchronize_prompt(&[0; 9]).is_err());
    assert_eq!(s.tokens(), &[1, 2, 3, 4]);
    s.synchronize_prompt(&[1, 5]).unwrap();
    reference.reset().unwrap();
    reference.prefill(&[1, 5]).unwrap();
    near(s.logits(), reference.logits());
}
