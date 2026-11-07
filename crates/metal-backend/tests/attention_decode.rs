use metal_backend::Gpu;

#[test]
fn partitioned_decode_matches_f64_causal_gqa() {
    let mut gpu = Gpu::new().unwrap();
    for (dim, tokens) in [
        (8usize, 129usize),
        (31, 129),
        (128, 257),
        (256, 4097),
        (512, 255),
        (31, 8193),
    ] {
        let (heads, kv_heads) = (6, 2);
        let q: Vec<_> = (0..heads * dim).map(|i| (i as f32 * 0.071).sin()).collect();
        let k: Vec<_> = (0..(tokens + 7) * kv_heads * dim)
            .map(|i| {
                if i < tokens * kv_heads * dim {
                    (i as f32 * 0.037).cos()
                } else {
                    1000.
                }
            })
            .collect();
        let v: Vec<_> = (0..k.len())
            .map(|i| {
                if i < tokens * kv_heads * dim {
                    (i as f32 * 0.019).sin()
                } else {
                    -1000.
                }
            })
            .collect();
        let mut expected = vec![0.; heads * dim];
        for head in 0..heads {
            let kh = head / (heads / kv_heads);
            let scores: Vec<f64> = (0..tokens)
                .map(|t| {
                    (0..dim)
                        .map(|d| q[head * dim + d] as f64 * k[(t * kv_heads + kh) * dim + d] as f64)
                        .sum::<f64>()
                        / (dim as f64).sqrt()
                })
                .collect();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<_> = scores.iter().map(|s| (s - maximum).exp()).collect();
            let total: f64 = weights.iter().sum();
            for d in 0..dim {
                expected[head * dim + d] = (weights
                    .iter()
                    .enumerate()
                    .map(|(t, w)| w * v[(t * kv_heads + kh) * dim + d] as f64)
                    .sum::<f64>()
                    / total) as f32;
            }
        }
        let qb = gpu.upload(&q).unwrap();
        let kb = gpu.upload(&k).unwrap();
        let vb = gpu.upload(&v).unwrap();
        let output = gpu.zeros(heads * dim).unwrap();
        let partial = gpu
            .zeros(heads * tokens.div_ceil(128).min(32) * (dim + 2))
            .unwrap();
        gpu.with_batch(|b| {
            b.attention_decode(
                &qb, &kb, &vb, &output, &partial, heads, kv_heads, dim, tokens,
            )
        })
        .unwrap();
        for (i, (a, b)) in gpu.read(&output).unwrap().iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() < 2e-4,
                "dim {dim}, context {tokens}, element {i}: {a} vs {b}"
            );
        }
    }
}
