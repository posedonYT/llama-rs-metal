use metal_backend::*;
fn near(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() < 2e-4 * (1. + b.abs()), "{i}: {a} != {b}")
    }
}
#[test]
fn actual_gpu_ops() {
    let mut g = Gpu::new().unwrap();
    g.warmup().unwrap();
    let data: Vec<_> = (0..65 * 7).map(|i| (i as f32 * 0.13).sin()).collect();
    let x: Vec<_> = (0..65).map(|i| i as f32 / 65.).collect();
    let w = g.weight_f32(&data, &[65, 7]).unwrap();
    let xb = g.upload(&x).unwrap();
    let y = g.zeros(7).unwrap();
    let norm = g.zeros(65).unwrap();
    let nw = g.weight_f32(&vec![1.; 65], &[65]).unwrap();
    let mut b = g.batch().unwrap();
    b.matmul(&w, &xb, &y, 1).unwrap();
    b.norm(&xb, &nw, &norm, 1, 65, 1e-5).unwrap();
    b.finish().unwrap();
    near(&g.read(&y).unwrap(), &tensor_core::matvec(&data, &x));
    near(
        &g.read(&norm).unwrap(),
        &tensor_core::rms_norm(&x, &vec![1.; 65], 1e-5),
    );
}
#[test]
fn attention_and_gqa() {
    let mut g = Gpu::new().unwrap();
    let dim = 8;
    let q = g.upload(&vec![1.; dim * 2]).unwrap();
    let k = g.upload(&vec![0.; dim * 3]).unwrap();
    let v = g
        .upload(&(0..dim * 3).map(|i| i as f32).collect::<Vec<_>>())
        .unwrap();
    let out = g.zeros(dim * 2).unwrap();
    let mut b = g.batch().unwrap();
    b.attention(&q, &k, &v, &out, 2, 1, dim, 3).unwrap();
    b.finish().unwrap();
    let expected: Vec<_> = (0..dim * 2).map(|i| (i % dim + dim) as f32).collect();
    near(&g.read(&out).unwrap(), &expected);
}

#[test]
fn packed_matmul_and_tiled_prefill() {
    use model_format::{DType, dequantize};
    let mut g = Gpu::new().unwrap();
    for ty in [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::Q4_0,
        DType::Q8_0,
        DType::Q4K,
        DType::Q5K,
        DType::Q6K,
    ] {
        let (block, size) = ty.block();
        let n = 256;
        let rows = 19;
        let mut data: Vec<u8> = (0..n / block * size * rows)
            .map(|i| ((i * 73 + 29) % 256) as u8)
            .collect();
        // Finite scales and deliberately nontrivial packed bitplanes.
        for chunk in data.chunks_exact_mut(size) {
            match ty {
                DType::F32 => chunk.copy_from_slice(&0.125f32.to_le_bytes()),
                DType::F16 => chunk.copy_from_slice(&0x3000u16.to_le_bytes()),
                DType::BF16 => chunk.copy_from_slice(&0x3e00u16.to_le_bytes()),
                DType::Q6K => chunk[208..210].copy_from_slice(&0x1800u16.to_le_bytes()),
                _ => {
                    chunk[..2].copy_from_slice(&0x2400u16.to_le_bytes());
                    if matches!(ty, DType::Q4K | DType::Q5K) {
                        chunk[2..4].copy_from_slice(&0x2000u16.to_le_bytes())
                    }
                }
            }
        }
        let ref_weights = dequantize(ty, &data, n * rows).unwrap();
        let w = g.weight_bytes(&data, ty, &[n, rows]).unwrap();
        for tokens in [1, 3, 17] {
            let x: Vec<_> = (0..n * tokens)
                .map(|i| (i as f32 * 0.31).sin() * 0.2)
                .collect();
            let xb = g.upload(&x).unwrap();
            let y = g.zeros(rows * tokens).unwrap();
            let mut b = g.batch().unwrap();
            b.matmul(&w, &xb, &y, tokens).unwrap();
            b.finish().unwrap();
            let expected: Vec<_> = x
                .chunks_exact(n)
                .flat_map(|x| tensor_core::matvec(&ref_weights, x))
                .collect();
            near(&g.read(&y).unwrap(), &expected);
        }
    }
}

#[test]
fn rejects_aliases_and_foreign_buffers() {
    let mut g = Gpu::new().unwrap();
    let mut other = Gpu::new().unwrap();
    let w = g.weight_f32(&vec![1.; 64], &[8, 8]).unwrap();
    let x = g.upload(&[1.; 8]).unwrap();
    let foreign = other.upload(&[1.; 8]).unwrap();
    let mut b = g.batch().unwrap();
    assert!(b.matmul(&w, &x, &x, 1).is_err());
    assert!(b.add(&x, &foreign, &x).is_err());
}

#[test]
fn delta_recurrence_matches_scalar() {
    let mut g = Gpu::new().unwrap();
    let dk = 8;
    let dv = 5;
    let q: Vec<f32> = (0..dk).map(|i| 0.1 + i as f32 * 0.03).collect();
    let k: Vec<f32> = (0..dk).map(|i| -0.2 + i as f32 * 0.04).collect();
    let v: Vec<f32> = (0..dv).map(|i| i as f32 - 2.).collect();
    let mut input = q.clone();
    input.extend(&k);
    input.extend(&v);
    let qkv = g.upload(&input).unwrap();
    let alpha = g.upload(&[0.3]).unwrap();
    let beta = g.upload(&[-0.2]).unwrap();
    let aw = g.upload(&[-0.5]).unwrap();
    let dt = g.upload(&[0.1]).unwrap();
    let state = g.zeros(dk * dv).unwrap();
    let out = g.zeros(dv).unwrap();
    let mut reference = vec![0.; dk * dv];
    let qnorm = q.iter().map(|x| x * x).sum::<f32>().sqrt() * (dk as f32).sqrt();
    let knorm = k.iter().map(|x| x * x).sum::<f32>().sqrt();
    let q: Vec<_> = q.iter().map(|x| x / qnorm).collect();
    let k: Vec<_> = k.iter().map(|x| x / knorm).collect();
    for _ in 0..3 {
        let mut b = g.batch().unwrap();
        b.delta(
            &qkv, &alpha, &beta, &aw, &dt, &state, &out, dk, 1, 1, dv, 1e-6, false, false,
        )
        .unwrap();
        b.finish().unwrap();
        let expected = tensor_core::delta_net(
            &q,
            &k,
            &v,
            (-0.5 * (1. + 0.4f32.exp()).ln()).exp(),
            1. / (1. + 0.2f32.exp()),
            &mut reference,
        );
        near(&g.read(&out).unwrap(), &expected);
        near(&g.read(&state).unwrap(), &reference);
    }
}

#[test]
fn gpu_greedy_ties_and_invalid_logits() {
    let mut g = Gpu::new().unwrap();
    let x = g.upload(&[1., 5., 2., 5., -8.]).unwrap();
    let out = g.zeros(1).unwrap();
    let mut b = g.batch().unwrap();
    b.argmax(&x, &out).unwrap();
    b.finish().unwrap();
    assert_eq!(g.read(&out).unwrap()[0].to_bits(), 1);
    let bad = g.upload(&[1., f32::NAN, 2.]).unwrap();
    let mut b = g.batch().unwrap();
    b.argmax(&bad, &out).unwrap();
    b.finish().unwrap();
    assert_eq!(g.read(&out).unwrap()[0].to_bits(), u32::MAX);
}
