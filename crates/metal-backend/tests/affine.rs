use metal_backend::Gpu;
use model_format::{AffineQuantization, DType, Tensor, TensorBytes, WeightData};
fn affine(dims: Vec<usize>, bits: u32, group_size: usize, scale_type: DType) -> Tensor {
    let n: usize = dims.iter().product();
    let mask = (1u32 << bits) - 1;
    let mut words = vec![0u32; n / (32 / bits as usize)];
    for i in 0..n {
        words[i / (32 / bits as usize)] |=
            ((i * 37 + i / 13) as u32 & mask) << ((i % (32 / bits as usize)) as u32 * bits)
    }
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for i in 0..n / group_size {
        let alternate = i % 2 == 1;
        match scale_type {
            DType::F32 => {
                scales.extend(if alternate { 0.25f32 } else { 0.125 }.to_le_bytes());
                biases.extend(if alternate { -0.25f32 } else { -0.5 }.to_le_bytes());
            }
            DType::F16 => {
                scales.extend(if alternate { 0x3400u16 } else { 0x3000 }.to_le_bytes());
                biases.extend(if alternate { 0xb400u16 } else { 0xb800 }.to_le_bytes());
            }
            DType::BF16 => {
                scales.extend(if alternate { 0x3e80u16 } else { 0x3e00 }.to_le_bytes());
                biases.extend(if alternate { 0xbe80u16 } else { 0xbf00 }.to_le_bytes());
            }
            _ => unreachable!(),
        }
    }
    Tensor::new(
        dims,
        WeightData::Affine {
            packed: TensorBytes::owned(words.iter().flat_map(|v| v.to_le_bytes()).collect()),
            quantization: AffineQuantization {
                bits,
                group_size,
                scale_type,
                scales: TensorBytes::owned(scales),
                biases: TensorBytes::owned(biases),
            },
        },
    )
    .unwrap()
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() < 0.0005 * (1. + b.abs()), "{i}: {a} != {b}")
    }
}
#[test]
fn affine_embedding_gemv_and_prefill_all_layouts() {
    let mut gpu = Gpu::new().unwrap();
    for bits in [2, 4, 8] {
        for group in [32, 64, 128] {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let tensor = affine(vec![256, 19], bits, group, dtype);
                let reference = tensor.decode().unwrap();
                let weight = gpu.weight_tensor(&tensor).unwrap();
                for count in [1, 3, 17] {
                    let x: Vec<_> = (0..256 * count)
                        .map(|i| (i as f32 * 0.27).sin() * 0.1)
                        .collect();
                    let input = gpu.upload(&x).unwrap();
                    let output = gpu.zeros(19 * count).unwrap();
                    let mut batch = gpu.batch().unwrap();
                    batch.matmul(&weight, &input, &output, count).unwrap();
                    batch.finish().unwrap();
                    let expected: Vec<_> = x
                        .chunks_exact(256)
                        .flat_map(|v| tensor_core::matvec(&reference, v))
                        .collect();
                    close(&gpu.read(&output).unwrap(), &expected)
                }
                let embedded = gpu.zeros(256 * 2).unwrap();
                let mut b = gpu.batch().unwrap();
                b.embeddings(&weight, &[0, 18], &embedded).unwrap();
                b.finish().unwrap();
                let expected: Vec<_> = reference[..256]
                    .iter()
                    .chain(&reference[18 * 256..])
                    .copied()
                    .collect();
                close(&gpu.read(&embedded).unwrap(), &expected);
            }
        }
    }
}
#[test]
fn affine_experts_use_gpu_selected_ids() {
    let mut gpu = Gpu::new().unwrap();
    let tensor = affine(vec![64, 3, 4], 4, 32, DType::F16);
    let values = tensor.decode().unwrap();
    let weight = gpu.weight_tensor(&tensor).unwrap();
    let logits = gpu.upload(&[1., 4., 3., 2.]).unwrap();
    let ids = gpu.zeros(2).unwrap();
    let probabilities = gpu.zeros(2).unwrap();
    let x: Vec<_> = (0..64).map(|i| i as f32 * 0.01).collect();
    let input = gpu.upload(&x).unwrap();
    let output = gpu.zeros(6).unwrap();
    let mut batch = gpu.batch().unwrap();
    batch.route(&logits, &ids, &probabilities, true).unwrap();
    batch
        .expert_matmul(&weight, &input, &ids, &output, false)
        .unwrap();
    batch.finish().unwrap();
    let expected: Vec<_> = [1, 2]
        .into_iter()
        .flat_map(|e| tensor_core::matvec(&values[e * 3 * 64..(e + 1) * 3 * 64], &x))
        .collect();
    close(&gpu.read(&output).unwrap(), &expected);
}
