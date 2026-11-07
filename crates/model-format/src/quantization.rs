use crate::error::invalid;
use crate::{DType, Result};
use half::{bf16, f16};
fn h(b: &[u8], i: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([b[i], b[i + 1]])).to_f32()
}
fn scale_min(s: &[u8], j: usize) -> (u8, u8) {
    if j < 4 {
        (s[j] & 63, s[j + 4] & 63)
    } else {
        (
            (s[j + 4] & 15) | ((s[j - 4] >> 6) << 4),
            (s[j + 4] >> 4) | ((s[j] >> 6) << 4),
        )
    }
}
/// Independent scalar decoder used as a numerical reference for GPU kernels.
pub fn dequantize(ty: DType, bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    let (block, size) = ty.block();
    if n % block != 0 || n.checked_div(block).and_then(|n| n.checked_mul(size)) != Some(bytes.len())
    {
        return Err(invalid("dequantization length"));
    }
    let mut out = Vec::with_capacity(n);
    for b in bytes.chunks_exact(size) {
        match ty {
            DType::F32 => out.push(f32::from_le_bytes(b.try_into().unwrap())),
            DType::F16 => out.push(h(b, 0)),
            DType::BF16 => {
                out.push(bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            }
            DType::Q4_0 => {
                for i in 0..32 {
                    let q = (b[2 + i % 16] >> (4 * (i / 16))) & 15;
                    out.push(h(b, 0) * (q as f32 - 8.));
                }
            }
            DType::Q8_0 => {
                for i in 0..32 {
                    out.push(h(b, 0) * b[2 + i] as i8 as f32)
                }
            }
            DType::Q4K | DType::Q5K => {
                for i in 0..256 {
                    let g = i / 32;
                    let (s, m) = scale_min(&b[4..16], g);
                    let qoff = if ty == DType::Q5K { 48 } else { 16 };
                    let mut q = (b[qoff + (i / 64) * 32 + i % 32] >> (4 * (g % 2))) & 15;
                    if ty == DType::Q5K {
                        q |= ((b[16 + i % 32] >> g) & 1) << 4;
                    }
                    out.push(h(b, 0) * s as f32 * q as f32 - h(b, 2) * m as f32);
                }
            }
            DType::Q6K => {
                for i in 0..256 {
                    let half = i / 128;
                    let j = i % 128;
                    let lane = j % 32;
                    let quadrant = j / 32;
                    let lo = b[half * 64 + lane + if quadrant % 2 == 0 { 0 } else { 32 }];
                    let lo = (lo >> if quadrant < 2 { 0 } else { 4 }) & 15;
                    let hi = (b[128 + half * 32 + lane] >> (quadrant * 2)) & 3;
                    let q = (lo | (hi << 4)) as i32 - 32;
                    out.push(h(b, 208) * b[192 + i / 16] as i8 as f32 * q as f32);
                }
            }
        }
    }
    Ok(out)
}
