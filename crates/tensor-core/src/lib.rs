//! Layouts and deliberately simple CPU oracles, independent of Metal dispatch.
use thiserror::Error;
#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid tensor layout: {0}")]
    Layout(&'static str),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}
impl Layout {
    pub fn contiguous(shape: &[usize]) -> Result<Self, Error> {
        let mut n = 1usize;
        let mut strides = Vec::new();
        for &d in shape {
            if d == 0 {
                return Err(Error::Layout("zero dimension"));
            }
            strides.push(n);
            n = n.checked_mul(d).ok_or(Error::Layout("overflow"))?;
        }
        Ok(Self {
            shape: shape.to_vec(),
            strides,
            offset: 0,
        })
    }
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn is_contiguous(&self) -> bool {
        let mut n = 1;
        for (&d, &s) in self.shape.iter().zip(&self.strides) {
            if d > 1 && s != n {
                return false;
            }
            n *= d;
        }
        true
    }
    pub fn reshape(&self, shape: &[usize]) -> Result<Self, Error> {
        let mut new = Self::contiguous(shape)?;
        if !self.is_contiguous() || new.elements() != self.elements() {
            return Err(Error::Layout(
                "reshape requires contiguous equal-sized view",
            ));
        }
        new.offset = self.offset;
        Ok(new)
    }
    pub fn transpose(&self, a: usize, b: usize) -> Result<Self, Error> {
        if a >= self.shape.len() || b >= self.shape.len() {
            return Err(Error::Layout("axis"));
        }
        let mut v = self.clone();
        v.shape.swap(a, b);
        v.strides.swap(a, b);
        Ok(v)
    }
}
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), w.len());
    let scale = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps)
        .sqrt()
        .recip();
    x.iter().zip(w).map(|(x, w)| x * w * scale).collect()
}
pub fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut v: Vec<_> = x.iter().map(|x| (x - m).exp()).collect();
    let s = v.iter().sum::<f32>();
    for x in &mut v {
        *x /= s
    }
    v
}
pub fn matvec(w: &[f32], x: &[f32]) -> Vec<f32> {
    assert_eq!(w.len() % x.len(), 0);
    w.chunks_exact(x.len())
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}
pub fn silu(x: f32) -> f32 {
    x / (1. + (-x).exp())
}
/// State layout: [head][value][key]. Updates read and write each value-row in place.
pub fn delta_net(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay: f32,
    beta: f32,
    state: &mut [f32],
) -> Vec<f32> {
    let dk = k.len();
    assert_eq!(q.len(), dk);
    assert_eq!(state.len(), dk * v.len());
    let mut out = vec![0.; v.len()];
    for (j, row) in state.chunks_exact_mut(dk).enumerate() {
        for s in row.iter_mut() {
            *s *= decay
        }
        let prediction: f32 = row.iter().zip(k).map(|(s, k)| s * k).sum();
        let d = beta * (v[j] - prediction);
        for (s, k) in row.iter_mut().zip(k) {
            *s += d * k
        }
        out[j] = row.iter().zip(q).map(|(s, q)| s * q).sum();
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn views() {
        let l = Layout::contiguous(&[2, 3]).unwrap();
        assert!(l.reshape(&[6]).is_ok());
        assert!(l.transpose(0, 1).unwrap().reshape(&[6]).is_err());
    }
    #[test]
    fn delta_state() {
        let mut s = vec![0.; 4];
        let a = delta_net(&[1., 0.], &[1., 0.], &[2., 3.], 0.9, 0.5, &mut s);
        assert_eq!(a, vec![1., 1.5]);
        let b = delta_net(&[1., 0.], &[1., 0.], &[2., 3.], 1., 1., &mut s);
        assert_eq!(b, vec![2., 3.]);
    }
}
