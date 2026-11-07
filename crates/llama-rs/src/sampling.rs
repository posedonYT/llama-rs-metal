use crate::{Error, Result};
#[derive(Clone, Debug)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}
impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.,
            repeat_last_n: 64,
            seed: 42,
        }
    }
}
impl Sampling {
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite()
            || self.temperature < 0.
            || !self.top_p.is_finite()
            || self.top_p <= 0.
            || self.top_p > 1.
            || !self.repeat_penalty.is_finite()
            || self.repeat_penalty <= 0.
        {
            return Err(Error::Options("invalid sampling parameters".into()));
        }
        Ok(())
    }
    pub(crate) fn sample(&self, logits: &[f32], history: &[u32], rng: &mut Rng) -> Result<u32> {
        if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
            return Err(Error::Options("empty or non-finite logits".into()));
        }
        if self.temperature == 0. && self.repeat_penalty == 1. {
            return Ok(logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
                .unwrap()
                .0 as u32);
        }
        let mut candidates: Vec<(u32, f64)> = logits
            .iter()
            .enumerate()
            .map(|(i, x)| (i as u32, *x as f64))
            .collect();
        if self.repeat_penalty != 1. {
            let mut seen = std::collections::HashSet::new();
            for &t in history.iter().rev().take(self.repeat_last_n) {
                if seen.insert(t) {
                    if let Some((_, v)) = candidates.get_mut(t as usize) {
                        *v = if *v <= 0. {
                            *v * self.repeat_penalty as f64
                        } else {
                            *v / self.repeat_penalty as f64
                        }
                    }
                }
            }
        }
        if self.top_k > 0 && self.top_k < candidates.len() {
            candidates
                .select_nth_unstable_by(self.top_k, |a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            candidates.truncate(self.top_k)
        }
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        if self.temperature == 0. {
            return Ok(candidates[0].0);
        }
        if self.top_k > 0 {
            candidates.truncate(self.top_k)
        }
        let mx = candidates[0].1;
        let mut total = 0.;
        for (_, v) in &mut candidates {
            *v = ((*v - mx) / self.temperature as f64).exp();
            total += *v
        }
        let mut sum = 0.;
        let mut keep = candidates.len();
        for (i, (_, v)) in candidates.iter().enumerate() {
            sum += *v;
            if sum / total >= self.top_p as f64 {
                keep = i + 1;
                break;
            }
        }
        candidates.truncate(keep);
        total = candidates.iter().map(|(_, v)| v).sum();
        let r = rng.uniform() * total;
        let mut acc = 0.;
        for (id, v) in &candidates {
            acc += v;
            if r < acc {
                return Ok(*id);
            }
        }
        Ok(candidates.last().unwrap().0)
    }
}
pub(crate) struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deterministic_sampling() {
        let p = Sampling {
            temperature: 0.,
            ..Default::default()
        };
        assert_eq!(p.sample(&[1., 3., 3.], &[], &mut Rng::new(0)).unwrap(), 1);
        let p = Sampling::default();
        let a: Vec<_> = (0..20)
            .map(|_| p.sample(&[1., 2., 3.], &[], &mut Rng::new(7)).unwrap())
            .collect();
        assert!(a.iter().all(|t| *t == a[0]));
    }
    #[test]
    fn reject_nan() {
        assert!(
            Sampling {
                temperature: f32::NAN,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
}
