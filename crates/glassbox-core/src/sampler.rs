//! Turning logits into the next token.

#[derive(Clone, Debug, PartialEq)]
pub struct SamplerConfig {
    /// 0 means greedy.
    pub temperature: f32,
    /// 0 disables.
    pub top_k: usize,
    /// 1 disables.
    pub top_p: f32,
    /// 0 disables.
    pub min_p: f32,
    /// 1 disables. Applied to tokens already in the context, as in llama.cpp / HF.
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            seed: 0,
        }
    }
}

/// xorshift64*: tiny, seedable, and the same on native and wasm.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9e3779b97f4a7c15) | 1)
    }
    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545f4914f6cdd1d);
        (x >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best
}

pub fn softmax(x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// The `k` largest entries as (index, value), largest first.
pub fn top_k(x: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<u32> = (0..x.len() as u32).collect();
    let k = k.min(x.len());
    if k == 0 {
        return Vec::new();
    }
    idx.select_nth_unstable_by(k - 1, |&a, &b| x[b as usize].total_cmp(&x[a as usize]));
    idx.truncate(k);
    idx.sort_by(|&a, &b| x[b as usize].total_cmp(&x[a as usize]));
    idx.into_iter().map(|i| (i, x[i as usize])).collect()
}

pub struct Sampler {
    pub config: SamplerConfig,
    rng: Rng,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        let rng = Rng::new(config.seed);
        Sampler { config, rng }
    }

    /// The distribution the sampler draws from, after every filter. Sparse:
    /// (token, probability), most likely first. Greedy returns one entry.
    pub fn distribution(&self, logits: &[f32], context: &[u32]) -> Vec<(u32, f32)> {
        let c = &self.config;
        let mut l = logits.to_vec();
        if c.repetition_penalty != 1.0 {
            for &t in context {
                let v = &mut l[t as usize];
                *v = if *v > 0.0 {
                    *v / c.repetition_penalty
                } else {
                    *v * c.repetition_penalty
                };
            }
        }
        if c.temperature <= 0.0 {
            return vec![(argmax(&l) as u32, 1.0)];
        }
        for v in &mut l {
            *v /= c.temperature;
        }
        let k = if c.top_k == 0 { l.len() } else { c.top_k };
        let mut cand = top_k(&l, k);
        let max = cand[0].1;
        let mut sum = 0.0;
        for (_, v) in &mut cand {
            *v = (*v - max).exp();
            sum += *v;
        }
        for (_, v) in &mut cand {
            *v /= sum;
        }
        if c.min_p > 0.0 {
            let floor = cand[0].1 * c.min_p;
            cand.retain(|&(_, p)| p >= floor);
        }
        if c.top_p < 1.0 {
            let mut acc = 0.0;
            let mut keep = cand.len();
            for (i, &(_, p)) in cand.iter().enumerate() {
                acc += p;
                if acc >= c.top_p {
                    keep = i + 1;
                    break;
                }
            }
            cand.truncate(keep);
        }
        let total: f32 = cand.iter().map(|c| c.1).sum();
        for (_, p) in &mut cand {
            *p /= total;
        }
        cand
    }

    pub fn sample(&mut self, logits: &[f32], context: &[u32]) -> u32 {
        let dist = self.distribution(logits, context);
        if dist.len() == 1 {
            return dist[0].0;
        }
        let r = self.rng.next_f32();
        let mut acc = 0.0;
        for &(t, p) in &dist {
            acc += p;
            if r < acc {
                return t;
            }
        }
        dist.last().unwrap().0
    }

    /// Draw from an arbitrary distribution with this sampler's RNG.
    pub fn draw(&mut self, probs: &[f32]) -> usize {
        let r = self.rng.next_f32() * probs.iter().sum::<f32>();
        let mut acc = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if r < acc {
                return i;
            }
        }
        probs.len() - 1
    }

    pub fn uniform(&mut self) -> f32 {
        self.rng.next_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_shape_the_distribution() {
        let logits = [1.0, 3.0, 2.0, 0.0, -1.0];
        let s = Sampler::new(SamplerConfig::default());
        assert_eq!(s.distribution(&logits, &[]), vec![(1, 1.0)]);
        let s = Sampler::new(SamplerConfig {
            temperature: 1.0,
            top_k: 2,
            ..Default::default()
        });
        let d = s.distribution(&logits, &[]);
        assert_eq!(d.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 2]);
        assert!((d[0].1 - 1.0 / (1.0 + (-1f32).exp())).abs() < 1e-6);
        let s = Sampler::new(SamplerConfig {
            temperature: 1.0,
            top_p: 0.5,
            ..Default::default()
        });
        assert_eq!(s.distribution(&logits, &[]).len(), 1);
        let s = Sampler::new(SamplerConfig {
            temperature: 1.0,
            min_p: 0.3,
            ..Default::default()
        });
        assert_eq!(s.distribution(&logits, &[]).len(), 2);
        let s = Sampler::new(SamplerConfig {
            repetition_penalty: 4.0,
            ..Default::default()
        });
        assert_eq!(s.distribution(&logits, &[1]), vec![(2, 1.0)]);
    }

    #[test]
    fn seeded_runs_repeat() {
        let logits: Vec<f32> = (0..50).map(|i| (i as f32 * 0.37).sin()).collect();
        let cfg = SamplerConfig {
            temperature: 1.0,
            seed: 7,
            ..Default::default()
        };
        let (mut a, mut b) = (Sampler::new(cfg.clone()), Sampler::new(cfg));
        let xa: Vec<u32> = (0..20).map(|_| a.sample(&logits, &[])).collect();
        let xb: Vec<u32> = (0..20).map(|_| b.sample(&logits, &[])).collect();
        assert_eq!(xa, xb);
    }
}
