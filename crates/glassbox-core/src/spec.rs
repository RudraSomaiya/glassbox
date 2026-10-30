//! Speculative decoding: a cheap drafter proposes `k` tokens, the target model
//! checks all of them in one forward pass, and rejection sampling keeps the
//! output distribution exactly the target's.
//!
//! Verification: with draft distribution q and target distribution p at a
//! position, accept the drafted token x with probability min(1, p(x)/q(x)).
//! On rejection, draw from the residual max(0, p - q) renormalized, and stop.
//! If every draft is accepted, draw one bonus token from the target's next
//! distribution. Under greedy decoding this reduces to "accept while the
//! target's argmax agrees", so the output is identical to the target alone.

use crate::sampler::{argmax, Sampler, SamplerConfig};
use crate::{Generator, Model};
use serde::Serialize;

pub enum Drafter<'m> {
    /// A smaller model with the same tokenizer.
    Model(&'m Model),
    /// Prompt lookup: find the most recent earlier occurrence of the last
    /// `ngram` tokens and propose what followed it. No extra model.
    Lookup { ngram: usize },
}

#[derive(Serialize, Clone, Debug)]
pub struct Round {
    pub drafted: Vec<u32>,
    pub accepted: usize,
    /// The token the target supplied after the accepted drafts (a correction
    /// or the bonus token).
    pub next: u32,
}

#[derive(Serialize, Clone, Debug, Default)]
pub struct Stats {
    pub rounds: Vec<Round>,
    pub drafted: usize,
    pub accepted: usize,
    pub target_passes: usize,
}

impl Stats {
    pub fn acceptance(&self) -> f64 {
        self.accepted as f64 / self.drafted.max(1) as f64
    }
}

/// Full filtered distribution (dense) for one row of logits.
fn dense(s: &Sampler, logits: &[f32], ctx: &[u32]) -> Vec<f32> {
    let mut p = vec![0.0; logits.len()];
    for (t, q) in s.distribution(logits, ctx) {
        p[t as usize] = q;
    }
    p
}

fn lookup(tokens: &[u32], ngram: usize, k: usize) -> Vec<u32> {
    for n in (1..=ngram).rev() {
        if tokens.len() <= n {
            continue;
        }
        let tail = &tokens[tokens.len() - n..];
        // Most recent earlier match.
        for start in (0..tokens.len() - n).rev() {
            if &tokens[start..start + n] == tail {
                let from = start + n;
                return tokens[from..(from + k).min(tokens.len())].to_vec();
            }
        }
    }
    Vec::new()
}

/// Generate `max` tokens after `prompt` with drafts of up to `k` tokens.
pub fn generate(
    target: &Model,
    drafter: &Drafter,
    prompt: &[u32],
    max: usize,
    k: usize,
    config: SamplerConfig,
    stop: Option<u32>,
) -> (Vec<u32>, Stats) {
    let greedy = config.temperature <= 0.0;
    let mut sampler = Sampler::new(config);
    let cap = prompt.len() + max + k + 2;
    let mut tg = Generator::new(target, cap);
    let mut dg = match drafter {
        Drafter::Model(m) => Some(Generator::new(m, cap)),
        Drafter::Lookup { .. } => None,
    };
    let mut stats = Stats::default();

    // Both models read the prompt; the target's logits give the first token.
    let t_logits = tg.feed(prompt, None);
    stats.target_passes += 1;
    if let Some(d) = dg.as_mut() {
        d.feed(&prompt[..prompt.len() - 1], None);
    }
    let mut out: Vec<u32> = Vec::new();
    let mut tokens = prompt.to_vec();
    // The newest token, sampled but not yet fed to either model.
    let mut pending = if greedy {
        argmax(&t_logits) as u32
    } else {
        sampler.sample(&t_logits, &tokens)
    };
    if let Some(d) = dg.as_mut() {
        // Keep the draft one token behind the target, like the target itself.
        d.truncate(prompt.len() - 1);
    }

    while out.len() < max && Some(pending) != stop {
        out.push(pending);
        tokens.push(pending);
        if out.len() >= max {
            break;
        }

        // Draft up to k tokens (and, for a draft model, its distributions).
        let room = (max - out.len()).min(k);
        let mut drafted = Vec::new();
        let mut q: Vec<Vec<f32>> = Vec::new();
        match (drafter, dg.as_mut()) {
            (Drafter::Model(_), Some(d)) => {
                let mut l = d.feed(&tokens[d.tokens.len()..], None);
                for _ in 0..room {
                    let qd = if greedy {
                        Vec::new()
                    } else {
                        dense(&sampler, &l, &tokens)
                    };
                    let x = if greedy {
                        argmax(&l) as u32
                    } else {
                        sampler.draw(&qd) as u32
                    };
                    drafted.push(x);
                    q.push(qd);
                    l = d.feed(&[x], None);
                }
            }
            (Drafter::Lookup { ngram }, _) => drafted = lookup(&tokens, *ngram, room),
            _ => unreachable!(),
        }

        // One target pass over the pending token plus every draft.
        let base = tg.tokens.len();
        let mut feed = vec![pending];
        feed.extend(&drafted);
        tg.cache.reserve(0, feed.len()).expect("cache sized for max + k");
        let pos: Vec<usize> = (base..base + feed.len()).collect();
        let want: Vec<usize> = (0..feed.len()).collect();
        let logits = target.forward(&feed, &pos, &vec![0; feed.len()], &want, &mut tg.cache, None);
        tg.cache.commit(0, &feed);
        tg.tokens.extend(&feed);
        stats.target_passes += 1;
        let vocab = target.config.vocab_size;
        let row = |i: usize| &logits[i * vocab..(i + 1) * vocab];

        let mut accepted = 0;
        let mut next = None;
        for (i, &x) in drafted.iter().enumerate() {
            let ctx = &tokens;
            if greedy {
                let best = argmax(row(i)) as u32;
                if best == x {
                    accepted += 1;
                    continue;
                }
                next = Some(best);
                break;
            }
            let p = dense(&sampler, row(i), ctx);
            let qx = if q.is_empty() { 1.0 } else { q[i][x as usize] };
            if sampler.uniform() * qx < p[x as usize] {
                accepted += 1;
                continue;
            }
            // Residual distribution max(0, p - q); for lookup q is a point mass on x.
            let resid: Vec<f32> = if q.is_empty() {
                p.iter()
                    .enumerate()
                    .map(|(t, &v)| if t == x as usize { 0.0 } else { v })
                    .collect()
            } else {
                p.iter().zip(&q[i]).map(|(a, b)| (a - b).max(0.0)).collect()
            };
            next = Some(sampler.draw(&resid) as u32);
            break;
        }
        // Tokens for accepted drafts become part of the output (stop at `stop`).
        let mut hit_stop = false;
        for &x in &drafted[..accepted] {
            if out.len() >= max || Some(x) == stop {
                hit_stop = Some(x) == stop;
                break;
            }
            out.push(x);
            tokens.push(x);
        }
        let next = next.unwrap_or_else(|| {
            let r = row(accepted);
            if greedy {
                argmax(r) as u32
            } else {
                sampler.draw(&dense(&sampler, r, &tokens)) as u32
            }
        });
        stats.drafted += drafted.len();
        stats.accepted += accepted;
        stats.rounds.push(Round {
            drafted: drafted.clone(),
            accepted,
            next,
        });

        // Roll both caches back to the accepted prefix (the target keeps the
        // pending token and accepted drafts; `next` is the new pending one).
        let keep = tokens.len();
        tg.truncate(keep);
        if let Some(d) = dg.as_mut() {
            d.truncate(d.tokens.len().min(keep));
        }
        if hit_stop {
            break;
        }
        pending = next;
    }
    out.truncate(max);
    (out, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{greedy, Config};

    fn tiny(seed: u64, hidden: usize) -> Model {
        let c = Config {
            hidden_size: hidden,
            intermediate_size: hidden * 2,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            vocab_size: 31,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            head_dim: None,
            max_position_embeddings: 4096,
        };
        Model::random(c, seed)
    }

    #[test]
    fn greedy_is_identical_to_target_alone() {
        let target = tiny(1, 64);
        let draft = tiny(2, 32);
        for (pi, prompt) in [vec![1u32, 2, 3], vec![5, 6, 7, 8, 9, 5, 6, 7], vec![30, 0, 11]]
            .iter()
            .enumerate()
        {
            let want = greedy(&target, prompt, 40);
            for k in [1, 2, 4, 7] {
                let (got, _) = generate(
                    &target,
                    &Drafter::Model(&draft),
                    prompt,
                    40,
                    k,
                    SamplerConfig::default(),
                    None,
                );
                assert_eq!(got, want, "draft model, prompt {pi}, k={k}");
                let (got, _) = generate(
                    &target,
                    &Drafter::Lookup { ngram: 3 },
                    prompt,
                    40,
                    k,
                    SamplerConfig::default(),
                    None,
                );
                assert_eq!(got, want, "prompt lookup, prompt {pi}, k={k}");
            }
        }
        // A model drafting for itself must have every draft accepted.
        let (_, s) = generate(
            &target,
            &Drafter::Model(&target),
            &[1, 2, 3],
            30,
            4,
            SamplerConfig::default(),
            None,
        );
        assert_eq!(s.accepted, s.drafted);
    }

    /// Sampling: the first two generated tokens must follow the target's own
    /// distribution. Chi-square over the second token's marginal, which
    /// depends on both acceptance and the residual draw.
    #[test]
    fn sampling_matches_target_distribution() {
        let target = tiny(3, 64);
        let draft = tiny(4, 32);
        // Repetitive, so prompt lookup finds n-gram matches to draft from.
        let prompt = [4u32, 9, 2, 4, 9, 2, 4];
        let cfg = SamplerConfig {
            temperature: 1.0,
            ..Default::default()
        };
        let s = Sampler::new(cfg.clone());
        let v = target.config.vocab_size;

        // Exact marginal of the second token: sum over x1 of p(x1) p(x2 | x1).
        let mut g = Generator::new(&target, 8);
        let p1 = dense(&s, &g.feed(&prompt, None), &prompt);
        let mut p2 = vec![0f64; v];
        for (x1, &px1) in p1.iter().enumerate() {
            let mut ctx = prompt.to_vec();
            ctx.push(x1 as u32);
            let mut g = Generator::new(&target, 8);
            let pp = dense(&s, &g.feed(&ctx, None), &ctx);
            for (x2, &q) in pp.iter().enumerate() {
                p2[x2] += px1 as f64 * q as f64;
            }
        }

        let n = 6000;
        for drafter in [Drafter::Model(&draft), Drafter::Lookup { ngram: 2 }] {
            let mut counts = vec![0f64; v];
            for seed in 0..n {
                let cfg = SamplerConfig {
                    seed: seed as u64,
                    ..cfg.clone()
                };
                let (out, _) = generate(&target, &drafter, &prompt, 2, 3, cfg, None);
                counts[out[1] as usize] += 1.0;
            }
            // Pool cells with expected count < 5 into one.
            let (mut chi, mut df, mut rest_e, mut rest_o) = (0.0, 0, 0.0, 0.0);
            for t in 0..v {
                let e = p2[t] * n as f64;
                if e < 5.0 {
                    rest_e += e;
                    rest_o += counts[t];
                } else {
                    chi += (counts[t] - e).powi(2) / e;
                    df += 1;
                }
            }
            if rest_e > 0.0 {
                chi += (rest_o - rest_e).powi(2) / rest_e.max(1e-9);
                df += 1;
            }
            df -= 1;
            // 99.9th percentile of chi-square is below df + 3.1 * sqrt(2 df) + 7 for these df.
            let limit = df as f64 + 3.1 * (2.0 * df as f64).sqrt() + 7.0;
            assert!(chi < limit, "chi-square {chi:.1} over {df} dof exceeds {limit:.1}");
        }
    }
}
