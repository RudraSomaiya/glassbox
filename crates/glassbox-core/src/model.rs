//! The Llama-family transformer: one forward pass for native, server and wasm.

use crate::kernels::{matmul, par_for, DType, Matrix};
use crate::kv::{KvCache, BLOCK};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
}

fn default_eps() -> f32 {
    1e-5
}
fn default_theta() -> f32 {
    10000.0
}
fn default_max_pos() -> usize {
    2048
}

impl Config {
    /// Parse `config.json`. Newer transformers nest `rope_theta` under `rope_parameters`.
    pub fn from_json(text: &str) -> serde_json::Result<Config> {
        let mut v: serde_json::Value = serde_json::from_str(text)?;
        if v.get("rope_theta").is_none() {
            if let Some(t) = v.pointer("/rope_parameters/rope_theta").cloned() {
                v["rope_theta"] = t;
            }
        }
        serde_json::from_value(v)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }
}

pub struct Layer {
    pub attn_norm: Vec<f32>,
    /// Q, K and V projections stacked by rows, so one matmul (one parallel
    /// region) computes all three.
    pub wqkv: Matrix,
    pub wo: Matrix,
    pub mlp_norm: Vec<f32>,
    /// Gate rows then up rows, stacked for the same reason.
    pub w_gate_up: Matrix,
    pub w_down: Matrix,
}

pub struct Model {
    pub config: Config,
    pub embed: Matrix,
    pub layers: Vec<Layer>,
    pub norm: Vec<f32>,
    /// `None` when the head is tied to the embedding.
    pub lm_head: Option<Matrix>,
    /// GGUF files from llama.cpp's converter store Q/K rows permuted for
    /// interleaved RoPE pairs; Hugging Face weights rotate (i, i + d/2).
    pub rope_interleaved: bool,
}

/// What the tracer captured for one token of the batch.
#[derive(Default, Clone)]
pub struct Trace {
    /// Batch index of the token to watch.
    pub token: usize,
    pub capture_residuals: bool,
    pub capture_attention: bool,
    /// Residual stream after each layer, `[layer][hidden]`.
    pub residuals: Vec<Vec<f32>>,
    /// Attention weights, `[layer][head][position]`.
    pub attention: Vec<Vec<Vec<f32>>>,
}

pub fn rms_norm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ms + eps).sqrt();
    for ((o, v), g) in out.iter_mut().zip(x).zip(w) {
        *o = g * (v * s);
    }
}

fn rope(x: &mut [f32], pos: usize, theta: f32, interleaved: bool) {
    let d = x.len();
    let half = d / 2;
    for i in 0..half {
        // f32 angle like transformers, so long positions round the same way.
        let inv = (1.0 / (theta as f64).powf((2 * i) as f64 / d as f64)) as f32;
        let a = pos as f32 * inv;
        let (s, c) = ((a as f64).sin() as f32, (a as f64).cos() as f32);
        let (i0, i1) = if interleaved { (2 * i, 2 * i + 1) } else { (i, i + half) };
        let (x0, x1) = (x[i0], x[i1]);
        x[i0] = x0 * c - x1 * s;
        x[i1] = x1 * c + x0 * s;
    }
}

struct SyncPtr<T>(*mut T);
// SAFETY: attention tasks write disjoint (token, head) slices through this pointer.
unsafe impl<T> Sync for SyncPtr<T> {}

impl Model {
    pub fn quantize(&mut self, to: DType) {
        for l in &mut self.layers {
            for m in [&mut l.wqkv, &mut l.wo, &mut l.w_gate_up, &mut l.w_down] {
                *m = m.quantize(to);
            }
        }
        self.embed = self.embed.quantize(to);
        if let Some(h) = &mut self.lm_head {
            *h = h.quantize(to);
        }
    }

    pub fn weight_bytes(&self) -> usize {
        let mut n = self.embed.bytes() + self.lm_head.as_ref().map_or(0, |m| m.bytes());
        for l in &self.layers {
            n += [&l.wqkv, &l.wo, &l.w_gate_up, &l.w_down]
                .iter()
                .map(|m| m.bytes())
                .sum::<usize>();
        }
        n
    }

    pub fn new_cache(&self, num_blocks: usize) -> KvCache {
        KvCache::new(self.config.num_hidden_layers, self.config.kv_dim(), num_blocks)
    }

    fn head(&self) -> &Matrix {
        self.lm_head.as_ref().unwrap_or(&self.embed)
    }

    /// Final norm and LM head applied to one hidden state.
    pub fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let mut h = vec![0.0; hidden.len()];
        rms_norm(&mut h, hidden, &self.norm, self.config.rms_norm_eps);
        let mut out = vec![0.0; self.config.vocab_size];
        matmul(&mut out, &h, 1, self.head());
        out
    }

    /// Run `tokens` through the model. Token `i` belongs to sequence `seqs[i]`
    /// at position `pos[i]`; each sequence's positions must be contiguous and
    /// follow what the cache already holds (call `cache.reserve` first).
    /// Returns logits `[want.len(), vocab]` for the batch indices in `want`.
    pub fn forward(
        &self,
        tokens: &[u32],
        pos: &[usize],
        seqs: &[usize],
        want: &[usize],
        cache: &mut KvCache,
        mut trace: Option<&mut Trace>,
    ) -> Vec<f32> {
        let c = &self.config;
        let n = tokens.len();
        let (d, hd, nh, nkv) = (
            c.hidden_size,
            c.head_dim(),
            c.num_attention_heads,
            c.num_key_value_heads,
        );
        let kvd = nkv * hd;
        let qd = nh * hd;
        let group = nh / nkv;

        let mut x = vec![0.0; n * d];
        for (i, &t) in tokens.iter().enumerate() {
            self.embed.row(t as usize, &mut x[i * d..(i + 1) * d]);
        }
        let slots: Vec<usize> = (0..n).map(|i| cache.slot(seqs[i], pos[i])).collect();
        let tables: Vec<Vec<u32>> = seqs.iter().map(|&s| cache.seq(s).blocks.clone()).collect();
        let mut h = vec![0.0; n * d];
        let mut qkv = vec![0.0; n * (qd + 2 * kvd)];
        let mut q = vec![0.0; n * qd];
        let mut k = vec![0.0; n * kvd];
        let mut v = vec![0.0; n * kvd];
        let mut att = vec![0.0; n * qd];
        let mut o = vec![0.0; n * d];
        let inter = c.intermediate_size;
        let mut gate_up = vec![0.0; n * 2 * inter];
        let mut act = vec![0.0; n * inter];
        let scale = 1.0 / (hd as f32).sqrt();

        if let Some(t) = trace.as_deref_mut() {
            t.residuals.clear();
            t.attention.clear();
        }

        for (li, layer) in self.layers.iter().enumerate() {
            for i in 0..n {
                rms_norm(
                    &mut h[i * d..(i + 1) * d],
                    &x[i * d..(i + 1) * d],
                    &layer.attn_norm,
                    c.rms_norm_eps,
                );
            }
            matmul(&mut qkv, &h, n, &layer.wqkv);
            for (i, row) in qkv.chunks_exact(qd + 2 * kvd).enumerate() {
                q[i * qd..(i + 1) * qd].copy_from_slice(&row[..qd]);
                k[i * kvd..(i + 1) * kvd].copy_from_slice(&row[qd..qd + kvd]);
                v[i * kvd..(i + 1) * kvd].copy_from_slice(&row[qd + kvd..]);
            }
            for i in 0..n {
                for hq in q[i * qd..(i + 1) * qd].chunks_exact_mut(hd) {
                    rope(hq, pos[i], c.rope_theta, self.rope_interleaved);
                }
                for hk in k[i * kvd..(i + 1) * kvd].chunks_exact_mut(hd) {
                    rope(hk, pos[i], c.rope_theta, self.rope_interleaved);
                }
                let s = slots[i] * kvd;
                cache.k[li][s..s + kvd].copy_from_slice(&k[i * kvd..(i + 1) * kvd]);
                cache.v[li][s..s + kvd].copy_from_slice(&v[i * kvd..(i + 1) * kvd]);
            }

            // Attention, one task per (token, head). Each token sees its own
            // sequence's positions 0..=pos, read through the block table.
            let watch = trace.as_deref().filter(|t| t.capture_attention).map(|t| t.token);
            let mut probs_out: Vec<Vec<f32>> = vec![Vec::new(); if watch.is_some() { nh } else { 0 }];
            let att_ptr = SyncPtr(att.as_mut_ptr());
            let probs_ptr = SyncPtr(probs_out.as_mut_ptr());
            let (kc, vc) = (&cache.k[li], &cache.v[li]);
            par_for(n * nh, |task| {
                let (att_ptr, probs_ptr) = (&att_ptr, &probs_ptr);
                let (i, head) = (task / nh, task % nh);
                let kvh = head / group;
                let qv = &q[i * qd + head * hd..i * qd + (head + 1) * hd];
                let ctx = pos[i] + 1;
                let slot = |p: usize| tables[i][p / BLOCK] as usize * BLOCK + p % BLOCK;
                let mut scores = Vec::with_capacity(ctx);
                for p in 0..ctx {
                    let s = slot(p) * kvd + kvh * hd;
                    scores.push(crate::kernels::dot_f32(qv, &kc[s..s + hd]) * scale);
                }
                let max = scores.iter().fold(f32::NEG_INFINITY, |m, &s| m.max(s));
                let mut sum = 0.0;
                for s in &mut scores {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                // SAFETY: task (i, head) owns att[i*qd + head*hd ..][..hd]; no other task touches it.
                let out = unsafe { std::slice::from_raw_parts_mut(att_ptr.0.add(i * qd + head * hd), hd) };
                out.fill(0.0);
                for (p, s) in scores.iter_mut().enumerate() {
                    *s /= sum;
                    let base = slot(p) * kvd + kvh * hd;
                    for (o, vv) in out.iter_mut().zip(&vc[base..base + hd]) {
                        *o += *s * vv;
                    }
                }
                if watch == Some(i) {
                    // SAFETY: only the tasks of the watched token write here, one slot per head.
                    unsafe { *probs_ptr.0.add(head) = scores };
                }
            });
            if let Some(t) = trace.as_deref_mut().filter(|t| t.capture_attention) {
                t.attention.push(probs_out);
            }

            matmul(&mut o, &att, n, &layer.wo);
            for (xv, ov) in x.iter_mut().zip(&o) {
                *xv += ov;
            }

            for i in 0..n {
                rms_norm(
                    &mut h[i * d..(i + 1) * d],
                    &x[i * d..(i + 1) * d],
                    &layer.mlp_norm,
                    c.rms_norm_eps,
                );
            }
            matmul(&mut gate_up, &h, n, &layer.w_gate_up);
            for (a, gu) in act.chunks_exact_mut(inter).zip(gate_up.chunks_exact(2 * inter)) {
                for ((a, g), u) in a.iter_mut().zip(&gu[..inter]).zip(&gu[inter..]) {
                    *a = g / (1.0 + (-g).exp()) * u;
                }
            }
            matmul(&mut o, &act, n, &layer.w_down);
            for (xv, ov) in x.iter_mut().zip(&o) {
                *xv += ov;
            }
            if let Some(t) = trace.as_deref_mut().filter(|t| t.capture_residuals) {
                t.residuals.push(x[t.token * d..(t.token + 1) * d].to_vec());
            }
        }

        let mut hs = vec![0.0; want.len() * d];
        for (j, &i) in want.iter().enumerate() {
            rms_norm(
                &mut hs[j * d..(j + 1) * d],
                &x[i * d..(i + 1) * d],
                &self.norm,
                c.rms_norm_eps,
            );
        }
        let mut logits = vec![0.0; want.len() * c.vocab_size];
        matmul(&mut logits, &hs, want.len(), self.head());
        logits
    }
}
