//! Browser bindings. The web app runs this in a Web Worker: one model, one
//! sequence, single-threaded, `simd128` kernels.

use glassbox_core::kv::BLOCK;
use glassbox_core::sampler::{softmax, top_k};
use glassbox_core::{DType, Generator, Model, Sampler, SamplerConfig, Tokenizer, Trace};
use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Engine {
    tok: Tokenizer,
    gen: Generator<'static>,
    logits: Vec<f32>,
    sampler: Sampler,
}

fn err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

/// Top `k` of a probability distribution as `[[id, p], ...]`, plus its entropy in nats.
fn summary(logits: &[f32], k: usize) -> Value {
    let mut p = logits.to_vec();
    softmax(&mut p);
    let entropy: f32 = p.iter().filter(|&&x| x > 0.0).map(|&x| -x * x.ln()).sum();
    json!({ "top": top_k(&p, k), "entropy": entropy })
}

#[wasm_bindgen]
impl Engine {
    /// `weights` is a safetensors file (or a GGUF file when `config` is empty).
    /// `quant` is "f32", "q8_0" or "q4_0".
    #[wasm_bindgen(constructor)]
    pub fn new(config: &str, tokenizer: &str, weights: &[u8], quant: &str, context: usize) -> Result<Engine, JsError> {
        let q = match quant {
            "q8_0" => Some(DType::Q8_0),
            "q4_0" => Some(DType::Q4_0),
            _ => None,
        };
        let model = if config.is_empty() {
            Model::from_gguf(weights).map_err(err)?
        } else {
            Model::from_safetensors(config, weights, q).map_err(err)?
        };
        // One model per page, alive until the worker dies.
        let model: &'static Model = Box::leak(Box::new(model));
        Ok(Engine {
            tok: Tokenizer::from_json(tokenizer).map_err(err)?,
            gen: Generator::new(model, context),
            logits: Vec::new(),
            sampler: Sampler::new(SamplerConfig::default()),
        })
    }

    pub fn info(&self) -> String {
        let c = &self.gen.model.config;
        json!({
            "layers": c.num_hidden_layers, "heads": c.num_attention_heads, "kv_heads": c.num_key_value_heads,
            "hidden": c.hidden_size, "vocab": c.vocab_size, "context": self.gen.capacity(), "block": BLOCK,
            "weight_bytes": self.gen.model.weight_bytes(), "kv_bytes_per_token": 2 * c.num_hidden_layers * c.kv_dim() * 4,
            "simd": glassbox_core::kernels::simd_available(), "eos": self.tok.eos_id,
        })
        .to_string()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tok.encode(text)
    }

    /// ChatML prompt for one user message, as token ids.
    pub fn chat(&self, user: &str) -> Vec<u32> {
        self.tok.encode(&Tokenizer::chat_prompt(&[("user", user)]))
    }

    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        self.tok.token_bytes(id)
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.tok.is_special(id)
    }

    /// Append tokens; returns the top-`k` next-token distribution as JSON.
    pub fn feed(&mut self, tokens: &[u32], k: usize) -> Result<String, JsError> {
        if self.gen.tokens.len() + tokens.len() > self.gen.capacity() {
            return Err(err("context is full"));
        }
        self.logits = self.gen.feed(tokens, None);
        Ok(summary(&self.logits, k).to_string())
    }

    /// Raw logits after the last token, for the sampling playground.
    pub fn logits(&self) -> Vec<f32> {
        self.logits.clone()
    }

    /// Sample the next token from the last logits with a sampler config
    /// (`{"temperature":..,"top_k":..,"top_p":..,"min_p":..,"seed":..}`).
    pub fn sample(&mut self, config: &str) -> Result<u32, JsError> {
        let v: Value = serde_json::from_str(config).map_err(err)?;
        let f = |k: &str, d: f64| v.get(k).and_then(|x| x.as_f64()).unwrap_or(d);
        let cfg = SamplerConfig {
            temperature: f("temperature", 0.0) as f32,
            top_k: f("top_k", 0.0) as usize,
            top_p: f("top_p", 1.0) as f32,
            min_p: f("min_p", 0.0) as f32,
            repetition_penalty: f("repetition_penalty", 1.0) as f32,
            seed: f("seed", 0.0) as u64,
        };
        if cfg != self.sampler.config {
            self.sampler = Sampler::new(cfg);
        }
        Ok(self.sampler.sample(&self.logits, &self.gen.tokens))
    }

    /// Forget everything after the first `len` tokens (forking the story).
    /// Returns the distribution after the kept prefix, recomputed.
    pub fn truncate(&mut self, len: usize, k: usize) -> Result<String, JsError> {
        let last = *self
            .gen
            .tokens
            .get(len.wrapping_sub(1))
            .ok_or_else(|| err("nothing to keep"))?;
        self.gen.truncate(len - 1);
        self.feed(&[last], k)
    }

    /// Start over with an empty context.
    pub fn reset(&mut self) {
        self.gen.truncate(0);
        self.logits.clear();
    }

    pub fn len(&self) -> usize {
        self.gen.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gen.tokens.is_empty()
    }

    /// What the model "thinks" at token index `pos`: for each layer, the top
    /// predictions if the residual stream stopped there (logit lens), and the
    /// attention weights of every head. Re-runs the prefix with a tracer, so
    /// it costs one prefill and nothing is captured during normal generation.
    pub fn inspect(&self, pos: usize, k: usize) -> Result<String, JsError> {
        let model = self.gen.model;
        let tokens = self
            .gen
            .tokens
            .get(..=pos)
            .ok_or_else(|| err("position out of range"))?;
        let mut g = Generator::new(model, tokens.len());
        let mut t = Trace {
            token: tokens.len() - 1,
            capture_residuals: true,
            capture_attention: true,
            ..Default::default()
        };
        g.feed(tokens, Some(&mut t));
        let lens: Vec<Value> = t.residuals.iter().map(|r| summary(&model.logits(r), k)).collect();
        Ok(json!({ "lens": lens, "attention": t.attention }).to_string())
    }

    /// Block table and reference counts, for the memory view.
    pub fn cache(&self) -> String {
        let c = &self.gen.cache;
        json!({
            "blocks": c.seq(0).blocks, "len": c.seq(0).len, "num_blocks": c.num_blocks,
            "used": c.used_blocks(), "block": BLOCK,
        })
        .to_string()
    }
}
