//! Glassbox core: tensors, kernels, quantization, model, tokenizer, sampler, KV cache.

pub mod kernels;
pub mod kv;
pub mod load;
pub mod model;
#[cfg(feature = "threads")]
pub mod pool;
pub mod quant;
pub mod sampler;
pub mod spec;
pub mod tokenizer;

pub use kernels::DType;
pub use model::{Config, Model, Trace};
pub use sampler::{Sampler, SamplerConfig};
pub use tokenizer::Tokenizer;

/// One sequence with its own cache: the simple path used by the CLI, the
/// browser and the reference tests. The server batches many sequences instead.
pub struct Generator<'m> {
    pub model: &'m Model,
    pub cache: kv::KvCache,
    pub tokens: Vec<u32>,
}

impl<'m> Generator<'m> {
    pub fn new(model: &'m Model, max_tokens: usize) -> Self {
        let mut cache = model.new_cache(max_tokens.div_ceil(kv::BLOCK));
        cache.add_seq(0, &[]);
        Generator {
            model,
            cache,
            tokens: Vec::new(),
        }
    }

    /// Append tokens and return the logits after the last one.
    pub fn feed(&mut self, tokens: &[u32], trace: Option<&mut Trace>) -> Vec<f32> {
        let start = self.tokens.len();
        self.cache.reserve(0, tokens.len()).expect("context is full");
        let pos: Vec<usize> = (start..start + tokens.len()).collect();
        let seqs = vec![0; tokens.len()];
        let logits = self
            .model
            .forward(tokens, &pos, &seqs, &[tokens.len() - 1], &mut self.cache, trace);
        self.cache.commit(0, tokens);
        self.tokens.extend_from_slice(tokens);
        logits
    }

    /// Forget everything after the first `len` tokens.
    pub fn truncate(&mut self, len: usize) {
        self.cache.truncate(0, len);
        self.tokens.truncate(len);
    }

    pub fn capacity(&self) -> usize {
        self.cache.num_blocks * kv::BLOCK
    }
}

/// Greedy decode `n` tokens after `prompt`.
pub fn greedy(model: &Model, prompt: &[u32], n: usize) -> Vec<u32> {
    let mut g = Generator::new(model, prompt.len() + n);
    let mut logits = g.feed(prompt, None);
    let mut out = Vec::new();
    for _ in 0..n {
        let t = sampler::argmax(&logits) as u32;
        out.push(t);
        logits = g.feed(&[t], None);
    }
    out
}
