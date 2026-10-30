//! Property tests for the scheduler on a tiny random model: batching,
//! chunked prefill, preemption, aborts and prefix sharing must never change
//! what greedy decoding produces, and must never leak a KV block.

use glassbox_core::sampler::Rng;
use glassbox_core::{greedy, Config, Model, SamplerConfig};
use glassbox_server::scheduler::{self, Event, Request, Scheduler};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};

fn tiny() -> Model {
    let config = Config {
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        vocab_size: 97,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        tie_word_embeddings: true,
        head_dim: None,
        max_position_embeddings: 4096,
    };
    Model::random(config, 7)
}

fn request(prompt: Vec<u32>, max_tokens: usize) -> (Request, UnboundedReceiver<Event>) {
    let (sink, rx) = unbounded_channel();
    let r = Request {
        prompt,
        max_tokens,
        sampler: SamplerConfig::default(),
        stop: vec![],
        sink,
        constraint: None,
    };
    (r, rx)
}

fn drain(rx: &mut UnboundedReceiver<Event>) -> (Vec<u32>, Option<scheduler::Finish>) {
    let mut out = Vec::new();
    let mut fin = None;
    while let Ok(e) = rx.try_recv() {
        match e {
            Event::Token(t) => out.push(t),
            Event::Done { reason, .. } => fin = Some(reason),
        }
    }
    (out, fin)
}

fn random_prompt(rng: &mut Rng, prefix: &[u32]) -> Vec<u32> {
    let mut p = prefix.to_vec();
    let n = 1 + (rng.next_f32() * 60.0) as usize;
    p.extend((0..n).map(|_| (rng.next_f32() * 97.0) as u32 % 97));
    p
}

fn run_all(
    model: &Model,
    config: scheduler::Config,
    jobs: &[(Vec<u32>, usize)],
) -> (Vec<Vec<u32>>, scheduler::Metrics) {
    let mut s = Scheduler::new(model, config);
    let mut rxs: Vec<_> = jobs
        .iter()
        .map(|(p, n)| {
            let (r, rx) = request(p.clone(), *n);
            s.submit(r);
            rx
        })
        .collect();
    let mut guard = 0;
    while !s.is_idle() {
        s.step();
        guard += 1;
        assert!(guard < 10_000, "scheduler made no progress");
    }
    s.cache.check_invariants();
    assert_eq!(s.cache.free_blocks(), s.cache.num_blocks, "KV blocks leaked");
    (rxs.iter_mut().map(|rx| drain(rx).0).collect(), s.metrics.clone())
}

#[test]
fn batched_equals_unbatched() {
    let model = tiny();
    for seed in 0..4 {
        let mut rng = Rng::new(seed);
        let jobs: Vec<(Vec<u32>, usize)> = (0..10)
            .map(|_| (random_prompt(&mut rng, &[]), 1 + (rng.next_f32() * 24.0) as usize))
            .collect();
        // Small budget forces chunked prefill alongside decoding.
        let config = scheduler::Config {
            token_budget: 16,
            max_seqs: 6,
            num_blocks: 256,
            prefix_caching: true,
        };
        let (outs, _) = run_all(&model, config, &jobs);
        for ((p, n), out) in jobs.iter().zip(&outs) {
            assert_eq!(out, &greedy(&model, p, *n), "seed {seed}: batched output differs");
        }
    }
}

#[test]
fn preemption_keeps_outputs_and_frees_blocks() {
    let model = tiny();
    let mut rng = Rng::new(99);
    let jobs: Vec<(Vec<u32>, usize)> = (0..12).map(|_| (random_prompt(&mut rng, &[]), 30)).collect();
    // 14 blocks = 224 token slots for 12 sequences that each need ~4: forces preemption.
    let config = scheduler::Config {
        token_budget: 64,
        max_seqs: 12,
        num_blocks: 14,
        prefix_caching: true,
    };
    let (outs, m) = run_all(&model, config, &jobs);
    assert!(m.preemptions > 0, "test did not exercise preemption");
    for ((p, n), out) in jobs.iter().zip(&outs) {
        assert_eq!(out, &greedy(&model, p, *n));
    }
}

#[test]
fn aborts_free_blocks() {
    let model = tiny();
    let mut rng = Rng::new(5);
    let mut s = Scheduler::new(
        &model,
        scheduler::Config {
            token_budget: 32,
            max_seqs: 8,
            num_blocks: 64,
            prefix_caching: true,
        },
    );
    let mut ids = Vec::new();
    let mut rxs = Vec::new();
    for _ in 0..16 {
        let (r, rx) = request(random_prompt(&mut rng, &[]), 40);
        ids.push(s.submit(r));
        rxs.push(rx);
    }
    for step in 0..400 {
        if s.is_idle() {
            break;
        }
        s.step();
        if step % 3 == 0 {
            let id = ids[(rng.next_f32() * ids.len() as f32) as usize];
            s.abort(id);
        }
        s.cache.check_invariants();
    }
    // A client hanging up counts as an abort too.
    drop(rxs.pop());
    while !s.is_idle() {
        s.step();
    }
    s.cache.check_invariants();
    assert_eq!(s.cache.free_blocks(), s.cache.num_blocks);
}

#[test]
fn prefix_sharing_never_changes_outputs() {
    let model = tiny();
    let mut rng = Rng::new(11);
    let system: Vec<u32> = (0..70).map(|i| (i * 13 % 97) as u32).collect();
    let jobs: Vec<(Vec<u32>, usize)> = (0..8).map(|_| (random_prompt(&mut rng, &system), 12)).collect();
    let on = scheduler::Config {
        token_budget: 48,
        max_seqs: 4,
        num_blocks: 128,
        prefix_caching: true,
    };
    let off = scheduler::Config {
        prefix_caching: false,
        ..on.clone()
    };
    let (a, ma) = run_all(&model, on, &jobs);
    let (b, mb) = run_all(&model, off, &jobs);
    assert_eq!(a, b);
    assert!(ma.prefix_hit_tokens >= 64, "shared system prompt was not reused");
    assert_eq!(mb.prefix_hit_tokens, 0);
}
