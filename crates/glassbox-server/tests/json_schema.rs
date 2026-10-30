//! 200 random schemas, each decoded by a random model through the scheduler
//! with sampling at temperature 1: every output must parse and validate.

use glassbox_core::sampler::Rng;
use glassbox_core::{Config, Model, SamplerConfig};
use glassbox_server::json_schema::{validate, JsonConstraint, Trie};
use glassbox_server::scheduler::{self, Event, Request, Scheduler};
use serde_json::{json, Value};
use std::sync::Arc;

/// Single bytes plus JSON-ish multi-byte tokens, so tokens straddle structure
/// (`",` `":` `{"`) the way a real BPE vocabulary does.
fn vocab() -> Vec<Vec<u8>> {
    let mut v: Vec<Vec<u8>> = (32u8..127).map(|b| vec![b]).collect();
    for s in [
        "{\"",
        "\":",
        "\",",
        "\"}",
        "\"]",
        "},",
        "}]",
        "true",
        "false",
        "null",
        "12",
        "3.",
        ".5",
        "-1",
        "ab",
        "the",
        " cat",
        "é",
        "\\n",
        "\\\"",
        "name",
        "\"name\":",
        "[1",
        "0,",
        "],",
        ", ",
        "\n",
    ] {
        v.push(s.as_bytes().to_vec());
    }
    v.push(Vec::new()); // end of sequence
    v
}

fn rand_schema(rng: &mut Rng, depth: usize) -> Value {
    let pick = (rng.next_f32() * if depth >= 3 { 6.0 } else { 8.0 }) as usize;
    fn n(rng: &mut Rng) -> usize {
        (rng.next_f32() * 4.0) as usize
    }
    match pick {
        0 => json!({"type": "string", "maxLength": 3 + n(rng) * 4}),
        1 => json!({"type": "integer"}),
        2 => json!({"type": "number"}),
        3 => json!({"type": "boolean"}),
        4 => {
            let all = [json!("red"), json!("green"), json!(3), json!(true), json!(null)];
            json!({"enum": all[..1 + n(rng)]})
        }
        5 => json!({"type": "null"}),
        6 => {
            let k = n(rng);
            let props: serde_json::Map<String, Value> = (0..k)
                .map(|i| (format!("f{i}_{}", depth), rand_schema(rng, depth + 1)))
                .collect();
            let required: Vec<String> = props.keys().filter(|_| rng.next_f32() < 0.6).cloned().collect();
            json!({"type": "object", "properties": props, "required": required})
        }
        _ => {
            let min = n(rng) / 2;
            let extra = n(rng);
            json!({"type": "array", "items": rand_schema(rng, depth + 1), "minItems": min, "maxItems": min + extra})
        }
    }
}

#[test]
fn oversized_enums_are_refused() {
    let vocab = Arc::new(vocab());
    let trie = Arc::new(Trie::new(vocab.clone(), &vec![false; vocab.len()], 0));
    let big: Vec<Value> = (0..65).map(|i| json!(i)).collect();
    assert!(JsonConstraint::new(&json!({ "enum": big }), trie).is_err());
}

#[test]
fn two_hundred_random_schemas_always_validate() {
    let vocab = Arc::new(vocab());
    let eos = (vocab.len() - 1) as u32;
    let special: Vec<bool> = (0..vocab.len()).map(|i| i == eos as usize).collect();
    let trie = Arc::new(Trie::new(vocab.clone(), &special, eos));
    let config = Config {
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        vocab_size: vocab.len(),
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        tie_word_embeddings: true,
        head_dim: None,
        max_position_embeddings: 8192,
    };
    let model = Model::random(config, 3);
    let mut s = Scheduler::new(
        &model,
        scheduler::Config {
            token_budget: 256,
            max_seqs: 16,
            num_blocks: 2048,
            prefix_caching: true,
        },
    );
    let mut rng = Rng::new(2024);
    let mut jobs = Vec::new();
    for i in 0..200 {
        let schema = rand_schema(&mut rng, 0);
        let (sink, rx) = tokio::sync::mpsc::unbounded_channel();
        let c = JsonConstraint::new(&schema, trie.clone()).unwrap();
        s.submit(Request {
            prompt: vec![(i % 90) as u32, 7, 9],
            max_tokens: 4000,
            sampler: SamplerConfig {
                temperature: 1.0,
                seed: i,
                ..Default::default()
            },
            stop: vec![eos],
            sink,
            constraint: Some(Box::new(c)),
        });
        jobs.push((schema, rx));
    }
    while !s.is_idle() {
        s.step();
    }
    for (schema, mut rx) in jobs {
        let mut bytes = Vec::new();
        let mut finished = None;
        while let Ok(e) = rx.try_recv() {
            match e {
                Event::Token(t) => bytes.extend(&vocab[t as usize]),
                Event::Done { reason, .. } => finished = Some(reason),
            }
        }
        assert_eq!(
            finished,
            Some(scheduler::Finish::Stop),
            "schema {schema} did not finish"
        );
        let text = String::from_utf8(bytes).expect("utf-8");
        let v: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}: {text} for {schema}"));
        assert!(validate(&schema, &v), "{text} does not validate against {schema}");
    }
}
