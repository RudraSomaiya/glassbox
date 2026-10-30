//! Parity with Hugging Face, against fixtures written by
//! `tools/reference/make_golden.py`. Skips models that haven't been fetched.

use glassbox_core::{Generator, Model, Tokenizer};
use serde_json::Value;
use std::path::PathBuf;

fn model_dirs() -> Vec<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root)
        .map(|r| {
            r.flatten()
                .map(|e| e.path())
                .filter(|p| p.join("golden.json").exists())
                .collect()
        })
        .unwrap_or_default();
    dirs.sort();
    if dirs.is_empty() {
        eprintln!("no models with golden.json; run tools/reference/make_golden.py");
    }
    dirs
}

fn load(path: PathBuf) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn tokenizer_matches_hf() {
    for dir in model_dirs() {
        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        let cases = load(dir.join("tokenizer_parity.json"));
        let cases = cases.as_array().unwrap();
        let mut bad = 0;
        for c in cases {
            let text = c["text"].as_str().unwrap();
            let want: Vec<u32> = c["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let got = tok.encode(text);
            if got != want {
                bad += 1;
                if bad <= 5 {
                    eprintln!("mismatch on {text:?}\n  hf:   {want:?}\n  ours: {got:?}");
                }
            }
            assert_eq!(tok.decode(&got), text, "decode roundtrip");
        }
        assert_eq!(bad, 0, "{}: {bad} of {} strings differ", dir.display(), cases.len());
        eprintln!("{}: {} strings match", dir.display(), cases.len());
    }
}

/// transformers runs the same f32 math in a different order (fused BLAS
/// kernels, other summation trees), so exact equality is impossible. 1e-3 on
/// logits of order 1-10 is ~1e-4 relative, a few hundred ulps after 30 layers.
const TOL: f32 = 1e-3;

#[test]
fn logits_and_greedy_match_hf() {
    for dir in model_dirs() {
        let model = Model::load_dir(&dir).unwrap();
        let g = load(dir.join("golden.json"));
        let mut worst = 0f32;
        for case in g["cases"].as_array().unwrap() {
            let ids: Vec<u32> = case["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let mut gen = Generator::new(&model, ids.len() + 64);
            gen.cache.reserve(0, ids.len()).unwrap();
            let pos: Vec<usize> = (0..ids.len()).collect();
            let want_pos: Vec<usize> = case["logits"]
                .as_object()
                .unwrap()
                .keys()
                .map(|k| k.parse().unwrap())
                .collect();
            let logits = model.forward(&ids, &pos, &vec![0; ids.len()], &want_pos, &mut gen.cache, None);
            let vocab = model.config.vocab_size;
            for (j, p) in want_pos.iter().enumerate() {
                let reference = &case["logits"][p.to_string()];
                let row = &logits[j * vocab..(j + 1) * vocab];
                for (id, v) in reference["ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(reference["logits"].as_array().unwrap())
                {
                    let diff = (row[id.as_u64().unwrap() as usize] - v.as_f64().unwrap() as f32).abs();
                    worst = worst.max(diff);
                    assert!(diff < TOL, "{}: logit diff {diff} at pos {p}", dir.display());
                }
            }

            let want: Vec<u32> = case["greedy"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let got = glassbox_core::greedy(&model, &ids, want.len());
            if let Some(i) = got.iter().zip(&want).position(|(a, b)| a != b) {
                let margin = case["margins"][i].as_f64().unwrap();
                panic!(
                    "{}: greedy differs at step {i} (reference top-2 margin {margin})",
                    dir.display()
                );
            }
        }
        eprintln!("{}: max |logit diff| {worst:.2e}", dir.display());
    }
}
