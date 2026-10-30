//! Benchmark harness. Every result file in bench/results records the CPU,
//! thread count, commit and date next to the numbers.

use glassbox_core::kernels::{matmul, set_simd, Matrix};
use glassbox_core::{pool, DType, Generator, Model};
use serde_json::{json, Value};
use std::time::Instant;

const USAGE: &str = "usage:
  glassbox-bench single      --model DIR [--gguf FILE] [--dtype f32|q8_0|q4_0] [--threads N] [--label L] [--out FILE]
  glassbox-bench progression --model DIR [--out FILE]
  glassbox-bench quality     --model DIR --file TEXT [--ctx 512] [--chunks 100] [--out FILE]
  glassbox-bench kernels";

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

pub fn meta() -> Value {
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .map(|l| l.split(':').nth(1).unwrap().trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into());
    let sh = |cmd: &[&str]| {
        std::process::Command::new(cmd[0])
            .args(&cmd[1..])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default()
    };
    json!({
        "cpu": cpu,
        "logical_cpus": std::thread::available_parallelism().map_or(1, |n| n.get()),
        "commit": sh(&["git", "rev-parse", "--short", "HEAD"]),
        "date": sh(&["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]),
        "simd": glassbox_core::kernels::simd_available(),
    })
}

fn load(dir: &str, gguf: Option<&str>, dtype: Option<&str>) -> Model {
    let mut m = match gguf {
        Some(f) => Model::load_gguf(f),
        None => Model::load_dir(dir),
    }
    .expect("load model");
    match dtype {
        Some("q8_0") => m.quantize(DType::Q8_0),
        Some("q4_0") => m.quantize(DType::Q4_0),
        _ => {}
    }
    m
}

fn stats(xs: &[f64]) -> (f64, f64) {
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
    (mean, var.sqrt())
}

/// Prefill `pp` tokens in one forward call, then decode `tg` tokens one at a
/// time: the same shape as llama-bench's ppN / tgN tests.
fn pp_tg(model: &Model, pp: usize, tg: usize, reps: usize) -> ((f64, f64), (f64, f64)) {
    let vocab = model.config.vocab_size as u32;
    let prompt: Vec<u32> = (0..pp as u32).map(|i| (i * 7919 + 13) % vocab).collect();
    let (mut pps, mut tgs) = (Vec::new(), Vec::new());
    for rep in 0..reps + 1 {
        let mut g = Generator::new(model, pp + tg + 1);
        let t = Instant::now();
        g.feed(&prompt, None);
        let p = pp as f64 / t.elapsed().as_secs_f64();
        let mut g = Generator::new(model, tg + 1);
        let t = Instant::now();
        let mut tok = 1;
        for _ in 0..tg {
            let l = g.feed(&[tok], None);
            tok = glassbox_core::sampler::argmax(&l) as u32;
        }
        let d = tg as f64 / t.elapsed().as_secs_f64();
        // First run warms caches and the thread pool.
        if rep > 0 {
            pps.push(p);
            tgs.push(d);
        }
    }
    (stats(&pps), stats(&tgs))
}

fn write(out: Option<&str>, v: &Value) {
    let text = serde_json::to_string_pretty(v).unwrap();
    match out {
        Some(f) => std::fs::write(f, text + "\n").expect("write results"),
        None => println!("{text}"),
    }
}

fn single(args: &[String]) {
    let dir = arg(args, "--model").expect(USAGE);
    let threads: usize = arg(args, "--threads")
        .and_then(|t| t.parse().ok())
        .unwrap_or(pool::threads());
    pool::set_threads(threads);
    let model = load(dir, arg(args, "--gguf"), arg(args, "--dtype"));
    let ((pp, pp_sd), (tg, tg_sd)) = pp_tg(&model, 512, 128, 3);
    eprintln!("pp512 {pp:.1} ± {pp_sd:.1} tok/s, tg128 {tg:.1} ± {tg_sd:.1} tok/s");
    let dtype = format!("{:?}", model.layers[0].wqkv.dtype());
    write(
        arg(args, "--out"),
        &json!({
            "engine": "glassbox",
            "label": arg(args, "--label").unwrap_or(dir),
            "model": dir,
            "source": if arg(args, "--gguf").is_some() { "gguf" } else { "safetensors" },
            "dtype": dtype,
            "threads": threads,
            "weight_mb": model.weight_bytes() as f64 / 1e6,
            "pp512_tok_s": pp, "pp512_sd": pp_sd,
            "tg128_tok_s": tg, "tg128_sd": tg_sd,
            "meta": meta(),
        }),
    );
}

/// How each step of the kernel work moved decode speed, on one model.
fn progression(args: &[String]) {
    let dir = arg(args, "--model").expect(USAGE);
    let all = pool::threads();
    let mut steps = Vec::new();
    for (name, dtype, simd, threads) in [
        ("scalar, 1 thread", "f32", false, 1),
        ("AVX2, 1 thread", "f32", true, 1),
        (&format!("AVX2, {all} threads")[..], "f32", true, all),
        ("Q8_0 int8 dot", "q8_0", true, all),
        ("Q4_0 int8 dot", "q4_0", true, all),
    ] {
        let model = load(dir, None, Some(dtype));
        set_simd(simd);
        pool::set_threads(threads);
        let ((pp, _), (tg, tg_sd)) = pp_tg(&model, 64, 32, 2);
        eprintln!("{name:22} pp64 {pp:7.1}  tg32 {tg:6.1} ± {tg_sd:.1} tok/s");
        steps.push(json!({"step": name, "dtype": dtype, "simd": simd, "threads": threads,
                          "pp64_tok_s": pp, "tg32_tok_s": tg, "tg32_sd": tg_sd}));
    }
    set_simd(true);
    write(
        arg(args, "--out"),
        &json!({"model": dir, "steps": steps, "meta": meta()}),
    );
}

/// Perplexity the way llama.cpp's `llama-perplexity` computes it: split the
/// tokens into `ctx`-sized chunks, run each chunk from an empty cache, and
/// score only the second half of each chunk (so every scored token has at
/// least ctx/2 tokens of context). Also returns, per scored position, the
/// argmax, to compare quantized models against f32.
fn perplexity(model: &Model, tokens: &[u32], ctx: usize, chunks: usize) -> (f64, Vec<u32>) {
    let vocab = model.config.vocab_size;
    let (mut nll, mut count, mut top1) = (0f64, 0usize, Vec::new());
    for chunk in tokens.chunks_exact(ctx).take(chunks) {
        let mut g = Generator::new(model, ctx);
        g.cache.reserve(0, ctx).unwrap();
        let pos: Vec<usize> = (0..ctx).collect();
        let want: Vec<usize> = (ctx / 2..ctx - 1).collect();
        let logits = model.forward(chunk, &pos, &vec![0; ctx], &want, &mut g.cache, None);
        for (j, &i) in want.iter().enumerate() {
            let row = &logits[j * vocab..(j + 1) * vocab];
            let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let lse = max as f64 + row.iter().map(|&v| ((v - max) as f64).exp()).sum::<f64>().ln();
            nll += lse - row[chunk[i + 1] as usize] as f64;
            count += 1;
            top1.push(glassbox_core::sampler::argmax(row) as u32);
        }
    }
    ((nll / count as f64).exp(), top1)
}

fn quality(args: &[String]) {
    let dir = arg(args, "--model").expect(USAGE);
    let ctx: usize = arg(args, "--ctx").and_then(|v| v.parse().ok()).unwrap_or(512);
    let chunks: usize = arg(args, "--chunks").and_then(|v| v.parse().ok()).unwrap_or(100);
    let file = arg(args, "--file").expect(USAGE);
    let text = std::fs::read_to_string(file).expect("read text");
    let tok = glassbox_core::Tokenizer::from_file(std::path::Path::new(dir).join("tokenizer.json")).unwrap();
    let tokens = tok.encode(&text);
    let chunks = chunks.min(tokens.len() / ctx);
    let mut rows = Vec::new();
    let mut reference = Vec::new();
    for dtype in ["f32", "q8_0", "q4_0"] {
        let model = load(dir, None, Some(dtype));
        let t = Instant::now();
        let (ppl, top1) = perplexity(&model, &tokens, ctx, chunks);
        if dtype == "f32" {
            reference = top1.clone();
        }
        let agree = top1.iter().zip(&reference).filter(|(a, b)| a == b).count() as f64 / top1.len() as f64;
        eprintln!(
            "{dtype}: ppl {ppl:.4}, top-1 agreement with f32 {:.2}% ({:.0}s)",
            agree * 100.0,
            t.elapsed().as_secs_f64()
        );
        rows.push(json!({"dtype": dtype, "ppl": ppl, "top1_agreement_vs_f32": agree, "weight_mb": model.weight_bytes() as f64 / 1e6}));
    }
    write(
        arg(args, "--out"),
        &json!({"model": dir, "text": file, "ctx": ctx, "chunks": chunks, "scored_tokens": chunks * (ctx / 2 - 1),
                "results": rows, "meta": meta()}),
    );
}

fn kernels() {
    for (rows, cols) in [(576, 576), (1536, 576), (576, 1536), (49152, 576)] {
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7919 % 1000) as f32 - 500.0) / 500.0)
            .collect();
        let base = Matrix::f32(rows, cols, w);
        for dt in [DType::F32, DType::Q8_0, DType::Q4_0] {
            let m = base.quantize(dt);
            for n in [1, 64] {
                let x = vec![0.5f32; n * cols];
                let mut out = vec![0.0; n * rows];
                matmul(&mut out, &x, n, &m);
                let iters = (2e9 / (rows * cols * n) as f64).clamp(3.0, 2000.0) as usize;
                let t = Instant::now();
                for _ in 0..iters {
                    matmul(&mut out, &x, n, &m);
                }
                let s = t.elapsed().as_secs_f64() / iters as f64;
                println!(
                    "{rows:6}x{cols:<5} {dt:?} n={n:<3} {:8.1} us  {:6.1} GB/s  {:6.1} GFLOP/s",
                    s * 1e6,
                    m.bytes() as f64 / s / 1e9,
                    2.0 * (rows * cols * n) as f64 / s / 1e9
                );
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("single") => single(&args),
        Some("progression") => progression(&args),
        Some("quality") => quality(&args),
        Some("kernels") => kernels(),
        _ => eprintln!("{USAGE}"),
    }
}
