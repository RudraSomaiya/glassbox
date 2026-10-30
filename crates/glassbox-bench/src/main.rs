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
  glassbox-bench serve       --model DIR [--dtype q8_0] [--out FILE]
  glassbox-bench spec        [--set random|trained] [--out FILE]   (needs the stand-ins under models/)
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

/// Exponential inter-arrival times (a Poisson process) from a seeded RNG.
fn arrivals(rate: f64, n: usize, rng: &mut glassbox_core::sampler::Rng) -> Vec<f64> {
    let mut t = 0.0;
    (0..n)
        .map(|_| {
            t += -(1.0 - rng.next_f32() as f64).ln() / rate;
            t
        })
        .collect()
}

struct Load {
    prompts: Vec<Vec<u32>>,
    outputs: Vec<usize>,
}

/// Run one workload through the scheduler in real time and summarise it.
fn serve_run(model: &Model, config: glassbox_server::scheduler::Config, load: &Load, rate: f64) -> Value {
    use glassbox_server::scheduler::{Event, Request, Scheduler};
    let mut rng = glassbox_core::sampler::Rng::new(42);
    let times = arrivals(rate, load.prompts.len(), &mut rng);
    let mut s = Scheduler::new(model, config.clone());
    let mut sinks = Vec::new();
    let t0 = Instant::now();
    let mut next = 0;
    while next < times.len() || !s.is_idle() {
        let now = t0.elapsed().as_secs_f64();
        while next < times.len() && times[next] <= now {
            let (sink, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
            let req = Request {
                prompt: load.prompts[next].clone(),
                max_tokens: load.outputs[next],
                sampler: glassbox_core::SamplerConfig::default(),
                stop: vec![],
                sink,
                constraint: None,
            };
            s.submit_at(req, t0 + std::time::Duration::from_secs_f64(times[next]));
            sinks.push(rx);
            next += 1;
        }
        if s.is_idle() {
            std::thread::sleep(std::time::Duration::from_secs_f64((times[next] - now).max(0.0)));
            continue;
        }
        s.step();
    }
    let wall = t0.elapsed().as_secs_f64();
    let m = &s.metrics;
    let q = |v: &[f64], p: f64| {
        let mut v = v.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        v.get(((v.len().max(1) - 1) as f64 * p).round() as usize)
            .copied()
            .unwrap_or(0.0)
    };
    json!({
        "rate_rps": rate, "requests": load.prompts.len(), "wall_s": wall,
        "throughput_tok_s": m.generated_tokens as f64 / wall,
        "ttft_p50_s": q(&m.ttft, 0.5), "ttft_p99_s": q(&m.ttft, 0.99),
        "itl_p50_s": q(&m.itl, 0.5), "itl_p99_s": q(&m.itl, 0.99),
        "prefix_hit_tokens": m.prefix_hit_tokens, "prompt_tokens": m.prompt_tokens,
        "preemptions": m.preemptions, "mean_batch_tokens": m.batch_tokens as f64 / m.steps.max(1) as f64,
    })
}

fn serve(args: &[String]) {
    use glassbox_server::scheduler::Config;
    let dir = arg(args, "--model").expect(USAGE);
    let model = load(dir, None, Some(arg(args, "--dtype").unwrap_or("q8_0")));
    let vocab = model.config.vocab_size as f32;
    let mut rng = glassbox_core::sampler::Rng::new(7);
    let mut rand_tokens = |n: usize| (0..n).map(|_| (rng.next_f32() * vocab) as u32).collect::<Vec<u32>>();
    let n = 48;
    // Chat-like lengths (ShareGPT-style: replies longer than prompts):
    // prompts 32-320 tokens, replies 32-256.
    let mut lens = glassbox_core::sampler::Rng::new(9);
    let mixed = Load {
        prompts: (0..n)
            .map(|_| rand_tokens(32 + (lens.next_f32() * 288.0) as usize))
            .collect(),
        outputs: (0..n).map(|_| 32 + (lens.next_f32() * 224.0) as usize).collect(),
    };
    // Every request starts with the same 384-token system prompt.
    let system = rand_tokens(384);
    let shared = Load {
        prompts: (0..n)
            .map(|_| {
                let mut p = system.clone();
                p.extend(rand_tokens(16 + (lens.next_f32() * 48.0) as usize));
                p
            })
            .collect(),
        outputs: (0..n).map(|_| 16 + (lens.next_f32() * 48.0) as usize).collect(),
    };
    let batched = Config {
        token_budget: 512,
        max_seqs: 32,
        num_blocks: 4096,
        prefix_caching: false,
    };
    let single = Config {
        max_seqs: 1,
        ..batched.clone()
    };
    let mut runs = Vec::new();
    for rate in [0.5, 1.0, 2.0, 4.0] {
        for (name, cfg) in [("continuous batching", &batched), ("one at a time", &single)] {
            let r = serve_run(&model, cfg.clone(), &mixed, rate);
            eprintln!(
                "{name:20} {rate:4} req/s: {:.0} tok/s, TTFT p50 {:.2}s p99 {:.2}s",
                r["throughput_tok_s"].as_f64().unwrap(),
                r["ttft_p50_s"].as_f64().unwrap(),
                r["ttft_p99_s"].as_f64().unwrap()
            );
            runs.push(json!({"workload": "mixed", "mode": name, "result": r}));
        }
    }
    for rate in [2.0, 8.0] {
        for (name, on) in [("prefix cache", true), ("no prefix cache", false)] {
            let r = serve_run(
                &model,
                Config {
                    prefix_caching: on,
                    ..batched.clone()
                },
                &shared,
                rate,
            );
            eprintln!(
                "{name:20} {rate:4} req/s: {:.0} tok/s, TTFT p50 {:.2}s p99 {:.2}s, {} prompt tokens reused",
                r["throughput_tok_s"].as_f64().unwrap(),
                r["ttft_p50_s"].as_f64().unwrap(),
                r["ttft_p99_s"].as_f64().unwrap(),
                r["prefix_hit_tokens"]
            );
            runs.push(json!({"workload": "shared system prompt", "mode": name, "result": r}));
        }
    }
    write(
        arg(args, "--out"),
        &json!({"model": dir, "dtype": format!("{:?}", model.layers[0].wqkv.dtype()),
        "threads": pool::threads(), "workloads": {
            "mixed": "48 requests, prompts 32-320 random tokens, replies 32-256 tokens, greedy, Poisson arrivals",
            "shared system prompt": "48 requests, one 384-token system prompt + 16-64 token user part, replies 16-64 tokens"},
        "runs": runs, "meta": meta()}),
    );
}

/// Speculative decoding: speedup and acceptance against draft length.
fn spec(args: &[String]) {
    use glassbox_core::spec::{generate, Drafter};
    let gguf = |m: &str, q: &str| Model::load_gguf(format!("models/{m}/model-{q}.gguf")).expect("gguf");
    let tok = glassbox_core::Tokenizer::from_file("models/standin-135M/tokenizer.json").unwrap();
    let text = std::fs::read_to_string("data/perplexity.txt").unwrap_or_default();
    let chat = |u: &str| tok.encode(&glassbox_core::Tokenizer::chat_prompt(&[("user", u)]));
    let questions = [
        "Explain how a bicycle stays upright.",
        "Write a short story about a lighthouse keeper.",
        "What are the trade-offs between arrays and linked lists?",
        "Describe the water cycle.",
    ];
    let qa: Vec<Vec<u32>> = questions.iter().map(|q| chat(q)).collect();
    // Summarisation: a passage plus an instruction, where the answer can copy from the prompt.
    let docs: Vec<Vec<u32>> = text
        .split("\n\n")
        .filter(|p| p.len() > 1200)
        .take(4)
        .map(|p| {
            chat(&format!(
                "{}\n\nSummarise the text above in a few sentences.",
                &p[..p.len().min(1600)]
            ))
        })
        .collect();
    let max = 64;
    let mut out = Vec::new();
    // Random-weight stand-ins for speed at real shapes, or the trained tiny
    // stand-in, whose distributions are peaked enough for acceptance to mean something.
    let trained = arg(args, "--set") == Some("trained");
    let load_random = || {
        (
            gguf("standin-135M", "q8_0"),
            gguf("standin-360M", "q8_0"),
            gguf("standin-1.7B", "q8_0"),
            gguf("standin-1.7B", "q4_0"),
        )
    };
    let load_tiny = || {
        let tiny = Model::load_dir("models/standin-tiny").expect("models/standin-tiny");
        let mut tiny4 = Model::load_dir("models/standin-tiny").expect("models/standin-tiny");
        tiny4.quantize(DType::Q4_0);
        (tiny, tiny4)
    };
    let (random, tiny) = if trained {
        (None, Some(load_tiny()))
    } else {
        (Some(load_random()), None)
    };
    let mut cases: Vec<(&str, &Model, Drafter, &Vec<Vec<u32>>)> = Vec::new();
    if let Some((small, mid, big, big4)) = &random {
        cases.push(("135M -> 360M", mid, Drafter::Model(small), &qa));
        cases.push(("135M -> 1.7B", big, Drafter::Model(small), &qa));
        cases.push(("1.7B Q4_0 -> 1.7B Q8_0", big, Drafter::Model(big4), &qa));
        cases.push((
            "prompt lookup -> 360M (summaries)",
            mid,
            Drafter::Lookup { ngram: 3 },
            &docs,
        ));
    }
    if let Some((tiny, tiny4)) = &tiny {
        cases.push(("tiny Q4_0 -> tiny f32", tiny, Drafter::Model(tiny4), &qa));
        cases.push((
            "prompt lookup -> tiny (summaries)",
            tiny,
            Drafter::Lookup { ngram: 3 },
            &docs,
        ));
    }
    for (name, target, drafter, prompts) in cases {
        // Speedups compare decode time only: prompt reading is the same work
        // either way (plus a draft-model prefill, counted as overhead below).
        let t = Instant::now();
        for p in prompts.iter() {
            Generator::new(target, p.len() + 1).feed(p, None);
        }
        let prefill = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut n = 0;
        for p in prompts.iter() {
            n += glassbox_core::greedy(target, p, max).len();
        }
        let base = n as f64 / (t.elapsed().as_secs_f64() - prefill);
        for k in [1, 2, 4, 8] {
            let t = Instant::now();
            let (mut n, mut drafted, mut accepted, mut passes) = (0, 0, 0, 0);
            for p in prompts.iter() {
                let (o, st) = generate(
                    target,
                    &drafter,
                    p,
                    max,
                    k,
                    glassbox_core::SamplerConfig::default(),
                    None,
                );
                n += o.len();
                drafted += st.drafted;
                accepted += st.accepted;
                passes += st.target_passes;
            }
            let tps = n as f64 / (t.elapsed().as_secs_f64() - prefill);
            let acc = accepted as f64 / drafted.max(1) as f64;
            eprintln!(
                "{name:34} k={k}: {tps:6.1} tok/s vs {base:6.1} alone ({:.2}x), acceptance {:.0}%",
                tps / base,
                acc * 100.0
            );
            out.push(
                json!({"pair": name, "k": k, "tok_s": tps, "baseline_tok_s": base, "speedup": tps / base,
                            "acceptance": acc, "drafted": drafted, "accepted": accepted, "target_passes": passes,
                            "tokens": n}),
            );
        }
    }
    write(
        arg(args, "--out"),
        &json!({"max_tokens": max, "decoding": "greedy", "runs": out, "threads": pool::threads(),
        "note": if trained { "trained tiny stand-in (6 layers, 384 wide): realistic distributions, but tiny" } else { "random-weight stand-ins: acceptance between different random models is near zero by construction, and random logits are so nearly tied that even 4-bit noise flips the argmax" },
        "timing": "decode only: tokens / (total time - target prefill time)",
        "meta": meta()}),
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
        Some("serve") => serve(&args),
        Some("spec") => spec(&args),
        Some("kernels") => kernels(),
        _ => eprintln!("{USAGE}"),
    }
}
