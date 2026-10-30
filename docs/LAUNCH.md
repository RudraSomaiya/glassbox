# Launch drafts

Drafts only. Before posting: re-run the benchmarks with the real SmolLM2 weights (see the README's Reproduce section) and replace any number marked "(stand-in)" if it changed. Speed numbers won't; quality and acceptance numbers will.

## LinkedIn

I wanted to understand what actually happens between a prompt and the first token, so I wrote an LLM inference engine from scratch in Rust. No PyTorch, no llama.cpp underneath: the tokenizer, the SIMD matrix kernels, 8- and 4-bit quantization, a paged KV cache, a continuous-batching server, speculative decoding and JSON-schema constrained output are all in the repo.

It's called Glassbox, because the other half is a web app that runs the model in your browser and shows you its insides while it writes: every token's probability, the ten tokens it almost chose (click one to fork the story), what each layer was predicting, where each attention head looked, and the cache filling up.

Some things I learned:
- The model matches Hugging Face transformers token for token. Getting there found a bug where a config field had moved and RoPE was silently using the wrong base.
- The biggest speed-up wasn't arithmetic. Decode runs ~150 tiny parallel jobs per token, and waking sleeping threads cost more than the work. A small spinning thread pool doubled throughput.
- It decodes at 50-80% of llama.cpp's speed on the same CPU and the same files, and prompt processing is 5-8x slower. The README says exactly why.

Demo and code: https://github.com/RudraSomaiya/glassbox

## Show HN

**Show HN: Glassbox, an LLM inference engine in Rust with a window into the model**

I wrote an inference engine for Llama-family models from scratch in Rust (no ML framework) to learn how inference works underneath, and a web app that runs it in the browser via WebAssembly and shows what the model is doing while it writes.

What's in the engine: byte-level BPE (identical to HF tokenizers on 2,175 test strings), f32/Q8_0/Q4_0 kernels with AVX2 and WASM SIMD, GGUF loading (Q8_0 blocks bit-identical to llama.cpp's quantizer, perplexity agrees to 5 significant figures), a paged KV cache with prefix sharing, a continuous-batching scheduler with chunked prefill and preemption behind an OpenAI-compatible API, lossless speculative decoding (draft model or prompt lookup), and JSON-schema constrained decoding via a token trie.

Numbers on a 4-vCPU cloud VM, same GGUF files: decode 181 tok/s vs llama.cpp's 272 for SmolLM2-135M Q8_0; prompt processing 296 vs 2,416. llama.cpp's 512-bit VNNI GEMM over repacked weights is the gap; that's next.

The web app: tokens shown as film frames whose density is the probability, a loupe with the top-10 alternatives (click to fork), a logit lens per layer, attention heads, live sampling dials.

One caveat: the build machine couldn't reach Hugging Face, so the repo's benchmarks use models with SmolLM2's shapes and stand-in weights (fine for speed; quality numbers will move). The live demo downloads the real 135M model.

https://github.com/RudraSomaiya/glassbox

## r/rust

**Glassbox: an LLM inference engine in Rust with no ML framework, plus a WASM build that runs in the browser**

Things that might interest this sub specifically:

- Hand-written AVX2/FMA kernels as const-generic register tiles (`dot_q8::<R, N>`), runtime feature detection, scalar twins for every kernel and tests on awkward lengths. Every `unsafe` block has a SAFETY comment.
- A false dependency I found with `objdump`: `_mm_cvtph_ps` on a single f16 compiled to `vpinsrw` into my accumulator register and serialized the loop. Fixed with a lookup table.
- rayon was the wrong tool for ~150 short parallel regions per token (futex wake-ups ~50 µs each). A ~150-line spinning fork-join pool doubled decode throughput.
- The same `forward` compiles to `wasm32-unknown-unknown` with `simd128`; a headless Chromium test checks it reproduces the native output.
- Axum + tokio for the OpenAI-compatible server, with one OS thread owning the model and the scheduler.

Decode is 50-80% of llama.cpp, prefill 5-8x slower; the README and write-up explain why. Feedback on the kernels very welcome.

https://github.com/RudraSomaiya/glassbox

## r/LocalLLaMA

**I built an inference engine from scratch to see inside the model: logit lens, attention, KV cache and speculative decoding, live in the browser**

Glassbox runs SmolLM2-135M in your browser (WebAssembly, 8-bit, cached after one download) and shows you what it's doing per token: probability, the top-10 alternatives (click to fork the story), what each layer would have predicted, where each head looked, and the paged KV cache filling up. Replays from the native server show continuous batching and speculative decoding.

The native side is an OpenAI-compatible server (works with the official `openai` client) with continuous batching, prefix caching, speculative decoding and `response_format: json_schema` that always returns valid JSON.

Honest numbers on a 4-core VM vs llama.cpp on the same GGUFs: decode 181 vs 272 tok/s (135M Q8_0), 13 vs 24 tok/s (1.7B Q8_0); prompt processing much slower. It isn't trying to replace llama.cpp; it's for understanding.

https://github.com/RudraSomaiya/glassbox

## CV bullet

**Glassbox** (Rust, WebAssembly, React): LLM inference engine written from scratch with no ML framework: byte-level BPE, AVX2/AVX-VNNI/WASM-SIMD f32/Q8_0/Q4_0 kernels, paged KV cache with prefix sharing, continuous-batching OpenAI-compatible server, lossless speculative decoding and JSON-schema constrained output. Matches Hugging Face transformers token for token; decodes SmolLM2-135M at 181 tok/s on 4 CPU cores (66% of llama.cpp on identical weights); doubled multi-core throughput by replacing rayon with a spinning thread pool. Browser glass box visualises tokens, layers, attention and the KV cache live.
