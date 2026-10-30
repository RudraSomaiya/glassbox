# Study pack

This is for Rudra: how to read the code, the ideas underneath it, the questions an interviewer is likely to ask, and exercises to prove you can change it. Line numbers are as of the commit that added this file; if they drift, search for the function name.

## 1. Reading order

Read in this order. Each step depends on the one before.

1. **The data types.** `crates/glassbox-core/src/model.rs:15` (`Config`), `:57` (`Layer`), `:67` (`Model`). A model is a list of matrices and norm vectors, nothing more.
2. **One forward pass.** `model.rs:207` (`Model::forward`). Follow one token through embed, then the layer loop (norm, QKV matmul, RoPE at `:102`, cache write, attention at `:278`, output projection, MLP), then the head. This is the whole model in ~120 lines.
3. **The matmul it calls.** `kernels.rs:519` (`matmul`), then `:451` (`tile`), then the AVX2 kernels: `dot_f32` (around `:100`), `mul_sum_i8` at `:130`, `dot_q8` at `:166`. Then the scalar twins at the top of the file and the tests at the bottom: the scalar versions define what "correct" means.
4. **Quantization.** `quant.rs:26` (`quantize_q8_0`) and `:47` (`quantize_q4_0`). Compare with `dequantize_*` below them; a block is 32 numbers and one scale.
5. **Why it's fast.** `kernels.rs:91` (`h2f`, the table that removed a false dependency), `:497` (`prefetch_rows`), and `pool.rs:133` (`par_for`) with `:96` (`worker`). Read the module comment in `pool.rs` first.
6. **The KV cache.** `kv.rs`: `add_seq` (`:135`, prefix sharing), `reserve` (`:161`, allocation and copy-on-write), `commit` (`:203`, hashing full blocks), `chain_hash` (`:45`), `fork` (`:248`), `check_invariants` (`:266`).
7. **Tokenizer.** `tokenizer.rs:182` (`encode`: special tokens first), `:208` (`encode_plain`: pre-tokenizer regex and digit splitting), `:255` (`bpe`: merge the lowest-ranked pair until none apply), `:49` (`bytes_to_unicode`).
8. **Sampling.** `sampler.rs:97` (`distribution`: repetition penalty, temperature, top-k, min-p, top-p), `:150` (`sample`).
9. **The single-sequence driver.** `lib.rs:21` (`Generator`) and `greedy` below it. This is what the CLI, the browser and the tests use.
10. **The scheduler.** `crates/glassbox-server/src/scheduler.rs:237` (`step`): admission (`:254`), planning (`:271`), reservation with preemption (`:289`, `preempt_last` at `:226`), the forward pass, sampling. Then `tests/scheduler.rs`.
11. **The HTTP layer.** `http.rs:46` (`spawn_engine`: one thread owns the model), `:180` (`generate`: SSE streaming).
12. **Speculative decoding.** `crates/glassbox-core/src/spec.rs:74` (`generate`), the verification loop, and `lookup` at `:56`. Then the two tests at the bottom, especially the chi-square one.
13. **Constrained decoding.** `crates/glassbox-server/src/json_schema.rs`: `compile`, `State::advance` (`:125`), `Trie::allowed` (`:330`).
14. **The browser.** `crates/glassbox-wasm/src/lib.rs` (`feed` at `:85`, `inspect` at `:147`), then `web/src/engine/worker.ts`.
15. **The tests that prove it.** `crates/glassbox-core/tests/golden.rs` against `tools/reference/make_golden.py`.

## 2. Concepts from first principles

### A transformer forward pass

Text becomes token ids. Each id selects a row of the embedding matrix: a vector of `hidden` numbers (576 for SmolLM2-135M). Every layer reads all the token vectors, computes something, and **adds** it back (the residual stream). After the last layer, the last token's vector is compared against every row of the output matrix (often the same embedding matrix, "tied"), giving one score per vocabulary entry: the logits. Softmax turns logits into probabilities.

Inside a layer there are two halves, each preceded by RMSNorm (divide the vector by its root-mean-square, multiply by learned gains):
- **Attention** moves information between tokens.
- **The MLP** (SwiGLU here: `down(silu(gate(x)) * up(x))`) transforms each token on its own.

### Attention

For each token, three projections: a query `q` (what am I looking for), a key `k` (what do I contain), a value `v` (what do I pass on). Token `i`'s output is `sum_j softmax_j(q_i · k_j / sqrt(d)) * v_j` over tokens `j <= i` (causal: no peeking ahead). Dividing by `sqrt(d)` keeps the dot products from growing with the head size and saturating the softmax.

**Multi-head:** the vector is split into heads (9 heads of 64 for 135M) that attend independently, then concatenated and projected. **Grouped-query attention:** fewer key/value heads than query heads (3 KV heads shared by 9 query heads), so the cache is 3x smaller. In code: `kvh = head / group` (`model.rs`, attention loop).

### RoPE

Attention needs position, but the dot product has none. Rotary position embedding rotates pairs of dimensions of `q` and `k` by an angle `pos * theta^(-2i/d)`. A rotation by `a` dotted with a rotation by `b` depends only on `a - b`, so the score depends on the **relative** position. Low `i` rotate fast (fine position), high `i` slowly (coarse). Two conventions exist for which dimensions pair up: Hugging Face rotates `(i, i + d/2)`, llama.cpp `(2i, 2i+1)` with Q/K rows permuted in the GGUF. Glassbox supports both (`rope_interleaved`). The bug that cost an hour: transformers 5 stores `rope_theta` inside `rope_parameters`, the config parser fell back to 10000, and positions > 0 were wrong.

### The KV cache

Generating token `n + 1` needs the keys and values of tokens `0..n`. They don't change once computed (causal attention: earlier tokens never see later ones), so compute them once and keep them. Without a cache, each new token costs a full pass over the whole sequence; with it, one token's worth of matmuls plus attention over the cached keys. Cost: `2 * layers * kv_heads * head_dim * 4 bytes` per token (46 kB for 135M in f32).

### Paging

A contiguous cache must reserve the maximum context for every sequence up front, so most of it sits empty, and it can't be shared. Paging (vLLM's PagedAttention idea) splits the cache into fixed blocks (16 tokens here) and gives each sequence a block table, like virtual memory. Memory is allocated as tokens arrive; blocks can be shared by reference count; a sequence that forks shares everything and copies a block only when it writes into a shared one (copy-on-write). Prefix sharing: hash each full block together with everything before it; an identical hash means identical tokens from the start, which means identical keys and values, so the block can be reused.

### Batching

On a CPU or GPU, a decode step for one sequence is memory-bound: every weight is read from memory to do two flops with it. Reading the weights once for 16 sequences costs almost the same as for one. So serving systems batch. **Static batching** waits for a whole batch to finish before starting the next; short requests wait for long ones. **Continuous batching** decides per step: sequences leave as they finish and new ones join immediately. **Chunked prefill** splits a long prompt across steps so the decoding sequences in the same batch keep getting tokens. **Preemption** frees memory when the cache is full by evicting a sequence and recomputing it later.

### Quantization

Store weights in fewer bits. Q8_0: blocks of 32 weights share one f16 scale `d = max|w| / 127`; each weight is `round(w / d)` in an i8. Q4_0: 4-bit values `0..15` meaning `-8..7` times `d = max / -8`. Size per weight: 34/32 bytes (Q8_0), 18/32 (Q4_0), versus 4 (f32). Decode is memory-bound, so fewer bytes means more tokens per second. The dot product also quantizes the activation row to Q8_0 so the inner loop is integer multiply-adds, with one float multiply per block (`d_w * d_x`). Quality cost: measured by perplexity and by how often the top-1 token matches f32.

### Speculative decoding

Verifying `k` guessed tokens takes one forward pass over `k + 1` tokens, which costs about the same as one token when decode is memory-bound. So let a cheap drafter guess, and keep the guesses the real model agrees with. Rejection sampling makes it **lossless**: accept draft `x` with probability `min(1, p(x)/q(x))`; on rejection draw from `max(0, p - q)` normalized. The resulting token has exactly distribution `p` (proof sketch: `P(token = x) = q(x) min(1, p/q) + P(reject) * resid(x) = min(q, p) + (p - min(p, q)) = p(x)`). Speedup depends on the acceptance rate and on how cheap the drafter is relative to the target.

### Constrained decoding

At each step, mask out tokens that would make the output invalid, then sample from what's left. The state of "what's valid next" is a machine compiled from the schema. The hard part is speed: the vocabulary has 49,152 tokens of up to ~16 bytes each. A trie of token bytes lets one walk share common prefixes and prune whole subtrees at the first invalid byte.

## 3. Interview questions with answers

**Basics**

1. *What is a KV cache?* The stored keys and values of every past token, per layer. Without it each new token would recompute them for the whole sequence. Glassbox's is paged: `kv.rs`, 16-token blocks, 46 kB per token for 135M in f32.
2. *Why is decode memory-bound and prefill compute-bound?* Decode multiplies each weight by one activation vector: 2 flops per weight read. Prefill multiplies each weight by hundreds of token vectors, so the same bytes feed hundreds of flops. In my numbers: 135M Q8_0 decode is 181 tok/s, prefill 296 tok/s for 512 tokens, but prefill does 512x the work per pass.
3. *What does RMSNorm do and why not LayerNorm?* Divides by the root-mean-square (no mean subtraction), times a learned gain. Cheaper, works as well in practice, and it's what Llama uses. `model.rs:94`.
4. *Why divide attention scores by sqrt(d)?* The variance of a dot product of random vectors grows with `d`; without scaling the softmax saturates and gradients vanish.
5. *What's grouped-query attention for?* Fewer KV heads than query heads: SmolLM2-135M has 9 query heads and 3 KV heads, so the KV cache and the K/V projections are 3x smaller with little quality loss.
6. *What's a tied LM head?* The output projection reuses the embedding matrix. SmolLM2 does this; `Model.lm_head` is `None` and `head()` returns the embedding.
7. *Explain BPE.* Start from bytes; repeatedly merge the adjacent pair with the lowest rank in the learned merge list until no pair is in the list. Byte-level means any input is encodable. `tokenizer.rs:255`, O(n^2) per word because words are short.
8. *Why does the tokenizer split digits individually?* SmolLM2's pre-tokenizer does (`Digits(individual_digits)`), which helps arithmetic. My implementation runs the same pre-tokenizer steps in order, then the GPT-2 regex.
9. *How did you verify the tokenizer?* 2,175 strings (English, code, emoji, Hindi, Hinglish, whitespace and special-token edge cases) encoded by Hugging Face `tokenizers` and by mine: identical ids for all of them.
10. *How did you verify the model?* Golden fixtures from transformers in f32: top-20 logits at three positions for 21 prompts match within 4.5e-6, and 64-token greedy continuations are identical.

**Performance**

11. *Walk me through how you made it fast.* Scalar 1 thread 8.7 tok/s, AVX2 24.1, 4 threads 65.5, Q8_0 189, Q4_0 219, Q4_0 with AVX-VNNI 230 (135M shapes, 32-token decode). Along the way: register-tiled kernels, an f16 lookup table that removed a false dependency, software prefetch, replacing rayon with a spinning pool, and VNNI int8 dot products.
12. *What was the false dependency?* Converting each block's f16 scale with `_mm_cvtph_ps` on a single value compiled to `vpinsrw` into the accumulator register, so every block's scale conversion waited on the previous block's FMA. I found it in `objdump`. Fixed with a 64K-entry table, as ggml does: 69 us to 43 us per 576x576 matvec.
13. *Why did replacing rayon help so much?* Decode runs about 150 parallel regions per token, each 30-300 us. rayon parks idle workers and wakes them with a futex (~50 us). My pool's workers spin for 2 ms after each job, so the next region starts immediately. Q8_0 decode on 4 threads: 74 to 159 tok/s.
14. *Why register tiles?* A 1x1 dot product loads two vectors per FMA; the load ports are the limit. A 4x1 tile loads the activation once for four weight rows and has four independent accumulators, so FMA latency overlaps. 2x4 for prefill reuses each weight block across four tokens.
15. *Why is your prefill so much slower than llama.cpp's?* 5-8x. llama.cpp on this CPU uses 512-bit VNNI and repacks weights into interleaved layouts so its GEMM streams contiguous memory. Mine uses 256-bit AVX-VNNI (`vpdpbusd`, one instruction for the multiply-add chain AVX2 does in three) over the original layout. Repacking and wider kernels are the next thing to do.
16. *Why is Q4_0 decode faster than Q8_0 but not twice as fast?* Half the bytes, but the nibble unpacking costs instructions, and at this size the kernel is partly compute-bound, not purely bandwidth-bound.
17. *What did prefetching fix?* A single core streaming weights from DRAM kept too few cache misses in flight: 4.8 GB/s. Prefetching the next 4-row tile while computing this one: 9.6 GB/s per core.
18. *How do you choose kernels at runtime?* `is_x86_feature_detected!` for AVX2, FMA and F16C, checked once per matmul; `simd128` is compile-time in WebAssembly. A global switch turns SIMD off for the benchmark.
19. *How do you know the SIMD kernels are right?* Each has a scalar twin; tests compare them on lengths 1, 7, 31, 33, 1537 and so on, and the matmul test checks every dtype against a naive dot product.
20. *Are batched results identical to unbatched?* Bitwise, because each output cell is one dot product whose summation order doesn't depend on the other tokens in the batch. The property tests assert exact equality.

**Memory and serving**

21. *Why paging instead of a contiguous cache?* No up-front reservation of the maximum context, no fragmentation, and blocks can be shared between sequences by reference count.
22. *How does prefix sharing work?* When a block fills, hash it chained with the hash of everything before it. `add_seq` walks a new prompt's full blocks and reuses any block whose chain hash is registered. Chaining matters: the same 16 tokens after a different prefix have different keys and values.
23. *Why never reuse the last prompt token's block?* The last token must run through the model to produce logits for the first generated token, so `add_seq` stops one block short.
24. *What happens when the cache is full?* `reserve` fails; the scheduler preempts the most recently admitted sequence, frees its blocks and requeues it at the front with computed = 0. When it comes back, the prefix cache often still holds its blocks.
25. *Why preempt the newest sequence?* It has done the least work, so recomputing it wastes the least, and the oldest requests keep their latency.
26. *What's chunked prefill and why?* A 2,000-token prompt prefilled in one step would stall every decoding sequence for that whole step. Splitting it under a per-step token budget bounds the step time.
27. *How do you detect KV block leaks?* `check_invariants` recomputes every reference count from the block tables and checks the free count. The property tests call it after random mixes of finish, abort, client hang-up and preemption.
28. *What does copy-on-write protect against?* Two sequences sharing a partly filled block after a fork; the first to write into it gets a private copy so the other's keys are untouched.
29. *How does TTFT behave under load and why?* Measured with Poisson arrivals: with batching TTFT grows slowly until prefill capacity runs out; one-at-a-time TTFT explodes as soon as arrivals outpace one request's service time. My prefill is the bottleneck in serving too.
30. *How does the HTTP server talk to the engine?* One OS thread owns the scheduler; handlers send requests over a channel and receive tokens on a per-request channel that becomes an SSE stream. A closed channel means the client left, and the scheduler aborts that sequence.

**Speculative and constrained decoding**

31. *Why is speculative decoding lossless?* Accept with probability `min(1, p/q)`, otherwise resample from `max(0, p - q)` normalized; the marginal distribution of each emitted token is exactly `p`. Under greedy it reduces to accepting while the target's argmax agrees. Tests: identical greedy output for several k; chi-square on sampled output.
32. *When does speculative decoding stop helping?* When acceptance is low (the drafter disagrees with the target), when the drafter isn't much cheaper than the target (SmolLM2-1.7B drafting for itself at Q4_0: 85% accepted, still 0.75x; 135M drafting for 360M: 70% accepted, 0.75x, because 135M is only 3x cheaper. 135M drafting for 1.7B, 13x cheaper, gives 1.24x), or when verification isn't nearly free (compute-bound, large batches, so extra tokens per pass aren't free).
33. *What's prompt lookup decoding?* Use the last n-gram of the context to find an earlier occurrence and propose what followed it. No extra model; great when the output copies the input (summaries, code edits). Measured on SmolLM2-360M summarising passages: 44% acceptance, 0.90x, because the real model paraphrases instead of copying.
34. *How do you keep two KV caches consistent?* After each round both are truncated back to the accepted prefix; the draft model stays one token behind the target.
35. *How does constrained decoding guarantee valid JSON?* The schema is a machine; at each step only tokens whose every byte the machine accepts are allowed, and EOS only when the document can end. Strings, arrays and numbers are capped so it always terminates. 200 random schemas: every output validates.
36. *Why a trie?* Tokens share prefixes. A depth-first walk carries the machine state down each edge and prunes a whole subtree at the first invalid byte. Inside a string body almost every token is valid, so there's a precomputed list of plain tokens instead.
37. *What are the limitations of your schema support?* No `oneOf`/`anyOf`, no patterns, no `additionalProperties`; objects emit all properties in schema order; output is compact. All chosen to keep the machine small and output always valid.

**Design and honesty**

38. *Why no ML framework?* The kernels are the point: being able to say why each instruction is there. The whole engine is `std`, a few small crates and hand-written SIMD.
39. *How is the browser version the same engine?* Same crate, same `forward`; only kernel selection differs (`simd128`, no threads). A headless Chromium test reproduces the native greedy output.
40. *What's the trace hook and why is it free when off?* `Option<&mut Trace>` checked once per layer; when present it copies the watched token's residual and attention weights. The browser re-runs a prefix with tracing only when you click a token, so generation never pays for it.
41. *What would you do next?* 512-bit kernels and repacked weights for prefill, multithreaded WebAssembly, and real-weight numbers for everything that depends on what the model knows.
42. *Why did you build a trained stand-in model?* Hugging Face was blocked from the build machine for most of the build. Random weights give flat distributions, so every view of the glass box looked the same. A 14-minute training run on Python docstrings gave realistic distributions to design against. Once the real weights were reachable, one script re-recorded everything with them, and the parity tests passed on the first run.
43. *What surprised you?* That the biggest single speed-up was thread wake-up latency, not arithmetic; and that a single instruction choice by LLVM (`vpinsrw` into a live register) cost 40% of the Q8_0 kernel.
44. *How do you know your benchmark is fair to llama.cpp?* Same GGUF bytes, same thread count, same machine, llama.cpp built from source with native flags, its own `llama-bench` tool. And I report where it's faster.

## 4. Exercises

Do these by hand; each has a way to check yourself.

1. **Add a sampler.** Implement typical sampling (keep tokens whose surprisal `-log p` is closest to the entropy, until their mass reaches `tau`) in `sampler.rs::distribution`. Check: with `tau = 1.0` it must return the same distribution as temperature sampling; add a unit test like `filters_shape_the_distribution`.
2. **Change the block size and predict the effect.** Set `BLOCK` in `kv.rs` to 32. Before running anything, write down: what happens to memory waste per sequence (on average half a block, so it doubles), to prefix-sharing hit rates (lower: only multiples of 32 can be shared), and to the scheduler tests (they should still pass). Then run `cargo test -p glassbox-server` and the `serve` benchmark's shared-prompt workload and compare.
3. **Explain a number.** In `bench/results/single`, Q4_0 decode on 360M is 69 tok/s and Q8_0 is 49. Compute bytes per token for each (weights only) and the implied bandwidth. Is Q4_0 bandwidth-bound? (Hint: run `glassbox-bench kernels` with `GLASSBOX_THREADS=4` and compare GB/s on the 49152x576 rows.)
4. **Break losslessness on purpose.** In `spec.rs`, change the acceptance test to `p[x] > 0.5` (a common "approximate" shortcut). Run `cargo test -p glassbox-core spec` and explain why the chi-square test now fails and the greedy test may not.
5. **Add `minLength` to the schema machine.** In `json_schema.rs`, refuse the closing quote while `len < min`. Add it to the random schema generator in `tests/json_schema.rs` and to `validate`. All 200 must still pass.
