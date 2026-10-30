# Architecture

Glassbox is four Rust crates, a Python reference toolkit and a React app.

```
crates/
  glassbox-core     tokenizer, kernels, quantization, model, sampler, paged KV cache,
                    speculative decoding, thread pool, trace hooks
  glassbox-server   continuous-batching scheduler, JSON-schema constraints,
                    OpenAI-compatible HTTP API, the `glassbox` CLI
  glassbox-wasm     wasm-bindgen bindings: one model, one sequence, in a Web Worker
  glassbox-bench    benchmark harness writing bench/results/*.json
tools/reference/    Python (uv): golden fixtures, GGUF writer, baselines, charts
web/                the glass box (React 19, TypeScript, Vite)
```

The rule that shapes everything: **one forward pass**. The CLI, the server, the benchmarks and the browser all call `Model::forward`. Platform differences live only in kernel selection.

## Data flow of one forward pass

`Model::forward(tokens, pos, seqs, want, cache, trace)` (`crates/glassbox-core/src/model.rs`) takes a flat batch of `n` tokens. Token `i` belongs to sequence `seqs[i]` at position `pos[i]`. A single sequence's prefill, one decode step for 32 sequences, or a mix of prefill chunks and decode steps all look the same to it.

1. **Embed.** Copy row `tokens[i]` of the embedding matrix into `x[i]` (dequantizing if the embedding is Q8_0/Q4_0). `x` is `[n, hidden]`, row-major f32.
2. **For each layer:**
   1. `h = rms_norm(x) * attn_norm`.
   2. `qkv = h · Wqkv^T`: one matmul for Q, K and V together (the three weight matrices are stacked by rows at load time), split per token into `q [n, heads*hd]`, `k`, `v [n, kv_heads*hd]`.
   3. RoPE on every head of `q` and `k`, at each token's own position. Half-split pairs `(i, i + hd/2)` for Hugging Face weights; interleaved pairs `(2i, 2i+1)` for GGUF, whose Q/K rows llama.cpp's converter permutes.
   4. Write `k` and `v` into the paged cache at each token's slot.
   5. Attention, one task per `(token, head)`: scores against every cached key of that token's sequence at positions `0..=pos`, softmax, weighted sum of values. Heads map to KV heads by `head / (heads / kv_heads)` (grouped-query attention). Causality falls out of the position bound, so a prefill chunk and a decode step need no mask.
   6. `x += att · Wo^T`.
   7. `h = rms_norm(x) * mlp_norm`, `gu = h · Wgate_up^T` (stacked), `act = silu(gate) * up`, `x += act · Wdown^T`.
   8. If tracing, copy the watched token's residual and its attention weights.
3. **Head.** For the batch indices in `want` only (the last token of each sequence that needs a next-token distribution), `rms_norm(x) * norm`, then `· Wlm^T` (the embedding matrix when tied). Returns `[want.len(), vocab]` logits.

Batched and unbatched runs are bitwise identical: every output cell of a matmul is one dot product whose summation order doesn't depend on which other tokens share the batch, and attention is per token. The scheduler property tests rely on this.

## Tensors in memory

- **f32 matrix**: `Vec<f32>` `[rows, cols]` row-major, `rows = out_features` (the `nn.Linear` layout). A matvec reads each row contiguously.
- **Q8_0**: blocks of 32 weights, `#[repr(C)] { d: f16 bits, qs: [i8; 32] }` = 34 bytes, value = `qs[j] * d`. Bit-identical to ggml's layout and reference quantizer.
- **Q4_0**: `{ d: f16 bits, qs: [u8; 16] }` = 18 bytes; low nibble of `qs[j]` is weight `j`, high nibble weight `j + 16`; value = `(nibble - 8) * d`.
- Quantized matmuls first quantize each activation row to Q8_0, so the inner loop is `i8 x i8` products (`maddubs`/`madd` on AVX2, a single `vpdpbusd` with AVX-VNNI, which needs one operand unsigned, so the sign of x is moved onto y first) with one f32 multiply per block. The int8 kernels are one macro instantiated for both instruction sets and picked at runtime.
- Weights are loaded from safetensors (bf16/f16/f32, memory-mapped and converted) or GGUF (F32/F16/Q8_0/Q4_0). The browser quantizes each matrix to Q8_0 as it's read, so peak memory stays near the file size.

## Kernels

`crates/glassbox-core/src/kernels.rs`. Every SIMD kernel has a scalar twin; tests compare them on awkward lengths (1, 7, 31, 33, 1537...).

- **Register tiles.** The AVX2 kernels compute `R` weight rows against `N` activation rows at once, `out[i][k] = dot(w[i], x[k])`. Decode uses 4x1 (four independent accumulators hide FMA latency, the activation row is loaded once per four weight rows); prefill uses 2x4 (each weight block is loaded and, for Q4_0, unpacked once for four tokens). Two or three tokens (speculative verification, small batches) use 4x2, so weights aren't re-read per token.
- **f16 scales through a table.** `_mm_cvtph_ps` on a single value compiled to a `vpinsrw` into the accumulator register, a false dependency that chained every block. A 64K-entry lookup table (as in ggml) removed it.
- **Prefetch.** Each task prefetches the next row tile's bytes. One core streaming from DRAM otherwise keeps too few misses in flight.
- **Threads.** `pool.rs`, a fork-join pool whose workers spin for 2 ms after each job before parking. A matmul is split into tasks of 16 output rows. Decode runs ~150 parallel regions per token, and rayon's futex wake-ups (~50 us each) cost more than the work.
- **WebAssembly.** `simd128` versions of the same three dot products, single-threaded.

## KV cache

`crates/glassbox-core/src/kv.rs`. One pool of blocks per layer, `[block][slot][kv_dim]` f32, 16 slots per block.

- Each sequence has a **block table** (logical block -> physical block) and a length. Token slot = `table[pos / 16] * 16 + pos % 16`.
- Blocks are **reference counted**. `fork` points a new sequence at the same blocks; the first write into a shared, partly filled block **copies it first** (copy-on-write).
- **Prefix sharing**: when a block fills, it is hashed together with every token before it (a chained FNV hash) and registered. A new sequence whose prompt starts with the same tokens reuses those blocks and skips their computation. Blocks no sequence uses stay in a cached list (oldest evicted first) so a later request can still hit them.
- `check_invariants` recomputes every reference count from the block tables; the scheduler tests call it after every step to catch leaks.

Memory per token for SmolLM2-135M: `2 (K and V) * 30 layers * 192 (3 KV heads * 64) * 4 bytes = 46 kB`.

## Scheduler

`crates/glassbox-server/src/scheduler.rs`. One `step()` is one forward pass:

1. Drop sequences whose client hung up.
2. **Admit** waiting requests while there is room for their first chunk (`add_seq` reports how many prompt tokens the prefix cache already holds).
3. **Plan** under the token budget: one token for every decoding sequence first, then prefill chunks (`min(remaining prompt, budget left)`), so a long prompt is read over several steps while others keep decoding.
4. **Reserve** cache blocks. When the pool is exhausted, **preempt** the most recently admitted sequence: free its blocks, reset it to zero computed tokens, and put it back at the front of the queue to be recomputed (often mostly from the prefix cache).
5. Build the flat batch, run `forward`, commit the written tokens, sample for each sequence that finished its prompt or is decoding (applying its JSON-schema mask if it has one), stream the token, and finish sequences on a stop token, `max_tokens` or a completed schema.

Metrics (TTFT, inter-token latency, throughput, queue depth, KV utilisation, preemptions, prefix hits) are exported at `/metrics`; `/v1/trace` records every step for the web app's scheduler view.

## Speculative decoding

`crates/glassbox-core/src/spec.rs`. A drafter (a smaller model with the same tokenizer, the same model at lower precision, or prompt lookup) proposes up to `k` tokens; the target runs **one** forward pass over the pending token plus all drafts and gets `k + 1` distributions. Draft `x` with draft probability `q(x)` and target probability `p(x)` is accepted with probability `min(1, p(x)/q(x))`; the first rejection is replaced by a draw from `max(0, p - q)` normalized; if all are accepted the target adds one bonus token. Both caches are truncated back to the accepted prefix. Greedy decoding reduces this to "accept while the target's argmax agrees", which is why the output is identical to the target alone; a chi-square test checks the sampled case.

## Constrained decoding

`crates/glassbox-server/src/json_schema.rs`. A JSON schema compiles to a byte-level pushdown machine (a stack of frames: fixed bytes, string body, number, literal choice, array items, value to expand). Each step the machine walks a trie of every token's bytes, depth first, carrying its state down each edge; tokens whose bytes are all accepted stay, the rest get `-inf` logits. Inside a string body the walk is skipped: tokens made only of plain string bytes are precomputed and allowed in bulk. Strings, arrays and numbers are capped, so output always ends.

## Browser

`crates/glassbox-wasm` wraps one `Generator` (one sequence and its cache). `web/src/engine/worker.ts` downloads `config.json`, `tokenizer.json` and `model.safetensors` from Hugging Face once, keeps them in the Cache API, and streams tokens to the page with the top-10 alternatives for each. On request it re-runs a prefix with the tracer on (`inspect`) to produce the logit lens and attention for one position, so tracing costs nothing during normal generation.

## Trace hooks

`Trace` (in `model.rs`) names one batch index to watch and what to capture: the residual after each layer (`[layer][hidden]`) and the attention weights (`[layer][head][position]`). With no tracer the forward pass only checks an `Option` per layer.
