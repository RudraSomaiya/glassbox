# Building an LLM inference engine from scratch, and a window into it

I've built a fair amount on top of language models: retrieval pipelines, OCR, agents for clients, a fine-tuned model in Kaun?. What I hadn't done was look underneath. When I call a model, what actually happens between my prompt and the first token? How does a server keep a CPU busy with ten people talking to it at once? Why does a 4-bit model run faster, and what does it lose?

So I wrote one. Glassbox is an inference engine in Rust with no ML framework: the tokenizer, the matrix kernels, quantization, the KV cache, the batching scheduler, speculative decoding and constrained JSON output are all mine. It runs SmolLM2 (135M, 360M and 1.7B) natively and, compiled to WebAssembly, in the browser, where a web app shows you everything the model does while it writes.

This is how it went, what made it fast, and where it's still slower than llama.cpp.

## Correct first

I didn't trust a single number until the engine matched Hugging Face `transformers` exactly. A Python script writes golden fixtures: token ids, the top-20 logits at a few positions, and a 64-token greedy continuation for 21 prompts. The Rust tests check the logits to within 1e-3 (the real error ended up around 4e-6) and the continuation token for token.

The fixtures caught a bug I'd never have spotted by reading. Logits matched at position 0 and drifted after it. Position 0 is where RoPE does nothing, so the rotation had to be wrong, but the code looked fine. It was the config: newer versions of `transformers` store `rope_theta` inside a `rope_parameters` object, my parser didn't find it at the top level, and quietly used the default of 10,000 instead of 100,000. One `Config::from_json` later, every greedy continuation matched.

The tokenizer got the same treatment: 2,175 strings in English, code, emoji, Hindi, Hinglish and nasty whitespace, encoded by the Hugging Face library and by mine. All identical. And when I wrote a GGUF loader, I checked that my Q8_0 quantizer produced the same bytes as llama.cpp's reference quantizer (884,736 blocks, all identical), and that both engines gave the same perplexity on the same file: 54,636.99 against 54,636.73. Two independent implementations agreeing to five significant figures is a stronger check than any unit test I could write.

## Making it fast

The first working version decoded the 135M model at 8.5 tokens a second on one core with scalar code. Here's how it got to 212.

![Kernel progression](../bench/results/charts/progression.svg)

**SIMD.** AVX2 and FMA versions of the dot products, chosen at runtime, each tested against the scalar version on awkward lengths like 1, 31 and 1537. Then register tiles: instead of one weight row times one activation vector, compute four weight rows against one vector (decode) or two rows against four vectors (prefill). Loads get reused and the four accumulators hide FMA latency. 8.5 to 22.3 tokens a second.

**Threads.** Splitting each matrix multiplication across four cores took it to 65, after one surprise I'll come back to.

**Quantization.** Decode is bound by memory bandwidth: each weight is read once per token and used for two flops. Store weights in 8 bits and there's less to read. Q8_0 keeps blocks of 32 weights with one f16 scale; Q4_0 packs two weights per byte. To keep the inner loop in integers, each activation vector is also quantized to Q8_0 before the multiply, so a block's dot product is 32 int8 multiply-adds and one float multiply. 65 to 178 (Q8_0) and 212 (Q4_0).

Three things along the way were worth more than any amount of clever arithmetic:

1. **A false dependency.** My Q8_0 kernel was inexplicably slow. In the disassembly, the conversion of each block's f16 scale (`_mm_cvtph_ps` on one value) had compiled to a `vpinsrw` into the register holding the accumulator. So every block's scale conversion waited on the previous block's FMA, serializing the whole loop. A 64K-entry lookup table, which is what ggml does, fixed it: 69 µs to 43 µs per 576x576 matrix.
2. **Prefetching.** In a microbenchmark that fit in cache my kernel ran at 15 GB/s; on the real model, 4. One core streaming weights from DRAM doesn't keep enough cache misses in flight. Prefetching the next tile of rows while computing the current one doubled it.
3. **Thread wake-ups.** With four threads, decode only went from 44 to 74 tokens a second. One decode step runs about 150 short parallel regions (one per matrix multiplication), each 30 to 300 µs, and rayon parks its idle workers and wakes them through the kernel, which cost about 50 µs a time. I replaced it with a 150-line fork-join pool whose workers spin for 2 ms after each job before sleeping. 74 to 159.

The lesson I'm keeping: measure first. Two of those three were invisible in the source and obvious in a profile or a disassembly.

## Where it's slower than llama.cpp, and why

Same machine, same GGUF files, same thread count:

![Decode speed](../bench/results/charts/single-decode.svg)

On decode, Glassbox runs at 50-80% of llama.cpp. On prompt processing it's 3-7x slower (317 against 2,416 tokens a second for 135M Q8_0). The gap is the matrix-matrix kernel: llama.cpp on this CPU uses AVX-512 with VNNI, a single instruction for the multiply-add sequence I do in three, and repacks weights into interleaved layouts so its GEMM streams contiguous memory. Mine is AVX2 dot products over the original layout. That's the next thing I'd build. It also shows up in serving, below: when prefill is slow, everything that waits on prefill is slow.

`transformers` is a useful other baseline: it beats me at prefill (it calls oneDNN), and I beat it at decode in f32 (63 against 37 tokens a second at 135M), because a Python loop around PyTorch has a lot of per-token overhead.

## Memory: a paged KV cache

To write the next token, attention needs the keys and values of every earlier token, so they're cached. A naive cache reserves the full context per sequence up front. Glassbox's is paged like virtual memory: a pool of 16-token blocks, a block table per sequence, reference counts. That makes three things cheap. Memory grows with the text. A forked sequence shares its parent's blocks and copies one only when it writes into it. And two requests that start with the same system prompt share those blocks: each full block is hashed together with everything before it, and a new request reuses any block whose hash it finds.

## Many users at once

The server batches requests into the same forward pass. Continuous batching means the batch is rebuilt every step: finished sequences leave, new ones join, decoding sequences each contribute one token, and new prompts are read in chunks so a long prompt can't stall everyone else. When the cache is full, the newest sequence is preempted and recomputed later.

What made me trust it was a property test: random mixes of prompts, lengths, aborts and forced preemptions on a tiny random model, checking that every sequence's greedy output is bit-for-bit what it would be alone, and that no block is ever leaked. Bit-for-bit holds because every output of a matrix multiplication is one dot product whose summation order doesn't depend on what else is in the batch.

SERVING_PARAGRAPH

## Guess ahead, check in bulk

Speculative decoding exploits the same fact as batching: checking five tokens costs about as much as generating one. A cheap drafter guesses several tokens; the real model checks them all in one pass and keeps the ones it agrees with. Rejection sampling (accept a draft with probability p/q, otherwise resample from what's left) makes the output distribution exactly the real model's. I test that two ways: greedy output must be identical to the target alone, and a chi-square test on thousands of sampled outputs must not tell them apart.

SPEC_PARAGRAPH

## Always-valid JSON

For `response_format: json_schema`, the schema compiles to a small byte-level machine (a stack of "expect these bytes", "inside a string", "a number", "one of these values", "more array items"). At each step I walk a trie of every token's bytes with the machine, pruning a branch at the first byte it rejects, and mask every token it can't complete. Inside a string, where almost everything is allowed, a precomputed list of plain tokens skips the walk. A test generates 200 random nested schemas and checks that every output parses and validates. It found two real bugs, including a backslash near a string's length cap being treated as a plain character, which left the string unterminated.

## The glass box

The part people see is a web app where the engine runs in a Web Worker. Text appears as a strip of film: each token is a frame, and its density is how sure the model was, on a log scale like real film density. Hover a frame and a loupe shows the ten tokens the model almost wrote; click one and the story forks from there. Click a frame to see what each of the model's layers would have predicted at that point (the "logit lens") and which earlier tokens each attention head looked at; those are computed on demand by re-running the prefix with a tracer, so generation never pays for them. There are live sampling dials, the KV cache filling up block by block, and replays of batching and speculative decoding recorded from the server.

## What surprised me

- The biggest single speed-up was thread wake-up latency, not arithmetic.
- One instruction LLVM chose cost 40% of a kernel.
- Testing against a second implementation (transformers for the model, llama.cpp for quantization and perplexity) found more bugs than reasoning did.
- A lot of "LLM serving" is ordinary systems work: memory allocators, schedulers, reference counts, invariants.

## An honest note on the numbers

The machine this was built on couldn't download from Hugging Face. Every speed number comes from models with SmolLM2's exact shapes and random weights, which is valid because speed depends only on shapes. Anything that depends on what the model knows (perplexity values, speculative acceptance between two different models, the text in the recorded replays) uses either random weights or a tiny stand-in model I trained locally for 14 minutes, and says so. One script re-records it all with the real weights.

## What's next

AVX-512 VNNI kernels and repacked weights for prefill; multithreaded WebAssembly; a second architecture (Qwen3) to prove nothing is hard-coded; the real-weight numbers.
