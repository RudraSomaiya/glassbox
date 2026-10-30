"""Single-stream baseline: Hugging Face transformers, f32, CPU.

Same shape as glassbox-bench single and llama-bench: prefill 512 tokens in one
forward call (pp512), then 128 greedy decode steps with the KV cache (tg128).

    uv run python bench_transformers.py ../../models/standin-135M out.json [threads]
"""

import json
import platform
import statistics
import subprocess
import sys
import time

import torch
from transformers import AutoModelForCausalLM


@torch.no_grad()
def run(model_dir, threads, reps=3):
    torch.set_num_threads(threads)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    vocab = model.config.vocab_size
    prompt = torch.tensor([[(i * 7919 + 13) % vocab for i in range(512)]])
    pps, tgs = [], []
    for rep in range(reps + 1):
        t = time.perf_counter()
        model(prompt)
        pp = 512 / (time.perf_counter() - t)
        tok, past = torch.tensor([[1]]), None
        t = time.perf_counter()
        for _ in range(128):
            r = model(tok, past_key_values=past, use_cache=True)
            past, tok = r.past_key_values, r.logits[:, -1:].argmax(-1)
        tg = 128 / (time.perf_counter() - t)
        if rep:
            pps.append(pp)
            tgs.append(tg)
        print(f"pp512 {pp:.1f} tg128 {tg:.1f}", file=sys.stderr)
    return pps, tgs


def cpu():
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


if __name__ == "__main__":
    model_dir, out = sys.argv[1], sys.argv[2]
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    pps, tgs = run(model_dir, threads)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    date = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump({
        "engine": "transformers", "model": model_dir, "dtype": "F32", "threads": threads,
        "pp512_tok_s": statistics.mean(pps), "pp512_sd": statistics.pstdev(pps),
        "tg128_tok_s": statistics.mean(tgs), "tg128_sd": statistics.pstdev(tgs),
        "meta": {"cpu": cpu(), "commit": commit, "date": date,
                 "torch": torch.__version__, "transformers": __import__("transformers").__version__},
    }, open(out, "w"), indent=2)
