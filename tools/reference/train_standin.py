"""Train a tiny Llama-architecture stand-in so the glass box has realistic
probability distributions to show when the real SmolLM2 weights can't be
downloaded. Random weights give near-uniform distributions (every token at
~1/49152), which makes every view degenerate.

The result is NOT SmolLM2: 6 layers, 384 wide, trained for a few minutes on
the English docstrings of Python's standard library with the stand-in
tokenizer. It writes text that looks like Python documentation.

    uv run python train_standin.py [minutes]
"""

import math
import os
import sys
import time

import torch
from tokenizers import Tokenizer
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(ROOT, "models", "standin-tiny")


def main(minutes):
    torch.manual_seed(0)
    torch.set_num_threads(4)
    tok = Tokenizer.from_file(os.path.join(ROOT, "models", "standin-tokenizer.json"))
    text = open(os.path.join(ROOT, "data", "perplexity.txt")).read()
    # Chat-shaped samples teach the template too, so the browser's chat prompt works.
    ids = tok.encode(text).ids
    print("corpus tokens", len(ids))
    data = torch.tensor(ids)
    cfg = LlamaConfig(
        vocab_size=49152, hidden_size=384, intermediate_size=1024, num_hidden_layers=6,
        num_attention_heads=6, num_key_value_heads=2, max_position_embeddings=2048,
        rms_norm_eps=1e-5, rope_theta=10000.0, tie_word_embeddings=True,
        bos_token_id=1, eos_token_id=2,
    )
    model = LlamaForCausalLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
    seq, batch = 256, 16
    deadline = time.time() + minutes * 60
    est_steps = max(1, int(minutes * 60 / 2.5))
    step = 0
    while time.time() < deadline:
        lr = 3e-3 * min(1.0, (step + 1) / 30) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / est_steps))))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.randint(0, len(data) - seq - 1, (batch,))
        x = torch.stack([data[i:i + seq] for i in idx])
        loss = model(x, labels=x).loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0:
            print(f"step {step} loss {loss.item():.3f} lr {lr:.2e}", flush=True)
        step += 1
    os.makedirs(OUT, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(OUT, safe_serialization=True)
    with open(os.path.join(ROOT, "models", "standin-tokenizer.json")) as f, open(os.path.join(OUT, "tokenizer.json"), "w") as g:
        g.write(f.read())
    with open(os.path.join(OUT, "STANDIN"), "w") as f:
        f.write(f"tiny Llama (6 layers, 384 wide) trained {step} steps on Python docstrings; not SmolLM2\n")
    print("wrote", OUT, "after", step, "steps, final loss", loss.item())


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 15)
