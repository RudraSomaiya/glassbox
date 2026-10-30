"""Golden fixtures from Hugging Face transformers (CPU, float32).

For each model directory under models/ (real SmolLM2 or stand-in), writes
golden.json next to the weights: prompt token ids, top-20 logits at a few
positions, a 64-token greedy continuation and the top-1/top-2 margin at each
greedy step. Also writes tokenizer_parity.json: ids from HF `tokenizers` for
the corpus in corpus.py.

    uv run python make_golden.py [model_dir ...]
"""

import glob
import json
import os
import sys

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

from corpus import strings

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "models")

USER_PROMPTS = [
    "What is the capital of France?",
    "Write a haiku about rain.",
    "Explain what a KV cache is in one sentence.",
    "List three prime numbers.",
    "Translate 'good morning' into Hindi.",
    "What is 17 times 23?",
    "Write a Python function that reverses a string.",
    "Why is the sky blue?",
    "Give me a name for a cat.",
    "Summarise the plot of Romeo and Juliet in two sentences.",
]
RAW_PROMPTS = [
    "Once upon a time",
    "The three laws of thermodynamics are",
    "def fibonacci(n):",
    "The quick brown fox",
    "In 1969, Neil Armstrong",
    "Rust is a systems programming language that",
    "नमस्ते, मेरा नाम",
    "1, 2, 3, 4, 5,",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
    "SELECT * FROM",
    "   indented text\n\twith tabs",
]
SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"


def chat(user):
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def top20(row):
    v, i = torch.topk(row, 20)
    return {"ids": i.tolist(), "logits": v.tolist()}


@torch.no_grad()
def golden(model_dir):
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()
    cases = []
    for text in [chat(p) for p in USER_PROMPTS] + RAW_PROMPTS:
        ids = tok.encode(text, add_special_tokens=False).ids
        x = torch.tensor([ids])
        logits = model(x).logits[0]
        positions = sorted({0, len(ids) // 2, len(ids) - 1})
        # Greedy with the model's own cache, recording how close each choice was.
        out, margins = [], []
        past, inp = None, x
        for _ in range(64):
            r = model(inp, past_key_values=past, use_cache=True)
            past = r.past_key_values
            last = r.logits[0, -1]
            v, i = torch.topk(last, 2)
            out.append(int(i[0]))
            margins.append(float(v[0] - v[1]))
            inp = i[:1].view(1, 1)
        cases.append({
            "text": text, "ids": ids,
            "logits": {str(p): top20(logits[p]) for p in positions},
            "greedy": out, "margins": margins,
        })
        print(f"  {len(ids):3d} tokens, min margin {min(margins):.4f}")
    with open(os.path.join(model_dir, "golden.json"), "w") as f:
        json.dump({"model": os.path.basename(model_dir), "cases": cases}, f)

    corpus = strings()
    enc = tok.encode_batch(corpus, add_special_tokens=False)
    with open(os.path.join(model_dir, "tokenizer_parity.json"), "w") as f:
        json.dump([{"text": t, "ids": e.ids} for t, e in zip(corpus, enc)], f, ensure_ascii=False)


if __name__ == "__main__":
    dirs = sys.argv[1:] or sorted(d for d in glob.glob(os.path.join(ROOT, "*")) if os.path.isdir(d))
    for d in dirs:
        if os.path.exists(os.path.join(d, "model.safetensors")) and "1.7B" not in d:
            print(d)
            golden(d)
