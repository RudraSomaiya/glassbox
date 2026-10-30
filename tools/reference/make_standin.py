"""Build stand-in models when the real SmolLM2 weights can't be downloaded.

A stand-in has SmolLM2's exact architecture (config copied from the model
card) with random weights, and a byte-level BPE tokenizer trained locally with
the same pre-tokenizer pipeline and special tokens as SmolLM2. Everything that
depends only on the architecture (forward-pass parity, tokenizer algorithm
parity, speed) can be checked against it. Text quality cannot.

    uv run python make_standin.py            # 135M and 360M
    uv run python make_standin.py 1.7B       # also the 1.7B shapes (benchmarks only)
"""

import glob
import json
import os
import random
import sys
import sysconfig

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "models")

SPECIAL = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<repo_name>", "<reponame>", "<file_sep>",
    "<filename>", "<gh_stars>", "<issue_start>", "<issue_comment>", "<issue_closed>",
    "<jupyter_start>", "<jupyter_text>", "<jupyter_code>", "<jupyter_output>",
    "<jupyter_script>", "<empty_output>",
]

# From the SmolLM2 model cards' config.json.
CONFIGS = {
    "135M": dict(hidden_size=576, intermediate_size=1536, num_hidden_layers=30,
                 num_attention_heads=9, num_key_value_heads=3, rope_theta=100000.0),
    "360M": dict(hidden_size=960, intermediate_size=2560, num_hidden_layers=32,
                 num_attention_heads=15, num_key_value_heads=5, rope_theta=100000.0),
    "1.7B": dict(hidden_size=2048, intermediate_size=8192, num_hidden_layers=24,
                 num_attention_heads=32, num_key_value_heads=32, rope_theta=130000.0),
}


def corpus():
    """Text to train the stand-in tokenizer: Python's own sources and docs."""
    lib = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(lib, "**", "*.py"), recursive=True))
    random.Random(0).shuffle(files)
    for f in files[:1500]:
        try:
            yield open(f, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            pass
    from corpus import strings
    yield from strings()


def train_tokenizer(path):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=49152, special_tokens=SPECIAL, show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(corpus(), trainer)
    tok.save(path)
    return tok.get_vocab_size()


def main(sizes):
    os.makedirs(ROOT, exist_ok=True)
    tok_path = os.path.join(ROOT, "standin-tokenizer.json")
    if not os.path.exists(tok_path):
        print("training tokenizer, vocab", train_tokenizer(tok_path))
    for size in sizes:
        out = os.path.join(ROOT, f"standin-{size}")
        if os.path.exists(os.path.join(out, "model.safetensors")):
            continue
        cfg = LlamaConfig(
            vocab_size=49152, max_position_embeddings=8192, rms_norm_eps=1e-5,
            tie_word_embeddings=True, bos_token_id=1, eos_token_id=2, hidden_act="silu",
            **CONFIGS[size],
        )
        torch.manual_seed(0)
        model = LlamaForCausalLM(cfg)
        # Random norms (not all ones) so a mistake in applying them shows up.
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "norm" in n:
                    p.uniform_(0.5, 1.5)
        # bf16 on disk, like the real checkpoints, so the loader's bf16 path is exercised.
        model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
        with open(tok_path) as f, open(os.path.join(out, "tokenizer.json"), "w") as g:
            g.write(f.read())
        with open(os.path.join(out, "STANDIN"), "w") as f:
            f.write("random weights with the SmolLM2 architecture; not the real model\n")
        print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:] or ["135M", "360M"])
