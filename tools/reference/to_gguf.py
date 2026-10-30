"""Write F32, Q8_0 and Q4_0 GGUF files for a Llama-family model directory.

    uv run python to_gguf.py ../../models/SmolLM2-135M-Instruct

Writes model-{f32,q8_0,q4_0}.gguf next to the weights, readable by both
Glassbox and llama.cpp. Follows llama.cpp's converter: Q/K rows are permuted
for interleaved RoPE, norms stay f32, every matrix (embedding included) is in
the requested type (like `llama-quantize --pure`). Quantization uses the
`gguf` package's port of ggml's reference quantizers.
"""

import json
import os
import sys

import gguf
import numpy as np
from safetensors.torch import load_file


def permute(w, n_head):
    # HF rotates pairs (i, i + d/2); llama.cpp rotates (2i, 2i + 1).
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def write(model_dir, qtype, out):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    theta = cfg.get("rope_theta") or cfg["rope_parameters"]["rope_theta"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    tensors = {k: v.float().numpy() for k, v in load_file(os.path.join(model_dir, "model.safetensors")).items()}

    w = gguf.GGUFWriter(out, "llama")
    w.add_context_length(cfg["max_position_embeddings"])
    w.add_embedding_length(cfg["hidden_size"])
    w.add_block_count(cfg["num_hidden_layers"])
    w.add_feed_forward_length(cfg["intermediate_size"])
    w.add_head_count(nh)
    w.add_head_count_kv(nkv)
    w.add_rope_dimension_count(cfg["hidden_size"] // nh)
    w.add_rope_freq_base(theta)
    w.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    w.add_vocab_size(cfg["vocab_size"])
    w.add_file_type({"F32": 0, "Q8_0": 7, "Q4_0": 2}[qtype.name])

    tok = json.load(open(os.path.join(model_dir, "tokenizer.json")))
    vocab = sorted(tok["model"]["vocab"].items(), key=lambda kv: kv[1])
    tokens = [t for t, _ in vocab]
    special = {t["id"] for t in tok["added_tokens"]}
    merges = [m if isinstance(m, str) else " ".join(m) for m in tok["model"]["merges"]]
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("smollm")
    w.add_token_list(tokens)
    w.add_token_types([gguf.TokenType.CONTROL if i in special else gguf.TokenType.NORMAL for i in range(len(tokens))])
    w.add_token_merges(merges)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_add_bos_token(False)

    names = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    per_layer = {
        "input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm",
        "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down",
    }
    for name, t in tensors.items():
        if name == "lm_head.weight" and cfg.get("tie_word_embeddings"):
            continue
        if name.startswith("model.layers."):
            i, rest = name[len("model.layers."):].split(".", 1)
            base = rest.rsplit(".", 1)[0]
            if base == "self_attn.q_proj":
                t = permute(t, nh)
            elif base == "self_attn.k_proj":
                t = permute(t, nkv)
            gname = f"blk.{i}.{per_layer[base]}.weight"
        else:
            gname = names[name]
        t = np.ascontiguousarray(t, dtype=np.float32)
        if t.ndim == 2 and qtype != gguf.GGMLQuantizationType.F32:
            w.add_tensor(gname, gguf.quants.quantize(t, qtype), raw_dtype=qtype)
        else:
            w.add_tensor(gname, t)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main(model_dir):
    for q in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q4_0]:
        out = os.path.join(model_dir, f"model-{q.name.lower()}.gguf")
        if not os.path.exists(out):
            write(model_dir, q, out)
        print(out)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        main(d)
