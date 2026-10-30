#!/usr/bin/env bash
# Downloads model files into models/<name>/. Weights never go in git.
# Usage: scripts/download_models.sh [135M|360M|1.7B ...]   (default: 135M 360M)
set -euo pipefail
cd "$(dirname "$0")/.."
sizes=("$@")
[ ${#sizes[@]} -eq 0 ] && sizes=(135M 360M)
for size in "${sizes[@]}"; do
  repo="HuggingFaceTB/SmolLM2-${size}-Instruct"
  dir="models/SmolLM2-${size}-Instruct"
  mkdir -p "$dir"
  for f in config.json tokenizer.json tokenizer_config.json generation_config.json model.safetensors; do
    [ -s "$dir/$f" ] && continue
    echo "fetching $repo/$f"
    curl -fL --retry 3 -o "$dir/$f.part" "https://huggingface.co/$repo/resolve/main/$f"
    mv "$dir/$f.part" "$dir/$f"
  done
done
