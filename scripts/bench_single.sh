#!/usr/bin/env bash
# Single-stream benchmarks: Glassbox (f32 / Q8_0 / Q4_0), llama.cpp on the
# same GGUF files, and transformers f32. Usage: scripts/bench_single.sh MODEL_DIR NAME
# Needs: cargo build --release; scripts/build_llamacpp.sh; to_gguf.py run on MODEL_DIR.
set -euo pipefail
cd "$(dirname "$0")/.."
dir=$1; name=$2; t=${THREADS:-4}
out=bench/results/single; mkdir -p $out
for q in f32 q8_0 q4_0; do
  ./target/release/glassbox-bench single --model "$dir" --gguf "$dir/model-$q.gguf" --threads $t \
    --label "$name" --out $out/glassbox-$name-$q.json
  third_party/llama.cpp/build/bin/llama-bench -m "$dir/model-$q.gguf" -t $t -p 512 -n 128 -r 3 -o json \
    > $out/llamacpp-$name-$q.json
done
(cd tools/reference && uv run python bench_transformers.py "../../$dir" "../../$out/transformers-$name-f32.json" $t)
