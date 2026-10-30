#!/usr/bin/env bash
# llama.cpp perplexity on the same GGUF files and text as `glassbox-bench quality`
# (512-token chunks, second half scored), for bench/results/quality-*.json.
# Usage: scripts/quality_llamacpp.sh MODEL_DIR OUT.json
set -euo pipefail
cd "$(dirname "$0")/.."
dir=$1; out=$2
{
  echo "{"
  first=1
  for q in f32 q8_0 q4_0; do
    [ -f "$dir/model-$q.gguf" ] || continue
    ppl=$(third_party/llama.cpp/build/bin/llama-perplexity -m "$dir/model-$q.gguf" -f data/perplexity.txt \
      -c 512 --chunks 100 -t ${THREADS:-4} 2>&1 | sed -n 's/.*Final estimate: PPL = \([0-9.]*\).*/\1/p')
    [ $first = 1 ] || echo ","
    first=0
    printf '  "%s": %s' "$q" "${ppl:-null}"
  done
  echo
  echo "}"
} > "$out"
cat "$out"
