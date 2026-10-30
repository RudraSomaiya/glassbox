#!/usr/bin/env bash
# Record the replay traces bundled with the web app (web/src/traces/*.json).
#   scripts/record_traces.sh                      # stand-in model (models/standin-tiny)
#   SMALL=models/SmolLM2-135M-Instruct BIG=models/SmolLM2-1.7B-Instruct scripts/record_traces.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SMALL=${SMALL:-models/standin-tiny}
BIG=${BIG:-$SMALL}
out=web/src/traces
bin=./target/release/glassbox
# The stand-in was trained on Python docstrings, so ask it something from its own world.
if [ -f "$SMALL/STANDIN" ]; then Q="What does the open() function return?"; else Q="Explain what a KV cache is in one sentence."; fi
cargo build --release -p glassbox-server >/dev/null

$bin trace --model "$SMALL" --dtype q8_0 --prompt "$Q" -n 48 --temp 0.6 --seed 2 --out $out/hero.json
$bin anatomy --model "$SMALL" --prompt "$Q" --out $out/anatomy.json
# Same model at 4 bits drafting for itself at full precision, and prompt lookup.
$bin spec-trace --model "$BIG" --draft "$BIG" --draft-dtype q4_0 -k 4 -n 64 \
  --prompt "Describe how a lighthouse works." --out $out/speculative.json
$bin spec-trace --model "$BIG" --lookup -k 4 -n 64 \
  --prompt "Repeat this sentence twice: the quick brown fox jumps over the lazy dog." --out $out/lookup.json

# Scheduler: a small KV pool so preemption shows up, ten staggered requests.
$bin serve --model "$BIG" --dtype q8_0 --port 8091 --blocks 48 --budget 128 >/dev/null 2>&1 &
pid=$!
trap 'kill $pid' EXIT
until curl -sf localhost:8091/health >/dev/null; do sleep 0.5; done
python3 - <<'PY' > /tmp/glassbox-trace-req.json
import json
qs = ["Name three rivers.", "What is a prime number?", "Write a haiku about rain.",
      "Explain gravity to a child.", "List four colours.", "Why do cats purr?",
      "Summarise the plot of Hamlet.", "What is 12 times 12?", "Describe a sunrise.",
      "Give me a tip for learning Rust."]
# Arrivals close enough together that requests overlap and share passes.
print(json.dumps({"requests": [{"messages": [{"role": "user", "content": q}], "delay_ms": i * 25,
                                "max_tokens": 40 + (i * 7) % 32} for i, q in enumerate(qs)]}))
PY
curl -sf localhost:8091/v1/trace -H 'content-type: application/json' -d @/tmp/glassbox-trace-req.json > $out/scheduler.json
python3 - "$out/scheduler.json" "$BIG" <<'PY'
import json, os, sys
f, model = sys.argv[1], sys.argv[2]
t = json.load(open(f))
t["standin"] = os.path.exists(os.path.join(model, "STANDIN"))
json.dump(t, open(f, "w"))
print("scheduler trace:", len(t["requests"]), "requests,", len(t["steps"]), "steps")
PY
ls -la $out
