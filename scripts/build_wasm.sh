#!/usr/bin/env bash
# Build the engine for the browser into web/src/wasm (simd128, single-threaded).
set -euo pipefail
cd "$(dirname "$0")/.."
CARGO_PROFILE_RELEASE_DEBUG=0 RUSTFLAGS="-C target-feature=+simd128" cargo build -p glassbox-wasm --target wasm32-unknown-unknown --release
wasm-bindgen --target web --out-dir web/src/wasm --out-name glassbox \
  target/wasm32-unknown-unknown/release/glassbox_wasm.wasm
ls -la web/src/wasm/glassbox_bg.wasm
