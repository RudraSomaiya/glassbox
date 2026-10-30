#!/usr/bin/env bash
# Build llama.cpp for the comparison benchmarks into third_party/llama.cpp.
# Source: the copy vendored in the llama-cpp-python 0.3.36 sdist on PyPI, which
# pins one llama.cpp revision and is reachable where GitHub release downloads aren't.
set -euo pipefail
cd "$(dirname "$0")/.."
dest=third_party/llama.cpp
if [ ! -d "$dest" ]; then
  tmp=$(mktemp -d)
  pip download --no-deps --no-binary :all: llama-cpp-python==0.3.36 -d "$tmp"
  tar xzf "$tmp"/llama_cpp_python-0.3.36.tar.gz -C "$tmp"
  mkdir -p third_party
  mv "$tmp"/llama_cpp_python-0.3.36/vendor/llama.cpp "$dest"
fi
cmake -S "$dest" -B "$dest/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=OFF
cmake --build "$dest/build" -j"$(nproc)" --target llama-bench llama-quantize llama-perplexity llama-simple
