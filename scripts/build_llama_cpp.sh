#!/usr/bin/env bash
# One-time build of llama.cpp for the GGUF conversion target.
# Run from anywhere; clones/builds into ~/llama.cpp by default.
#
# Usage: bash scripts/build_llama_cpp.sh [install_dir]

set -euo pipefail

INSTALL_DIR="${1:-$HOME/llama.cpp}"

if [ -d "$INSTALL_DIR/.git" ]; then
  echo "[build_llama_cpp] $INSTALL_DIR already exists, pulling latest"
  git -C "$INSTALL_DIR" pull --ff-only
else
  git clone https://github.com/ggml-org/llama.cpp "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# CUDA build if nvcc is on PATH (WSL2 + CUDA toolkit installed), else CPU-only.
if command -v nvcc >/dev/null 2>&1; then
  echo "[build_llama_cpp] nvcc found, building with CUDA support"
  cmake -B build -DGGML_CUDA=ON
else
  echo "[build_llama_cpp] nvcc not found, building CPU-only (GGUF conversion itself is CPU-only anyway)"
  cmake -B build
fi

cmake --build build --config Release -j "$(nproc)"

echo ""
echo "Build complete. Add this to your shell profile or run before conversion:"
echo "  export LLAMA_CPP_DIR=$INSTALL_DIR"
