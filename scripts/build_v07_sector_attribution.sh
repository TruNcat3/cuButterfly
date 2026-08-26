#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-"$ROOT/build-ncu-lineinfo"}
CUDA_COMPILER=${CUDA_COMPILER:-/usr/local/cuda-11.8/bin/nvcc}
CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES:-70}
JOBS=${JOBS:-$(nproc)}

cmake -S "$ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
    -DCMAKE_CUDA_FLAGS=-lineinfo
cmake --build "$BUILD_DIR" --target cuntt_bench -j"$JOBS"

printf 'lineinfo profiling binary: %s\n' "$BUILD_DIR/cuntt_bench"
