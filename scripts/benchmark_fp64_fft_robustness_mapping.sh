#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build/cubutterfly_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results"}
SWEEP="$ROOT/scripts/sweep_cubutterfly_designs.py"
SUMMARY="$ROOT/scripts/summarize_cubutterfly_designs.py"

if [[ ! -x "$BIN" ]]; then
    echo "benchmark binary is missing: $BIN" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

python3 "$SWEEP" --binary "$BIN" --operators fft --precisions fp64 \
    --logNs 14 15 16 17 18 --target-points 4194304 \
    --directions forward --normalizations none --placements out-of-place \
    --backends online-reorder --fft-cores cufftdx-block \
    --hierarchical-local-stages 7 8 9 \
    --prefix-thread-options 128 256 512 --suffix-thread-options 128 256 512 \
    --prefix-ept-options 4 8 --suffix-ept-options 4 8 \
    --cross-twiddles recurrence --shared-layouts linear xor-swizzle \
    --warmup 20 --repeat 20 --trials 1 \
    --output "$OUTPUT_DIR/fp64_robustness_mapping_coarse_raw.csv"

python3 "$SUMMARY" "$OUTPUT_DIR/fp64_robustness_mapping_coarse_raw.csv" \
    --output "$OUTPUT_DIR/fp64_robustness_mapping_coarse_summary.csv"

printf 'FP64 robustness mapping summary: %s\n' \
    "$OUTPUT_DIR/fp64_robustness_mapping_coarse_summary.csv"
