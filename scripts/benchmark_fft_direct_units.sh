#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BINARY=${BINARY:-"$ROOT/build/cubutterfly_bench"}
RAW=${RAW:-"$ROOT/results/fft_direct_units_v100_raw.csv"}
SUMMARY=${SUMMARY:-"$ROOT/results/fft_direct_units_v100_summary.csv"}

python3 "$ROOT/scripts/sweep_cubutterfly_designs.py" \
    --binary "$BINARY" --operators fft --precisions fp32 \
    --logNs 11 12 13 14 --directions forward --normalizations none \
    --placements out-of-place --backends temporal-tile cufft \
    --complex-multiplies four-mul --cross-twiddles table \
    --local-exchanges shared --fft-cores scalar cufftdx-direct \
    --target-points 4194304 --tile-thread-options 256 \
    --resident-thread-options 256 512 1024 \
    --warmup 20 --repeat 100 --trials 5 --output "$RAW"

python3 "$ROOT/scripts/summarize_cubutterfly_designs.py" "$RAW" --output "$SUMMARY"
printf 'Raw trials: %s\nSummary: %s\n' "$RAW" "$SUMMARY"
