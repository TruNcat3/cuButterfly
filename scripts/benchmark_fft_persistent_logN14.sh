#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BINARY=${BINARY:-"$ROOT/build/cubutterfly_bench"}
RAW=${RAW:-"$ROOT/results/fft_persistent_logN14_v100_raw.csv"}
SUMMARY=${SUMMARY:-"$ROOT/results/fft_persistent_logN14_v100_summary.csv"}

python3 "$ROOT/scripts/sweep_cubutterfly_designs.py" \
    --binary "$BINARY" --operators fft --precisions fp32 --logNs 14 \
    --directions forward --normalizations none --placements out-of-place \
    --backends online-reorder cufft --compute-units radix4 \
    --complex-multiplies four-mul --cross-twiddles table recurrence \
    --local-exchanges shared --fft-cores scalar cufftdx-block cufftdx-resident \
    --target-points 4194304 --tile-thread-options 256 \
    --resident-thread-options 256 512 1024 \
    --hierarchical-local-stages 5 7 9 --reorder-column-options 1 \
    --warmup 20 --repeat 100 --trials 5 --output "$RAW"

python3 "$ROOT/scripts/summarize_cubutterfly_designs.py" "$RAW" --output "$SUMMARY"
printf 'Raw trials: %s\nSummary: %s\n' "$RAW" "$SUMMARY"
