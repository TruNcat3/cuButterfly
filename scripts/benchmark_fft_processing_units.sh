#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BINARY=${BINARY:-"$ROOT/build/cubutterfly_bench"}
RAW=${RAW:-"$ROOT/results/fft_processing_units_v100_raw.csv"}
SUMMARY=${SUMMARY:-"$ROOT/results/fft_processing_units_v100_summary.csv"}
TARGET_POINTS=${TARGET_POINTS:-4194304}
WARMUP=${WARMUP:-20}
REPEAT=${REPEAT:-100}
TRIALS=${TRIALS:-5}

python3 "$ROOT/scripts/sweep_cubutterfly_designs.py" \
    --binary "$BINARY" \
    --operators fft \
    --precisions fp32 \
    --logNs 3 4 5 6 7 8 9 10 \
    --directions forward \
    --normalizations none \
    --placements out-of-place \
    --backends temporal-tile cufft \
    --compute-units radix8 \
    --complex-multiplies four-mul \
    --local-exchanges shared \
    --fft-cores scalar cta-dft8 cufftdx-block turbofft-generated \
    --target-points "$TARGET_POINTS" \
    --warmup "$WARMUP" \
    --repeat "$REPEAT" \
    --trials "$TRIALS" \
    --output "$RAW"

python3 "$ROOT/scripts/summarize_fft_processing_units.py" "$RAW" --output "$SUMMARY"
printf 'Raw trials: %s\nSummary: %s\n' "$RAW" "$SUMMARY"
