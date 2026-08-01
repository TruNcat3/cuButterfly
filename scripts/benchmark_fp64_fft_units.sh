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

python3 "$SWEEP" --binary "$BIN" --operators fft --precisions fp64 --logNs 16 --batch 64 \
    --directions forward --normalizations none --placements out-of-place \
    --backends hierarchical online-reorder cufft --fft-cores scalar \
    --tile-thread-options 32 64 128 256 --hierarchical-local-stages 5 6 8 10 \
    --reorder-column-options 1 2 4 --compute-units radix2 radix4 radix8 \
    --complex-multiplies four-mul gauss3 --warmup 10 --repeat 20 --trials 3 \
    --output "$OUTPUT_DIR/fp64_scalar_logN16_coarse_raw.csv"
python3 "$SUMMARY" "$OUTPUT_DIR/fp64_scalar_logN16_coarse_raw.csv" \
    --output "$OUTPUT_DIR/fp64_scalar_logN16_coarse_summary.csv"

python3 "$SWEEP" --binary "$BIN" --operators fft --precisions fp64 --logNs 16 --batch 64 \
    --directions forward --normalizations none --placements out-of-place \
    --backends online-reorder --fft-cores cufftdx-block --hierarchical-local-stages 8 \
    --prefix-thread-options 128 256 512 --suffix-thread-options 128 256 512 \
    --prefix-ept-options 4 8 --suffix-ept-options 4 8 --cross-twiddles table \
    --warmup 100 --repeat 50 --trials 3 \
    --output "$OUTPUT_DIR/fp64_cufftdx_logN16_mapping_raw.csv"
python3 "$SUMMARY" "$OUTPUT_DIR/fp64_cufftdx_logN16_mapping_raw.csv" \
    --output "$OUTPUT_DIR/fp64_cufftdx_logN16_mapping_summary.csv"

python3 "$SWEEP" --binary "$BIN" --operators fft --precisions fp64 --logNs 16 --batch 64 \
    --directions forward --normalizations none --placements out-of-place \
    --backends online-reorder --fft-cores cufftdx-block --hierarchical-local-stages 8 \
    --prefix-thread-options 128 256 512 --suffix-thread-options 128 256 512 \
    --prefix-ept-options 4 8 --suffix-ept-options 4 8 --cross-twiddles recurrence \
    --warmup 100 --repeat 50 --trials 3 \
    --output "$OUTPUT_DIR/fp64_cufftdx_recurrence_mapping_raw.csv"
python3 "$SUMMARY" "$OUTPUT_DIR/fp64_cufftdx_recurrence_mapping_raw.csv" \
    --output "$OUTPUT_DIR/fp64_cufftdx_recurrence_mapping_summary.csv"

COMMON=(--binary "$BIN" --operators fft --precisions fp64 --logNs 16 --batch 64
        --directions forward --normalizations none --placements out-of-place
        --warmup 1000 --repeat 100 --trials 5)
python3 "$SWEEP" "${COMMON[@]}" --backends online-reorder --fft-cores scalar \
    --tile-thread-options 128 --hierarchical-local-stages 8 --reorder-column-options 1 \
    --compute-units radix4 --complex-multiplies four-mul \
    --output "$OUTPUT_DIR/fp64_scalar_logN16_confirm_raw.csv"
python3 "$SWEEP" "${COMMON[@]}" --backends online-reorder --fft-cores cufftdx-block \
    --hierarchical-local-stages 8 --prefix-thread-options 256 --suffix-thread-options 128 \
    --prefix-ept-options 4 --suffix-ept-options 8 --cross-twiddles table \
    --output "$OUTPUT_DIR/fp64_cufftdx_logN16_confirm_raw.csv"
python3 "$SWEEP" "${COMMON[@]}" --backends online-reorder --fft-cores cufftdx-block \
    --hierarchical-local-stages 8 --prefix-thread-options 256 --suffix-thread-options 128 \
    --prefix-ept-options 4 --suffix-ept-options 8 --cross-twiddles recurrence \
    --output "$OUTPUT_DIR/fp64_cufftdx_recurrence_logN16_confirm_raw.csv"
python3 "$SWEEP" "${COMMON[@]}" --backends cufft --fft-cores scalar \
    --output "$OUTPUT_DIR/fp64_cufft_logN16_confirm_raw.csv"
python3 "$SUMMARY" "$OUTPUT_DIR/fp64_scalar_logN16_confirm_raw.csv" \
    "$OUTPUT_DIR/fp64_cufftdx_logN16_confirm_raw.csv" \
    "$OUTPUT_DIR/fp64_cufftdx_recurrence_logN16_confirm_raw.csv" \
    "$OUTPUT_DIR/fp64_cufft_logN16_confirm_raw.csv" \
    --output "$OUTPUT_DIR/fp64_logN16_twiddle_summary.csv"

printf 'FP64 confirmation summary: %s\n' "$OUTPUT_DIR/fp64_logN16_twiddle_summary.csv"
