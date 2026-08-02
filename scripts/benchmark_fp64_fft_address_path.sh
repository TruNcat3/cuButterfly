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

COMMON=(--binary "$BIN" --operators fft --precisions fp64 --logNs 16 --batch 64
        --directions forward --normalizations none --placements out-of-place
        --warmup 1000 --repeat 100 --trials 5)
MAPPING=(--hierarchical-local-stages 8
         --prefix-thread-options 256 --suffix-thread-options 128
         --prefix-ept-options 4 --suffix-ept-options 8)

python3 "$SWEEP" "${COMMON[@]}" --backends online-reorder --fft-cores scalar \
    --tile-thread-options 128 --hierarchical-local-stages 8 --reorder-column-options 1 \
    --compute-units radix4 --complex-multiplies four-mul \
    --output "$OUTPUT_DIR/fp64_address_scalar_logN16_raw.csv"
python3 "$SWEEP" "${COMMON[@]}" --backends online-reorder --fft-cores cufftdx-block \
    "${MAPPING[@]}" --cross-twiddles table --shared-layouts linear \
    --output "$OUTPUT_DIR/fp64_address_table_logN16_raw.csv"
python3 "$SWEEP" "${COMMON[@]}" --backends online-reorder --fft-cores cufftdx-block \
    "${MAPPING[@]}" --cross-twiddles recurrence --shared-layouts linear xor-swizzle \
    --output "$OUTPUT_DIR/fp64_address_recurrence_logN16_raw.csv"
python3 "$SWEEP" "${COMMON[@]}" --backends cufft --fft-cores scalar \
    --output "$OUTPUT_DIR/fp64_address_cufft_logN16_raw.csv"

python3 "$SUMMARY" \
    "$OUTPUT_DIR/fp64_address_scalar_logN16_raw.csv" \
    "$OUTPUT_DIR/fp64_address_table_logN16_raw.csv" \
    "$OUTPUT_DIR/fp64_address_recurrence_logN16_raw.csv" \
    "$OUTPUT_DIR/fp64_address_cufft_logN16_raw.csv" \
    --output "$OUTPUT_DIR/fp64_address_logN16_summary.csv"

printf 'FP64 address-path summary: %s\n' "$OUTPUT_DIR/fp64_address_logN16_summary.csv"
