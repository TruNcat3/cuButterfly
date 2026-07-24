#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build/cubutterfly_bench"}
OUTPUT=${OUTPUT:-"$ROOT/results/fft_cubutterfly_selected_v100_raw.csv"}
WARMUP=${WARMUP:-20}
REPEAT=${REPEAT:-100}
TRIALS=${TRIALS:-5}

if [[ ! -x "$BIN" ]]; then
    printf 'Missing %s. Build cubutterfly_bench first.\n' "$BIN" >&2
    exit 1
fi

run_point() {
    local log_n=$1
    local batch=$2
    shift 2
    "$BIN" --operator fft --precision fp32 --logN "$log_n" --batch "$batch" \
        --placement in-place --normalization none --warmup "$WARMUP" --repeat "$REPEAT" \
        --verify --csv "$@"
}

mkdir -p "$(dirname "$OUTPUT")"
rm -f "$OUTPUT"
for trial in $(seq 1 "$TRIALS"); do
    while IFS=' ' read -r log_n batch arguments; do
        read -r -a extra <<<"$arguments"
        csv=$(run_point "$log_n" "$batch" "${extra[@]}")
        if [[ ! -s "$OUTPUT" ]]; then
            printf 'trial,%s\n' "$(printf '%s\n' "$csv" | head -n 1)" >"$OUTPUT"
        fi
        printf '%s,%s\n' "$trial" "$(printf '%s\n' "$csv" | tail -n 1)" >>"$OUTPUT"
    done <<'POINTS'
3 524288 --backend temporal-tile --compute-unit radix8 --fft-core cta-dft8 --tile-threads 32
8 16384 --backend temporal-tile --compute-unit radix4 --tile-threads 128
12 1024 --backend online-reorder --compute-unit radix4 --tile-threads 256 --local-stages 10 --reorder-columns 64
16 64 --backend online-reorder --compute-unit radix4 --tile-threads 256 --local-stages 10 --reorder-columns 8
20 4 --backend online-reorder --compute-unit radix4 --tile-threads 256 --local-stages 10 --reorder-columns 1
POINTS
done

printf 'Wrote %s\n' "$OUTPUT"
