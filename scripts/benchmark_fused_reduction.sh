#!/usr/bin/env bash
set -euo pipefail

CUNTT_BIN=${CUNTT_BIN:-./build/cuntt_bench}
OUTPUT=${OUTPUT:-results/fused_reduction_ab.csv}
MODULUS=${MODULUS:-576460756061519873}
WARMUP=${WARMUP:-1000}
REPEAT=${REPEAT:-200}
TRIALS=${TRIALS:-3}

if [[ ! -x "$CUNTT_BIN" ]]; then
    echo "cuNTT benchmark not found: $CUNTT_BIN" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
header_written=0

run_case() {
    local log_n=$1
    local batch=$2
    local n1_log=$3
    local rows=$4
    local threads=$5
    local multiply=$6
    local trial=$7
    local csv

    csv=$("$CUNTT_BIN" \
        --logN "$log_n" \
        --batch "$batch" \
        --backend hybrid2d \
        --n1-log "$n1_log" \
        --rows-per-block "$rows" \
        --threads-per-block "$threads" \
        --compute-unit radix4 \
        --cross-twiddle fused \
        --mod-multiply "$multiply" \
        --word-bits 64 \
        --modulus "$MODULUS" \
        --warmup "$WARMUP" \
        --repeat "$REPEAT" \
        --csv)

    if (( header_written == 0 )); then
        printf 'trial,%s\n' "$(printf '%s\n' "$csv" | tail -n 2 | head -n 1)" >"$OUTPUT"
        header_written=1
    fi
    printf '%s,%s\n' "$trial" "$(printf '%s\n' "$csv" | tail -n 1)" >>"$OUTPUT"
}

for multiply in shoup barrett; do
    for trial in $(seq 1 "$TRIALS"); do
        run_case 16 64 8 4 256 "$multiply" "$trial"
        run_case 18 16 9 4 512 "$multiply" "$trial"
        run_case 20 4 10 2 512 "$multiply" "$trial"
    done
done

echo "Raw A/B results written to $OUTPUT"
