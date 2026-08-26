#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/three_level_partitions"}
WARMUP=${WARMUP:-10}
REPEAT=${REPEAT:-50}
TRIALS=${TRIALS:-3}
BATCHES=${BATCHES:-"1 4 16"}
mkdir -p "$OUTPUT_DIR"
RAW="$OUTPUT_DIR/raw.csv"
rm -f "$RAW"

common=(--logN 20 --word-bits 32 --modulus 998244353
        --backend hierarchical-dataflow --boundary-storage full-scratch
        --segment-cores dataflow-radix4 --segment-threads 256
        --segment-units 32 --target-ctas-per-sm 2
        --cross-twiddle fused --mod-multiply shoup)

append() {
    local label=$1
    shift
    local output
    output=$("$BIN" "${common[@]}" "$@" --warmup "$WARMUP" \
             --repeat "$REPEAT" --verify --csv)
    if [[ ! -s "$RAW" ]]; then
        printf 'label,%s\n' "$(printf '%s\n' "$output" | sed -n '1p')" >"$RAW"
    fi
    printf '%s,%s\n' "$label" "$(printf '%s\n' "$output" | sed -n '2p')" >>"$RAW"
}

partitions=(6,7,7 7,6,7 7,7,6 6,6,8 6,8,6 8,6,6)
for batch in $BATCHES; do
    for ((trial = 1; trial <= TRIALS; ++trial)); do
        offset=$((trial % ${#partitions[@]}))
        for ((index = 0; index < ${#partitions[@]}; ++index)); do
            partition=${partitions[$(((index + offset) % ${#partitions[@]}))]}
            append "p${partition//,/-}_b${batch}_t${trial}" \
                --batch "$batch" --stage-partition "$partition" \
                --segment-cta-weights "$partition"
        done
    done
done

python3 "$ROOT/scripts/summarize_three_level_partitions.py" "$RAW" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'wrote %s\n' "$OUTPUT_DIR/analysis.md"
