#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/packed_stage6_lane32_schedule"}
TRIALS=${TRIALS:-2}
WARMUP=${WARMUP:-5}
REPEAT=${REPEAT:-30}
BATCH=${BATCH:-16}
PRODUCER_BLOCKS=${PRODUCER_BLOCKS:-"72 74 76 78 80 82 84 85 86 88 90"}
DATA_TIME_PAIRS=${DATA_TIME_PAIRS:-"1:1 1:2 2:1 2:2"}

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
raw="$OUTPUT_DIR/raw.csv"
: > "$raw"

common=(--logN 20 --batch "$BATCH" --word-bits 32 --modulus 998244353
        --backend hierarchical-dataflow --stage-partition 10,10
        --boundary-storage full-scratch --cross-twiddle fused
        --mod-multiply shoup)
lane32=(--segment-cores
        homogeneous-warp128-vector-radix4-packed-stage6-distributed-static-io
        --segment-threads 256 --segment-units 8 --segment-data-space 4
        --segment-coefficient-reuse-stages 6 --target-ctas-per-sm 2)

run() {
    local trial=$1 variant=$2 producer_dt=$3 consumer_dt=$4
    local producer_blocks=$5 consumer_blocks=$6
    shift 6
    local output header record
    output=$("$BIN" "$@" --warmup "$WARMUP" --repeat "$REPEAT" --csv)
    header=$(printf '%s\n' "$output" | sed -n '/^device,/{p;q;}')
    record=$(printf '%s\n' "$output" | tail -n 1)
    if [[ ! -s "$raw" ]]; then
        printf 'trial,variant,producer_dt,consumer_dt,producer_blocks,consumer_blocks,%s\n' \
            "$header" > "$raw"
    fi
    printf '%s,%s,%s,%s,%s,%s,%s\n' "$trial" "$variant" \
        "$producer_dt" "$consumer_dt" "$producer_blocks" "$consumer_blocks" \
        "$record" >> "$raw"
}

"$BIN" "${common[@]}" "${lane32[@]}" --segment-data-time 1,1 \
    --segment-cta-weights 85,75 --warmup 0 --repeat 1 --verify >/dev/null

for trial in $(seq 1 "$TRIALS"); do
    run "$trial" v06 1 1 91 69 "${common[@]}" \
        --segment-cores dataflow-radix4 --segment-data-time 1,1 \
        --segment-cta-weights 17,13 --target-ctas-per-sm 4
    run "$trial" d7 1 1 85 75 "${common[@]}" \
        --segment-cores homogeneous-warp128-vector-radix4-static-io \
        --segment-threads 256 --segment-units 8 --segment-data-space 4 \
        --segment-coefficient-reuse-stages 7 --segment-data-time 1,1 \
        --segment-cta-weights 85,75 --target-ctas-per-sm 2
    for data_time_pair in $DATA_TIME_PAIRS; do
        producer_dt=${data_time_pair%%:*}
        consumer_dt=${data_time_pair##*:}
        for producer in $PRODUCER_BLOCKS; do
            consumer=$((160 - producer))
            run "$trial" lane32 "$producer_dt" "$consumer_dt" \
                "$producer" "$consumer" "${common[@]}" "${lane32[@]}" \
                --segment-data-time "$producer_dt,$consumer_dt" \
                --segment-cta-weights "$producer,$consumer"
        done
    done
    run "$trial" v06 1 1 91 69 "${common[@]}" \
        --segment-cores dataflow-radix4 --segment-data-time 1,1 \
        --segment-cta-weights 17,13 --target-ctas-per-sm 4
done

python3 "$ROOT/scripts/summarize_packed_stage6_lane32_schedule.py" "$raw" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'Packed lane32 schedule search: %s\n' "$OUTPUT_DIR/analysis.md"
