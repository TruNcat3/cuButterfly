#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/homogeneous_10x10"}
TRIALS=${TRIALS:-2}
WARMUP=${WARMUP:-2}
REPEAT=${REPEAT:-10}
BATCHES=${BATCHES:-1,4,16}
WORD_BITS=${WORD_BITS:-32,64}
DATA_TIME_PAIRS=${DATA_TIME_PAIRS:-"1:1 1:2 1:4 2:1 2:2 4:1"}
CTA_WEIGHT_PAIRS=${CTA_WEIGHT_PAIRS:-"8:12 9:11 10:10"}

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
raw="$OUTPUT_DIR/raw.csv"
: > "$raw"

run() {
    local trial=$1 variant=$2 producer_dt=$3 consumer_dt=$4 weights=$5 resident=$6
    shift 6
    local output header record
    output=$("$BIN" "$@" --warmup "$WARMUP" --repeat "$REPEAT" --csv)
    header=$(printf '%s\n' "$output" | sed -n '/^device,/{p;q;}')
    record=$(printf '%s\n' "$output" | tail -n 1)
    if [[ ! -s "$raw" ]]; then
        printf 'trial,variant,producer_dt,consumer_dt,weights,resident_ctas,%s\n' \
            "$header" > "$raw"
    fi
    printf '%s,%s,%s,%s,%s,%s,%s\n' "$trial" "$variant" \
        "$producer_dt" "$consumer_dt" "${weights//:/-}" "$resident" \
        "$record" >> "$raw"
}

IFS=, read -ra selected_batches <<< "$BATCHES"
IFS=, read -ra selected_bits <<< "$WORD_BITS"
for bits in "${selected_bits[@]}"; do
    if [[ $bits == 32 ]]; then
        modulus=998244353
        resident=${RESIDENT_U32:-4}
    elif [[ $bits == 64 ]]; then
        modulus=576460756061519873
        resident=${RESIDENT_U64:-2}
    else
        echo "unsupported word bits: $bits" >&2
        exit 1
    fi
    for batch in "${selected_batches[@]}"; do
        common=(--logN 20 --batch "$batch" --word-bits "$bits"
                --modulus "$modulus" --backend hierarchical-dataflow
                --stage-partition 10,10 --boundary-storage full-scratch
                --target-ctas-per-sm "$resident")
        for trial in $(seq 1 "$TRIALS"); do
            run "$trial" v06 1 1 9:11 "$resident" "${common[@]}" \
                --segment-cores dataflow-radix4 \
                --segment-data-time 1,1 --segment-cta-weights 9,11
            for data_time_pair in $DATA_TIME_PAIRS; do
                producer_dt=${data_time_pair%%:*}
                consumer_dt=${data_time_pair##*:}
                for weight_pair in $CTA_WEIGHT_PAIRS; do
                    producer_weight=${weight_pair%%:*}
                    consumer_weight=${weight_pair##*:}
                    run "$trial" homogeneous "$producer_dt" "$consumer_dt" \
                        "$weight_pair" "$resident" "${common[@]}" \
                        --segment-cores homogeneous-radix4 \
                        --segment-threads 256 --segment-units 4 \
                        --segment-data-time "$producer_dt,$consumer_dt" \
                        --segment-cta-weights \
                            "$producer_weight,$consumer_weight"
                done
            done
            # Bracket the search with the same physical control so process
            # startup, clock ramp, and thermal drift do not bias one side.
            run "$trial" v06 1 1 9:11 "$resident" "${common[@]}" \
                --segment-cores dataflow-radix4 \
                --segment-data-time 1,1 --segment-cta-weights 9,11
        done
    done
done

python3 "$ROOT/scripts/summarize_homogeneous_10x10.py" "$raw" \
    --csv "$OUTPUT_DIR/summary.csv" \
    --markdown "$OUTPUT_DIR/analysis.md"
printf 'Homogeneous 10+10 comparison: %s\n' "$OUTPUT_DIR/analysis.md"
