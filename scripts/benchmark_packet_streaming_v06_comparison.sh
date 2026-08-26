#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/packet_streaming_v06_comparison"}
RAW=${RAW:-"$OUTPUT_DIR/raw.csv"}
BATCHES=${BATCHES:-"16 32 64"}
TRIALS=${TRIALS:-3}
WARMUP=${WARMUP:-10}
REPEAT=${REPEAT:-50}

mkdir -p "$OUTPUT_DIR"
[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }

common=(--logN 20 --word-bits 32 --modulus 998244353
        --backend hierarchical-dataflow --stage-partition 10,10
        --boundary-storage full-scratch --segment-data-time 1,1
        --cross-twiddle fused --mod-multiply shoup)
packet=(--segment-cores homogeneous-warp128-packet-shared-radix4-static-io
        --segment-threads 128 --segment-units 4 --segment-data-space 4
        --target-ctas-per-sm 4)
v06=(--segment-cores dataflow-radix4 --segment-cta-weights 17,13
     --target-ctas-per-sm 4)

header_written=0
run() {
    local label=$1
    shift
    local temporary header row
    temporary=$(mktemp)
    "$BIN" "$@" --warmup "$WARMUP" --repeat "$REPEAT" --csv >"$temporary"
    header=$(grep '^device,' "$temporary" | tail -n 1)
    row=$(grep '^"' "$temporary" | tail -n 1)
    if [[ -z "$header" || -z "$row" ]]; then
        cat "$temporary" >&2
        rm -f "$temporary"
        echo "benchmark did not produce CSV for $label" >&2
        return 1
    fi
    if (( header_written == 0 )); then
        printf 'label,%s\n' "$header" >"$RAW"
        header_written=1
    fi
    printf '%s,%s\n' "$label" "$row" >>"$RAW"
    rm -f "$temporary"
}

packet_weights() {
    case "$1" in
        16) printf '7,13' ;;
        32|64) printf '5,15' ;;
        *) echo "unsupported batch: $1" >&2; return 1 ;;
    esac
}

run_variant() {
    local variant=$1 batch=$2 trial=$3 weights=$4
    case "$variant" in
        v06)
            run "v06_b${batch}_t${trial}" "${common[@]}" --batch "$batch" \
                "${v06[@]}"
            ;;
        aggregate)
            run "aggregate_b${batch}_t${trial}" "${common[@]}" --batch "$batch" \
                "${packet[@]}" --segment-cta-weights "$weights" \
                --segment-token-interleave 0,0
            ;;
        online)
            run "online_b${batch}_t${trial}" "${common[@]}" --batch "$batch" \
                "${packet[@]}" --segment-cta-weights "$weights" \
                --segment-token-interleave 0,16
            ;;
    esac
}

echo '[preflight] forward and inverse correctness'
"$BIN" "${common[@]}" --batch 1 "${v06[@]}" \
    --warmup 0 --repeat 1 --verify >/dev/null
"$BIN" "${common[@]}" --batch 1 "${packet[@]}" \
    --segment-cta-weights 10,10 --segment-token-interleave 0,16 \
    --warmup 0 --repeat 1 --verify >/dev/null
"$BIN" "${common[@]}" --batch 1 "${packet[@]}" \
    --segment-cta-weights 10,10 --segment-token-interleave 0,16 --inverse \
    --warmup 0 --repeat 1 --verify >/dev/null

rm -f "$RAW"
orders=("v06 aggregate online" "online v06 aggregate" "aggregate online v06")
for batch in $BATCHES; do
    weights=$(packet_weights "$batch")
    for ((trial = 1; trial <= TRIALS; ++trial)); do
        order=${orders[$(((trial - 1) % ${#orders[@]}))]}
        echo "[batch=$batch trial=$trial] $order"
        for variant in $order; do
            run_variant "$variant" "$batch" "$trial" "$weights"
        done
    done
done

python3 "$ROOT/scripts/summarize_packet_streaming_v06_comparison.py" "$RAW" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'wrote %s, %s, and %s\n' "$RAW" "$OUTPUT_DIR/summary.csv" \
    "$OUTPUT_DIR/analysis.md"
