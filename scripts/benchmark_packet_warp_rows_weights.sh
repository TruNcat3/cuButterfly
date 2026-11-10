#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/packet_warp_rows_weights"}
WARMUP=${WARMUP:-10}
REPEAT=${REPEAT:-50}
TRIALS=${TRIALS:-3}
mkdir -p "$OUTPUT_DIR"
RAW="$OUTPUT_DIR/raw.csv"
rm -f "$RAW"

common=(--logN 20 --word-bits 32 --modulus 998244353
        --backend hierarchical-dataflow --stage-partition 10,10
        --boundary-storage full-scratch --segment-data-time 1,1
        --segment-cores homogeneous-warp128-packet-shared-radix4-static-io
        --segment-threads 128 --segment-units 4 --segment-data-space 4
        --target-ctas-per-sm 4 --cross-twiddle fused --mod-multiply shoup)

append() {
    local label=$1
    shift
    local output
    output=$("$BIN" "${common[@]}" "$@" --warmup "$WARMUP" --repeat "$REPEAT" --csv)
    if [[ ! -s "$RAW" ]]; then
        printf 'label,%s\n' "$(printf '%s\n' "$output" | sed -n '1p')" >"$RAW"
    fi
    printf '%s,%s\n' "$label" "$(printf '%s\n' "$output" | sed -n '2p')" >>"$RAW"
}

echo "[preflight] warp-row selected protocol"
"$BIN" "${common[@]}" --batch 1 --segment-cta-weights 10,10 \
    --segment-token-interleave 0,16 --ready-window 2 \
    --packet-readiness wave-bitmap --packet-compute-layout warp-rows \
    --warmup 0 --repeat 1 --verify >/dev/null

for batch in 16 32 64; do
    case "$batch" in
        16) old_producer=7; ready_window=16 ;;
        32) old_producer=5; ready_window=8 ;;
        64) old_producer=5; ready_window=8 ;;
    esac
    for ((trial = 1; trial <= TRIALS; ++trial)); do
        echo "[batch=$batch trial=$trial] producer/consumer weights"
        append "aggregate_b${batch}_t${trial}" --batch "$batch" \
            --segment-cta-weights "${old_producer},$((20 - old_producer))" \
            --segment-token-interleave 0,0
        append "interleaved_b${batch}_p${old_producer}_t${trial}" \
            --batch "$batch" \
            --segment-cta-weights "${old_producer},$((20 - old_producer))" \
            --segment-token-interleave 0,16 --ready-window "$ready_window" \
            --packet-readiness wave-bitmap
        for producer in 4 5 6 7 8 9 10 11 12; do
            append "warp_rows_b${batch}_p${producer}_t${trial}" \
                --batch "$batch" \
                --segment-cta-weights "${producer},$((20 - producer))" \
                --segment-token-interleave 0,16 --ready-window "$ready_window" \
                --packet-readiness wave-bitmap --packet-compute-layout warp-rows
        done
    done
done

python3 "$ROOT/scripts/summarize_packet_warp_rows_weights.py" "$RAW" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'wrote %s\n' "$OUTPUT_DIR/analysis.md"
