#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/packet_compute_layout"}
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

echo "[preflight] warp-row forward/inverse"
for direction in forward inverse; do
    args=(--batch 1 --segment-cta-weights 10,10
          --segment-token-interleave 0,16 --ready-window 2
          --packet-readiness wave-bitmap --packet-compute-layout warp-rows
          --warmup 0 --repeat 1 --verify)
    [[ "$direction" == inverse ]] && args+=(--inverse)
    "$BIN" "${common[@]}" "${args[@]}" >/dev/null
done

for batch in 16 32 64; do
    case "$batch" in
        16) weights=7,13; packet_window=4; bitmap_window=16 ;;
        32) weights=5,15; packet_window=2; bitmap_window=8 ;;
        64) weights=5,15; packet_window=16; bitmap_window=8 ;;
    esac
    for ((trial = 1; trial <= TRIALS; ++trial)); do
        echo "[batch=$batch trial=$trial] packet compute layouts"
        append "aggregate_b${batch}_t${trial}" --batch "$batch" \
            --segment-cta-weights "$weights" --segment-token-interleave 0,0
        for layout in interleaved-rows warp-rows; do
            label=${layout//-/_}
            append "per_packet_${label}_b${batch}_t${trial}" --batch "$batch" \
                --segment-cta-weights "$weights" --segment-token-interleave 0,16 \
                --ready-window "$packet_window" --packet-readiness per-packet \
                --packet-compute-layout "$layout"
            append "wave_bitmap_${label}_b${batch}_t${trial}" --batch "$batch" \
                --segment-cta-weights "$weights" --segment-token-interleave 0,16 \
                --ready-window "$bitmap_window" --packet-readiness wave-bitmap \
                --packet-compute-layout "$layout"
        done
    done
done

python3 "$ROOT/scripts/summarize_packet_compute_layout.py" "$RAW" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'wrote %s\n' "$OUTPUT_DIR/analysis.md"
