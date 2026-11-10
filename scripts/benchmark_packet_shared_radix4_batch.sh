#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/packet_shared_radix4_batch"}
RAW=${RAW:-"$OUTPUT_DIR/raw.csv"}
BATCHES=${BATCHES:-"1 2 4 8 16 32 64"}
PACKET_PRODUCERS=${PACKET_PRODUCERS:-"4 5 6 7 8 9 10"}
REPEAT=${REPEAT:-50}
WARMUP=${WARMUP:-5}

mkdir -p "$OUTPUT_DIR"
[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }

header_written=0
run() {
    local label=$1
    shift
    local temporary
    temporary=$(mktemp)
    "$BIN" "$@" --warmup "$WARMUP" --repeat "$REPEAT" --csv >"$temporary"
    local header row
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

common=(--logN 20 --word-bits 32 --modulus 998244353
        --backend hierarchical-dataflow --stage-partition 10,10
        --boundary-storage full-scratch --segment-data-time 1,1
        --cross-twiddle fused --mod-multiply shoup)

rm -f "$RAW"
for batch in $BATCHES; do
    echo "[batch=$batch] controls"
    run "v06_b${batch}" "${common[@]}" --batch "$batch" \
        --segment-cores dataflow-radix4 --segment-cta-weights 17,13 \
        --target-ctas-per-sm 4
    run "lane32_b${batch}" "${common[@]}" --batch "$batch" \
        --segment-cores homogeneous-warp128-vector-radix4-packed-stage6-distributed-static-io \
        --segment-threads 256 --segment-units 8 --segment-data-space 4 \
        --segment-coefficient-reuse-stages 6 --segment-cta-weights 84,76 \
        --target-ctas-per-sm 2
    echo "[batch=$batch] packet weights"
    for producer in $PACKET_PRODUCERS; do
        consumer=$((20 - producer))
        run "packet128_b${batch}_w${producer}_${consumer}" \
            "${common[@]}" --batch "$batch" \
            --segment-cores homogeneous-warp128-packet-shared-radix4-static-io \
            --segment-threads 128 --segment-units 4 --segment-data-space 4 \
            --segment-cta-weights "$producer,$consumer" --target-ctas-per-sm 4
    done
done

python3 "$ROOT/scripts/summarize_packet_shared_radix4_batch.py" "$RAW" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'wrote %s, %s, and %s\n' "$RAW" "$OUTPUT_DIR/summary.csv" \
    "$OUTPUT_DIR/analysis.md"
