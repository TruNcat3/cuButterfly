#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/packet_polling_backoff"}
RAW=${RAW:-"$OUTPUT_DIR/raw.csv"}
BATCHES=${BATCHES:-"16 32 64"}
READY_WINDOWS=${READY_WINDOWS:-"1 2 4 8 16"}
TRIALS=${TRIALS:-3}
WARMUP=${WARMUP:-10}
REPEAT=${REPEAT:-50}

mkdir -p "$OUTPUT_DIR"
[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }

common=(--logN 20 --word-bits 32 --modulus 998244353
        --backend hierarchical-dataflow --stage-partition 10,10
        --boundary-storage full-scratch --segment-data-time 1,1
        --segment-cores homogeneous-warp128-packet-shared-radix4-static-io
        --segment-threads 128 --segment-units 4 --segment-data-space 4
        --target-ctas-per-sm 4 --cross-twiddle fused --mod-multiply shoup)

weights_for_batch() {
    case "$1" in
        16) printf '7,13' ;;
        32|64) printf '5,15' ;;
        *) echo "unsupported batch: $1" >&2; return 1 ;;
    esac
}

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

echo '[preflight] polling-window correctness'
for ready_window in $READY_WINDOWS; do
    "$BIN" "${common[@]}" --batch 1 --segment-cta-weights 10,10 \
        --segment-token-interleave 0,16 --ready-window "$ready_window" \
        --warmup 0 --repeat 1 --verify >/dev/null
done

rm -f "$RAW"
for batch in $BATCHES; do
    weights=$(weights_for_batch "$batch")
    for ((trial = 1; trial <= TRIALS; ++trial)); do
        echo "[batch=$batch trial=$trial] aggregate and polling windows"
        run "aggregate_b${batch}_t${trial}" "${common[@]}" --batch "$batch" \
            --segment-cta-weights "$weights" --segment-token-interleave 0,0 \
            --ready-window 2
        for ready_window in $READY_WINDOWS; do
            run "online_b${batch}_rw${ready_window}_t${trial}" \
                "${common[@]}" --batch "$batch" \
                --segment-cta-weights "$weights" \
                --segment-token-interleave 0,16 \
                --ready-window "$ready_window"
        done
    done
done

python3 "$ROOT/scripts/summarize_packet_polling_backoff.py" "$RAW" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'wrote %s, %s, and %s\n' "$RAW" "$OUTPUT_DIR/summary.csv" \
    "$OUTPUT_DIR/analysis.md"
