#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-11.8/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/hierarchical_streaming"}
WARMUP=${WARMUP:-20}
REPEAT=${REPEAT:-100}

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
raw="$OUTPUT_DIR/raw.csv"
: > "$raw"

run() {
    local label=$1
    shift
    printf '%s\n' "$label" >&2
    "$BIN" "$@" --warmup "$WARMUP" --repeat "$REPEAT" --verify --csv >> "$raw"
}

for word_bits in 32 64; do
    if [[ $word_bits -eq 32 ]]; then modulus=998244353; else modulus=576460756061519873; fi
    for log_n in 12 15 17 20; do
        for batch in 1 2 4 8; do
            common=(--word-bits "$word_bits" --modulus "$modulus" --logN "$log_n" --batch "$batch")
            run "barrier-w${word_bits}-n${log_n}-b${batch}" "${common[@]}" --backend hierarchical-barrier
            run "stream-auto-w${word_bits}-n${log_n}-b${batch}" "${common[@]}" --backend hierarchical-dataflow
            if [[ $log_n -eq 15 ]]; then
                run "stream-555-w${word_bits}-b${batch}" "${common[@]}" --backend hierarchical-dataflow \
                    --stage-partition 5,5,5 --segment-cta-weights 5,5,5
            elif [[ $log_n -eq 20 ]]; then
                run "stream-776-w${word_bits}-b${batch}" "${common[@]}" --backend hierarchical-dataflow \
                    --stage-partition 7,7,6 --segment-cta-weights 7,7,6
                run "stream-1010-w${word_bits}-b${batch}" "${common[@]}" --backend hierarchical-dataflow \
                    --stage-partition 10,10 --segment-cores dataflow-radix4 \
                    --segment-threads 256 --segment-cta-weights 9,11
            fi
        done
    done
done

printf 'Raw streaming comparison: %s\n' "$raw"
