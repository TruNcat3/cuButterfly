#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-11.8/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/hierarchical_streaming/resident_776"}
WARMUP=${WARMUP:-10}
REPEAT=${REPEAT:-50}

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
    for batch in 1 2 4 8; do
        common=(--word-bits "$word_bits" --modulus "$modulus" --logN 20 --batch "$batch")
        run "barrier-w${word_bits}-b${batch}" "${common[@]}" --backend hierarchical-barrier
        run "resident-1010-w${word_bits}-b${batch}" "${common[@]}" \
            --backend hierarchical-dataflow --stage-partition 10,10 \
            --segment-cores dataflow-radix4 --segment-cta-weights 9,11
        for target in 1 2 3 4 5; do
            for units in 8 16 32; do
                for weights in 7,7,6 8,8,4 6,8,6 8,6,6 6,6,8; do
                    run "resident-776-w${word_bits}-b${batch}-t${target}-u${units}-${weights//,/-}" \
                        "${common[@]}" --backend hierarchical-dataflow \
                        --stage-partition 7,7,6 --segment-cores dataflow-radix4 \
                        --segment-threads 256 --segment-units "$units" \
                        --segment-cta-weights "$weights" --target-ctas-per-sm "$target"
                done
            done
        done
    done
done

printf 'Resident 7+7+6 scan: %s\n' "$raw"
