#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-v07/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/hierarchical_streaming/comprehensive"}
WARMUP=${WARMUP:-10}
REPEAT=${REPEAT:-50}
TRIALS=${TRIALS:-3}

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
raw="$OUTPUT_DIR/raw.csv"
: > "$raw"

run() {
    local trial=$1
    local label=$2
    shift 2
    local output header record
    printf '%s\n' "$label" >&2
    output=$("$BIN" "$@" --warmup "$WARMUP" --repeat "$REPEAT" --csv)
    header=$(printf '%s\n' "$output" | sed -n '/^device,/{p;q;}')
    record=$(printf '%s\n' "$output" | tail -n 1)
    if [[ ! -s "$raw" ]]; then
        printf 'trial,label,%s\n' "$header" >> "$raw"
    fi
    printf '%s,%s,%s\n' "$trial" "$label" "$record" >> "$raw"
}

for word_bits in 32 64; do
    if [[ $word_bits -eq 32 ]]; then modulus=998244353; else modulus=576460756061519873; fi
    common=(--word-bits "$word_bits" --modulus "$modulus" --logN 20)

    # Correctness is checked once per generated physical kernel; performance
    # runs avoid the CPU reference cost at large batch.
    "$BIN" "${common[@]}" --batch 1 --backend hierarchical-dataflow \
        --stage-partition 7,7,6 --segment-cores dataflow-radix4 \
        --warmup 0 --repeat 1 --verify >/dev/null

    for trial in $(seq 1 "$TRIALS"); do
        for batch in 1 4 8 16 32; do
            shape=("${common[@]}" --batch "$batch")
            run "$trial" "barrier-w${word_bits}-b${batch}" "${shape[@]}" \
                --backend hierarchical-barrier
            run "$trial" "resident1010-w${word_bits}-b${batch}" "${shape[@]}" \
                --backend hierarchical-dataflow --stage-partition 10,10 \
                --segment-cores dataflow-radix4 --segment-cta-weights 9,11
            run "$trial" "generic776-w${word_bits}-b${batch}" "${shape[@]}" \
                --backend hierarchical-dataflow --stage-partition 7,7,6 \
                --segment-cores radix4 --segment-cta-weights 7,7,6
            run "$trial" "resident776-w${word_bits}-b${batch}" "${shape[@]}" \
                --backend hierarchical-dataflow --stage-partition 7,7,6 \
                --segment-cores dataflow-radix4
        done
    done
done

python3 "$ROOT/scripts/summarize_hierarchical_streaming_comparison.py" "$raw" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'Hierarchical streaming comparison: %s\n' "$OUTPUT_DIR/analysis.md"
