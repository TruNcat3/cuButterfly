#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-v07/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/appt_core_space"}
WORD_BITS_LIST=${WORD_BITS_LIST:-"32 64"}
BATCH_LIST=${BATCH_LIST:-"1 4"}
CORE_LIST=${CORE_LIST:-"appt-online appt-online-radix4"}
TRIALS=${TRIALS:-2}
WARMUP=${WARMUP:-2}
REPEAT=${REPEAT:-8}
WEIGHT_TOTAL=${WEIGHT_TOTAL:-20}
WEIGHT_MIN=${WEIGHT_MIN:-3}
WEIGHT_MAX=${WEIGHT_MAX:-10}
WEIGHT_LIST=${WEIGHT_LIST:-""}

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
raw="$OUTPUT_DIR/raw.csv"
: > "$raw"

weight_points=()
if [[ -n "$WEIGHT_LIST" ]]; then
    read -r -a weight_points <<< "$WEIGHT_LIST"
else
    for ((w0=WEIGHT_MIN; w0<=WEIGHT_MAX; ++w0)); do
        for ((w1=WEIGHT_MIN; w1<=WEIGHT_MAX; ++w1)); do
            w2=$((WEIGHT_TOTAL - w0 - w1))
            if ((w2 >= WEIGHT_MIN && w2 <= WEIGHT_MAX)); then
                weight_points+=("$w0,$w1,$w2")
            fi
        done
    done
fi

run() {
    local trial=$1 family=$2 core=$3 weights=$4
    shift 4
    local output header record
    output=$("$BIN" "$@" --warmup "$WARMUP" --repeat "$REPEAT" --csv)
    header=$(printf '%s\n' "$output" | sed -n '/^device,/{p;q;}')
    record=$(printf '%s\n' "$output" | tail -n 1)
    if [[ ! -s "$raw" ]]; then
        printf 'trial,family,physical_core,weights,%s\n' "$header" > "$raw"
    fi
    printf '%s,%s,%s,%s,%s\n' "$trial" "$family" "$core" "$weights" "$record" >> "$raw"
}

for bits in $WORD_BITS_LIST; do
    if [[ $bits == 32 ]]; then
        modulus=998244353
        ctas=3
    else
        modulus=576460756061519873
        ctas=2
    fi
    for batch in $BATCH_LIST; do
        common=(--word-bits "$bits" --modulus "$modulus" --logN 20 --batch "$batch")
        for trial in $(seq 1 "$TRIALS"); do
            run "$trial" v06 dataflow-radix4 9-11 "${common[@]}" \
                --backend hierarchical-dataflow --stage-partition 10,10 \
                --segment-cores dataflow-radix4 --segment-cta-weights 9,11
            for core in $CORE_LIST; do
                for weights in "${weight_points[@]}"; do
                    run "$trial" appt "$core" "${weights//,/-}" \
                        "${common[@]}" --backend hierarchical-dataflow \
                        --stage-partition 7,7,6 --segment-cores "$core" \
                        --segment-units 8 --segment-data-space 16,16,16 \
                        --segment-data-time 4 --segment-role-stages 2 \
                        --segment-token-interleave 1 \
                        --segment-cta-weights "$weights" \
                        --boundary-storage ring --boundary-buffers 2 \
                        --target-ctas-per-sm "$ctas"
                done
            done
        done
    done
done

python3 "$ROOT/scripts/summarize_appt_core_space.py" "$raw" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'APPT physical-core design space: %s\n' "$OUTPUT_DIR/analysis.md"
if [[ -n ${SUDO_USER:-} ]]; then
    chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$OUTPUT_DIR"
fi
