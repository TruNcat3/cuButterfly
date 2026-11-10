#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-11.8/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/hybrid_dataflow_roles"}
REPEAT=${REPEAT:-50}
WARMUP=${WARMUP:-10}
TIMEOUT=${TIMEOUT:-120}

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR/raw"
reports=()

run() {
    local label=$1
    shift
    local report="$OUTPUT_DIR/raw/${label}.csv"
    printf '%s\n' "$label"
    timeout --foreground "$TIMEOUT" "$BIN" "$@" --warmup "$WARMUP" --repeat "$REPEAT" \
        --verify --csv > "$report"
    reports+=("$report")
}

for word_bits in 32 64; do
    if [[ $word_bits -eq 32 ]]; then
        modulus=1073479681
        data_space=32
    else
        modulus=1152921504606584833
        data_space=16
    fi
    for batch in 1 16 256; do
        common=(--batch "$batch" --word-bits "$word_bits" --modulus "$modulus")
        if [[ $word_bits -eq 64 ]]; then
            run "tile256_w64_n10_b${batch}" "${common[@]}" --logN 10 --backend tile256
            run "tile256_w64_n12_b${batch}" "${common[@]}" --logN 12 --backend tile256
        else
            run "hybrid2d_w32_n12_b${batch}" "${common[@]}" --logN 12 --backend hybrid2d --compute-unit radix2
        fi

        for role_stages in 1 2 5; do
            run "dataflow_w${word_bits}_n10_b${batch}_ur${role_stages}_td4" "${common[@]}" \
                --logN 10 --backend hybrid-dataflow --flow-tile-log 5 --stage-space 5 \
                --role-stages "$role_stages" --data-space "$data_space" --data-time 4 --pipeline-buffers 2
        done
        for role_stages in 1 2 3 6; do
            run "dataflow_w${word_bits}_n12_b${batch}_ur${role_stages}_td4" "${common[@]}" \
                --logN 12 --backend hybrid-dataflow --flow-tile-log 6 --stage-space 6 \
                --role-stages "$role_stages" --data-space "$data_space" --data-time 4 --pipeline-buffers 2
            run "dataflow_w${word_bits}_n12_b${batch}_ur${role_stages}_td8" "${common[@]}" \
                --logN 12 --backend hybrid-dataflow --flow-tile-log 6 --stage-space 6 \
                --role-stages "$role_stages" --data-space "$data_space" --data-time 8 --pipeline-buffers 2
        done
        if [[ $word_bits -eq 64 ]]; then
            run "dataflow_w64_n12_b${batch}_ur6_td8_rb3" "${common[@]}" \
                --logN 12 --backend hybrid-dataflow --flow-tile-log 6 --stage-space 6 \
                --role-stages 6 --target-ctas-per-sm 3 --data-space 16 --data-time 8 --pipeline-buffers 2
        fi
    done
done

python3 "$ROOT/scripts/summarize_hybrid_dataflow.py" "${reports[@]}" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/summary.md"
printf 'HybridDataflow role-fusion summary: %s\n' "$OUTPUT_DIR/summary.md"
