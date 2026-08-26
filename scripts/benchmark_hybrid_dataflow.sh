#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-11.8/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/hybrid_dataflow"}
MODE=${MODE:-quick}
REPEAT=${REPEAT:-10}
WARMUP=${WARMUP:-3}
TIMEOUT=${TIMEOUT:-120}

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR/raw"
case "$MODE" in
    quick)
        log_ns=(6 8 10 12)
        batches=(1 16)
        ;;
    full)
        log_ns=(6 8 10 12 max)
        batches=(1 16 256)
        ;;
    *)
        echo "MODE must be quick or full" >&2
        exit 1
        ;;
esac
case_index=0
RAW_DIR="$OUTPUT_DIR/raw/$MODE"
mkdir -p "$RAW_DIR"

run() {
    local label=$1
    shift
    case_index=$((case_index + 1))
    printf '[%03d] %s\n' "$case_index" "$label"
    timeout --foreground "$TIMEOUT" "$BIN" "$@" --warmup "$WARMUP" --repeat "$REPEAT" \
        --verify --csv > "$RAW_DIR/${label}.csv"
}

for word_bits in 32 64; do
    if [[ $word_bits -eq 32 ]]; then
        modulus=1073479681
        data_space=32
        max_log_n=14
    else
        modulus=1152921504606584833
        data_space=16
        max_log_n=13
    fi
    for requested_log_n in "${log_ns[@]}"; do
        log_n=$requested_log_n
        [[ $requested_log_n == max ]] && log_n=$max_log_n
        for batch in "${batches[@]}"; do
            common=(--logN "$log_n" --batch "$batch" --word-bits "$word_bits" --modulus "$modulus")
            if [[ $word_bits -eq 64 ]]; then
                run "tile256_w64_n${log_n}_b${batch}" "${common[@]}" --backend tile256
            elif [[ $log_n -ge 12 ]]; then
                run "hybrid2d_w32_n${log_n}_b${batch}" "${common[@]}" --backend hybrid2d --compute-unit radix2
            fi
            flow=8
            stages=8
            [[ $log_n -lt 8 ]] && { flow=6; stages=4; }
            run "dataflow_default_w${word_bits}_n${log_n}_b${batch}" "${common[@]}" \
                --backend hybrid-dataflow --data-space "$data_space" --pipeline-buffers 2
            for stage_space in 2 4; do
                run "dataflow_search_w${word_bits}_n${log_n}_b${batch}_us${stage_space}" "${common[@]}" \
                    --backend hybrid-dataflow --flow-tile-log "$flow" --stage-space "$stage_space" \
                    --data-space "$data_space" --pipeline-buffers 2
            done
            if [[ $log_n -eq 8 && $word_bits -eq 64 ]]; then
                run "stage_pipeline_w64_n8_b${batch}" "${common[@]}" --backend stage-pipeline --stage-space 8
            fi
        done
    done
done

python3 "$ROOT/scripts/summarize_hybrid_dataflow.py" "$RAW_DIR"/*.csv \
    --csv "$OUTPUT_DIR/summary_${MODE}.csv" --markdown "$OUTPUT_DIR/summary_${MODE}.md"
printf 'HybridDataflow benchmark summary: %s\n' "$OUTPUT_DIR/summary_${MODE}.md"
