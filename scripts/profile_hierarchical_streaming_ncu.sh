#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build-11.8/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_hierarchical_streaming"}
METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block,launch__waves_per_multiprocessor}

mkdir -p "$OUTPUT_DIR"
for executable in "$NCU" "$BIN"; do
    [[ -x "$executable" ]] || { echo "missing executable: $executable" >&2; exit 1; }
done

profile() {
    local label=$1
    shift
    "$NCU" --target-processes all --replay-mode kernel --cache-control none --clock-control base \
        --launch-count 1 --metrics "$METRICS" --page raw --csv --force-overwrite \
        --log-file "$OUTPUT_DIR/${label}.csv" "$BIN" "$@" --warmup 0 --repeat 1 --csv
}

common=(--word-bits 64 --modulus 576460756061519873)
profile barrier_n15_b2 "${common[@]}" --logN 15 --batch 2 --backend hierarchical-barrier
profile streaming_n15_b2 "${common[@]}" --logN 15 --batch 2 --backend hierarchical-dataflow \
    --stage-partition 5,5,5 --segment-cores dataflow-radix4 --segment-cta-weights 5,5,5
profile barrier_n20_b4 "${common[@]}" --logN 20 --batch 4 --backend hierarchical-barrier
profile streaming_generic_n20_b4 "${common[@]}" --logN 20 --batch 4 --backend hierarchical-dataflow \
    --stage-partition 7,7,6 --segment-cores radix4 --segment-cta-weights 7,7,6
profile streaming_resident_776_n20_b1 "${common[@]}" --logN 20 --batch 1 \
    --backend hierarchical-dataflow --stage-partition 7,7,6 \
    --segment-cores dataflow-radix4 --segment-units 32 \
    --segment-cta-weights 6,6,8 \
    --target-ctas-per-sm 2
profile streaming_resident_776_n20_b4 "${common[@]}" --logN 20 --batch 4 \
    --backend hierarchical-dataflow --stage-partition 7,7,6 \
    --segment-cores dataflow-radix4 --segment-units 32 \
    --segment-cta-weights 6,6,8 \
    --target-ctas-per-sm 2
profile streaming_specialized_n20_b4 "${common[@]}" --logN 20 --batch 4 \
    --backend hierarchical-dataflow --stage-partition 10,10 \
    --segment-cores dataflow-radix4 --segment-cta-weights 9,11

common32=(--word-bits 32 --modulus 998244353)
profile barrier_w32_n20_b4 "${common32[@]}" --logN 20 --batch 4 \
    --backend hierarchical-barrier
profile streaming_specialized_w32_n20_b4 "${common32[@]}" --logN 20 --batch 4 \
    --backend hierarchical-dataflow --stage-partition 10,10 \
    --segment-cores dataflow-radix4 --segment-cta-weights 9,11
profile streaming_resident_776_w32_n20_b1 "${common32[@]}" --logN 20 --batch 1 \
    --backend hierarchical-dataflow --stage-partition 7,7,6 \
    --segment-cores dataflow-radix4 --segment-units 32 \
    --segment-cta-weights 6,6,8 \
    --target-ctas-per-sm 3

python3 "$ROOT/scripts/summarize_ncu.py" \
    "$OUTPUT_DIR/barrier_n15_b2.csv" \
    "$OUTPUT_DIR/streaming_n15_b2.csv" \
    "$OUTPUT_DIR/barrier_n20_b4.csv" \
    "$OUTPUT_DIR/streaming_generic_n20_b4.csv" \
    "$OUTPUT_DIR/streaming_resident_776_n20_b1.csv" \
    "$OUTPUT_DIR/streaming_resident_776_n20_b4.csv" \
    "$OUTPUT_DIR/streaming_specialized_n20_b4.csv" \
    "$OUTPUT_DIR/barrier_w32_n20_b4.csv" \
    "$OUTPUT_DIR/streaming_specialized_w32_n20_b4.csv" \
    "$OUTPUT_DIR/streaming_resident_776_w32_n20_b1.csv" \
    --output "$OUTPUT_DIR/summary.csv"
printf 'Streaming NCU summary: %s\n' "$OUTPUT_DIR/summary.csv"
