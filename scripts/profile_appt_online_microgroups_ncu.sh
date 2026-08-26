#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build-v07/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_appt_online_microgroups"}
METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block,launch__waves_per_multiprocessor}

mkdir -p "$OUTPUT_DIR"
for executable in "$NCU" "$BIN"; do
    [[ -x "$executable" ]] || { echo "missing executable: $executable" >&2; exit 1; }
done

profile() {
    local label=$1 batch=$2 spaces=$3 weights=$4
    "$NCU" --target-processes all --replay-mode kernel --cache-control none \
        --clock-control base --launch-count 1 --metrics "$METRICS" --page raw \
        --csv --force-overwrite --log-file "$OUTPUT_DIR/${label}_b${batch}.csv" \
        "$BIN" --word-bits 64 --modulus 576460756061519873 --logN 20 \
        --batch "$batch" --backend hierarchical-dataflow \
        --stage-partition 7,7,6 --segment-cores appt-online \
        --segment-units 8 --segment-data-space "$spaces" \
        --segment-data-time 8 --segment-role-stages 2 \
        --segment-token-interleave 2 --segment-cta-weights "$weights" \
        --boundary-storage ring --boundary-buffers 2 \
        --target-ctas-per-sm 2 --warmup 0 --repeat 1 --csv
}

for batch in 1 4; do
    weights=5,7,8
    [[ $batch == 1 ]] && weights=8,8,4
    profile fine "$batch" 16,16,16 "$weights"
    profile publish2 "$batch" 16,8,16 "$weights"
    profile output4 "$batch" 16,16,32 "$weights"
    profile publish2_output4 "$batch" 16,8,32 "$weights"
done

raw_reports=("$OUTPUT_DIR"/*_b1.csv "$OUTPUT_DIR"/*_b4.csv)
python3 "$ROOT/scripts/summarize_ncu.py" "${raw_reports[@]}" \
    --output "$OUTPUT_DIR/summary.csv"
printf 'APPT online microgroup NCU summary: %s\n' "$OUTPUT_DIR/summary.csv"
if [[ -n ${SUDO_USER:-} ]]; then
    chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$OUTPUT_DIR"
fi
