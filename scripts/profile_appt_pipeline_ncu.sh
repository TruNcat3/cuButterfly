#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build-v07/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_appt_pipeline"}
METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block,launch__waves_per_multiprocessor}

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

common=(--word-bits 64 --modulus 576460756061519873 --logN 20)
for batch in 1 4; do
    profile "barrier_b${batch}" "${common[@]}" --batch "$batch" --backend hierarchical-barrier
    profile "resident1010_b${batch}" "${common[@]}" --batch "$batch" \
        --backend hierarchical-dataflow --stage-partition 10,10 \
        --segment-cores dataflow-radix4 --segment-cta-weights 9,11
    profile "appt_us8_td8_b${batch}" "${common[@]}" --batch "$batch" \
        --backend hierarchical-dataflow --stage-partition 8,8,4 \
        --segment-cores appt-pipeline --segment-data-space 16 --segment-data-time 8 \
        --segment-role-stages 1 --segment-token-interleave 1 --boundary-storage ring \
        --boundary-buffers 2 --target-ctas-per-sm 2
    profile "appt_us8_rs2_td8_b${batch}" "${common[@]}" --batch "$batch" \
        --backend hierarchical-dataflow --stage-partition 8,8,4 \
        --segment-cores appt-pipeline --segment-data-space 16 --segment-data-time 8 \
        --segment-role-stages 2 --segment-token-interleave 1 --boundary-storage ring \
        --boundary-buffers 2 --target-ctas-per-sm 2
    profile "appt_us7_rs2_rep2_ti2_b${batch}" "${common[@]}" --batch "$batch" \
        --backend hierarchical-dataflow --stage-partition 7,7,6 \
        --segment-cores appt-pipeline --segment-units 8 --segment-data-space 16 \
        --segment-data-time 8 --segment-role-stages 2 --segment-token-interleave 2 \
        --boundary-storage ring --boundary-buffers 2 --target-ctas-per-sm 2
    online_td=8
    online_ti=2
    online_weights=7,7,6
    if [[ $batch == 1 ]]; then
        online_td=4
        online_ti=1
        online_weights=8,8,4
    fi
    profile "appt_online_b${batch}" "${common[@]}" --batch "$batch" \
        --backend hierarchical-dataflow --stage-partition 7,7,6 \
        --segment-cores appt-online --segment-units 8 --segment-data-space 16 \
        --segment-data-time "$online_td" --segment-role-stages 2 \
        --segment-token-interleave "$online_ti" \
        --segment-cta-weights "$online_weights" --boundary-storage ring \
        --boundary-buffers 2 --target-ctas-per-sm 2
done

shopt -s nullglob
raw_reports=(
    "$OUTPUT_DIR"/barrier_b*.csv
    "$OUTPUT_DIR"/resident1010_b*.csv
    "$OUTPUT_DIR"/appt_*.csv
)
if (( ${#raw_reports[@]} == 0 )); then
    echo "no NCU raw reports found in $OUTPUT_DIR" >&2
    exit 1
fi
python3 "$ROOT/scripts/summarize_ncu.py" "${raw_reports[@]}" \
    --output "$OUTPUT_DIR/summary.csv"
python3 "$ROOT/scripts/analyze_appt_pipeline_ncu.py" "$OUTPUT_DIR/summary.csv" \
    --csv "$OUTPUT_DIR/analysis.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'APPT NCU summary: %s\n' "$OUTPUT_DIR/summary.csv"
printf 'APPT NCU analysis: %s\n' "$OUTPUT_DIR/analysis.md"
if [[ -n ${SUDO_USER:-} ]]; then
    chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$OUTPUT_DIR"
fi
