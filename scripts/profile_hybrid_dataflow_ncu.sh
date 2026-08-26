#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build-11.8/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_hybrid_dataflow_td"}
METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sector_hit_rate.pct,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block,launch__waves_per_multiprocessor,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps}

for executable in "$NCU" "$BIN"; do
    [[ -x "$executable" ]] || { echo "missing executable: $executable" >&2; exit 1; }
done
mkdir -p "$OUTPUT_DIR"
reports=()

restore_owner() {
    if [[ ${EUID:-$(id -u)} -eq 0 && -n ${SUDO_USER:-} ]]; then
        chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$OUTPUT_DIR"
    fi
}
trap restore_owner EXIT

profile() {
    local label=$1
    local report="$OUTPUT_DIR/${label}.csv"
    shift
    "$NCU" --target-processes all --replay-mode kernel --cache-control none --clock-control base \
        --launch-count 1 --metrics "$METRICS" --page raw --csv --force-overwrite \
        --log-file "$report" "$BIN" "$@" --warmup 0 --repeat 1 --csv
    reports+=("$report")
}

common=(--backend hybrid-dataflow --word-bits 64 --modulus 1152921504606584833 --data-space 16)
profile primary_n8_b256 "${common[@]}" --logN 8 --batch 256 --flow-tile-log 8 --stage-space 8 --pipeline-buffers 2
profile balanced_n10_td4_b16 "${common[@]}" --logN 10 --batch 16 --flow-tile-log 5 --stage-space 5 --data-time 4 --pipeline-buffers 2
profile balanced_n12_td1_b16 "${common[@]}" --logN 12 --batch 16 --flow-tile-log 6 --stage-space 6 --data-time 1 --pipeline-buffers 2
profile balanced_n12_td4_b16 "${common[@]}" --logN 12 --batch 16 --flow-tile-log 6 --stage-space 6 --data-time 4 --pipeline-buffers 2
profile unbalanced_n12_td1_b16 "${common[@]}" --logN 12 --batch 16 --flow-tile-log 8 --stage-space 8 --data-time 1 --pipeline-buffers 2
# Td=4 needs 114736 B at logN=13/Us=7; V100 permits 98304 B per CTA.
profile balanced_n13_td2_b1 "${common[@]}" --logN 13 --batch 1 --flow-tile-log 7 --stage-space 7 --data-time 2 --pipeline-buffers 2
profile us2_n12_td1_b16 "${common[@]}" --logN 12 --batch 16 --flow-tile-log 8 --stage-space 2 --data-time 1 --pipeline-buffers 2
profile linear_n12_b16 "${common[@]}" --logN 12 --batch 16 --flow-tile-log 8 --stage-space 8 --pipeline-buffers 2 --dataflow-layout linear
profile atomic_n12_b16 "${common[@]}" --logN 12 --batch 16 --flow-tile-log 8 --stage-space 8 --pipeline-buffers 2 --stage-handoff atomic

python3 "$ROOT/scripts/summarize_ncu.py" "${reports[@]}" --output "$OUTPUT_DIR/summary.csv"
printf 'HybridDataflow NCU summary: %s\n' "$OUTPUT_DIR/summary.csv"
