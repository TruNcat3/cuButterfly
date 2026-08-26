#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build-11.8/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_hybrid_dataflow_roles"}
METRICS=${METRICS:-gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block,launch__waves_per_multiprocessor,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps}

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
    shift
    local report="$OUTPUT_DIR/${label}.csv"
    "$NCU" --target-processes all --replay-mode kernel --cache-control none --clock-control base \
        --launch-count 1 --metrics "$METRICS" --page raw --csv --force-overwrite \
        --log-file "$report" "$BIN" "$@" --warmup 0 --repeat 1 --csv
    reports+=("$report")
}

common=(--backend hybrid-dataflow --word-bits 64 --modulus 1152921504606584833 \
        --data-space 16 --pipeline-buffers 2 --batch 16)
for role_stages in 1 2 5; do
    profile "n10_ur${role_stages}_td4" "${common[@]}" --logN 10 --flow-tile-log 5 \
        --stage-space 5 --role-stages "$role_stages" --data-time 4
done
for role_stages in 1 2 3 6; do
    profile "n12_ur${role_stages}_td4" "${common[@]}" --logN 12 --flow-tile-log 6 \
        --stage-space 6 --role-stages "$role_stages" --data-time 4
    profile "n12_ur${role_stages}_td8" "${common[@]}" --logN 12 --flow-tile-log 6 \
        --stage-space 6 --role-stages "$role_stages" --data-time 8
done
profile "n12_ur6_td8_rb3" "${common[@]}" --logN 12 --flow-tile-log 6 \
    --stage-space 6 --role-stages 6 --target-ctas-per-sm 3 --data-time 8

python3 "$ROOT/scripts/summarize_ncu.py" "${reports[@]}" --output "$OUTPUT_DIR/summary.csv"
printf 'HybridDataflow role-fusion NCU summary: %s\n' "$OUTPUT_DIR/summary.csv"
