#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bench="${CUNTT_BENCH:-$root/build-11.8/cuntt_bench}"
out="${1:-$root/results/ncu_hybrid_dataflow_interleave}"
mkdir -p "$out"
NCU="${NCU:-ncu}"
section_folder="${NCU_SECTION_FOLDER:-/usr/lib/nsight-compute/sections}"

restore_owner() {
    if [[ ${EUID:-$(id -u)} -eq 0 && -n ${SUDO_USER:-} ]]; then
        chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$out"
    fi
}
trap restore_owner EXIT

metrics="sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__inst_executed.sum,smsp__inst_executed.avg.per_cycle_active,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,launch__registers_per_thread,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem"
common=(--backend hybrid-dataflow --word-bits 64 --batch 256 --data-space 16 \
        --pipeline-buffers 2 --warmup 0 --repeat 1)
reports=()

profile() {
    local name="$1"
    shift
    local report="$out/$name.csv"
    local section_args=()
    if [[ -d "$section_folder" ]]; then
        section_args=(--section-folder "$section_folder" --apply-rules no)
    fi
    echo "[NCU] profiling $name"
    if ! "$NCU" "${section_args[@]}" --target-processes all --kernel-name-base demangled \
        --replay-mode kernel --cache-control none --clock-control base --launch-count 1 \
        --metrics "$metrics" --page raw --csv --force-overwrite --log-file "$report" \
        "$bench" "$@" >/dev/null; then
        echo "[NCU] failed: $name; diagnostic from $report:" >&2
        tail -n 20 "$report" >&2 || true
        return 1
    fi
    if ! grep -q '^"ID","Process ID"' "$report" || ! grep -q '^"0",' "$report"; then
        echo "[NCU] invalid report: $report" >&2
        tail -n 20 "$report" >&2 || true
        return 1
    fi
    echo "[NCU] wrote $report"
    reports+=("$report")
}

profile n10_ti1 "${common[@]}" --logN 10 --flow-tile-log 5 --stage-space 5 \
    --role-stages 5 --data-time 4 --token-interleave 1
profile n10_td10_ti1 "${common[@]}" --logN 10 --flow-tile-log 5 --stage-space 5 \
    --role-stages 5 --data-time 10 --token-interleave 1
profile n10_ti2 "${common[@]}" --logN 10 --flow-tile-log 5 --stage-space 5 \
    --role-stages 5 --data-time 10 --token-interleave 2
profile n12_ti1 "${common[@]}" --logN 12 --flow-tile-log 6 --stage-space 6 \
    --role-stages 6 --data-time 8 --token-interleave 1
profile n12_td12_ti1 "${common[@]}" --logN 12 --flow-tile-log 6 --stage-space 6 \
    --role-stages 6 --data-time 12 --token-interleave 1
profile n12_ti2 "${common[@]}" --logN 12 --flow-tile-log 6 --stage-space 6 \
    --role-stages 6 --data-time 12 --token-interleave 2

python3 "$root/scripts/summarize_ncu.py" "${reports[@]}" --output "$out/summary.csv"
echo "NCU CSV files written to $out"
