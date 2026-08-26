#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build-v07/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_appt_tail_cores"}
METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block,launch__waves_per_multiprocessor}

mkdir -p "$OUTPUT_DIR"
for executable in "$NCU" "$BIN"; do
    [[ -x "$executable" ]] || { echo "missing executable: $executable" >&2; exit 1; }
done

profile() {
    local label=$1
    shift
    "$NCU" --target-processes all --replay-mode kernel --cache-control none \
        --clock-control base --launch-count 1 --metrics "$METRICS" --page raw \
        --csv --force-overwrite --log-file "$OUTPUT_DIR/${label}.csv" \
        "$BIN" "$@" --warmup 0 --repeat 1 --csv
}

for bits in 32 64; do
    if [[ $bits == 32 ]]; then modulus=998244353; resident=3
    else modulus=576460756061519873; resident=2
    fi
    for batch in 1 4; do
        common=(--word-bits "$bits" --modulus "$modulus" --logN 20 --batch "$batch")
        if [[ $bits == 32 && $batch == 1 ]]; then warp=6,6,8; cta=7,6,7; full=9,6,5; split=7,7,6
        elif [[ $bits == 32 ]]; then warp=7,7,6; cta=7,6,7; full=4,8,8; split=8,6,6
        elif [[ $batch == 1 ]]; then warp=8,8,4; cta=4,8,8; full=9,6,5; split=4,6,10
        else warp=6,8,6; cta=5,7,8; full=7,7,6; split=4,8,8
        fi
        if [[ $bits == 32 && $batch == 1 ]]; then register=8,7,5
        elif [[ $bits == 32 ]]; then register=8,7,5
        elif [[ $batch == 1 ]]; then register=11,5,4
        else register=10,6,4
        fi

        profile "v06_u${bits}_b${batch}" "${common[@]}" \
            --backend hierarchical-dataflow --stage-partition 10,10 \
            --segment-cores dataflow-radix4 --segment-cta-weights 9,11
        for point in "warp:${warp}:appt-online:${resident}" \
                     "cta-radix4:${cta}:appt-online-radix4:${resident}" \
                     "fused-tail:${full}:appt-online-fused-tail:$([[ $bits == 32 ]] && printf 2 || printf 1)" \
                     "split-tail:${split}:appt-online-split-tail:${resident}" \
                     "register-tail:${register}:appt-online-register-tail:${resident}"; do
            IFS=: read -r label weights core target <<< "$point"
            profile "${label}_u${bits}_b${batch}" "${common[@]}" \
                --backend hierarchical-dataflow --stage-partition 7,7,6 \
                --segment-cores "$core" --segment-units 8 \
                --segment-data-space 16,16,16 --segment-data-time 4 \
                --segment-role-stages 2 --segment-token-interleave 1 \
                --segment-cta-weights "$weights" --boundary-storage ring \
                --boundary-buffers 2 --target-ctas-per-sm "$target"
        done
    done
done

shopt -s nullglob
reports=("$OUTPUT_DIR"/*.csv)
filtered=()
for report in "${reports[@]}"; do
    [[ $(basename "$report") == summary.csv ]] || filtered+=("$report")
done
(( ${#filtered[@]} > 0 )) || { echo "no NCU raw reports found" >&2; exit 1; }
python3 "$ROOT/scripts/summarize_ncu.py" "${filtered[@]}" \
    --output "$OUTPUT_DIR/summary.csv"
analysis_args=(
    "$OUTPUT_DIR/summary.csv"
    --csv "$OUTPUT_DIR/analysis.csv"
    --markdown "$OUTPUT_DIR/analysis.md"
)
if [[ -n ${ABLATION_BASELINE:-} ]]; then
    analysis_args+=(--ablation-baseline "$ABLATION_BASELINE")
fi
python3 "$ROOT/scripts/analyze_appt_tail_cores_ncu.py" "${analysis_args[@]}"
printf 'APPT tail-core NCU analysis: %s\n' "$OUTPUT_DIR/analysis.md"
if [[ -n ${SUDO_USER:-} ]]; then
    chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$OUTPUT_DIR"
fi
