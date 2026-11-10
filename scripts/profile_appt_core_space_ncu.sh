#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build-v07/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_appt_core_space"}
METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block,launch__waves_per_multiprocessor}

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
    if [[ $bits == 32 ]]; then
        modulus=998244353
        ctas=3
    else
        modulus=576460756061519873
        ctas=2
    fi
    for batch in 1 4; do
        common=(--word-bits "$bits" --modulus "$modulus" --logN 20 --batch "$batch")
        profile "v06_u${bits}_b${batch}" "${common[@]}" \
            --backend hierarchical-dataflow --stage-partition 10,10 \
            --segment-cores dataflow-radix4 --segment-cta-weights 9,11

        if [[ $bits == 32 && $batch == 1 ]]; then warp_weights=6,6,8
        elif [[ $bits == 32 ]]; then warp_weights=7,7,6
        elif [[ $batch == 1 ]]; then warp_weights=8,8,4
        else warp_weights=6,8,6
        fi
        if [[ $bits == 32 ]]; then cta_weights=7,6,7
        elif [[ $batch == 1 ]]; then cta_weights=4,8,8
        else cta_weights=5,7,8
        fi

        for point in "warp:${warp_weights}:appt-online" \
                     "cta-radix4:${cta_weights}:appt-online-radix4"; do
            IFS=: read -r label weights core <<< "$point"
            profile "${label}_u${bits}_b${batch}" "${common[@]}" \
                --backend hierarchical-dataflow --stage-partition 7,7,6 \
                --segment-cores "$core" --segment-units 8 \
                --segment-data-space 16,16,16 --segment-data-time 4 \
                --segment-role-stages 2 --segment-token-interleave 1 \
                --segment-cta-weights "$weights" --boundary-storage ring \
                --boundary-buffers 2 --target-ctas-per-sm "$ctas"
        done
    done
done

shopt -s nullglob
raw_reports=("$OUTPUT_DIR"/*.csv)
filtered=()
for report in "${raw_reports[@]}"; do
    [[ $(basename "$report") == summary.csv ]] || filtered+=("$report")
done
(( ${#filtered[@]} > 0 )) || { echo "no NCU raw reports found" >&2; exit 1; }
python3 "$ROOT/scripts/summarize_ncu.py" "${filtered[@]}" \
    --output "$OUTPUT_DIR/summary.csv"
python3 "$ROOT/scripts/analyze_appt_core_space_ncu.py" \
    "$OUTPUT_DIR/summary.csv" --csv "$OUTPUT_DIR/analysis.csv" \
    --markdown "$OUTPUT_DIR/analysis.md"
printf 'APPT physical-core NCU summary: %s\n' "$OUTPUT_DIR/summary.csv"
printf 'APPT physical-core NCU analysis: %s\n' "$OUTPUT_DIR/analysis.md"
if [[ -n ${SUDO_USER:-} ]]; then
    chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$OUTPUT_DIR"
fi
