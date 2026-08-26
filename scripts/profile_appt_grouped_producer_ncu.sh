#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build-cuda118/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_appt_grouped_producer"}
BATCHES=${BATCHES:-1,4,16}
A_GROUPS=${A_GROUPS:-8,16,32}
METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block,launch__waves_per_multiprocessor}

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
        "$BIN" "$@" --warmup 0 --repeat 1 --csv >/dev/null
}

IFS=, read -ra selected_batches <<< "$BATCHES"
IFS=, read -ra selected_groups <<< "$A_GROUPS"
for bits in 32 64; do
    if [[ $bits == 32 ]]; then modulus=998244353; resident=3
    else modulus=576460756061519873; resident=2
    fi
    for batch in "${selected_batches[@]}"; do
        case "$bits:$batch" in
            32:1)  fragment=16; radix4_roles=7,13,3; grouped_roles=7,13,3 ;;
            32:4)  fragment=16; radix4_roles=6,14,2; grouped_roles=6,14,2 ;;
            32:16) fragment=32; radix4_roles=6,16,2; grouped_roles=6,16,2 ;;
            64:1)  fragment=8;  radix4_roles=10,9,3; grouped_roles=10,9,3 ;;
            64:4)  fragment=16; radix4_roles=9,11,2; grouped_roles=8,10,2 ;;
            64:16) fragment=16; radix4_roles=9,11,2; grouped_roles=8,10,2 ;;
            *) echo "unsupported bits/batch point: $bits/$batch" >&2; exit 1 ;;
        esac
        common=(--word-bits "$bits" --modulus "$modulus" --logN 20
                --batch "$batch")
        appt=("${common[@]}" --backend hierarchical-dataflow
              --stage-partition 7,7,6 --segment-units 8
              --segment-data-time 4 --segment-role-stages 2
              --segment-token-interleave 1 --boundary-storage ring
              --boundary-buffers 2 --target-ctas-per-sm "$resident"
              --appt-fragment-width "$fragment" --appt-writer-tiles 2
              --output-order natural)
        profile "v06_u${bits}_b${batch}" "${common[@]}" \
            --backend hierarchical-dataflow --stage-partition 10,10 \
            --segment-cores dataflow-radix4 --segment-cta-weights 9,11
        profile "radix4_u${bits}_b${batch}" "${appt[@]}" \
            --segment-cores appt-online-register-tail \
            --segment-data-space 16 --appt-role-weights "$radix4_roles"
        for group in "${selected_groups[@]}"; do
            profile "group${group}_u${bits}_b${batch}" "${appt[@]}" \
                --segment-cores appt-online-register-tail-grouped \
                --segment-data-space "$group",16,16 \
                --appt-role-weights "$grouped_roles"
        done
    done
done

shopt -s nullglob
reports=("$OUTPUT_DIR"/*.csv)
filtered=()
for report in "${reports[@]}"; do
    [[ $(basename "$report") == summary.csv ]] || filtered+=("$report")
done
python3 "$ROOT/scripts/summarize_ncu.py" "${filtered[@]}" \
    --output "$OUTPUT_DIR/summary.csv"
python3 "$ROOT/scripts/analyze_appt_grouped_producer_ncu.py" \
    "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'APPT grouped-producer NCU analysis: %s\n' "$OUTPUT_DIR/analysis.md"
if [[ -n ${SUDO_USER:-} ]]; then
    chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$OUTPUT_DIR"
fi
