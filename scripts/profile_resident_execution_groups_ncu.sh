#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_resident_execution_groups"}
BATCHES=${BATCHES:-"1 16"}
METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,smsp__inst_executed_pipe_adu.sum,smsp__inst_executed_pipe_cbu.sum,smsp__inst_executed_pipe_lsu.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block,launch__waves_per_multiprocessor,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps}

for executable in "$NCU" "$BIN"; do
    [[ -x "$executable" ]] || {
        echo "missing executable: $executable" >&2
        exit 1
    }
done
mkdir -p "$OUTPUT_DIR"
reports=()

restore_owner() {
    if [[ ${EUID:-$(id -u)} -eq 0 && -n ${SUDO_USER:-} ]]; then
        chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$OUTPUT_DIR"
    fi
}
trap restore_owner EXIT

common=(--logN 20 --word-bits 32 --modulus 998244353
        --backend hierarchical-dataflow --cross-twiddle fused
        --mod-multiply shoup --warmup 0 --repeat 1 --csv)

verify() {
    local label=$1
    shift
    echo "[preflight] $label"
    "$BIN" "${common[@]}" "$@" --verify >/dev/null
}

profile() {
    local label=$1
    shift
    local report="$OUTPUT_DIR/${label}.csv"
    echo "[profile] $label"
    "$NCU" --target-processes all --replay-mode kernel --cache-control all \
        --clock-control base --launch-count 1 --metrics "$METRICS" \
        --page raw --csv --force-overwrite --log-file "$report" \
        "$BIN" "${common[@]}" "$@"
    reports+=("$report")
}

for batch in $BATCHES; do
    m4_g4=(--batch "$batch" --stage-partition 5,5,5,5
           --segment-cores radix4 --segment-threads 256
           --segment-cta-weights 5,5,5,5
           --boundary-storage full-scratch --target-ctas-per-sm 2)
    m4_g2_generic=(--batch "$batch" --stage-partition 5,5,5,5
                   --segment-cores radix4 --segment-threads 256
                   --segment-cta-weights 5,5,5,5
                   --boundary-storage resident-fused,full-scratch,resident-fused
                   --target-ctas-per-sm 2)
    m4_g2_dataflow=(--batch "$batch" --stage-partition 5,5,5,5
                    --segment-cores radix4 --segment-threads 256
                    --segment-cta-weights 5,5,5,5
                    --boundary-storage resident-fused,full-scratch,resident-fused
                    --execution-group-cores dataflow-radix4
                    --execution-group-cta-weights 10,10
                    --target-ctas-per-sm 4)
    m2_g2_dataflow=(--batch "$batch" --stage-partition 10,10
                    --segment-cores dataflow-radix4 --segment-threads 256
                    --segment-cta-weights 10,10 --boundary-storage full-scratch
                    --target-ctas-per-sm 4)

    verify "b${batch}_m4_g4_generic" "${m4_g4[@]}"
    verify "b${batch}_m4_g2_generic" "${m4_g2_generic[@]}"
    verify "b${batch}_m4_g2_dataflow" "${m4_g2_dataflow[@]}"
    verify "b${batch}_m2_g2_dataflow" "${m2_g2_dataflow[@]}"
    profile "b${batch}_m4_g4_generic" "${m4_g4[@]}"
    profile "b${batch}_m4_g2_generic" "${m4_g2_generic[@]}"
    profile "b${batch}_m4_g2_dataflow" "${m4_g2_dataflow[@]}"
    profile "b${batch}_m2_g2_dataflow" "${m2_g2_dataflow[@]}"
done

python3 "$ROOT/scripts/summarize_ncu.py" "${reports[@]}" \
    --output "$OUTPUT_DIR/summary.csv"
python3 "$ROOT/scripts/analyze_resident_execution_groups_ncu.py" \
    "$OUTPUT_DIR/summary.csv" --csv "$OUTPUT_DIR/analysis.csv" \
    --markdown "$OUTPUT_DIR/analysis.md"
printf 'wrote %s\n' "$OUTPUT_DIR/summary.csv"
