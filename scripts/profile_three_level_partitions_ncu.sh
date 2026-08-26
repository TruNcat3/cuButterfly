#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_three_level_partitions"}
METRICS=${METRICS:-gpu__time_duration.sum,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,smsp__inst_executed_pipe_adu.sum,smsp__inst_executed_pipe_cbu.sum,smsp__inst_executed_pipe_lsu.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block,launch__occupancy_limit_blocks,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem}

mkdir -p "$OUTPUT_DIR"
for executable in "$NCU" "$BIN"; do
    [[ -x "$executable" ]] || { echo "missing executable: $executable" >&2; exit 1; }
done

common=(--logN 20 --batch 4 --word-bits 32 --modulus 998244353
        --backend hierarchical-dataflow --cross-twiddle fused
        --mod-multiply shoup --warmup 0 --repeat 1 --csv)

profile() {
    local label=$1
    shift
    echo "[profile] $label"
    "$NCU" --target-processes all --replay-mode kernel --cache-control all \
        --clock-control base --launch-count 1 --metrics "$METRICS" \
        --page raw --csv --force-overwrite \
        --log-file "$OUTPUT_DIR/${label}.csv" \
        "$BIN" "${common[@]}" "$@"
}

profile packet128_10x10 \
    --stage-partition 10,10 --boundary-storage full-scratch \
    --segment-cores homogeneous-warp128-packet-shared-radix4-static-io \
    --segment-threads 128 --segment-units 4 --segment-data-space 4 \
    --segment-data-time 1 --segment-cta-weights 10,10 \
    --target-ctas-per-sm 4
profile resident_7x7x6_rows32 \
    --stage-partition 7,7,6 --segment-cores dataflow-radix4 \
    --segment-threads 256 --segment-units 32 \
    --segment-cta-weights 7,7,6 --target-ctas-per-sm 2
profile resident_6x6x8_rows32 \
    --stage-partition 6,6,8 --segment-cores dataflow-radix4 \
    --segment-threads 256 --segment-units 32 \
    --segment-cta-weights 4,4,12 --target-ctas-per-sm 2
profile resident_6x6x8_rows4 \
    --stage-partition 6,6,8 --segment-cores dataflow-radix4 \
    --segment-threads 128 --segment-units 4 \
    --segment-cta-weights 4,4,12 --target-ctas-per-sm 4

python3 "$ROOT/scripts/summarize_ncu.py" "$OUTPUT_DIR"/*.csv \
    --output "$OUTPUT_DIR/summary.csv"
printf 'wrote %s\n' "$OUTPUT_DIR/summary.csv"
