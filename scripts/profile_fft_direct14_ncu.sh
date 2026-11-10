#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build/cubutterfly_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_fft_direct14"}
BATCH=${BATCH:-256}

if [[ ! -x "$NCU" ]]; then
    echo "ncu not found: $NCU" >&2
    exit 1
fi
if [[ ! -x "$BIN" ]]; then
    echo "cuButterfly benchmark not found: $BIN" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

NCU_METRICS=(
    gpu__time_duration.sum
    dram__bytes_read.sum
    dram__bytes_write.sum
    dram__throughput.avg.pct_of_peak_sustained_elapsed
    lts__t_sector_hit_rate.pct
    l1tex__t_sector_hit_rate.pct
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
    smsp__inst_executed.sum
    smsp__sass_thread_inst_executed_op_fp32_pred_on.sum
    sm__warps_active.avg.pct_of_peak_sustained_active
    smsp__warp_issue_stalled_barrier_per_warp_active.pct
    smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
    launch__registers_per_thread
    launch__shared_mem_per_block
)
METRICS=$(IFS=,; echo "${NCU_METRICS[*]}")
COMMON=(--operator fft --precision fp32 --logN 14 --batch "$BATCH" --placement out-of-place
        --normalization none --warmup 0 --repeat 1 --csv)

profile() {
    local label=$1
    shift
    "$NCU" --target-processes all --replay-mode kernel --cache-control none \
        --clock-control base --launch-count 2 --metrics "$METRICS" --page raw --csv \
        --force-overwrite --log-file "$OUTPUT_DIR/${label}_logN14.csv" "$@"
}

profile direct_1024 "$BIN" "${COMMON[@]}" --backend temporal-tile \
    --fft-core cufftdx-direct --tile-threads 1024
profile cufft "$BIN" "${COMMON[@]}" --backend cufft

python3 "$ROOT/scripts/summarize_ncu.py" \
    "$OUTPUT_DIR/direct_1024_logN14.csv" \
    "$OUTPUT_DIR/cufft_logN14.csv" \
    --output "$OUTPUT_DIR/summary_logN14.csv"
printf 'NCU summary: %s\n' "$OUTPUT_DIR/summary_logN14.csv"
