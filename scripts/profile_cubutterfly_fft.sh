#!/usr/bin/env bash
set -euo pipefail

LOG_N=${LOG_N:-20}
BATCH=${BATCH:-4}
LOCAL_STAGES=${LOCAL_STAGES:-10}
TILE_THREADS=${TILE_THREADS:-256}
REORDER_COLUMNS=${REORDER_COLUMNS:-1}
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-./build/cubutterfly_bench}
OUTPUT_DIR=${OUTPUT_DIR:-results/ncu_fft}
CUFFT_LAUNCH_COUNT=${CUFFT_LAUNCH_COUNT:-2}

if [[ ! -x "$NCU" ]]; then
    echo "ncu not found: $NCU" >&2
    exit 1
fi
if [[ ! -x "$BIN" ]]; then
    echo "cuButterfly benchmark not found: $BIN" >&2
    exit 1
fi
if (( LOG_N <= LOCAL_STAGES )); then
    echo "LOG_N must exceed LOCAL_STAGES" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

profile() {
    local label=$1
    local count=$2
    shift 2
    "$NCU" \
        --target-processes all \
        --replay-mode kernel \
        --cache-control none \
        --clock-control base \
        --launch-count "$count" \
        --metrics "$METRICS" \
        --page raw \
        --csv \
        --force-overwrite \
        --log-file "$OUTPUT_DIR/${label}_logN${LOG_N}.csv" \
        "$@"
}

NCU_METRICS=(
    gpu__time_duration.sum
    dram__bytes_read.sum
    dram__bytes_write.sum
    dram__throughput.avg.pct_of_peak_sustained_elapsed
    lts__t_sector_hit_rate.pct
    l1tex__t_sector_hit_rate.pct
    smsp__inst_executed.sum
    smsp__sass_thread_inst_executed_op_fp32_pred_on.sum
    sm__warps_active.avg.pct_of_peak_sustained_active
    smsp__warp_issue_stalled_barrier_per_warp_active.pct
    smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
    launch__registers_per_thread
    launch__shared_mem_per_block
)
METRICS=$(IFS=,; echo "${NCU_METRICS[*]}")

COMMON=(
    "$BIN" --operator fft --precision fp32 --logN "$LOG_N" --batch "$BATCH"
    --warmup 0 --repeat 1 --csv
)

profile cubutterfly_fft "$((LOG_N - LOCAL_STAGES + 1))" \
    "${COMMON[@]}" --backend hierarchical --compute-unit radix4 \
    --local-stages "$LOCAL_STAGES" --tile-threads "$TILE_THREADS"

profile online_reorder_fft 2 \
    "${COMMON[@]}" --backend online-reorder --compute-unit radix4 \
    --local-stages "$LOCAL_STAGES" --reorder-columns "$REORDER_COLUMNS" --tile-threads "$TILE_THREADS"

profile cufft "$CUFFT_LAUNCH_COUNT" \
    "${COMMON[@]}" --backend cufft

python3 scripts/summarize_ncu.py \
    "$OUTPUT_DIR"/*_logN"$LOG_N".csv \
    --output "$OUTPUT_DIR/summary_logN${LOG_N}.csv"

echo "NCU summary written to $OUTPUT_DIR/summary_logN${LOG_N}.csv"
