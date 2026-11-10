#!/usr/bin/env bash
set -euo pipefail

LOG_N=${LOG_N:-3}
BATCH=${BATCH:-524288}
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-./build/cubutterfly_bench}
OUTPUT_DIR=${OUTPUT_DIR:-results/ncu_fft_units}

if [[ ! -x "$NCU" ]]; then
    echo "ncu not found: $NCU" >&2
    exit 1
fi
if [[ ! -x "$BIN" ]]; then
    echo "cuButterfly benchmark not found: $BIN" >&2
    exit 1
fi
if (( LOG_N != 3 )); then
    echo "generated thread-dft8, cta-dft8, and wmma-dft8 currently require LOG_N=3" >&2
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
    smsp__inst_executed.sum
    smsp__sass_thread_inst_executed_op_fp32_pred_on.sum
    smsp__inst_executed_pipe_tensor.sum
    sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
    sm__warps_active.avg.pct_of_peak_sustained_active
    smsp__warp_issue_stalled_barrier_per_warp_active.pct
    smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
    launch__registers_per_thread
    launch__shared_mem_per_block
)
METRICS=$(IFS=,; echo "${NCU_METRICS[*]}")

profile() {
    local label=$1
    shift
    "$NCU" \
        --target-processes all \
        --replay-mode kernel \
        --cache-control none \
        --clock-control base \
        --launch-count 1 \
        --metrics "$METRICS" \
        --page raw \
        --csv \
        --force-overwrite \
        --log-file "$OUTPUT_DIR/${label}_logN${LOG_N}.csv" \
        "$@"
}

COMMON=(--operator fft --backend temporal-tile --logN "$LOG_N" --batch "$BATCH" --warmup 0 --repeat 1 --csv)

profile scalar_radix8_fp32 "$BIN" "${COMMON[@]}" \
    --precision fp32 --fft-core scalar --compute-unit radix8 --tile-threads 128
profile thread_dft8_fp32 "$BIN" "${COMMON[@]}" \
    --precision fp32 --fft-core thread-dft8
profile cta_dft8_fp32 "$BIN" "${COMMON[@]}" \
    --precision fp32 --fft-core cta-dft8
profile wmma_dft8_mixed "$BIN" "${COMMON[@]}" \
    --precision fp16-fp32 --fft-core wmma-dft8
profile cufft_fp32 "$BIN" --operator fft --backend cufft --precision fp32 \
    --logN "$LOG_N" --batch "$BATCH" --warmup 0 --repeat 1 --csv

python3 scripts/summarize_ncu.py \
    "$OUTPUT_DIR"/*_logN"$LOG_N".csv \
    --output "$OUTPUT_DIR/summary_logN${LOG_N}.csv"

echo "NCU summary written to $OUTPUT_DIR/summary_logN${LOG_N}.csv"
