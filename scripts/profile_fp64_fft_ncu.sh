#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build/cubutterfly_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_fp64_fft"}
BATCH=${BATCH:-64}

if [[ ! -x "$NCU" || ! -x "$BIN" ]]; then
    echo "missing profiler or benchmark: NCU=$NCU BIN=$BIN" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_fp64_pred_on.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block,launch__waves_per_multiprocessor,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps}
COMMON=(--operator fft --precision fp64 --logN 16 --batch "$BATCH" --placement out-of-place
        --normalization none --warmup 0 --repeat 1 --csv)

profile() {
    local label=$1
    shift
    "$NCU" --target-processes all --replay-mode kernel --cache-control none --clock-control base \
        --launch-count 4 --metrics "$METRICS" --page raw --csv --force-overwrite \
        --log-file "$OUTPUT_DIR/${label}.csv" "$@"
}

profile "fp64_scalar_b${BATCH}" "$BIN" "${COMMON[@]}" \
    --backend online-reorder --fft-core scalar --compute-unit radix4 \
    --tile-threads 128 --local-stages 8 --reorder-columns 1 --complex-multiply four-mul
profile "fp64_cufftdx_b${BATCH}" "$BIN" "${COMMON[@]}" \
    --backend online-reorder --fft-core cufftdx-block --local-stages 8 \
    --prefix-threads 256 --prefix-ept 4 --suffix-threads 128 --suffix-ept 8
profile "fp64_recurrence_b${BATCH}" "$BIN" "${COMMON[@]}" \
    --backend online-reorder --fft-core cufftdx-block --local-stages 8 \
    --prefix-threads 256 --prefix-ept 4 --suffix-threads 128 --suffix-ept 8 \
    --cross-twiddle recurrence
profile "fp64_cufft_b${BATCH}" "$BIN" "${COMMON[@]}" --backend cufft

python3 "$ROOT/scripts/summarize_ncu.py" "$OUTPUT_DIR"/fp64_*.csv \
    --output "$OUTPUT_DIR/summary.csv"
python3 "$ROOT/scripts/analyze_fp64_fft_ncu.py" "$OUTPUT_DIR/summary.csv" \
    --batch "$BATCH" \
    --timing-summary "$ROOT/results/fp64_logN16_twiddle_summary.csv" \
    --output "$OUTPUT_DIR/analysis.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'FP64 NCU attribution: %s\n' "$OUTPUT_DIR/analysis.md"
