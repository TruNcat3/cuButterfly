#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build/cubutterfly_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_fft_vectorized"}
BATCH=${BATCH:-16}

if [[ ! -x "$NCU" || ! -x "$BIN" ]]; then
    echo "missing profiler or benchmark: NCU=$NCU BIN=$BIN" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block,launch__waves_per_multiprocessor}
COMMON=(--operator fft --precision fp32 --logN 20 --batch "$BATCH" --placement in-place
        --normalization none --warmup 0 --repeat 1 --csv)

profile() {
    local label=$1
    shift
    "$NCU" --target-processes all --replay-mode kernel --cache-control none --clock-control base \
        --launch-count 4 --metrics "$METRICS" --page raw --csv --force-overwrite \
        --log-file "$OUTPUT_DIR/${label}.csv" "$@"
}

ONLINE=(--backend online-reorder --fft-core cufftdx-block --local-stages 10
        --reorder-columns 1 --cross-twiddle recurrence)
profile "post_vector_fixed_b${BATCH}" "$BIN" "${COMMON[@]}" "${ONLINE[@]}" \
    --prefix-threads 512 --suffix-threads 512 --prefix-ept 8 --suffix-ept 8
profile "post_vector_selected_b${BATCH}" "$BIN" "${COMMON[@]}" "${ONLINE[@]}" \
    --prefix-threads 256 --suffix-threads 128 --prefix-ept 16 --suffix-ept 16
profile "post_vector_cufft_b${BATCH}" "$BIN" "${COMMON[@]}" --backend cufft

python3 "$ROOT/scripts/summarize_ncu.py" "$OUTPUT_DIR"/post_vector_*.csv \
    --output "$OUTPUT_DIR/summary.csv"
python3 "$ROOT/scripts/analyze_fft_vectorized_ncu.py" \
    "$ROOT/results/ncu_scaling_crossovers/summary.csv" "$OUTPUT_DIR/summary.csv" \
    --pre-timing-summary "$ROOT/results/v100_fft_pipeline_summary.csv" \
    --post-timing-summary "$ROOT/results/v100_fft_pipeline_vectorized_summary.csv" \
    --batch "$BATCH" --output "$OUTPUT_DIR/attribution.csv" \
    --markdown "$OUTPUT_DIR/attribution.md"
printf 'NCU attribution: %s\n' "$OUTPUT_DIR/attribution.md"
