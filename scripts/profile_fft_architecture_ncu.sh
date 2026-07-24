#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build/cubutterfly_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_fft_architecture"}
LOG_N=${LOG_N:-18}
BATCH=${BATCH:-16}
LOCAL_STAGES=${LOCAL_STAGES:-9}

if [[ ! -x "$NCU" || ! -x "$BIN" ]]; then
    echo "missing profiler or benchmark: NCU=$NCU BIN=$BIN" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__inst_executed.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block}
COMMON=(--operator fft --precision fp32 --logN "$LOG_N" --batch "$BATCH" --placement out-of-place
        --normalization none --warmup 0 --repeat 1 --csv)
ONLINE=(--backend online-reorder --fft-core cufftdx-block --local-stages "$LOCAL_STAGES"
        --cross-twiddle recurrence --tile-threads 32 --reorder-columns 1)

profile() {
    local label=$1
    shift
    "$NCU" --target-processes all --replay-mode kernel --cache-control none --clock-control base \
        --launch-count 2 --metrics "$METRICS" --page raw --csv --force-overwrite \
        --log-file "$OUTPUT_DIR/${label}_logN${LOG_N}.csv" "$@"
}

profile tuned_256_256 "$BIN" "${COMMON[@]}" "${ONLINE[@]}" --prefix-threads 256 --suffix-threads 256
profile fixed_512_512 "$BIN" "${COMMON[@]}" "${ONLINE[@]}" --prefix-threads 512 --suffix-threads 512
profile asymmetric_256_512 "$BIN" "${COMMON[@]}" "${ONLINE[@]}" --prefix-threads 256 --suffix-threads 512
profile cufft "$BIN" "${COMMON[@]}" --backend cufft

python3 "$ROOT/scripts/summarize_ncu.py" "$OUTPUT_DIR"/*_logN"$LOG_N".csv \
    --output "$OUTPUT_DIR/summary_logN${LOG_N}.csv"
printf 'NCU summary: %s\n' "$OUTPUT_DIR/summary_logN${LOG_N}.csv"
