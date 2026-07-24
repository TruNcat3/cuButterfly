#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build/cubutterfly_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_scaling_crossovers"}

if [[ ! -x "$NCU" || ! -x "$BIN" ]]; then
    echo "missing profiler or benchmark: NCU=$NCU BIN=$BIN" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block,launch__waves_per_multiprocessor}

profile() {
    local label=$1
    shift
    "$NCU" --target-processes all --replay-mode kernel --cache-control none \
        --clock-control base --launch-count 4 --metrics "$METRICS" --page raw --csv \
        --force-overwrite --log-file "$OUTPUT_DIR/${label}.csv" "$@"
}

common_fft=(--operator fft --precision fp32 --placement in-place --normalization none \
            --warmup 0 --repeat 1 --csv)

for batch in 16 256 1024; do
    profile "fft14_direct_b${batch}" "$BIN" "${common_fft[@]}" --logN 14 --batch "$batch" \
        --backend temporal-tile --fft-core cufftdx-direct --tile-threads 1024
    profile "fft14_cufft_b${batch}" "$BIN" "${common_fft[@]}" --logN 14 --batch "$batch" \
        --backend cufft
done

for batch in 2 16 64; do
    profile "fft18_online_b${batch}" "$BIN" "${common_fft[@]}" --logN 18 --batch "$batch" \
        --backend online-reorder --fft-core cufftdx-block --local-stages 9 \
        --reorder-columns 1 --cross-twiddle recurrence --prefix-threads 256 \
        --suffix-threads 256 --prefix-ept 8 --suffix-ept 8
    profile "fft18_cufft_b${batch}" "$BIN" "${common_fft[@]}" --logN 18 --batch "$batch" \
        --backend cufft
done

for batch in 2 8 16; do
    profile "fft20_online_b${batch}" "$BIN" "${common_fft[@]}" --logN 20 --batch "$batch" \
        --backend online-reorder --fft-core cufftdx-block --local-stages 10 \
        --reorder-columns 1 --cross-twiddle recurrence --prefix-threads 512 \
        --suffix-threads 512 --prefix-ept 8 --suffix-ept 8
    profile "fft20_cufft_b${batch}" "$BIN" "${common_fft[@]}" --logN 20 --batch "$batch" \
        --backend cufft
done

for batch in 4 16; do
    common_fwht=(--operator fwht --precision fp32 --logN 15 --batch "$batch" \
                 --normalization none --warmup 0 --repeat 1 --csv)
    profile "fwht15_online_b${batch}" "$BIN" "${common_fwht[@]}" \
        --backend online-reorder --compute-unit radix4 --tile-threads 256 \
        --local-stages 8 --reorder-columns 1
    profile "fwht15_warp_b${batch}" "$BIN" "${common_fwht[@]}" \
        --backend temporal-tile --local-exchange warp-register --compute-unit radix2 \
        --tile-threads 256
done

python3 "$ROOT/scripts/summarize_ncu.py" "$OUTPUT_DIR"/*.csv \
    --output "$OUTPUT_DIR/summary.csv"
printf 'NCU summary: %s\n' "$OUTPUT_DIR/summary.csv"
