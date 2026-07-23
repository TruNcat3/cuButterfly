#!/usr/bin/env bash
set -euo pipefail

NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-./build/cubutterfly_bench}
OUTPUT_DIR=${OUTPUT_DIR:-results/ncu_fft_cta_mapping}
TARGET_POINTS=${TARGET_POINTS:-4194304}

if [[ ! -x "$NCU" ]]; then
    echo "ncu not found: $NCU" >&2
    exit 1
fi
if [[ ! -x "$BIN" ]]; then
    echo "cuButterfly benchmark not found: $BIN" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

METRICS=$(IFS=,; echo "gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block")

profile() {
    local label=$1
    shift
    "$NCU" --target-processes all --replay-mode kernel --cache-control none \
        --clock-control base --launch-count 1 --metrics "$METRICS" --page raw \
        --csv --force-overwrite --log-file "$OUTPUT_DIR/${label}.csv" "$@"
}

for log_n in 3 6 8 10; do
    batch=$((TARGET_POINTS >> log_n))
    case "$log_n" in
        3) cta_threads=32; scalar_threads=64 ;;
        6) cta_threads=32; scalar_threads=32 ;;
        8) cta_threads=64; scalar_threads=64 ;;
        10) cta_threads=128; scalar_threads=256 ;;
    esac
    common=(--operator fft --logN "$log_n" --batch "$batch" --precision fp32 --warmup 0 --repeat 1 --csv)
    profile "cta_dft8_logN${log_n}_t${cta_threads}" "$BIN" "${common[@]}" \
        --backend temporal-tile --fft-core cta-dft8 --compute-unit radix8 --tile-threads "$cta_threads"
    profile "scalar_radix8_logN${log_n}_t${scalar_threads}" "$BIN" "${common[@]}" \
        --backend temporal-tile --fft-core scalar --compute-unit radix8 --tile-threads "$scalar_threads"
    profile "cufft_logN${log_n}" "$BIN" "${common[@]}" --backend cufft
done

python3 scripts/summarize_ncu.py "$OUTPUT_DIR"/cta_*.csv "$OUTPUT_DIR"/scalar_*.csv "$OUTPUT_DIR"/cufft_*.csv \
    --output "$OUTPUT_DIR/summary.csv"
echo "NCU summary written to $OUTPUT_DIR/summary.csv"
