#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build/cubutterfly_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_structured_2x2"}

if [[ ! -x "$NCU" || ! -x "$BIN" ]]; then
    echo "missing profiler or benchmark: NCU=$NCU BIN=$BIN" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,smsp__sass_thread_inst_executed_op_fp64_pred_on.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block,launch__waves_per_multiprocessor}

profile() {
    local label=$1
    shift
    echo "Profiling $label"
    "$NCU" --target-processes all --replay-mode kernel --cache-control none \
        --clock-control base --launch-count 4 --metrics "$METRICS" --page raw --csv \
        --force-overwrite --log-file "$OUTPUT_DIR/${label}.csv" "$@"
}

common=(--precision fp32 --normalization none --placement out-of-place \
        --warmup 0 --repeat 1 --csv)
structured=(--operator structured-2x2 --stage-matrix 1,0.25,-0.5,1)
structured_per_stage=(--operator structured-2x2)
for ((stage=0; stage<12; ++stage)); do
    structured_per_stage+=(--stage-matrix 1,0.25,-0.5,1)
done
fwht=(--operator fwht)

# Same-mapping pairs isolate local arithmetic from the architecture schedule.
profile structured8_temporal_r4 "$BIN" "${common[@]}" "${structured[@]}" \
    --logN 8 --batch 16384 --backend temporal-tile --compute-unit radix4 --tile-threads 128
profile fwht8_temporal_r4 "$BIN" "${common[@]}" "${fwht[@]}" \
    --logN 8 --batch 16384 --backend temporal-tile --compute-unit radix4 --tile-threads 128

profile structured12_hierarchical_r8 "$BIN" "${common[@]}" "${structured[@]}" \
    --logN 12 --batch 1024 --backend hierarchical --compute-unit radix8 \
    --tile-threads 256 --local-stages 10
profile structured12_warp_register_broadcast "$BIN" "${common[@]}" "${structured[@]}" \
    --logN 12 --batch 1024 --backend temporal-tile --local-exchange warp-register \
    --compute-unit radix2 --tile-threads 256
profile structured12_warp_register_per_stage "$BIN" "${common[@]}" "${structured_per_stage[@]}" \
    --logN 12 --batch 1024 --backend temporal-tile --local-exchange warp-register \
    --compute-unit radix2 --tile-threads 256
profile fwht12_hierarchical_r8 "$BIN" "${common[@]}" "${fwht[@]}" \
    --logN 12 --batch 1024 --backend hierarchical --compute-unit radix8 \
    --tile-threads 256 --local-stages 10

# FWHT is the same-transport arithmetic control for the Structured register
# codelet above.
profile fwht12_warp_register "$BIN" "${common[@]}" "${fwht[@]}" \
    --logN 12 --batch 1024 --backend temporal-tile --local-exchange warp-register \
    --compute-unit radix2 --tile-threads 256

profile structured20_online_r8 "$BIN" "${common[@]}" "${structured[@]}" \
    --logN 20 --batch 4 --backend online-reorder --compute-unit radix8 \
    --tile-threads 256 --local-stages 10 --reorder-columns 1
profile fwht20_online_r8 "$BIN" "${common[@]}" "${fwht[@]}" \
    --logN 20 --batch 4 --backend online-reorder --compute-unit radix8 \
    --tile-threads 256 --local-stages 10 --reorder-columns 1

python3 "$ROOT/scripts/summarize_ncu.py" "$OUTPUT_DIR"/*.csv \
    --output "$OUTPUT_DIR/summary.csv"
python3 "$ROOT/scripts/analyze_structured_2x2_ncu.py" "$OUTPUT_DIR/summary.csv" \
    --output "$OUTPUT_DIR/attribution.csv" \
    --markdown "$OUTPUT_DIR/attribution.md"
if (( EUID == 0 )) && [[ -n ${SUDO_UID:-} && -n ${SUDO_GID:-} ]]; then
    chown -R "$SUDO_UID:$SUDO_GID" "$OUTPUT_DIR"
fi
printf 'NCU summary: %s\n' "$OUTPUT_DIR/summary.csv"
