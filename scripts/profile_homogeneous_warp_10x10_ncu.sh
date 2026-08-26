#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_homogeneous_warp_10x10"}
METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block,launch__waves_per_multiprocessor,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps}
BITS=${BITS:-32}
BATCH=${BATCH:-16}

mkdir -p "$OUTPUT_DIR"
restore_owner() {
    if [[ ${EUID:-$(id -u)} -eq 0 && -n ${SUDO_USER:-} ]]; then
        chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$OUTPUT_DIR"
    fi
}
trap restore_owner EXIT
for executable in "$NCU" "$BIN"; do
    [[ -x "$executable" ]] || { echo "missing executable: $executable" >&2; exit 1; }
done
if [[ $BITS == 32 ]]; then modulus=998244353; v06_resident=4
else modulus=576460756061519873; v06_resident=2
fi
if [[ $BITS == 32 ]]; then half_packet=4
else half_packet=2
fi
io_half_weights=9,11
if [[ $BITS == 32 ]]; then
    case $BATCH in
        1) io_half_weights=11,9 ;;
        4) io_half_weights=8,7 ;;
        16) io_half_weights=17,13 ;;
        *) io_half_weights=11,9 ;;
    esac
fi

profile() {
    local label=$1
    shift
    "$NCU" --target-processes all --replay-mode kernel --cache-control none \
        --clock-control base --launch-count 1 --metrics "$METRICS" \
        --page raw --csv --force-overwrite \
        --log-file "$OUTPUT_DIR/${label}.csv" \
        "$BIN" "$@" --warmup 0 --repeat 1 --csv
}

common=(--logN 20 --batch "$BATCH" --word-bits "$BITS" --modulus "$modulus"
        --backend hierarchical-dataflow --stage-partition 10,10
        --boundary-storage full-scratch --segment-data-time 1,1
        --segment-cta-weights 9,11)
profile v06 "${common[@]}" --segment-cores dataflow-radix4 \
    --target-ctas-per-sm "$v06_resident"
profile warp1024 "${common[@]}" --segment-cores homogeneous-warp-radix2 \
    --segment-threads 256 --segment-units 8 --target-ctas-per-sm 1
profile warp256_t128 "${common[@]}" \
    --segment-cores homogeneous-warp256-radix2 \
    --segment-threads 128 --segment-units 4 --target-ctas-per-sm 3
profile warp256_t256 "${common[@]}" \
    --segment-cores homogeneous-warp256-radix2 \
    --segment-threads 256 --segment-units 8 --target-ctas-per-sm 1
profile warp256_static_t128 "${common[@]}" \
    --segment-cores homogeneous-warp256-static-radix2 \
    --segment-threads 128 --segment-units 4 --target-ctas-per-sm 2
profile warp256_static_t256 "${common[@]}" \
    --segment-cores homogeneous-warp256-static-radix2 \
    --segment-threads 256 --segment-units 8 --target-ctas-per-sm 1
profile warp256_static_io_t128 "${common[@]}" \
    --segment-cores homogeneous-warp256-static-io-radix2 \
    --segment-threads 128 --segment-units 4 --target-ctas-per-sm 2
profile warp256_static_io_t256 "${common[@]}" \
    --segment-cores homogeneous-warp256-static-io-radix2 \
    --segment-threads 256 --segment-units 8 --target-ctas-per-sm 1
profile warp256_static_half_t128 "${common[@]}" \
    --segment-cores homogeneous-warp256-static-radix2 \
    --segment-threads 128 --segment-units 4 \
    --segment-data-space "$half_packet" --target-ctas-per-sm 3
profile warp256_static_io_half_t128 "${common[@]}" \
    --segment-cores homogeneous-warp256-static-io-radix2 \
    --segment-threads 128 --segment-units 4 \
    --segment-data-space "$half_packet" \
    --segment-cta-weights "$io_half_weights" --target-ctas-per-sm 3
if [[ $BITS == 32 ]]; then
    profile warp128_static_io_half_t128 "${common[@]}" \
        --segment-cores homogeneous-warp128-static-io-radix2 \
        --segment-threads 128 --segment-units 4 \
        --segment-data-space 4 \
        --segment-cta-weights "$io_half_weights" --target-ctas-per-sm 4
    profile warp64_static_io_half_t128 "${common[@]}" \
        --segment-cores homogeneous-warp64-static-io-radix2 \
        --segment-threads 128 --segment-units 4 \
        --segment-data-space 4 \
        --segment-cta-weights "$io_half_weights" --target-ctas-per-sm 4
fi

ncu_files=(
    "$OUTPUT_DIR/v06.csv" "$OUTPUT_DIR/warp1024.csv"
    "$OUTPUT_DIR/warp256_t128.csv" "$OUTPUT_DIR/warp256_t256.csv"
    "$OUTPUT_DIR/warp256_static_t128.csv"
    "$OUTPUT_DIR/warp256_static_t256.csv"
    "$OUTPUT_DIR/warp256_static_io_t128.csv"
    "$OUTPUT_DIR/warp256_static_io_t256.csv"
    "$OUTPUT_DIR/warp256_static_half_t128.csv"
    "$OUTPUT_DIR/warp256_static_io_half_t128.csv"
)
if [[ $BITS == 32 ]]; then
    ncu_files+=(
        "$OUTPUT_DIR/warp128_static_io_half_t128.csv"
        "$OUTPUT_DIR/warp64_static_io_half_t128.csv"
    )
fi
python3 "$ROOT/scripts/summarize_ncu.py" "${ncu_files[@]}" \
    --output "$OUTPUT_DIR/summary.csv"
printf 'Warp-granular 10+10 NCU summary: %s\n' "$OUTPUT_DIR/summary.csv"
