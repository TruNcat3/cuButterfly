#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_homogeneous_online_tiles"}
BATCH=${BATCH:-16}
METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block,launch__grid_size,launch__waves_per_multiprocessor,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps}

mkdir -p "$OUTPUT_DIR"
restore_owner() {
    if [[ ${EUID:-$(id -u)} -eq 0 && -n ${SUDO_USER:-} ]]; then
        chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$OUTPUT_DIR"
    fi
}
trap restore_owner EXIT
for executable in "$NCU" "$BIN"; do
    [[ -x "$executable" ]] || {
        echo "missing executable: $executable" >&2
        exit 1
    }
done

case $BATCH in
    1) role_weights=11,9 ;;
    4) role_weights=8,7 ;;
    16) role_weights=17,13 ;;
    *) role_weights=11,9 ;;
esac

profile() {
    local label=$1
    shift
    local output="$OUTPUT_DIR/${label}.csv"
    local temporary="$output.tmp"
    rm -f "$temporary"
    if ! "$NCU" --target-processes all --replay-mode kernel --cache-control none \
        --clock-control base --launch-count 1 --metrics "$METRICS" \
        --page raw --csv --force-overwrite \
        --log-file "$temporary" \
        "$BIN" "$@" --warmup 0 --repeat 1 --csv; then
        [[ -f "$temporary" ]] && cat "$temporary" >&2
        rm -f "$temporary"
        return 1
    fi
    if ! grep -q '"Kernel Name"' "$temporary"; then
        cat "$temporary" >&2
        rm -f "$temporary"
        echo "NCU did not produce a raw CSV table for $label" >&2
        return 1
    fi
    mv "$temporary" "$output"
}

common=(--logN 20 --batch "$BATCH" --word-bits 32 --modulus 998244353
        --backend hierarchical-dataflow --stage-partition 10,10
        --boundary-storage full-scratch --segment-data-time 1,1
        --segment-cta-weights "$role_weights" --cross-twiddle fused
        --mod-multiply shoup)

profile v06 "${common[@]}" --segment-cores dataflow-radix4 \
    --target-ctas-per-sm 4
profile warp256_static_io_half "${common[@]}" \
    --segment-cores homogeneous-warp256-static-io-radix2 \
    --segment-threads 128 --segment-units 4 --segment-data-space 4 \
    --target-ctas-per-sm 3
profile warp256_coefficient_reuse_static_io_half "${common[@]}" \
    --segment-cores \
        homogeneous-warp256-coefficient-reuse-static-io-radix2 \
    --segment-threads 128 --segment-units 4 --segment-data-space 4 \
    --segment-coefficient-reuse-stages 6 \
    --target-ctas-per-sm 3
for reuse_stages in 1 2 3 4 5; do
    profile "warp256_coefficient_reuse${reuse_stages}_static_io_half" \
        "${common[@]}" \
        --segment-cores \
            homogeneous-warp256-coefficient-reuse-static-io-radix2 \
        --segment-threads 128 --segment-units 4 --segment-data-space 4 \
        --segment-coefficient-reuse-stages "$reuse_stages" \
        --target-ctas-per-sm 3
done
profile warp128_static_io_half "${common[@]}" \
    --segment-cores homogeneous-warp128-static-io-radix2 \
    --segment-threads 128 --segment-units 4 --segment-data-space 4 \
    --target-ctas-per-sm 4
profile warp128_static_io_half_dual "${common[@]}" \
    --segment-cores homogeneous-warp128-static-io-radix2 \
    --segment-threads 256 --segment-units 8 --segment-data-space 4 \
    --segment-cta-weights 8,7 \
    --target-ctas-per-sm 2
profile warp128_coefficient_reuse_static_io_half_dual "${common[@]}" \
    --segment-cores \
        homogeneous-warp128-coefficient-reuse-static-io-radix2 \
    --segment-threads 256 --segment-units 8 --segment-data-space 4 \
    --segment-coefficient-reuse-stages 6 \
    --segment-cta-weights 8,7 \
    --target-ctas-per-sm 2
for reuse_stages in 1 2 3 4 5; do
    profile "warp128_coefficient_reuse${reuse_stages}_static_io_half_dual" \
        "${common[@]}" \
        --segment-cores \
            homogeneous-warp128-coefficient-reuse-static-io-radix2 \
        --segment-threads 256 --segment-units 8 --segment-data-space 4 \
        --segment-coefficient-reuse-stages "$reuse_stages" \
        --segment-cta-weights 8,7 \
        --target-ctas-per-sm 2
done
profile warp128_vector_radix4_static_io_half_dual "${common[@]}" \
    --segment-cores homogeneous-warp128-vector-radix4-static-io \
    --segment-threads 256 --segment-units 8 --segment-data-space 4 \
    --segment-cta-weights 8,7 \
    --target-ctas-per-sm 2
profile warp64_static_io_half "${common[@]}" \
    --segment-cores homogeneous-warp64-static-io-radix2 \
    --segment-threads 128 --segment-units 4 --segment-data-space 4 \
    --target-ctas-per-sm 4
profile warp128_online_static_io_half "${common[@]}" \
    --segment-cores homogeneous-warp128-pipeline-static-io-radix2 \
    --segment-threads 128 --segment-units 4 --segment-data-space 4 \
    --pipeline-buffers 1 --target-ctas-per-sm 4
profile warp128_cooperative_static_io_half_occ4 "${common[@]}" \
    --segment-cores homogeneous-warp128-cooperative-static-io-radix2 \
    --segment-threads 128 --segment-units 4 --segment-data-space 4 \
    --target-ctas-per-sm 4
profile warp128_cooperative_static_io_half_occ3 "${common[@]}" \
    --segment-cores homogeneous-warp128-cooperative-static-io-radix2 \
    --segment-threads 128 --segment-units 4 --segment-data-space 4 \
    --target-ctas-per-sm 3

python3 "$ROOT/scripts/summarize_ncu.py" \
    "$OUTPUT_DIR/v06.csv" \
    "$OUTPUT_DIR/warp256_static_io_half.csv" \
    "$OUTPUT_DIR/warp256_coefficient_reuse1_static_io_half.csv" \
    "$OUTPUT_DIR/warp256_coefficient_reuse2_static_io_half.csv" \
    "$OUTPUT_DIR/warp256_coefficient_reuse3_static_io_half.csv" \
    "$OUTPUT_DIR/warp256_coefficient_reuse4_static_io_half.csv" \
    "$OUTPUT_DIR/warp256_coefficient_reuse5_static_io_half.csv" \
    "$OUTPUT_DIR/warp256_coefficient_reuse_static_io_half.csv" \
    "$OUTPUT_DIR/warp128_static_io_half.csv" \
    "$OUTPUT_DIR/warp128_static_io_half_dual.csv" \
    "$OUTPUT_DIR/warp128_coefficient_reuse1_static_io_half_dual.csv" \
    "$OUTPUT_DIR/warp128_coefficient_reuse2_static_io_half_dual.csv" \
    "$OUTPUT_DIR/warp128_coefficient_reuse3_static_io_half_dual.csv" \
    "$OUTPUT_DIR/warp128_coefficient_reuse4_static_io_half_dual.csv" \
    "$OUTPUT_DIR/warp128_coefficient_reuse5_static_io_half_dual.csv" \
    "$OUTPUT_DIR/warp128_coefficient_reuse_static_io_half_dual.csv" \
    "$OUTPUT_DIR/warp128_vector_radix4_static_io_half_dual.csv" \
    "$OUTPUT_DIR/warp64_static_io_half.csv" \
    "$OUTPUT_DIR/warp128_online_static_io_half.csv" \
    "$OUTPUT_DIR/warp128_cooperative_static_io_half_occ4.csv" \
    "$OUTPUT_DIR/warp128_cooperative_static_io_half_occ3.csv" \
    --output "$OUTPUT_DIR/summary.csv"
printf 'Small-tile NCU summary: %s\n' "$OUTPUT_DIR/summary.csv"
