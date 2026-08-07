#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BUTTERFLY_BIN=${BUTTERFLY_BIN:-"$ROOT/build/cubutterfly_bench"}
NTT_BIN=${NTT_BIN:-"$ROOT/build/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_numeric_boundaries"}
METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,smsp__sass_thread_inst_executed_op_fp64_pred_on.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block,launch__waves_per_multiprocessor,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps}

for executable in "$NCU" "$BUTTERFLY_BIN" "$NTT_BIN"; do
    [[ -x "$executable" ]] || { echo "missing executable: $executable" >&2; exit 1; }
done
mkdir -p "$OUTPUT_DIR"

restore_owner() {
    if [[ ${EUID:-$(id -u)} -eq 0 && -n ${SUDO_USER:-} ]]; then
        chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$OUTPUT_DIR"
    fi
}
trap restore_owner EXIT

profile() {
    local label=$1
    shift
    "$NCU" --target-processes all --replay-mode kernel --cache-control none --clock-control base \
        --launch-count 4 --metrics "$METRICS" --page raw --csv --force-overwrite \
        --log-file "$OUTPUT_DIR/${label}.csv" "$@"
}

profile crossover_009_not_confirmed_fft_fp16_log15_b5_online_r4_t256 "$BUTTERFLY_BIN" --operator fft --precision fp16 --accumulation fp32 --backend online-reorder --compute-unit radix4 --tile-threads 256 --normalization none --local-stages 8 --reorder-columns 1 --logN 15 --batch 5 --warmup 0 --repeat 1 --csv
profile crossover_009_not_confirmed_fft_fp16_log15_b5_online_r2_t256 "$BUTTERFLY_BIN" --operator fft --precision fp16 --accumulation fp32 --backend online-reorder --compute-unit radix2 --tile-threads 256 --normalization none --local-stages 8 --reorder-columns 1 --logN 15 --batch 5 --warmup 0 --repeat 1 --csv
profile crossover_026_confirmed_reversed_structured_2x2_fp32_log15_b8_hier_r4_t256 "$BUTTERFLY_BIN" --operator structured-2x2 --precision fp32 --accumulation native --backend hierarchical --compute-unit radix4 --tile-threads 256 --normalization none --local-stages 8 --stage-matrix 0.9238795,-0.3826834,0.3826834,0.9238795 --logN 15 --batch 8 --warmup 0 --repeat 1 --csv
profile crossover_026_confirmed_reversed_structured_2x2_fp32_log15_b8_online_r4_t256 "$BUTTERFLY_BIN" --operator structured-2x2 --precision fp32 --accumulation native --backend online-reorder --compute-unit radix4 --tile-threads 256 --normalization none --local-stages 8 --reorder-columns 1 --stage-matrix 0.9238795,-0.3826834,0.3826834,0.9238795 --logN 15 --batch 8 --warmup 0 --repeat 1 --csv
profile crossover_044_not_confirmed_subset_zeta_uint32_log8_b1281_temporal_r2_t128 "$BUTTERFLY_BIN" --operator subset-zeta --precision uint32 --accumulation native --backend temporal-tile --compute-unit radix2 --tile-threads 128 --normalization none --logN 8 --batch 1281 --warmup 0 --repeat 1 --csv
profile crossover_044_not_confirmed_subset_zeta_uint32_log8_b1281_temporal_r4_t128 "$BUTTERFLY_BIN" --operator subset-zeta --precision uint32 --accumulation native --backend temporal-tile --compute-unit radix4 --tile-threads 128 --normalization none --logN 8 --batch 1281 --warmup 0 --repeat 1 --csv
profile crossover_053_confirmed_reversed_fft_fp16_log8_b1281_temporal_r4_t128 "$BUTTERFLY_BIN" --operator fft --precision fp16 --accumulation fp32 --backend temporal-tile --compute-unit radix4 --tile-threads 128 --normalization none --logN 8 --batch 1281 --warmup 0 --repeat 1 --csv
profile crossover_053_confirmed_reversed_fft_fp16_log8_b1281_temporal_r2_t128 "$BUTTERFLY_BIN" --operator fft --precision fp16 --accumulation fp32 --backend temporal-tile --compute-unit radix2 --tile-threads 128 --normalization none --logN 8 --batch 1281 --warmup 0 --repeat 1 --csv
profile crossover_064_confirmed_reversed_ntt_uint64_log15_b241_ntt_hybrid_r4_t256 "$NTT_BIN" --backend hybrid2d --compute-unit radix4 --threads-per-block 256 --word-bits 64 --cross-twiddle fused --output-order natural --logN 15 --batch 241 --modulus 1152921504606748673 --warmup 0 --repeat 1 --csv
profile crossover_064_confirmed_reversed_ntt_uint64_log15_b241_ntt_hybrid_r2_t256 "$NTT_BIN" --backend hybrid2d --compute-unit radix2 --threads-per-block 256 --word-bits 64 --cross-twiddle fused --output-order natural --logN 15 --batch 241 --modulus 1152921504606748673 --warmup 0 --repeat 1 --csv
profile crossover_069_not_confirmed_fft_fp64_log8_b961_temporal_r2_t128 "$BUTTERFLY_BIN" --operator fft --precision fp64 --accumulation native --backend temporal-tile --compute-unit radix2 --tile-threads 128 --normalization none --logN 8 --batch 961 --warmup 0 --repeat 1 --csv
profile crossover_069_not_confirmed_fft_fp64_log8_b961_temporal_r4_t128 "$BUTTERFLY_BIN" --operator fft --precision fp64 --accumulation native --backend temporal-tile --compute-unit radix4 --tile-threads 128 --normalization none --logN 8 --batch 961 --warmup 0 --repeat 1 --csv

python3 "$ROOT/scripts/summarize_ncu.py" "$OUTPUT_DIR"/*.csv --output "$OUTPUT_DIR/summary.csv"
python3 "$ROOT/scripts/analyze_numeric_boundary_ncu.py" "$OUTPUT_DIR/summary.csv" \
    --output "$OUTPUT_DIR/analysis.csv" --report "$OUTPUT_DIR/analysis.md"
printf "Numeric-boundary NCU attribution: %s\n" "$OUTPUT_DIR/analysis.md"
