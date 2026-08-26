#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build-ncu-lineinfo/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_v07_sector_attribution"}
METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block,launch__grid_size,launch__waves_per_multiprocessor,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps}

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
        [[ "$executable" == "$BIN" ]] && \
            echo "run $ROOT/scripts/build_v07_sector_attribution.sh first" >&2
        exit 1
    }
done

common=(--logN 20 --batch 16 --word-bits 32 --modulus 998244353
        --backend hierarchical-dataflow --stage-partition 10,10
        --boundary-storage full-scratch --segment-data-time 1,1
        --cross-twiddle fused --mod-multiply shoup)

v06=("${common[@]}" --segment-cores dataflow-radix4
     --segment-cta-weights 17,13 --target-ctas-per-sm 4)
vector_common=("${common[@]}"
     --segment-cores homogeneous-warp128-vector-radix4-static-io
     --segment-threads 256 --segment-units 8 --segment-data-space 4
     --segment-cta-weights 8,7 --target-ctas-per-sm 2)

printf '[preflight] verifying v0.6 and v0.7 d6\n'
"$BIN" "${v06[@]}" --warmup 0 --repeat 1 --verify --csv >/dev/null
"$BIN" "${v06[@]}" --inverse --warmup 0 --repeat 1 --verify --csv >/dev/null
"$BIN" "${vector_common[@]}" --segment-coefficient-reuse-stages 6 \
    --warmup 0 --repeat 1 --verify --csv >/dev/null
"$BIN" "${vector_common[@]}" --segment-coefficient-reuse-stages 6 \
    --inverse --warmup 0 --repeat 1 --verify --csv >/dev/null

profile_aggregate() {
    local label=$1
    shift
    local output="$OUTPUT_DIR/${label}.csv"
    local temporary="$output.tmp"
    printf '[aggregate] %s\n' "$label"
    rm -f "$temporary"
    if ! "$NCU" --target-processes all --replay-mode kernel \
        --cache-control none --clock-control base --launch-count 1 \
        --metrics "$METRICS" --page raw --csv --force-overwrite \
        --log-file "$temporary" "$BIN" "$@" --warmup 0 --repeat 1 --csv; then
        [[ -f "$temporary" ]] && cat "$temporary" >&2
        rm -f "$temporary"
        return 1
    fi
    if ! grep -q '"Kernel Name"' "$temporary"; then
        cat "$temporary" >&2
        rm -f "$temporary"
        echo "NCU did not produce aggregate CSV for $label" >&2
        return 1
    fi
    mv "$temporary" "$output"
}

profile_source() {
    local label=$1
    shift
    local report_base="$OUTPUT_DIR/${label}_source"
    local collect_log="$OUTPUT_DIR/${label}_source_collect.log"
    local source_csv="$OUTPUT_DIR/${label}_source.csv"
    printf '[source] %s (SourceCounters may require several replay passes)\n' "$label"
    rm -f "$report_base" "$report_base.ncu-rep" "$collect_log" "$source_csv"
    if ! "$NCU" --target-processes all --replay-mode kernel \
        --cache-control none --clock-control base --launch-count 1 \
        --section SourceCounters --import-source yes --force-overwrite \
        --export "$report_base" --log-file "$collect_log" \
        "$BIN" "$@" --warmup 0 --repeat 1 --csv; then
        cat "$collect_log" >&2
        return 1
    fi
    local report="$report_base.ncu-rep"
    [[ -f "$report" ]] || report="$report_base"
    if ! "$NCU" --import "$report" --page source --print-source cuda,sass \
        --csv --log-file "$source_csv"; then
        cat "$source_csv" >&2
        return 1
    fi
    if ! grep -Eq 'Theoretical Sectors|memory_l2_theoretical_sectors_global' \
        "$source_csv"; then
        cat "$source_csv" >&2
        echo "NCU source page is missing global-sector columns for $label" >&2
        return 1
    fi
}

profile_aggregate v06 "${v06[@]}"
profile_aggregate warp128_vector_radix4_dual "${vector_common[@]}"
for reuse_stages in 3 4 5 6; do
    profile_aggregate "warp128_vector_radix4_reuse${reuse_stages}_dual" \
        "${vector_common[@]}" \
        --segment-coefficient-reuse-stages "$reuse_stages"
done

python3 "$ROOT/scripts/summarize_ncu.py" \
    "$OUTPUT_DIR/v06.csv" \
    "$OUTPUT_DIR/warp128_vector_radix4_dual.csv" \
    "$OUTPUT_DIR/warp128_vector_radix4_reuse3_dual.csv" \
    "$OUTPUT_DIR/warp128_vector_radix4_reuse4_dual.csv" \
    "$OUTPUT_DIR/warp128_vector_radix4_reuse5_dual.csv" \
    "$OUTPUT_DIR/warp128_vector_radix4_reuse6_dual.csv" \
    --output "$OUTPUT_DIR/summary.csv"

profile_source v06 "${v06[@]}"
profile_source v07_d6 "${vector_common[@]}" \
    --segment-coefficient-reuse-stages 6

python3 "$ROOT/scripts/analyze_v07_sector_attribution.py" \
    "$OUTPUT_DIR/summary.csv" \
    --source-v06 "$OUTPUT_DIR/v06_source.csv" \
    --source-v07 "$OUTPUT_DIR/v07_d6_source.csv" \
    --csv "$OUTPUT_DIR/stage_attribution.csv" \
    --pc-csv "$OUTPUT_DIR/source_pc_attribution.csv" \
    --markdown "$OUTPUT_DIR/analysis.md"

printf 'wrote %s\n' "$OUTPUT_DIR/analysis.md"
