#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_vector_stage6_distribution"}
METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,smsp__inst_executed.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block,launch__waves_per_multiprocessor,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps}

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

common=(--logN 20 --batch 16 --word-bits 32 --modulus 998244353
        --backend hierarchical-dataflow --stage-partition 10,10
        --boundary-storage full-scratch --segment-data-time 1,1
        --cross-twiddle fused --mod-multiply shoup)
vector=(--segment-cores homogeneous-warp128-vector-radix4-static-io
        --segment-threads 256 --segment-units 8 --segment-data-space 4
        --segment-cta-weights 8,7 --target-ctas-per-sm 2)

echo "[preflight] d6 forward/inverse"
"$BIN" "${common[@]}" "${vector[@]}" \
    --segment-coefficient-reuse-stages 6 --warmup 0 --repeat 1 --verify >/dev/null
"$BIN" "${common[@]}" "${vector[@]}" \
    --segment-coefficient-reuse-stages 6 --inverse \
    --warmup 0 --repeat 1 --verify >/dev/null
echo "[preflight] d7 forward/inverse"
"$BIN" "${common[@]}" "${vector[@]}" \
    --segment-coefficient-reuse-stages 7 --warmup 0 --repeat 1 --verify >/dev/null
"$BIN" "${common[@]}" "${vector[@]}" \
    --segment-coefficient-reuse-stages 7 --inverse \
    --warmup 0 --repeat 1 --verify >/dev/null

profile() {
    local label=$1
    shift
    local output="$OUTPUT_DIR/${label}.csv"
    local temporary="$output.tmp"
    rm -f "$temporary"
    "$NCU" --target-processes all --replay-mode kernel --cache-control none \
        --clock-control base --launch-count 1 --metrics "$METRICS" \
        --page raw --csv --force-overwrite --log-file "$temporary" \
        "$BIN" "$@" --warmup 0 --repeat 1 --csv
    if ! grep -q '"Kernel Name"' "$temporary"; then
        cat "$temporary" >&2
        rm -f "$temporary"
        echo "NCU did not produce a raw CSV table for $label" >&2
        return 1
    fi
    mv "$temporary" "$output"
}

profile v06 "${common[@]}" --segment-cores dataflow-radix4 \
    --segment-cta-weights 17,13 --target-ctas-per-sm 4
profile vector_d6 "${common[@]}" "${vector[@]}" \
    --segment-coefficient-reuse-stages 6
profile vector_d7 "${common[@]}" "${vector[@]}" \
    --segment-coefficient-reuse-stages 7

python3 "$ROOT/scripts/summarize_ncu.py" \
    "$OUTPUT_DIR/v06.csv" "$OUTPUT_DIR/vector_d6.csv" \
    "$OUTPUT_DIR/vector_d7.csv" --output "$OUTPUT_DIR/summary.csv"
python3 "$ROOT/scripts/analyze_vector_stage6_distribution.py" \
    "$OUTPUT_DIR/summary.csv" --output "$OUTPUT_DIR/analysis.md"

printf 'wrote %s and %s\n' \
    "$OUTPUT_DIR/summary.csv" "$OUTPUT_DIR/analysis.md"
