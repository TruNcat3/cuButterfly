#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build-11.8/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_hierarchical_cores"}
METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block,launch__waves_per_multiprocessor,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps}

for executable in "$NCU" "$BIN"; do
    [[ -x "$executable" ]] || { echo "missing executable: $executable" >&2; exit 1; }
done
mkdir -p "$OUTPUT_DIR"
reports=()

restore_owner() {
    if [[ ${EUID:-$(id -u)} -eq 0 && -n ${SUDO_USER:-} ]]; then
        chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$OUTPUT_DIR"
    fi
}
trap restore_owner EXIT

profile() {
    local label=$1
    shift
    local report="$OUTPUT_DIR/${label}.csv"
    "$NCU" --target-processes all --replay-mode kernel --cache-control none --clock-control base \
        --launch-count 1 --metrics "$METRICS" --page raw --csv --force-overwrite \
        --log-file "$report" "$BIN" "$@" --warmup 0 --repeat 1 --csv
    reports+=("$report")
}

for word_bits in 32 64; do
    if [[ $word_bits -eq 32 ]]; then modulus=998244353; rows=4; threads=256
    else modulus=576460756061519873; rows=2; threads=512; fi
    for core in dataflow-radix4 hybrid2d-radix4; do
        profile "w${word_bits}_${core}" --word-bits "$word_bits" --modulus "$modulus" \
            --logN 20 --batch 4 --backend hierarchical-barrier --hierarchical-core "$core" \
            --rows-per-block "$rows" --data-time 1 --threads-per-block "$threads"
    done
done

python3 "$ROOT/scripts/summarize_ncu.py" "${reports[@]}" --output "$OUTPUT_DIR/summary.csv"
python3 "$ROOT/scripts/analyze_hierarchical_cores_ncu.py" "$OUTPUT_DIR/summary.csv" \
    --csv "$OUTPUT_DIR/analysis.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'Hierarchical core NCU summary: %s\n' "$OUTPUT_DIR/summary.csv"
