#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_homogeneous_10x10"}
METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block,launch__waves_per_multiprocessor}
BITS=${BITS:-32}
BATCH=${BATCH:-16}
BEST_DT=${BEST_DT:-1,2}
BEST_WEIGHTS=${BEST_WEIGHTS:-9,11}

mkdir -p "$OUTPUT_DIR"
for executable in "$NCU" "$BIN"; do
    [[ -x "$executable" ]] || { echo "missing executable: $executable" >&2; exit 1; }
done
if [[ $BITS == 32 ]]; then modulus=998244353; resident=4
else modulus=576460756061519873; resident=2
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
        --boundary-storage full-scratch --target-ctas-per-sm "$resident")
profile v06 "${common[@]}" --segment-cores dataflow-radix4 \
    --segment-data-time 1,1 --segment-cta-weights 9,11
profile homogeneous_dt1 "${common[@]}" --segment-cores homogeneous-radix4 \
    --segment-threads 256 --segment-units 4 --segment-data-time 1,1 \
    --segment-cta-weights 9,11
profile homogeneous_best "${common[@]}" --segment-cores homogeneous-radix4 \
    --segment-threads 256 --segment-units 4 --segment-data-time "$BEST_DT" \
    --segment-cta-weights "$BEST_WEIGHTS"

python3 "$ROOT/scripts/summarize_ncu.py" \
    "$OUTPUT_DIR/v06.csv" "$OUTPUT_DIR/homogeneous_dt1.csv" \
    "$OUTPUT_DIR/homogeneous_best.csv" --output "$OUTPUT_DIR/summary.csv"
printf 'Homogeneous 10+10 NCU summary: %s\n' "$OUTPUT_DIR/summary.csv"
