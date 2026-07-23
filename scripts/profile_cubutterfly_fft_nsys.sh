#!/usr/bin/env bash
set -euo pipefail

LOG_N=${LOG_N:-20}
BATCH=${BATCH:-4}
LOCAL_STAGES=${LOCAL_STAGES:-10}
TILE_THREADS=${TILE_THREADS:-256}
REORDER_COLUMNS=${REORDER_COLUMNS:-1}
NSYS=${NSYS:-/usr/local/cuda-11.8/bin/nsys}
BIN=${BIN:-./build/cubutterfly_bench}
OUTPUT_DIR=${OUTPUT_DIR:-results/nsys_fft}

mkdir -p "$OUTPUT_DIR"

profile() {
    local label=$1
    shift
    "$NSYS" profile --force-overwrite=true --sample=none --trace=cuda,nvtx \
        --output="$OUTPUT_DIR/${label}_logN${LOG_N}" "$@"
    "$NSYS" stats --force-overwrite=true --force-export=true --report gpukernsum --format csv \
        --output "$OUTPUT_DIR/${label}_logN${LOG_N}_kernel_summary" \
        "$OUTPUT_DIR/${label}_logN${LOG_N}.nsys-rep"
    "$NSYS" stats --force-overwrite=true --force-export=true --report gputrace --format csv \
        --output "$OUTPUT_DIR/${label}_logN${LOG_N}_trace" \
        "$OUTPUT_DIR/${label}_logN${LOG_N}.nsys-rep"
}

COMMON=(
    "$BIN" --operator fft --precision fp32 --logN "$LOG_N" --batch "$BATCH"
    --warmup 0 --repeat 1 --csv
)

profile cubutterfly_fft "${COMMON[@]}" --backend hierarchical --compute-unit radix4 \
    --local-stages "$LOCAL_STAGES" --tile-threads "$TILE_THREADS"
profile online_reorder_fft "${COMMON[@]}" --backend online-reorder --compute-unit radix4 \
    --local-stages "$LOCAL_STAGES" --reorder-columns "$REORDER_COLUMNS" --tile-threads "$TILE_THREADS"
profile cufft "${COMMON[@]}" --backend cufft

echo "Nsight Systems reports written to $OUTPUT_DIR"
