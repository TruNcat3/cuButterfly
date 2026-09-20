#!/usr/bin/env bash
set -euo pipefail

# Profile one searched cuButterfly FFT route and cuFFT with the same workload.
# The only privileged operation this script may request is removing Nsight
# Compute's stale global lock file.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BINARY=${BINARY:-"$ROOT/build-a100-cufftdx/cubutterfly_bench"}
GPU=${GPU:-0}
LOG_N=${LOG_N:-12}
BATCH=${BATCH:-1024}
LOCAL_STAGES=${LOCAL_STAGES:-10}
REORDER_COLUMNS=${REORDER_COLUMNS:-64}
DIRECT_BOUNDARY=${DIRECT_BOUNDARY:-direct-strided}
TILE_THREADS=${TILE_THREADS:-256}
COMPUTE_UNIT=${COMPUTE_UNIT:-radix4}
PREFIX_THREADS=${PREFIX_THREADS:-512}
SUFFIX_THREADS=${SUFFIX_THREADS:-512}
PREFIX_EPT=${PREFIX_EPT:-8}
SUFFIX_EPT=${SUFFIX_EPT:-8}
OUT_DIR=${OUT_DIR:-"$ROOT/results/ncu_compare_log${LOG_N}"}
LOCK_FILE=${NCU_LOCK_FILE:-/tmp/nsight-compute-lock}

if [[ ! -x "$BINARY" ]]; then
    echo "benchmark binary not found: $BINARY" >&2
    exit 1
fi
if ! command -v ncu >/dev/null 2>&1; then
    echo "ncu is not installed or not on PATH" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

if [[ -e "$LOCK_FILE" ]]; then
    if [[ ! -w "$LOCK_FILE" ]]; then
        echo "Nsight Compute lock is not writable: $LOCK_FILE"
        echo "Run this exact command with sudo, then rerun this script:"
        echo "  sudo rm -f '$LOCK_FILE'"
        exit 2
    fi
    echo "Nsight Compute lock exists and is writable: $LOCK_FILE"
    echo "Remove it only if no other ncu process is running:"
    echo "  rm -f '$LOCK_FILE'"
    exit 2
fi

profile_one() {
    local name=$1
    shift
    CUDA_VISIBLE_DEVICES="$GPU" ncu \
        --target-processes all \
        --kernel-name-base demangled \
        --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct \
        --csv \
        --log-file "$OUT_DIR/${name}.csv" \
        "$BINARY" "$@"
}

profile_one cubutterfly_online \
    --operator fft --backend online-reorder --logN "$LOG_N" --batch "$BATCH" \
    --local-stages "$LOCAL_STAGES" --reorder-columns "$REORDER_COLUMNS" \
    --tile-threads "$TILE_THREADS" --compute-unit "$COMPUTE_UNIT" \
    --direct-boundary "$DIRECT_BOUNDARY" \
    --prefix-threads "$PREFIX_THREADS" --suffix-threads "$SUFFIX_THREADS" \
    --prefix-ept "$PREFIX_EPT" --suffix-ept "$SUFFIX_EPT" \
    --warmup 0 --repeat 1 --verify --csv

profile_one cufft \
    --operator fft --backend cufft --logN "$LOG_N" --batch "$BATCH" \
    --warmup 0 --repeat 1 --verify --csv

echo "Nsight Compute profiles written to $OUT_DIR"
