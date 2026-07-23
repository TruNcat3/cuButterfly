#!/usr/bin/env bash
set -euo pipefail

LOG_N=${1:-16}
BATCH=${2:-64}
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
CUNTT_BIN=${CUNTT_BIN:-./build/cuntt_bench}
GPUNTT_BIN=${GPUNTT_BIN:-}
PROFILE_CUNTT=${PROFILE_CUNTT:-1}
PROFILE_COMPACT=${PROFILE_COMPACT:-0}
WARMUP=${WARMUP:-1000}
REPEAT=${REPEAT:-1}
OUTPUT_DIR=${OUTPUT_DIR:-results/ncu}
MODULUS=${MODULUS:-576460756061519873}

if [[ ${3:-} == "--gpuntt-only" ]]; then
    if [[ -z ${4:-} ]]; then
        echo "--gpuntt-only requires the GPU-NTT benchmark path" >&2
        exit 1
    fi
    PROFILE_CUNTT=0
    GPUNTT_BIN=$4
elif [[ ${3:-} == "--compact-only" ]]; then
    PROFILE_CUNTT=0
    PROFILE_COMPACT=1
elif [[ -n ${3:-} ]]; then
    echo "unknown argument: $3" >&2
    exit 1
fi

if [[ ! -x "$NCU" ]]; then
    echo "ncu not found: $NCU" >&2
    exit 1
fi
if [[ ! -x "$CUNTT_BIN" ]]; then
    echo "cuNTT benchmark not found: $CUNTT_BIN" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

profile() {
    local label=$1
    local skip=$2
    local count=$3
    shift 3
    echo "Profiling $label (skip=$skip, count=$count)"
    "$NCU" \
        --target-processes all \
        --replay-mode kernel \
        --cache-control none \
        --clock-control base \
        --launch-skip "$skip" \
        --launch-count "$count" \
        --metrics "$METRICS" \
        --page raw \
        --csv \
        --force-overwrite \
        --log-file "$OUTPUT_DIR/${label}_logN${LOG_N}.csv" \
        "$@"
}

NCU_METRICS=(
    gpu__time_duration.sum
    dram__bytes_read.sum
    dram__bytes_write.sum
    dram__throughput.avg.pct_of_peak_sustained_elapsed
    lts__t_sector_hit_rate.pct
    l1tex__t_sector_hit_rate.pct
    smsp__inst_executed.sum
    smsp__sass_thread_inst_executed_op_integer_pred_on.sum
    sm__warps_active.avg.pct_of_peak_sustained_active
    smsp__warp_issue_stalled_barrier_per_warp_active.pct
    smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
    launch__registers_per_thread
    launch__shared_mem_per_block
)
METRICS=$(IFS=,; echo "${NCU_METRICS[*]}")

CUNTT_COMMON=(
    "$CUNTT_BIN"
    --logN "$LOG_N"
    --batch "$BATCH"
    --backend hybrid2d
    --word-bits 64
    --modulus "$MODULUS"
    --warmup "$WARMUP"
    --repeat "$REPEAT"
    --csv
)

# Keep the cuNTT A/B mapping identical so the counter delta isolates twiddle
# organization rather than block geometry.
case "$LOG_N" in
    16)
        CUNTT_COMMON+=(--n1-log 8 --rows-per-block 4 --threads-per-block 256)
        ;;
    18)
        CUNTT_COMMON+=(--n1-log 9 --rows-per-block 4 --threads-per-block 512)
        ;;
    20)
        CUNTT_COMMON+=(--n1-log 10 --rows-per-block 2 --threads-per-block 512)
        ;;
esac

if (( PROFILE_CUNTT != 0 )); then
    profile cuntt_first "$((WARMUP * 2))" 2 "${CUNTT_COMMON[@]}" --cross-twiddle first
    profile cuntt_fused "$((WARMUP * 2))" 2 "${CUNTT_COMMON[@]}" --cross-twiddle fused
    profile cuntt_fused_barrett "$((WARMUP * 2))" 2 "${CUNTT_COMMON[@]}" --cross-twiddle fused --mod-multiply barrett
fi

if (( PROFILE_COMPACT != 0 )); then
    if (( LOG_N != 20 )); then
        echo "PROFILE_COMPACT=1 currently requires logN=20" >&2
        exit 1
    fi
    profile cuntt_compact "$((WARMUP * 3))" 3 \
        "$CUNTT_BIN" \
        --logN "$LOG_N" \
        --batch "$BATCH" \
        --backend compact-stage \
        --output-order bit-reversed \
        --word-bits 64 \
        --modulus "$MODULUS" \
        --warmup "$WARMUP" \
        --repeat "$REPEAT" \
        --csv
fi

if [[ -n "$GPUNTT_BIN" ]]; then
    if [[ ! -x "$GPUNTT_BIN" ]]; then
        echo "GPU-NTT benchmark not found: $GPUNTT_BIN" >&2
        exit 1
    fi
    if (( LOG_N <= 16 )); then
        GPUNTT_KERNELS=2
    else
        GPUNTT_KERNELS=3
    fi
    profile gpuntt_merge "$((WARMUP * GPUNTT_KERNELS))" "$GPUNTT_KERNELS" \
        "$GPUNTT_BIN" "$LOG_N" "$BATCH" "$WARMUP" "$REPEAT"
else
    echo "GPUNTT_BIN is unset; skipping GPU-NTT. Set it to the comparator harness binary to include it."
fi

echo "Raw NCU CSV files are in $OUTPUT_DIR"
