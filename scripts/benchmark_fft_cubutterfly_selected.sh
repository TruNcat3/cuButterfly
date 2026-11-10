#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build/cubutterfly_bench"}
OUTPUT=${OUTPUT:-"$ROOT/results/fft_cubutterfly_selected_v100_raw.csv"}
WARMUP=${WARMUP:-20}
REPEAT=${REPEAT:-100}
TRIALS=${TRIALS:-5}
PLACEMENT=${PLACEMENT:-in-place}
REQUIRE_EXCLUSIVE_GPU=${REQUIRE_EXCLUSIVE_GPU:-0}
POINTS_FILE=${POINTS_FILE:-}

if [[ "$PLACEMENT" != in-place && "$PLACEMENT" != out-of-place ]]; then
    printf 'PLACEMENT must be in-place or out-of-place\n' >&2
    exit 2
fi

require_exclusive_gpu() {
    (( REQUIRE_EXCLUSIVE_GPU )) || return 0
    local visible=${CUDA_VISIBLE_DEVICES:-0}
    local gpu=${visible%%,*}
    local processes
    processes=$(nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)
    if [[ -n "$processes" ]]; then
        printf 'Target GPU %s is not exclusive; active compute processes:\n%s\n' "$gpu" "$processes" >&2
        return 1
    fi
}

if [[ ! -x "$BIN" ]]; then
    printf 'Missing %s. Build cubutterfly_bench first.\n' "$BIN" >&2
    exit 1
fi

run_point() {
    local log_n=$1
    local batch=$2
    shift 2
    "$BIN" --operator fft --precision fp32 --logN "$log_n" --batch "$batch" \
        --placement "$PLACEMENT" --normalization none --warmup "$WARMUP" --repeat "$REPEAT" \
        --verify --csv "$@"
}

mkdir -p "$(dirname "$OUTPUT")"
temporary="${OUTPUT}.tmp.$$"
trap 'rm -f "$temporary"' EXIT
rm -f "$temporary"
require_exclusive_gpu
for trial in $(seq 1 "$TRIALS"); do
    if [[ -n "$POINTS_FILE" ]]; then
        points_command=(cat "$POINTS_FILE")
    else
        points_command=(printf '%s\n' \
            '3 524288 --backend temporal-tile --compute-unit radix8 --fft-core cta-dft8 --tile-threads 32' \
            '8 16384 --backend temporal-tile --compute-unit radix4 --tile-threads 128' \
            '12 1024 --backend online-reorder --compute-unit radix4 --tile-threads 256 --local-stages 10 --reorder-columns 64' \
            '16 64 --backend online-reorder --compute-unit radix4 --tile-threads 256 --local-stages 10 --reorder-columns 8' \
            '20 4 --backend online-reorder --compute-unit radix4 --tile-threads 256 --local-stages 10 --reorder-columns 1')
    fi
    while IFS=' ' read -r log_n batch arguments; do
        read -r -a extra <<<"$arguments"
        require_exclusive_gpu
        csv=$(run_point "$log_n" "$batch" "${extra[@]}")
        require_exclusive_gpu
        if [[ ! -s "$temporary" ]]; then
            printf 'trial,%s\n' "$(printf '%s\n' "$csv" | head -n 1)" >"$temporary"
        fi
        printf '%s,%s\n' "$trial" "$(printf '%s\n' "$csv" | tail -n 1)" >>"$temporary"
    done < <("${points_command[@]}")
done

mv "$temporary" "$OUTPUT"
trap - EXIT

printf 'Wrote %s\n' "$OUTPUT"
