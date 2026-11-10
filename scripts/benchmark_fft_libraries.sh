#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build/vkfft_bench"}
OUTPUT=${OUTPUT:-"$ROOT/results/fft_libraries_v100_raw.csv"}
LOG_NS=${LOG_NS:-"3 8 12 14 16 18 20"}
TOTAL_POINTS=${TOTAL_POINTS:-4194304}
WARMUP=${WARMUP:-20}
REPEAT=${REPEAT:-100}
TRIALS=${TRIALS:-5}
PLACEMENT=${PLACEMENT:-in-place}
REQUIRE_EXCLUSIVE_GPU=${REQUIRE_EXCLUSIVE_GPU:-0}

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
    printf 'Missing %s. Configure with -DCUBUTTERFLY_ENABLE_VKFFT=ON first.\n' "$BIN" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
temporary="${OUTPUT}.tmp.$$"
trap 'rm -f "$temporary"' EXIT
rm -f "$temporary"
require_exclusive_gpu
for trial in $(seq 1 "$TRIALS"); do
    for log_n in $LOG_NS; do
        for library in cufft vkfft; do
            require_exclusive_gpu
            csv=$(
                "$BIN" --library "$library" --logN "$log_n" --total-points "$TOTAL_POINTS" \
                    --placement "$PLACEMENT" --warmup "$WARMUP" --repeat "$REPEAT" --verify --csv
            )
            require_exclusive_gpu
            if [[ ! -s "$temporary" ]]; then
                printf 'trial,%s\n' "$(printf '%s\n' "$csv" | head -n 1)" >"$temporary"
            fi
            printf '%s,%s\n' "$trial" "$(printf '%s\n' "$csv" | tail -n 1)" >>"$temporary"
        done
    done
done
mv "$temporary" "$OUTPUT"
trap - EXIT

printf 'Wrote %s\n' "$OUTPUT"
