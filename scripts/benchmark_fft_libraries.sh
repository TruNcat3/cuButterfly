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

if [[ ! -x "$BIN" ]]; then
    printf 'Missing %s. Configure with -DCUBUTTERFLY_ENABLE_VKFFT=ON first.\n' "$BIN" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
rm -f "$OUTPUT"
for trial in $(seq 1 "$TRIALS"); do
    for log_n in $LOG_NS; do
        for library in cufft vkfft; do
            csv=$(
                "$BIN" --library "$library" --logN "$log_n" --total-points "$TOTAL_POINTS" \
                    --warmup "$WARMUP" --repeat "$REPEAT" --verify --csv
            )
            if [[ ! -s "$OUTPUT" ]]; then
                printf 'trial,%s\n' "$(printf '%s\n' "$csv" | head -n 1)" >"$OUTPUT"
            fi
            printf '%s,%s\n' "$trial" "$(printf '%s\n' "$csv" | tail -n 1)" >>"$OUTPUT"
        done
    done
done

printf 'Wrote %s\n' "$OUTPUT"
