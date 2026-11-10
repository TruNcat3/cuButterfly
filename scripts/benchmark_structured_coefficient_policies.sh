#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build/cubutterfly_bench"}
OUTPUT=${OUTPUT:-"$ROOT/results/structured_broadcast_policy_v100_raw.csv"}
LOG_NS=${LOG_NS:-"8 10 12 15"}
TRIALS=${TRIALS:-5}
TOTAL_LOG_POINTS=${TOTAL_LOG_POINTS:-22}
WARMUP=${WARMUP:-20}
REPEAT=${REPEAT:-100}

if [[ ! -x "$BIN" ]]; then
    echo "benchmark executable not found: $BIN" >&2
    exit 1
fi

first=1
: > "$OUTPUT"

run_case() {
    local policy=$1
    local log_n=$2
    local trial=$3
    local batch=$(( (1 << TOTAL_LOG_POINTS) / (1 << log_n) ))
    local args=(--operator structured-2x2 --precision fp32 --backend temporal-tile
                --local-exchange warp-register --compute-unit radix2
                --logN "$log_n" --batch "$batch" --warmup "$WARMUP" --repeat "$REPEAT" --csv)
    if [[ $policy == broadcast ]]; then
        args+=(--stage-matrix 1,0.25,-0.5,1)
    else
        for ((stage=0; stage<log_n; ++stage)); do
            args+=(--stage-matrix 1,0.25,-0.5,1)
        done
    fi

    local result
    result=$("$BIN" "${args[@]}")
    if (( first )); then
        printf 'trial,coefficient_policy,%s\n' "$(printf '%s\n' "$result" | head -n 1)" >> "$OUTPUT"
        first=0
    fi
    printf '%s,%s,%s\n' "$trial" "$policy" "$(printf '%s\n' "$result" | tail -n 1)" >> "$OUTPUT"
}

for log_n in $LOG_NS; do
    for ((trial=1; trial<=TRIALS; ++trial)); do
        if (( trial % 2 )); then
            policies=(broadcast per-stage)
        else
            policies=(per-stage broadcast)
        fi
        for policy in "${policies[@]}"; do
            run_case "$policy" "$log_n" "$trial"
        done
    done
done

python3 "$ROOT/scripts/summarize_structured_coefficient_policies.py" "$OUTPUT" \
    --output "${OUTPUT%_raw.csv}_summary.csv"
