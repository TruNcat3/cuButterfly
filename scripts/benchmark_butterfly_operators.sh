#!/usr/bin/env bash
set -euo pipefail

BIN=${BIN:-./build/cubutterfly_bench}
TRIALS=${TRIALS:-5}
BATCH=${BATCH:-16384}
WARMUP=${WARMUP:-20}
REPEAT=${REPEAT:-100}
OUTPUT=${OUTPUT:-results/butterfly_operators_v100_raw.csv}

if [[ ! -x "$BIN" ]]; then
    echo "cuButterfly benchmark not found: $BIN" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
: > "$OUTPUT"
first=1

run_trial() {
    local operator=$1
    local backend=$2
    local stage_space=$3
    local trial=$4
    local temp
    temp=$(mktemp)
    trap 'unlink "$temp"' EXIT
    command=("$BIN" --operator "$operator" --backend "$backend" --batch "$BATCH" --warmup "$WARMUP" --repeat "$REPEAT" --csv)
    if [[ "$backend" == "stage-pipeline" ]]; then
        command+=(--stage-space "$stage_space" --stage-handoff named-barrier)
    fi
    "${command[@]}" > "$temp"
    if ((first)); then
        awk -v trial="$trial" 'NR == 1 {print "trial," $0} NR == 2 {print trial "," $0}' "$temp" >> "$OUTPUT"
        first=0
    else
        awk -v trial="$trial" 'NR == 2 {print trial "," $0}' "$temp" >> "$OUTPUT"
    fi
    unlink "$temp"
    trap - EXIT
}

for operator in fwht xor-zeta fft; do
    for ((trial = 1; trial <= TRIALS; ++trial)); do
        run_trial "$operator" temporal-tile 0 "$trial"
    done
    for stage_space in 1 2 4 8; do
        for ((trial = 1; trial <= TRIALS; ++trial)); do
            run_trial "$operator" stage-pipeline "$stage_space" "$trial"
        done
    done
done

for ((trial = 1; trial <= TRIALS; ++trial)); do
    run_trial fft cufft 0 "$trial"
done

echo "Raw cross-operator trials written to $OUTPUT"
