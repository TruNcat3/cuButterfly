#!/usr/bin/env bash
set -euo pipefail

BIN=${BIN:-./build/cuntt_hardware_microbench}
TRIALS=${TRIALS:-5}
OUTPUT=${OUTPUT:-results/hardware_capabilities_raw.csv}

if [[ ! -x "$BIN" ]]; then
    echo "hardware microbenchmark not found: $BIN" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
: > "$OUTPUT"
for ((trial = 1; trial <= TRIALS; ++trial)); do
    temp=$(mktemp)
    trap 'unlink "$temp"' EXIT
    "$BIN" "$@" > "$temp"
    if (( trial == 1 )); then
        awk -v trial="$trial" 'NR == 1 {print "trial," $0} NR == 2 {print trial "," $0}' "$temp" >> "$OUTPUT"
    else
        awk -v trial="$trial" 'NR == 2 {print trial "," $0}' "$temp" >> "$OUTPUT"
    fi
    unlink "$temp"
    trap - EXIT
done

echo "Raw capability trials written to $OUTPUT"
