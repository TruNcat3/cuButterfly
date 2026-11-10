#!/usr/bin/env bash
set -euo pipefail

BIN=${BIN:-./build/cuntt_bench}
TRIALS=${TRIALS:-5}
BATCH=${BATCH:-16384}
WARMUP=${WARMUP:-20}
REPEAT=${REPEAT:-100}
OUTPUT=${OUTPUT:-results/stage_pipeline_ntt256_v100_raw.csv}

if [[ ! -x "$BIN" ]]; then
    echo "cuNTT benchmark not found: $BIN" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
: > "$OUTPUT"
first=1
for backend in tile256 stage-pipeline; do
    stage_spaces=(0)
    handoffs=(atomic)
    if [[ "$backend" == "stage-pipeline" ]]; then
        stage_spaces=(1 2 4 8)
        handoffs=(atomic named-barrier)
    fi
    for handoff in "${handoffs[@]}"; do
      for stage_space in "${stage_spaces[@]}"; do
        for ((trial = 1; trial <= TRIALS; ++trial)); do
            temp=$(mktemp)
            trap 'unlink "$temp"' EXIT
            command=("$BIN" --logN 8 --batch "$BATCH" --backend "$backend" --stage-handoff "$handoff" --warmup "$WARMUP" --repeat "$REPEAT" --csv)
            if [[ "$backend" == "stage-pipeline" ]]; then
                command+=(--stage-space "$stage_space")
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
        done
      done
    done
done

echo "Raw NTT256 stage-pipeline trials written to $OUTPUT"
