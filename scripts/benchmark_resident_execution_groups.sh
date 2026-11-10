#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/resident_execution_groups"}
WARMUP=${WARMUP:-10}
REPEAT=${REPEAT:-50}
TRIALS=${TRIALS:-3}
BATCHES=${BATCHES:-"1 4 16"}
MIN_SEGMENTS=${MIN_SEGMENTS:-2}
MAX_SEGMENTS=${MAX_SEGMENTS:-8}
EXECUTION_GROUPS=${EXECUTION_GROUPS:-}
INCLUDE_MATCHED_CONTROLS=${INCLUDE_MATCHED_CONTROLS:-1}
SEGMENT_CORE=${SEGMENT_CORE:-radix4}
EXECUTION_GROUP_CORE=${EXECUTION_GROUP_CORE:-}
TARGET_CTAS_PER_SM=${TARGET_CTAS_PER_SM:-2}
export EXECUTION_GROUPS INCLUDE_MATCHED_CONTROLS
mkdir -p "$OUTPUT_DIR"
SPACE="$OUTPUT_DIR/space.json"
RAW="$OUTPUT_DIR/raw.csv"
rm -f "$RAW"

python3 "$ROOT/scripts/generate_ntt_partition_space.py" \
    --logN 20 --batch 1 --word-bits 32 \
    --min-segments "$MIN_SEGMENTS" --max-segments "$MAX_SEGMENTS" \
    --top-per-count 1 --output "$SPACE"

mapfile -t points < <(python3 - "$SPACE" <<'PY'
import json
import os
import sys
with open(sys.argv[1]) as handle:
    document = json.load(handle)
entries = {}
allowed_g = {int(value) for value in os.environ["EXECUTION_GROUPS"].split(',')
             if value}
include_controls = os.environ["INCLUDE_MATCHED_CONTROLS"] != "0"
for point in document["execution_shortlist"]:
    realization = point["execution_template"]
    if allowed_g and realization["execution_group_count"] not in allowed_g:
        continue
    key = (tuple(point["stage_partition"]),
           realization["execution_group_count"],
           tuple(realization["execution_stage_partition"]))
    entries[key] = (point, realization)
    if (include_controls and
            realization["execution_group_count"] < point["segment_count"]):
        control = {
            "execution_group_count": point["segment_count"],
            "execution_stage_partition": point["stage_partition"],
            "boundary_storage": ["full-scratch"] *
                                (point["segment_count"] - 1),
        }
        control_key = (tuple(point["stage_partition"]),
                       point["segment_count"],
                       tuple(point["stage_partition"]))
        entries[control_key] = (point, control)
for _, (point, realization) in sorted(entries.items()):
    partition = ",".join(map(str, point["stage_partition"]))
    weights = ",".join(map(str, point["default_cta_weights"]))
    storage = ",".join(realization["boundary_storage"])
    execution = "-".join(map(str, realization["execution_stage_partition"]))
    print(f"{point['segment_count']}|{realization['execution_group_count']}|{partition}|{execution}|{weights}|{storage}")
PY
)

append() {
    local label=$1
    shift
    local output
    output=$("$BIN" "$@" --warmup "$WARMUP" --repeat "$REPEAT" --csv)
    if [[ ! -s "$RAW" ]]; then
        printf 'label,%s\n' "$(printf '%s\n' "$output" | sed -n '1p')" >"$RAW"
    fi
    printf '%s,%s\n' "$label" "$(printf '%s\n' "$output" | sed -n '2p')" >>"$RAW"
}

common=(--logN 20 --word-bits 32 --modulus 998244353
        --backend hierarchical-dataflow --segment-cores "$SEGMENT_CORE"
        --segment-threads 256 --target-ctas-per-sm "$TARGET_CTAS_PER_SM"
        --cross-twiddle fused --mod-multiply shoup)
if [[ -n "$EXECUTION_GROUP_CORE" ]]; then
    common+=(--execution-group-cores "$EXECUTION_GROUP_CORE")
fi

for batch in $BATCHES; do
    for ((trial = 1; trial <= TRIALS; ++trial)); do
        offset=$((trial % ${#points[@]}))
        for ((index = 0; index < ${#points[@]}; ++index)); do
            point=${points[$(((index + offset) % ${#points[@]}))]}
            IFS='|' read -r m g partition execution weights storage <<<"$point"
            append "m${m}_g${g}_p${partition//,/-}_e${execution}_b${batch}_t${trial}" \
                "${common[@]}" --batch "$batch" \
                --stage-partition "$partition" \
                --segment-cta-weights "$weights" \
                --boundary-storage "$storage"
        done
    done
done

python3 "$ROOT/scripts/summarize_resident_execution_groups.py" "$RAW" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'wrote %s\n' "$OUTPUT_DIR/analysis.md"
