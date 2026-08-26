#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/stage_count_space"}
WARMUP=${WARMUP:-10}
REPEAT=${REPEAT:-50}
TRIALS=${TRIALS:-3}
BATCHES=${BATCHES:-"1 4 16"}
MIN_SEGMENTS=${MIN_SEGMENTS:-2}
MAX_SEGMENTS=${MAX_SEGMENTS:-8}
PREFERRED_STAGE_LOGS=${PREFERRED_STAGE_LOGS:-"6,8,10"}
mkdir -p "$OUTPUT_DIR"
SPACE="$OUTPUT_DIR/space.json"
RAW="$OUTPUT_DIR/raw.csv"
rm -f "$RAW"

python3 "$ROOT/scripts/generate_ntt_partition_space.py" \
    --logN 20 --batch 1 --word-bits 32 \
    --min-segments "$MIN_SEGMENTS" --max-segments "$MAX_SEGMENTS" \
    --preferred-stage-logs "$PREFERRED_STAGE_LOGS" --top-per-count 1 \
    --output "$SPACE"

mapfile -t points < <(python3 - "$SPACE" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    document = json.load(handle)
for point in document["shortlist"]:
    partition = ",".join(map(str, point["stage_partition"]))
    weights = ",".join(map(str, point["default_cta_weights"]))
    print(f"{point['segment_count']}|{partition}|{weights}")
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
        --backend hierarchical-dataflow --segment-cores radix4
        --segment-threads 256 --target-ctas-per-sm 2
        --boundary-storage full-scratch --cross-twiddle fused
        --mod-multiply shoup)

for batch in $BATCHES; do
    for ((trial = 1; trial <= TRIALS; ++trial)); do
        offset=$((trial % ${#points[@]}))
        for ((index = 0; index < ${#points[@]}; ++index)); do
            point=${points[$(((index + offset) % ${#points[@]}))]}
            IFS='|' read -r segments partition weights <<<"$point"
            append "m${segments}_p${partition//,/-}_b${batch}_t${trial}" \
                "${common[@]}" --batch "$batch" \
                --stage-partition "$partition" \
                --segment-cta-weights "$weights"
        done
    done
done

python3 "$ROOT/scripts/summarize_stage_count_space.py" "$RAW" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'wrote %s\n' "$OUTPUT_DIR/analysis.md"
