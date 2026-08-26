#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-cuda118/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/appt_physical_cores"}
TRIALS=${TRIALS:-3}
WARMUP=${WARMUP:-3}
REPEAT=${REPEAT:-20}
BATCHES=${BATCHES:-1,4,16}

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
raw="$OUTPUT_DIR/raw.csv"
: > "$raw"

run() {
    local trial=$1 label=$2
    shift 2
    local output header record
    output=$("$BIN" "$@" --warmup "$WARMUP" --repeat "$REPEAT" --csv)
    header=$(printf '%s\n' "$output" | sed -n '/^device,/{p;q;}')
    record=$(printf '%s\n' "$output" | tail -n 1)
    if [[ ! -s "$raw" ]]; then
        printf 'trial,variant,%s\n' "$header" > "$raw"
    fi
    printf '%s,%s,%s\n' "$trial" "$label" "$record" >> "$raw"
}

IFS=, read -ra selected_batches <<< "$BATCHES"
for bits in 32 64; do
    if [[ $bits == 32 ]]; then
        modulus=998244353; resident=3
    else
        modulus=576460756061519873; resident=2
    fi
    for batch in "${selected_batches[@]}"; do
        case "$bits:$batch" in
            32:1)  fragment=16; matched_roles=7,13,3; default_roles=8,12,2 ;;
            32:4)  fragment=16; matched_roles=6,14,2; default_roles=7,14,2 ;;
            32:16) fragment=32; matched_roles=6,16,2; default_roles=7,15,2 ;;
            64:1)  fragment=8;  matched_roles=10,9,3; default_roles=11,9,2 ;;
            64:4)  fragment=16; matched_roles=9,11,2; default_roles=11,9,2 ;;
            64:16) fragment=16; matched_roles=9,11,2; default_roles=11,9,2 ;;
            *) echo "unsupported bits/batch point: $bits/$batch" >&2; exit 1 ;;
        esac
        common=(--word-bits "$bits" --modulus "$modulus" --logN 20
                --batch "$batch")
        appt=("${common[@]}" --backend hierarchical-dataflow
              --stage-partition 7,7,6 --segment-units 8
              --segment-data-space 16 --segment-data-time 4
              --segment-role-stages 2 --segment-token-interleave 1
              --boundary-storage ring --boundary-buffers 2
              --target-ctas-per-sm "$resident"
              --appt-fragment-width "$fragment" --appt-writer-tiles 2
              --output-order natural)
        for trial in $(seq 1 "$TRIALS"); do
            run "$trial" v06 "${common[@]}" --backend hierarchical-dataflow \
                --stage-partition 10,10 --segment-cores dataflow-radix4 \
                --segment-cta-weights 9,11
            run "$trial" radix4-default "${appt[@]}" \
                --segment-cores appt-online-register-tail \
                --appt-role-weights "$default_roles"
            run "$trial" radix4-matched "${appt[@]}" \
                --segment-cores appt-online-register-tail \
                --appt-role-weights "$matched_roles"
            run "$trial" radix8-matched "${appt[@]}" \
                --segment-cores appt-online-register-tail-radix8 \
                --appt-role-weights "$matched_roles"
        done
    done
done

python3 "$ROOT/scripts/summarize_appt_physical_cores.py" "$raw" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'APPT physical-core comparison: %s\n' "$OUTPUT_DIR/analysis.md"
