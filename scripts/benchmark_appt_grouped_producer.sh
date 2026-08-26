#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-cuda118/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/appt_grouped_producer"}
TRIALS=${TRIALS:-2}
WARMUP=${WARMUP:-3}
REPEAT=${REPEAT:-20}
BATCHES=${BATCHES:-1,4,16}
A_GROUPS=${A_GROUPS:-8,16,32}
U32_ROLES=${U32_ROLES:-"4,16,2 5,15,2 5,16,2 6,14,2 6,16,2 7,13,3"}
U64_ROLES=${U64_ROLES:-"6,12,2 7,11,2 8,10,2 9,11,2 10,9,3"}

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
raw="$OUTPUT_DIR/raw.csv"
: > "$raw"

run() {
    local trial=$1 variant=$2 group=$3 roles=$4
    shift 4
    local output header record
    output=$("$BIN" "$@" --warmup "$WARMUP" --repeat "$REPEAT" --csv)
    header=$(printf '%s\n' "$output" | sed -n '/^device,/{p;q;}')
    record=$(printf '%s\n' "$output" | tail -n 1)
    if [[ ! -s "$raw" ]]; then
        printf 'trial,variant,a_group,roles,%s\n' "$header" > "$raw"
    fi
    printf '%s,%s,%s,%s,%s\n' "$trial" "$variant" "$group" \
        "${roles//,/-}" "$record" >> "$raw"
}

IFS=, read -ra selected_batches <<< "$BATCHES"
IFS=, read -ra selected_groups <<< "$A_GROUPS"
for bits in 32 64; do
    if [[ $bits == 32 ]]; then
        modulus=998244353
        resident=3
        role_space=$U32_ROLES
    else
        modulus=576460756061519873
        resident=2
        role_space=$U64_ROLES
    fi
    for batch in "${selected_batches[@]}"; do
        case "$bits:$batch" in
            32:1)  fragment=16; matched_roles=7,13,3 ;;
            32:4)  fragment=16; matched_roles=6,14,2 ;;
            32:16) fragment=32; matched_roles=6,16,2 ;;
            64:1)  fragment=8;  matched_roles=10,9,3 ;;
            64:4|64:16) fragment=16; matched_roles=9,11,2 ;;
            *) echo "unsupported bits/batch point: $bits/$batch" >&2; exit 1 ;;
        esac
        common=(--word-bits "$bits" --modulus "$modulus" --logN 20
                --batch "$batch")
        appt=("${common[@]}" --backend hierarchical-dataflow
              --stage-partition 7,7,6 --segment-units 8
              --segment-data-time 4 --segment-role-stages 2
              --segment-token-interleave 1 --boundary-storage ring
              --boundary-buffers 2 --target-ctas-per-sm "$resident"
              --appt-fragment-width "$fragment" --appt-writer-tiles 2
              --output-order natural)
        for trial in $(seq 1 "$TRIALS"); do
            run "$trial" v06 0 9-11 "${common[@]}" \
                --backend hierarchical-dataflow --stage-partition 10,10 \
                --segment-cores dataflow-radix4 --segment-cta-weights 9,11
            run "$trial" radix4-matched 1 "$matched_roles" "${appt[@]}" \
                --segment-cores appt-online-register-tail \
                --segment-data-space 16 --appt-role-weights "$matched_roles"
            for group in "${selected_groups[@]}"; do
                for roles in $role_space; do
                    run "$trial" grouped "$group" "$roles" "${appt[@]}" \
                        --segment-cores appt-online-register-tail-grouped \
                        --segment-data-space "$group",16,16 \
                        --appt-role-weights "$roles"
                done
            done
        done
    done
done

python3 "$ROOT/scripts/summarize_appt_grouped_producer.py" "$raw" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'APPT grouped-producer comparison: %s\n' "$OUTPUT_DIR/analysis.md"
