#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-cuda118/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/appt_resident_quarter"}
TRIALS=${TRIALS:-2}
WARMUP=${WARMUP:-3}
REPEAT=${REPEAT:-20}
BATCHES=${BATCHES:-1,4,16}
U32_ROLES=${U32_ROLES:-"8,10,4 10,8,4 12,8,2 14,6,2"}
U64_ROLES=${U64_ROLES:-"8,8,4 10,7,3 12,6,2 14,5,1"}

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
raw="$OUTPUT_DIR/raw.csv"
: > "$raw"

run() {
    local trial=$1 variant=$2 roles=$3
    shift 3
    local output header record
    output=$("$BIN" "$@" --warmup "$WARMUP" --repeat "$REPEAT" --csv)
    header=$(printf '%s\n' "$output" | sed -n '/^device,/{p;q;}')
    record=$(printf '%s\n' "$output" | tail -n 1)
    if [[ ! -s "$raw" ]]; then
        printf 'trial,variant,scan_roles,%s\n' "$header" > "$raw"
    fi
    printf '%s,%s,%s,%s\n' "$trial" "$variant" "${roles//,/-}" \
        "$record" >> "$raw"
}

IFS=, read -ra selected_batches <<< "$BATCHES"
for bits in 32 64; do
    if [[ $bits == 32 ]]; then
        modulus=998244353; role_space=$U32_ROLES; writer_resident=3
    else
        modulus=576460756061519873; role_space=$U64_ROLES; writer_resident=2
    fi
    for batch in "${selected_batches[@]}"; do
        case "$bits:$batch" in
            32:1) fragment=8; writer_tiles=1; writer_roles=6,12,4; resident_roles=10,8,4 ;;
            32:4) fragment=8; writer_tiles=2; writer_roles=6,12,4; resident_roles=12,8,2 ;;
            32:16) fragment=16; writer_tiles=2; writer_roles=6,12,4; resident_roles=12,8,2 ;;
            64:1) fragment=8; writer_tiles=2; writer_roles=8,8,4; resident_roles=8,8,4 ;;
            64:4) fragment=8; writer_tiles=2; writer_roles=8,8,4; resident_roles=12,6,2 ;;
            64:16) fragment=16; writer_tiles=2; writer_roles=8,8,4; resident_roles=12,6,2 ;;
            *) echo "unsupported bits/batch point: $bits/$batch" >&2; exit 1 ;;
        esac
        common=(--word-bits "$bits" --modulus "$modulus" --logN 20
                --batch "$batch")
        appt=("${common[@]}" --backend hierarchical-dataflow
              --stage-partition 7,7,6 --segment-units 8
              --segment-data-space 32,16,16 --segment-data-time 1
              --segment-role-stages 2 --segment-token-interleave 1
              --boundary-storage ring --boundary-buffers 2
              --output-order natural --appt-fragment-width "$fragment"
              --appt-writer-tiles "$writer_tiles")
        for trial in $(seq 1 "$TRIALS"); do
            run "$trial" v06 9-11 "${common[@]}" \
                --backend hierarchical-dataflow --stage-partition 10,10 \
                --segment-cores dataflow-radix4 --segment-cta-weights 9,11
            run "$trial" writer-final "$writer_roles" "${appt[@]}" \
                --target-ctas-per-sm "$writer_resident" \
                --segment-cores appt-online-register-tail-grouped-writer-final \
                --appt-role-weights "$writer_roles"
            run "$trial" resident-2d "$resident_roles" "${appt[@]}" \
                --target-ctas-per-sm 2 \
                --segment-cores appt-online-register-tail-grouped-writer-final-resident \
                --appt-role-weights "$resident_roles" --appt-data-time-roles 5
            for roles in $role_space; do
                run "$trial" resident-quarter "$roles" "${appt[@]}" \
                    --target-ctas-per-sm 2 \
                    --segment-cores appt-online-register-tail-grouped-writer-final-resident-quarter \
                    --appt-role-weights "$roles" --appt-data-time-roles 5
            done
        done
    done
done

python3 "$ROOT/scripts/summarize_appt_resident_quarter.py" "$raw" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'APPT resident-quarter comparison: %s\n' "$OUTPUT_DIR/analysis.md"
