#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-cuda118/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/appt_data_time"}
TRIALS=${TRIALS:-2}
WARMUP=${WARMUP:-3}
REPEAT=${REPEAT:-20}
BATCHES=${BATCHES:-1,4,16}
DATA_TIMES=${DATA_TIMES:-1,2,4,8}
ROLE_MASKS=${ROLE_MASKS:-4,6,7}
U32_ROLES=${U32_ROLES:-"6,12,4 6,11,5 7,11,4"}
U64_ROLES=${U64_ROLES:-"8,8,4 8,7,5 9,8,3"}

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
raw="$OUTPUT_DIR/raw.csv"
: > "$raw"

run() {
    local trial=$1 variant=$2 data_time=$3 roles=$4 role_mask=$5
    shift 5
    local output header record
    output=$("$BIN" "$@" --warmup "$WARMUP" --repeat "$REPEAT" --csv)
    header=$(printf '%s\n' "$output" | sed -n '/^device,/{p;q;}')
    record=$(printf '%s\n' "$output" | tail -n 1)
    if [[ ! -s "$raw" ]]; then
        printf 'trial,variant,scan_data_time,roles,role_mask,%s\n' "$header" > "$raw"
    fi
    printf '%s,%s,%s,%s,%s,%s\n' "$trial" "$variant" "$data_time" \
        "${roles//,/-}" "$role_mask" "$record" >> "$raw"
}

IFS=, read -ra selected_batches <<< "$BATCHES"
IFS=, read -ra selected_data_times <<< "$DATA_TIMES"
IFS=, read -ra selected_role_masks <<< "$ROLE_MASKS"
for bits in 32 64; do
    if [[ $bits == 32 ]]; then modulus=998244353; resident=3; role_space=$U32_ROLES
    else modulus=576460756061519873; resident=2; role_space=$U64_ROLES
    fi
    for batch in "${selected_batches[@]}"; do
        case "$bits:$batch" in
            32:1) fragment=8; writer_tiles=1; base_roles=6,12,4 ;;
            32:4) fragment=8; writer_tiles=2; base_roles=6,12,4 ;;
            32:16) fragment=16; writer_tiles=2; base_roles=6,12,4 ;;
            64:1|64:4) fragment=8; writer_tiles=2; base_roles=8,8,4 ;;
            64:16) fragment=16; writer_tiles=2; base_roles=8,8,4 ;;
            *) echo "unsupported bits/batch point: $bits/$batch" >&2; exit 1 ;;
        esac
        common=(--word-bits "$bits" --modulus "$modulus" --logN 20
                --batch "$batch")
        appt=("${common[@]}" --backend hierarchical-dataflow
              --stage-partition 7,7,6 --segment-units 8
              --segment-data-space 32,16,16 --segment-role-stages 2
              --segment-token-interleave 1 --boundary-storage ring
              --boundary-buffers 2 --target-ctas-per-sm "$resident"
              --appt-fragment-width "$fragment"
              --appt-writer-tiles "$writer_tiles" --output-order natural)
        for trial in $(seq 1 "$TRIALS"); do
            run "$trial" v06 0 9,11 0 "${common[@]}" \
                --backend hierarchical-dataflow --stage-partition 10,10 \
                --segment-cores dataflow-radix4 --segment-cta-weights 9,11
            run "$trial" writer-final 1 "$base_roles" 0 "${appt[@]}" \
                --segment-cores appt-online-register-tail-grouped-writer-final \
                --segment-data-time 1 --appt-role-weights "$base_roles"
            for data_time in "${selected_data_times[@]}"; do
                for role_mask in "${selected_role_masks[@]}"; do
                    for roles in $role_space; do
                        run "$trial" data-time "$data_time" "$roles" \
                            "$role_mask" "${appt[@]}" \
                            --segment-cores \
                                appt-online-register-tail-grouped-writer-final-data-time \
                            --segment-data-time "$data_time" \
                            --appt-role-weights "$roles" \
                            --appt-data-time-roles "$role_mask"
                    done
                done
            done
        done
    done
done

python3 "$ROOT/scripts/summarize_appt_data_time.py" "$raw" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'APPT data-time comparison: %s\n' "$OUTPUT_DIR/analysis.md"
