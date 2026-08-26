#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-cuda118/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/appt_static_layout"}
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
    if [[ ! -s "$raw" ]]; then printf 'trial,variant,%s\n' "$header" > "$raw"; fi
    printf '%s,%s,%s\n' "$trial" "$label" "$record" >> "$raw"
}

IFS=, read -ra batches <<< "$BATCHES"
for bits in 32 64; do
    if [[ $bits == 32 ]]; then
        modulus=998244353; resident=3
        role_points=(8,12,1 8,12,2 8,12,3 7,13,2 7,14,2 7,15,2 7,14,3 9,11,2 8,14,2)
    else
        modulus=576460756061519873; resident=2
        role_points=(10,10,1 10,10,2 9,11,2 11,9,2 10,12,2)
    fi
    for batch in "${batches[@]}"; do
        if [[ $bits == 32 ]]; then
            (( batch >= 16 )) && static_roles=7,14,1 || static_roles=8,12,1
        else
            (( batch == 1 )) && static_roles=11,9,1 || static_roles=10,10,1
        fi
        common=(--word-bits "$bits" --modulus "$modulus" --logN 20 --batch "$batch")
        appt=("${common[@]}" --backend hierarchical-dataflow --stage-partition 7,7,6
              --segment-cores appt-online-register-tail --segment-units 8
              --segment-data-space 16 --segment-data-time 4 --segment-role-stages 2
              --segment-token-interleave 1 --boundary-storage ring --boundary-buffers 2
              --target-ctas-per-sm "$resident")
        for trial in $(seq 1 "$TRIALS"); do
            run "$trial" v06 "${common[@]}" --backend hierarchical-dataflow \
                --stage-partition 10,10 --segment-cores dataflow-radix4 \
                --segment-cta-weights 9,11
            for fragment in 8 16 32; do
                run "$trial" "static-fw${fragment}" "${appt[@]}" \
                    --output-order appt-static --appt-fragment-width "$fragment" \
                    --appt-role-weights "$static_roles" --appt-writer-tiles 1
                for tiles in 1 2 4; do
                    for roles in "${role_points[@]}"; do
                        run "$trial" "natural-fw${fragment}-wt${tiles}-r${roles//,/-}" \
                            "${appt[@]}" --output-order natural \
                            --appt-fragment-width "$fragment" --appt-writer-tiles "$tiles" \
                            --appt-role-weights "$roles"
                    done
                done
            done
        done
    done
done

python3 "$ROOT/scripts/summarize_appt_static_layout.py" "$raw" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'APPT static-layout search: %s\n' "$OUTPUT_DIR/analysis.md"
