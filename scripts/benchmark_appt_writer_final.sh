#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-cuda118/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/appt_writer_final"}
TRIALS=${TRIALS:-2}
WARMUP=${WARMUP:-3}
REPEAT=${REPEAT:-20}
BATCHES=${BATCHES:-1,4,16}
FRAGMENTS=${FRAGMENTS:-8,16,32}
WRITER_TILES=${WRITER_TILES:-1,2}
U32_ROLES=${U32_ROLES:-"7,13,3 7,12,4 7,11,5 6,14,2 6,13,3 6,12,4 6,11,5 6,10,6"}
U64_ROLES=${U64_ROLES:-"10,9,3 10,8,4 9,9,3 9,8,4 9,7,5 8,10,2 8,9,3 8,8,4 8,7,5"}

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
raw="$OUTPUT_DIR/raw.csv"
: > "$raw"

run() {
    local trial=$1 variant=$2 fragment=$3 writer_tiles=$4 roles=$5
    shift 5
    local output header record
    output=$("$BIN" "$@" --warmup "$WARMUP" --repeat "$REPEAT" --csv)
    header=$(printf '%s\n' "$output" | sed -n '/^device,/{p;q;}')
    record=$(printf '%s\n' "$output" | tail -n 1)
    if [[ ! -s "$raw" ]]; then
        printf 'trial,variant,fragment_width,writer_tiles,roles,%s\n' "$header" > "$raw"
    fi
    printf '%s,%s,%s,%s,%s,%s\n' "$trial" "$variant" "$fragment" \
        "$writer_tiles" "${roles//,/-}" "$record" >> "$raw"
}

IFS=, read -ra selected_batches <<< "$BATCHES"
IFS=, read -ra selected_fragments <<< "$FRAGMENTS"
IFS=, read -ra selected_writer_tiles <<< "$WRITER_TILES"
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
            32:1)  grouped_fragment=16; grouped_roles=7,13,3 ;;
            32:4)  grouped_fragment=16; grouped_roles=6,14,2 ;;
            32:16) grouped_fragment=32; grouped_roles=6,16,2 ;;
            64:1)  grouped_fragment=8;  grouped_roles=10,9,3 ;;
            64:4|64:16) grouped_fragment=16; grouped_roles=8,10,2 ;;
            *) echo "unsupported bits/batch point: $bits/$batch" >&2; exit 1 ;;
        esac
        common=(--word-bits "$bits" --modulus "$modulus" --logN 20
                --batch "$batch")
        appt=("${common[@]}" --backend hierarchical-dataflow
              --stage-partition 7,7,6 --segment-units 8
              --segment-data-space 32,16,16 --segment-data-time 4
              --segment-role-stages 2 --segment-token-interleave 1
              --boundary-storage ring --boundary-buffers 2
              --target-ctas-per-sm "$resident" --output-order natural)
        for trial in $(seq 1 "$TRIALS"); do
            run "$trial" v06 0 0 9,11 "${common[@]}" \
                --backend hierarchical-dataflow --stage-partition 10,10 \
                --segment-cores dataflow-radix4 --segment-cta-weights 9,11
            run "$trial" grouped "$grouped_fragment" 2 "$grouped_roles" \
                "${appt[@]}" \
                --segment-cores appt-online-register-tail-grouped \
                --appt-fragment-width "$grouped_fragment" \
                --appt-writer-tiles 2 --appt-role-weights "$grouped_roles"
            for fragment in "${selected_fragments[@]}"; do
                for writer_tiles in "${selected_writer_tiles[@]}"; do
                    for roles in $role_space; do
                        run "$trial" writer-final "$fragment" "$writer_tiles" \
                            "$roles" "${appt[@]}" \
                            --segment-cores \
                                appt-online-register-tail-grouped-writer-final \
                            --appt-fragment-width "$fragment" \
                            --appt-writer-tiles "$writer_tiles" \
                            --appt-role-weights "$roles"
                    done
                done
            done
        done
    done
done

python3 "$ROOT/scripts/summarize_appt_writer_final.py" "$raw" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'APPT writer-final comparison: %s\n' "$OUTPUT_DIR/analysis.md"
