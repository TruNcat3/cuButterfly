#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-cuda118/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/appt_writer_final/confirmed"}
TRIALS=${TRIALS:-3}
WARMUP=${WARMUP:-5}
REPEAT=${REPEAT:-30}

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

for bits in 32 64; do
    if [[ $bits == 32 ]]; then modulus=998244353; resident=3
    else modulus=576460756061519873; resident=2
    fi
    for batch in 1 4 16; do
        case "$bits:$batch" in
            32:1)  grouped_fragment=16; grouped_roles=7,13,3
                   fragment=8; writer_tiles=1; roles=6,12,4 ;;
            32:4)  grouped_fragment=16; grouped_roles=6,14,2
                   fragment=8; writer_tiles=2; roles=6,12,4 ;;
            32:16) grouped_fragment=32; grouped_roles=6,16,2
                   fragment=16; writer_tiles=2; roles=6,12,4 ;;
            64:1)  grouped_fragment=8; grouped_roles=10,9,3
                   fragment=8; writer_tiles=2; roles=8,8,4 ;;
            64:4|64:16) grouped_fragment=16; grouped_roles=8,10,2
                   fragment=$([[ $batch == 4 ]] && echo 8 || echo 16)
                   writer_tiles=2; roles=8,8,4 ;;
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
            run "$trial" writer-final "$fragment" "$writer_tiles" "$roles" \
                "${appt[@]}" \
                --segment-cores \
                    appt-online-register-tail-grouped-writer-final \
                --appt-fragment-width "$fragment" \
                --appt-writer-tiles "$writer_tiles" \
                --appt-role-weights "$roles"
        done
    done
done

python3 "$ROOT/scripts/summarize_appt_writer_final.py" "$raw" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'Confirmed APPT writer-final comparison: %s\n' \
    "$OUTPUT_DIR/analysis.md"
