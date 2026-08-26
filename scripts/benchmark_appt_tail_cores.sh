#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-v07/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/appt_tail_cores"}
TRIALS=${TRIALS:-3}
WARMUP=${WARMUP:-3}
REPEAT=${REPEAT:-20}

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
raw="$OUTPUT_DIR/raw.csv"
: > "$raw"

run() {
    local trial=$1 label=$2 weights=$3
    shift 3
    local output header record
    output=$("$BIN" "$@" --warmup "$WARMUP" --repeat "$REPEAT" --csv)
    header=$(printf '%s\n' "$output" | sed -n '/^device,/{p;q;}')
    record=$(printf '%s\n' "$output" | tail -n 1)
    if [[ ! -s "$raw" ]]; then
        printf 'trial,physical_core,weights,%s\n' "$header" > "$raw"
    fi
    printf '%s,%s,%s,%s\n' "$trial" "$label" "${weights//,/-}" "$record" >> "$raw"
}

for bits in 32 64; do
    if [[ $bits == 32 ]]; then modulus=998244353; resident=3
    else modulus=576460756061519873; resident=2
    fi
    for batch in 1 4; do
        common=(--word-bits "$bits" --modulus "$modulus" --logN 20 --batch "$batch")
        if [[ $bits == 32 && $batch == 1 ]]; then warp=6,6,8; cta=7,6,7; full=9,6,5; split=7,7,6
        elif [[ $bits == 32 ]]; then warp=7,7,6; cta=7,6,7; full=4,8,8; split=8,6,6
        elif [[ $batch == 1 ]]; then warp=8,8,4; cta=4,8,8; full=9,6,5; split=4,6,10
        else warp=6,8,6; cta=5,7,8; full=7,7,6; split=4,8,8
        fi
        if [[ $bits == 32 && $batch == 1 ]]; then register=8,7,5
        elif [[ $bits == 32 ]]; then register=8,7,5
        elif [[ $batch == 1 ]]; then register=11,5,4
        else register=10,6,4
        fi

        for trial in $(seq 1 "$TRIALS"); do
            run "$trial" v06 9,11 "${common[@]}" \
                --backend hierarchical-dataflow --stage-partition 10,10 \
                --segment-cores dataflow-radix4 --segment-cta-weights 9,11
            for point in "warp:${warp}:appt-online:${resident}" \
                         "cta-radix4:${cta}:appt-online-radix4:${resident}" \
                         "fused-tail:${full}:appt-online-fused-tail:$([[ $bits == 32 ]] && printf 2 || printf 1)" \
                         "split-tail:${split}:appt-online-split-tail:${resident}" \
                         "register-tail:${register}:appt-online-register-tail:${resident}"; do
                IFS=: read -r label weights core target <<< "$point"
                run "$trial" "$label" "$weights" "${common[@]}" \
                    --backend hierarchical-dataflow --stage-partition 7,7,6 \
                    --segment-cores "$core" --segment-units 8 \
                    --segment-data-space 16,16,16 --segment-data-time 4 \
                    --segment-role-stages 2 --segment-token-interleave 1 \
                    --segment-cta-weights "$weights" --boundary-storage ring \
                    --boundary-buffers 2 --target-ctas-per-sm "$target"
            done
        done
    done
done

python3 "$ROOT/scripts/summarize_appt_tail_cores.py" "$raw" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'APPT tail-core comparison: %s\n' "$OUTPUT_DIR/analysis.md"
if [[ -n ${SUDO_USER:-} ]]; then
    chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$OUTPUT_DIR"
fi
