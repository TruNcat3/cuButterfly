#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-v07/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/appt_role_breakdown"}
TRIALS=${TRIALS:-3}
WARMUP=${WARMUP:-2}
REPEAT=${REPEAT:-5}

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
raw="$OUTPUT_DIR/raw.csv"
printf 'kind,bits,batch,trial,kernel_ms,role,wait_ns,compute_ns,boundary_ns,tasks,span_ns,start_delay_ns\n' > "$raw"

parse_segment() {
    local output=$1 segment=$2 line payload start end first
    line=$(printf '%s\n' "$output" | grep "^pipeline_segment_${segment}:")
    payload=${line#*: }
    start=${payload%%,*}; start=${start#start=}
    end=${payload##*end=}
    first=$(printf '%s\n' "$output" | sed -n 's/^pipeline_segment_[0-9]*: start=\([0-9]*\).*/\1/p' | sort -n | head -n 1)
    printf '%s,%s' "$((end-start))" "$((start-first))"
}

for trial in $(seq 1 "$TRIALS"); do
    for bits in 32 64; do
        if [[ $bits == 32 ]]; then
            modulus=998244353
            ctas=3
        else
            modulus=576460756061519873
            ctas=2
        fi
        for batch in 1 4; do
            if [[ $bits == 32 && $batch == 1 ]]; then
                weights=6,6,8
            elif [[ $bits == 32 ]]; then
                weights=7,7,6
            elif [[ $batch == 1 ]]; then
                weights=8,8,4
            else
                weights=5,7,8
            fi
            common=(--word-bits "$bits" --modulus "$modulus" --logN 20 --batch "$batch")
            output=$("$BIN" "${common[@]}" --backend hierarchical-dataflow \
                --stage-partition 7,7,6 --segment-cores appt-online \
                --segment-units 8 --segment-data-space 16,16,16 \
                --segment-data-time 8 --segment-role-stages 2 \
                --segment-token-interleave 2 --segment-cta-weights "$weights" \
                --boundary-storage ring --boundary-buffers 2 \
                --target-ctas-per-sm "$ctas" --profile-appt-roles \
                --warmup "$WARMUP" --repeat "$REPEAT")
            kernel=$(printf '%s\n' "$output" | awk '/^kernel_ms:/ {print $2}')
            for role in 0 1 2; do
                line=$(printf '%s\n' "$output" | grep "^pipeline_role_${role}:")
                payload=${line#*: }; payload=${payload//, /,}
                IFS=, read -r wait compute boundary tasks <<< "$payload"
                IFS=, read -r span delay <<< "$(parse_segment "$output" "$role")"
                printf 'online,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                    "$bits" "$batch" "$trial" "$kernel" "$role" \
                    "${wait#wait=}" "${compute#compute=}" \
                    "${boundary#boundary=}" "${tasks#tasks=}" "$span" "$delay" >> "$raw"
            done

            output=$("$BIN" "${common[@]}" --backend hierarchical-dataflow \
                --stage-partition 10,10 --segment-cores dataflow-radix4 \
                --segment-cta-weights 9,11 --trace-pipeline \
                --warmup "$WARMUP" --repeat "$REPEAT")
            kernel=$(printf '%s\n' "$output" | awk '/^kernel_ms:/ {print $2}')
            for role in 0 1; do
                IFS=, read -r span delay <<< "$(parse_segment "$output" "$role")"
                printf 'v06,%s,%s,%s,%s,%s,0,0,0,0,%s,%s\n' \
                    "$bits" "$batch" "$trial" "$kernel" "$role" "$span" "$delay" >> "$raw"
            done
        done
    done
done

python3 "$ROOT/scripts/summarize_appt_role_breakdown.py" "$raw" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'APPT role breakdown: %s\n' "$OUTPUT_DIR/analysis.md"
