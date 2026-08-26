#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-v07/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/appt_pipeline"}
WARMUP=${WARMUP:-5}
REPEAT=${REPEAT:-20}
TRIALS=${TRIALS:-2}
WORD_BITS_LIST=${WORD_BITS_LIST:-"32 64"}
LOGN_LIST=${LOGN_LIST:-"20"}
BATCH_LIST=${BATCH_LIST:-"1 4 16"}
US_LIST=${US_LIST:-"6 7 8"}
TD_LIST=${TD_LIST:-"2 4 8 16"}
ROLE_STAGES_LIST=${ROLE_STAGES_LIST:-"1 2"}
REPLICA_LIST=${REPLICA_LIST:-"1 2"}
TOKEN_INTERLEAVE_LIST=${TOKEN_INTERLEAVE_LIST:-"1 2"}
BUFFER_LIST=${BUFFER_LIST:-"1 2"}
CTA_LIST=${CTA_LIST:-"1 2"}
ONLINE_DATA_SPACE_LIST=${ONLINE_DATA_SPACE_LIST:-"16,16,16 16,8,16 16,32,16 16,16,8 16,16,32 16,8,8 16,8,32 16,32,8 16,32,32"}

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
raw="$OUTPUT_DIR/raw.csv"
: > "$raw"

partition() {
    local remaining=$1 us=$2 result=""
    while (( remaining > us )); do
        result+="${result:+,}${us}"
        ((remaining -= us))
    done
    result+="${result:+,}${remaining}"
    printf '%s' "$result"
}

run() {
    local trial=$1 family=$2 label=$3
    shift 3
    local output header record
    if ! output=$("$BIN" "$@" --warmup "$WARMUP" --repeat "$REPEAT" --csv); then
        printf 'skip unsupported point: %s\n' "$label" >&2
        return 0
    fi
    header=$(printf '%s\n' "$output" | sed -n '/^device,/{p;q;}')
    record=$(printf '%s\n' "$output" | tail -n 1)
    if [[ ! -s "$raw" ]]; then
        printf 'trial,family,label,%s\n' "$header" > "$raw"
    fi
    printf '%s,%s,%s,%s\n' "$trial" "$family" "$label" "$record" >> "$raw"
}

for bits in $WORD_BITS_LIST; do
    if [[ $bits == 32 ]]; then modulus=998244353; else modulus=576460756061519873; fi
    for logn in $LOGN_LIST; do
        verify_partition=$(partition "$logn" 8)
        "$BIN" --backend hierarchical-dataflow --word-bits "$bits" --modulus "$modulus" \
            --logN "$logn" --batch 1 --stage-partition "$verify_partition" \
            --segment-cores appt-pipeline --segment-data-space 16 --segment-data-time 8 \
            --boundary-storage ring --boundary-buffers 2 --target-ctas-per-sm 1 \
            --warmup 0 --repeat 1 --verify >/dev/null
        for trial in $(seq 1 "$TRIALS"); do
            for batch in $BATCH_LIST; do
                common=(--word-bits "$bits" --modulus "$modulus" --logN "$logn" --batch "$batch")
                run "$trial" baseline barrier "${common[@]}" --backend hierarchical-barrier
                if [[ $logn == 20 ]]; then
                    run "$trial" v06 resident1010 "${common[@]}" \
                        --backend hierarchical-dataflow --stage-partition 10,10 \
                        --segment-cores dataflow-radix4 --segment-cta-weights 9,11
                    online_ctas=2
                    online_weights=7,7,6
                    online_td=8
                    online_ti=2
                    if [[ $bits == 32 ]]; then online_ctas=3; fi
                    if [[ $batch == 1 ]]; then
                        online_td=4
                        if [[ $bits == 32 ]]; then
                            online_weights=6,6,8
                        else
                            online_weights=8,8,4
                            online_ti=1
                        fi
                    elif [[ $bits == 64 ]]; then
                        online_weights=5,7,8
                    fi
                    for online_space in $ONLINE_DATA_SPACE_LIST; do
                        run "$trial" appt-online \
                            "online-ds${online_space//,/-}-w${online_weights//,/-}-td${online_td}-ti${online_ti}-cta${online_ctas}" \
                            "${common[@]}" --backend hierarchical-dataflow \
                            --stage-partition 7,7,6 --segment-cores appt-online \
                            --segment-units 8 --segment-data-space "$online_space" \
                            --segment-data-time "$online_td" --segment-role-stages 2 \
                            --segment-token-interleave "$online_ti" \
                            --segment-cta-weights "$online_weights" \
                            --boundary-storage ring --boundary-buffers 2 \
                            --target-ctas-per-sm "$online_ctas"
                    done
                fi
                for us in $US_LIST; do
                    stages=$(partition "$logn" "$us")
                    for td in $TD_LIST; do
                      for role_stages in $ROLE_STAGES_LIST; do
                        roles=$(((us + role_stages - 1) / role_stages))
                        for replicas in $REPLICA_LIST; do
                          units=$((roles * replicas))
                          for token_interleave in $TOKEN_INTERLEAVE_LIST; do
                            if (( token_interleave > 1 &&
                                  (role_stages == 1 || td < role_stages * token_interleave) )); then
                                continue
                            fi
                            for buffers in $BUFFER_LIST; do
                              for ctas in $CTA_LIST; do
                                label="appt-us${us}-rs${role_stages}-rep${replicas}-td${td}-ti${token_interleave}-buf${buffers}-cta${ctas}"
                                run "$trial" appt "$label" "${common[@]}" \
                                    --backend hierarchical-dataflow --stage-partition "$stages" \
                                    --segment-cores appt-pipeline --segment-units "$units" \
                                    --segment-data-space 16 --segment-data-time "$td" \
                                    --segment-role-stages "$role_stages" \
                                    --segment-token-interleave "$token_interleave" \
                                    --boundary-storage ring --boundary-buffers "$buffers" \
                                    --target-ctas-per-sm "$ctas"
                              done
                            done
                          done
                        done
                      done
                    done
                done
            done
        done
    done
done

python3 "$ROOT/scripts/summarize_appt_pipeline.py" "$raw" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'APPT pipeline analysis: %s\n' "$OUTPUT_DIR/analysis.md"
if [[ -n ${SUDO_USER:-} ]]; then
    chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$OUTPUT_DIR"
fi
