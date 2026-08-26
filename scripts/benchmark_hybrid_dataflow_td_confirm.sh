#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bench="${CUNTT_BENCH:-$root/build-11.8/cuntt_bench}"
out="${1:-$root/results/hybrid_dataflow_td_confirm}"
mkdir -p "$out/raw"
td10_forward=(4 5 6 9 10 11 14 15 16)
td10_reverse=(16 15 14 11 10 9 6 5 4)
td12_forward=(5 6 7 11 12 13 16)
td12_reverse=(16 13 12 11 7 6 5)

run() {
    local name="$1"
    shift
    "$bench" "$@" --backend hybrid-dataflow --token-interleave 1 \
        --pipeline-buffers 2 --warmup 30 --repeat 500 --verify --csv > "$out/raw/$name.csv"
}

for trial in 1 2 3; do
    if (( trial % 2 == 1 )); then
        td10=("${td10_forward[@]}")
        td12=("${td12_forward[@]}")
    else
        td10=("${td10_reverse[@]}")
        td12=("${td12_reverse[@]}")
    fi
    for word_bits in 32 64; do
        if [[ "$word_bits" == 32 ]]; then
            numeric=(--word-bits 32 --modulus 1073479681 --data-space 32)
        else
            numeric=(--word-bits 64 --data-space 16)
        fi
        for batch in 1 16 80 160 256 512; do
            for data_time in "${td10[@]}"; do
                run "w${word_bits}_n10_b${batch}_td${data_time}_t${trial}" "${numeric[@]}" \
                    --logN 10 --batch "$batch" --flow-tile-log 5 --stage-space 5 \
                    --role-stages 5 --data-time "$data_time"
            done
            for data_time in "${td12[@]}"; do
                run "w${word_bits}_n12_b${batch}_td${data_time}_t${trial}" "${numeric[@]}" \
                    --logN 12 --batch "$batch" --flow-tile-log 6 --stage-space 6 \
                    --role-stages 6 --data-time "$data_time"
            done
        done
    done
done

python3 "$root/scripts/summarize_hybrid_dataflow_td_confirm.py" "$out"/raw/*.csv \
    --csv "$out/summary.csv" --markdown "$out/summary.md"
