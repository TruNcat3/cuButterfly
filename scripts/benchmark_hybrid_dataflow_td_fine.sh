#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bench="${CUNTT_BENCH:-$root/build-11.8/cuntt_bench}"
out="${1:-$root/results/hybrid_dataflow_td_fine}"
mkdir -p "$out/raw"

run() {
    local name="$1"
    shift
    "$bench" "$@" --backend hybrid-dataflow --token-interleave 1 \
        --pipeline-buffers 2 --warmup 20 --repeat 200 --verify --csv > "$out/raw/$name.csv"
}

for word_bits in 32 64; do
    if [[ "$word_bits" == 32 ]]; then
        numeric=(--word-bits 32 --modulus 1073479681 --data-space 32)
    else
        numeric=(--word-bits 64 --data-space 16)
    fi
    for batch in 1 16 80 160 256 512; do
        for data_time in $(seq 1 16); do
            run "w${word_bits}_n10_b${batch}_td${data_time}" "${numeric[@]}" \
                --logN 10 --batch "$batch" --flow-tile-log 5 --stage-space 5 \
                --role-stages 5 --data-time "$data_time"
            run "w${word_bits}_n12_b${batch}_td${data_time}" "${numeric[@]}" \
                --logN 12 --batch "$batch" --flow-tile-log 6 --stage-space 6 \
                --role-stages 6 --data-time "$data_time"
        done
    done
done

python3 "$root/scripts/summarize_hybrid_dataflow_td_fine.py" "$out"/raw/*.csv \
    --csv "$out/summary.csv" --markdown "$out/summary.md"
