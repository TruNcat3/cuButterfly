#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bench="${CUNTT_BENCH:-$root/build-11.8/cuntt_bench}"
out="${1:-$root/results/hybrid_dataflow_td_batch_fine}"
mkdir -p "$out/raw"
batches=(1 $(seq 16 16 640))

run() {
    local name="$1"
    shift
    "$bench" "$@" --backend hybrid-dataflow --token-interleave 1 \
        --pipeline-buffers 2 --warmup 20 --repeat 300 --verify --csv > "$out/raw/$name.csv"
}

for batch in "${batches[@]}"; do
    for data_time in 9 10 15; do
        run "w32_n10_b${batch}_td${data_time}" --word-bits 32 --modulus 1073479681 \
            --data-space 32 --logN 10 --batch "$batch" --flow-tile-log 5 \
            --stage-space 5 --role-stages 5 --data-time "$data_time"
    done
    for data_time in 5 9 10; do
        run "w64_n10_b${batch}_td${data_time}" --word-bits 64 --data-space 16 \
            --logN 10 --batch "$batch" --flow-tile-log 5 --stage-space 5 \
            --role-stages 5 --data-time "$data_time"
    done
    for data_time in 6 12 16; do
        run "w32_n12_b${batch}_td${data_time}" --word-bits 32 --modulus 1073479681 \
            --data-space 32 --logN 12 --batch "$batch" --flow-tile-log 6 \
            --stage-space 6 --role-stages 6 --data-time "$data_time"
    done
    for data_time in 11 12 16; do
        run "w64_n12_b${batch}_td${data_time}" --word-bits 64 --data-space 16 \
            --logN 12 --batch "$batch" --flow-tile-log 6 --stage-space 6 \
            --role-stages 6 --data-time "$data_time"
    done
done

python3 "$root/scripts/summarize_hybrid_dataflow_td_batch_fine.py" "$out"/raw/*.csv \
    --csv "$out/summary.csv" --markdown "$out/summary.md"
