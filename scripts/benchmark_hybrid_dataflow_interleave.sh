#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bench="${CUNTT_BENCH:-$root/build-11.8/cuntt_bench}"
out="${1:-$root/results/hybrid_dataflow_interleave}"
mkdir -p "$out/raw"

run() {
    local name="$1"
    shift
    "$bench" "$@" --backend hybrid-dataflow --pipeline-buffers 2 \
        --warmup 20 --repeat 200 --verify --csv > "$out/raw/$name.csv"
}

for word_bits in 32 64; do
    if [[ "$word_bits" == 32 ]]; then
        numeric=(--word-bits 32 --modulus 1073479681 --data-space 32)
    else
        numeric=(--word-bits 64 --data-space 16)
    fi
    for batch in 1 16 80 160 256 512; do
        common=("${numeric[@]}" --batch "$batch")
        run "w${word_bits}_n10_b${batch}_ur1" "${common[@]}" --logN 10 \
            --flow-tile-log 5 --stage-space 5 --role-stages 1 --data-time 4 --token-interleave 1
        run "w${word_bits}_n10_b${batch}_ti1" "${common[@]}" --logN 10 \
            --flow-tile-log 5 --stage-space 5 --role-stages 5 --data-time 4 --token-interleave 1
        run "w${word_bits}_n10_b${batch}_td10_ti1" "${common[@]}" --logN 10 \
            --flow-tile-log 5 --stage-space 5 --role-stages 5 --data-time 10 --token-interleave 1
        run "w${word_bits}_n10_b${batch}_ti2" "${common[@]}" --logN 10 \
            --flow-tile-log 5 --stage-space 5 --role-stages 5 --data-time 10 --token-interleave 2
        run "w${word_bits}_n12_b${batch}_ti1" "${common[@]}" --logN 12 \
            --flow-tile-log 6 --stage-space 6 --role-stages 6 --data-time 8 --token-interleave 1
        run "w${word_bits}_n12_b${batch}_ur1" "${common[@]}" --logN 12 \
            --flow-tile-log 6 --stage-space 6 --role-stages 1 --data-time 4 --token-interleave 1
        run "w${word_bits}_n12_b${batch}_td12_ti1" "${common[@]}" --logN 12 \
            --flow-tile-log 6 --stage-space 6 --role-stages 6 --data-time 12 --token-interleave 1
        run "w${word_bits}_n12_b${batch}_ti2" "${common[@]}" --logN 12 \
            --flow-tile-log 6 --stage-space 6 --role-stages 6 --data-time 12 --token-interleave 2
    done
done

python3 "$root/scripts/summarize_hybrid_dataflow.py" "$out"/raw/*.csv \
    --csv "$out/summary.csv" --markdown "$out/summary.md"
