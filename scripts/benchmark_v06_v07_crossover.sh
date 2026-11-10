#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/v06_v07_crossover"}
WARMUP=${WARMUP:-30}
REPEAT=${REPEAT:-500}
TRIALS=${TRIALS:-3}
VERIFY=${VERIFY:-1}
mkdir -p "$OUTPUT_DIR"
RAW="$OUTPUT_DIR/raw.csv"
: >"$RAW"

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }

append() {
    local case_id=$1 variant=$2 trial=$3
    shift 3
    local output header record
    output=$("$BIN" "$@" --warmup "$WARMUP" --repeat "$REPEAT" --csv)
    header=$(printf '%s\n' "$output" | sed -n '/^device,/{p;q;}')
    record=$(printf '%s\n' "$output" | tail -n 1)
    if [[ ! -s "$RAW" ]]; then
        printf 'case_id,variant,trial,%s\n' "$header" >"$RAW"
    fi
    printf '%s,%s,%s,%s\n' "$case_id" "$variant" "$trial" "$record" >>"$RAW"
}

n12_common=(--logN 12 --word-bits 32 --modulus 1073479681
            --cross-twiddle fused --mod-multiply shoup
            --output-order natural)
hybrid_common=(--backend hybrid-dataflow --compute-unit radix4
               --flow-tile-log 6 --stage-space 6 --data-space 32
               --role-stages 6 --target-ctas-per-sm 1
               --token-interleave 1)

n20_common=(--logN 20 --batch 16 --word-bits 32 --modulus 998244353
            --cross-twiddle fused --mod-multiply shoup
            --output-order natural)
resident=(--backend hierarchical-dataflow --stage-partition 10,10
          --boundary-storage full-scratch --segment-data-time 1,1)

if [[ $VERIFY != 0 ]]; then
    printf '[preflight] checking focused crossover paths\n'
    "$BIN" "${n12_common[@]}" --batch 1 --backend hybrid2d \
        --warmup 0 --repeat 1 --verify >/dev/null
    "$BIN" "${n12_common[@]}" --batch 1 "${hybrid_common[@]}" \
        --data-time 12 --warmup 0 --repeat 1 --verify >/dev/null
    "$BIN" "${n20_common[@]}" "${resident[@]}" \
        --segment-cores homogeneous-warp128-packet-shared-radix4-static-io \
        --segment-threads 128 --segment-units 4 --segment-data-space 4 \
        --segment-cta-weights 7,13 --target-ctas-per-sm 4 \
        --warmup 0 --repeat 1 --verify >/dev/null
fi

for batch in 80 160 256 512; do
    for ((trial = 1; trial <= TRIALS; ++trial)); do
        append n12_u32 "hybrid2d_b${batch}" "$trial" \
            "${n12_common[@]}" --batch "$batch" \
            --backend hybrid2d --compute-unit radix4
        append n12_u32 "hybrid_td6_b${batch}" "$trial" \
            "${n12_common[@]}" --batch "$batch" \
            "${hybrid_common[@]}" --data-time 6
        append n12_u32 "hybrid_td12_b${batch}" "$trial" \
            "${n12_common[@]}" --batch "$batch" \
            "${hybrid_common[@]}" --data-time 12
    done
done

for ((trial = 1; trial <= TRIALS; ++trial)); do
    append n20_u32_b16 hybrid2d "$trial" "${n20_common[@]}" \
        --backend hybrid2d --compute-unit radix4
    append n20_u32_b16 resident_9_11 "$trial" "${n20_common[@]}" \
        "${resident[@]}" --segment-cores dataflow-radix4 \
        --segment-cta-weights 9,11 --target-ctas-per-sm 4
    append n20_u32_b16 packet_7_13 "$trial" "${n20_common[@]}" \
        "${resident[@]}" \
        --segment-cores homogeneous-warp128-packet-shared-radix4-static-io \
        --segment-threads 128 --segment-units 4 --segment-data-space 4 \
        --segment-cta-weights 7,13 --target-ctas-per-sm 4
    append n20_u32_b16 lane32 "$trial" "${n20_common[@]}" \
        "${resident[@]}" \
        --segment-cores homogeneous-warp128-vector-radix4-packed-stage6-distributed-static-io \
        --segment-threads 256 --segment-units 8 --segment-data-space 4 \
        --segment-coefficient-reuse-stages 6 \
        --segment-cta-weights 84,76 --target-ctas-per-sm 2
done

python3 "$ROOT/scripts/analyze_v06_v07_crossover.py" "$RAW" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'wrote %s\n' "$OUTPUT_DIR/analysis.md"
