#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/resident_v06_comparison"}
LOGNS=${LOGNS:-"12 14 16 18 20"}
BATCHES=${BATCHES:-"1 4 16"}
WORD_BITS=${WORD_BITS:-"32 64"}
TRIALS=${TRIALS:-3}
WARMUP=${WARMUP:-10}
REPEAT=${REPEAT:-50}
VERIFY=${VERIFY:-1}

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
RAW="$OUTPUT_DIR/raw.csv"
: >"$RAW"

logical_partition() {
    case "$1" in
        12) echo 3,3,3,3 ;;
        13) echo 3,3,3,4 ;;
        14) echo 3,4,3,4 ;;
        15) echo 3,4,4,4 ;;
        16) echo 4,4,4,4 ;;
        17) echo 4,4,4,5 ;;
        18) echo 4,5,4,5 ;;
        19) echo 4,5,5,5 ;;
        20) echo 5,5,5,5 ;;
        *) echo "unsupported logN for four-subgraph comparison: $1" >&2; return 1 ;;
    esac
}

execution_partition() {
    local logical
    IFS=, read -ra logical <<<"$(logical_partition "$1")"
    echo "$((logical[0] + logical[1])),$((logical[2] + logical[3]))"
}

execution_weights() {
    # Preserve the established V100 10+10 role balance at the physical
    # equivalence point. Other lengths start from stage-proportional weights.
    [[ $1 == 20 ]] && echo 9,11 || execution_partition "$1"
}

modulus_for() {
    case "$1" in
        32) echo 998244353 ;;
        64) echo 576460756061519873 ;;
        *) echo "unsupported word width: $1" >&2; return 1 ;;
    esac
}

target_for() {
    [[ $1 == 32 ]] && echo 4 || echo 2
}

variant_args() {
    local variant=$1 bits=$2 logn=$3
    local logical execution weights target
    logical=$(logical_partition "$logn")
    execution=$(execution_partition "$logn")
    weights=$(execution_weights "$logn")
    target=$(target_for "$bits")
    case "$variant" in
        v06_hybrid2d)
            printf '%s\0' --backend hybrid2d --compute-unit radix4 \
                --cross-twiddle fused --mod-multiply shoup \
                --output-order natural
            ;;
        v06_resident)
            [[ $logn == 20 ]] || return 2
            printf '%s\0' --backend hierarchical-dataflow \
                --stage-partition 10,10 --boundary-storage full-scratch \
                --segment-cores dataflow-radix4 \
                --segment-data-time 1,1 --segment-cta-weights 9,11 \
                --target-ctas-per-sm "$target" \
                --cross-twiddle fused --mod-multiply shoup \
                --output-order natural
            ;;
        v07_full)
            printf '%s\0' --backend hierarchical-dataflow \
                --stage-partition "$logical" \
                --boundary-storage full-scratch,full-scratch,full-scratch \
                --segment-cores radix4 --segment-threads 256 \
                --segment-cta-weights "$logical" --target-ctas-per-sm 2 \
                --cross-twiddle fused --mod-multiply shoup \
                --output-order natural
            ;;
        v07_resident_generic)
            printf '%s\0' --backend hierarchical-dataflow \
                --stage-partition "$logical" \
                --boundary-storage resident-fused,full-scratch,resident-fused \
                --segment-cores radix4 --segment-threads 256 \
                --segment-cta-weights "$logical" \
                --execution-group-cores radix4,radix4 \
                --execution-group-cta-weights "$execution" \
                --target-ctas-per-sm 2 \
                --cross-twiddle fused --mod-multiply shoup \
                --output-order natural
            ;;
        v07_resident_core)
            printf '%s\0' --backend hierarchical-dataflow \
                --stage-partition "$logical" \
                --boundary-storage resident-fused,full-scratch,resident-fused \
                --segment-cores radix4 --segment-threads 256 \
                --segment-cta-weights "$logical" \
                --execution-group-cores dataflow-radix4,dataflow-radix4 \
                --execution-group-cta-weights "$weights" \
                --target-ctas-per-sm "$target" \
                --cross-twiddle fused --mod-multiply shoup \
                --output-order natural
            ;;
        *) echo "unknown variant: $variant" >&2; return 1 ;;
    esac
}

read_variant_args() {
    local variant=$1 bits=$2 logn=$3
    VARIANT_ARGS=()
    while IFS= read -r -d '' item; do VARIANT_ARGS+=("$item"); done \
        < <(variant_args "$variant" "$bits" "$logn")
}

run_bench() {
    local bits=$1 logn=$2 batch=$3 variant=$4 trial=$5 timed=$6
    local modulus output header record
    modulus=$(modulus_for "$bits")
    read_variant_args "$variant" "$bits" "$logn"
    local common=(--word-bits "$bits" --modulus "$modulus" \
                  --logN "$logn" --batch "$batch")
    if [[ $timed == 0 ]]; then
        "$BIN" "${common[@]}" "${VARIANT_ARGS[@]}" \
            --warmup 0 --repeat 1 --verify --csv >/dev/null
        return
    fi
    output=$("$BIN" "${common[@]}" "${VARIANT_ARGS[@]}" \
        --warmup "$WARMUP" --repeat "$REPEAT" --csv)
    header=$(printf '%s\n' "$output" | sed -n '/^device,/{p;q;}')
    record=$(printf '%s\n' "$output" | tail -n 1)
    [[ -n "$header" && -n "$record" ]] || {
        echo "missing benchmark CSV for $variant u$bits logN=$logn batch=$batch" >&2
        return 1
    }
    if [[ ! -s "$RAW" ]]; then
        printf 'variant,trial,%s\n' "$header" >"$RAW"
    fi
    printf '%s,%s,%s\n' "$variant" "$trial" "$record" >>"$RAW"
}

variants=(v06_hybrid2d v07_full v07_resident_generic v07_resident_core)

if [[ $VERIFY != 0 ]]; then
    printf '[preflight] checking forward correctness at batch=1\n'
    for bits in $WORD_BITS; do
        for logn in $LOGNS; do
            for variant in "${variants[@]}"; do
                run_bench "$bits" "$logn" 1 "$variant" 0 0
            done
            if [[ $logn == 20 ]]; then
                run_bench "$bits" "$logn" 1 v06_resident 0 0
            fi
        done
    done
fi

printf '[timed] word_bits={%s} logN={%s} batch={%s} trials=%s\n' \
    "$WORD_BITS" "$LOGNS" "$BATCHES" "$TRIALS"
for bits in $WORD_BITS; do
    for logn in $LOGNS; do
        point_variants=("${variants[@]}")
        [[ $logn == 20 ]] && point_variants+=(v06_resident)
        for batch in $BATCHES; do
            for ((trial = 1; trial <= TRIALS; ++trial)); do
                offset=$(((trial - 1) % ${#point_variants[@]}))
                for ((index = 0; index < ${#point_variants[@]}; ++index)); do
                    variant=${point_variants[$(((index + offset) % ${#point_variants[@]}))]}
                    printf '  u%s logN=%s batch=%s trial=%s %s\n' \
                        "$bits" "$logn" "$batch" "$trial" "$variant"
                    run_bench "$bits" "$logn" "$batch" "$variant" "$trial" 1
                done
            done
        done
    done
done

python3 "$ROOT/scripts/summarize_resident_v06_comparison.py" "$RAW" \
    --csv "$OUTPUT_DIR/summary.csv" --markdown "$OUTPUT_DIR/analysis.md"
printf 'wrote %s\n' "$OUTPUT_DIR/analysis.md"
