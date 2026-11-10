#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/homogeneous_warp_10x10"}
BATCHES=${BATCHES:-1,4,16}
TRIALS=${TRIALS:-2}
WARMUP=${WARMUP:-2}
REPEAT=${REPEAT:-10}

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
raw="$OUTPUT_DIR/raw.csv"
: > "$raw"

run() {
    local trial=$1 variant=$2 threads=$3 units=$4 resident=$5
    shift 5
    local output header record
    output=$("$BIN" "$@" --warmup "$WARMUP" --repeat "$REPEAT" --csv)
    header=$(printf '%s\n' "$output" | sed -n '/^device,/{p;q;}')
    record=$(printf '%s\n' "$output" | tail -n 1)
    if [[ ! -s "$raw" ]]; then
        printf 'trial,variant,scan_threads,scan_units,scan_resident,%s\n' \
            "$header" > "$raw"
    fi
    printf '%s,%s,%s,%s,%s,%s\n' "$trial" "$variant" "$threads" \
        "$units" "$resident" "$record" >> "$raw"
}

IFS=, read -ra selected_batches <<< "$BATCHES"
for bits in 32 64; do
    if [[ $bits == 32 ]]; then
        modulus=998244353; v06_resident=4; half_packet=4
    else
        modulus=576460756061519873; v06_resident=2; half_packet=2
    fi
    for batch in "${selected_batches[@]}"; do
        io_half_weights=9,11
        dual_warp128_weights=8,7
        if [[ $bits == 32 ]]; then
            case $batch in
                1) io_half_weights=11,9; dual_warp128_weights=11,9 ;;
                4) io_half_weights=8,7; dual_warp128_weights=17,13 ;;
                16) io_half_weights=17,13; dual_warp128_weights=8,7 ;;
                *) io_half_weights=11,9 ;;
            esac
        fi
        common=(--logN 20 --batch "$batch" --word-bits "$bits"
                --modulus "$modulus" --backend hierarchical-dataflow
                --stage-partition 10,10 --boundary-storage full-scratch
                --segment-data-time 1,1 --segment-cta-weights 9,11)
        for trial in $(seq 1 "$TRIALS"); do
            run "$trial" v06 256 4 "$v06_resident" "${common[@]}" \
                --segment-cores dataflow-radix4 \
                --target-ctas-per-sm "$v06_resident"
            run "$trial" warp1024 256 8 1 "${common[@]}" \
                --segment-cores homogeneous-warp-radix2 \
                --segment-threads 256 --segment-units 8 \
                --target-ctas-per-sm 1
            run "$trial" warp256 128 4 3 "${common[@]}" \
                --segment-cores homogeneous-warp256-radix2 \
                --segment-threads 128 --segment-units 4 \
                --target-ctas-per-sm 3
            run "$trial" warp256 256 8 1 "${common[@]}" \
                --segment-cores homogeneous-warp256-radix2 \
                --segment-threads 256 --segment-units 8 \
                --target-ctas-per-sm 1
            run "$trial" warp256_static 128 4 2 "${common[@]}" \
                --segment-cores homogeneous-warp256-static-radix2 \
                --segment-threads 128 --segment-units 4 \
                --target-ctas-per-sm 2
            run "$trial" warp256_static 256 8 1 "${common[@]}" \
                --segment-cores homogeneous-warp256-static-radix2 \
                --segment-threads 256 --segment-units 8 \
                --target-ctas-per-sm 1
            run "$trial" warp256_static_io 128 4 2 "${common[@]}" \
                --segment-cores homogeneous-warp256-static-io-radix2 \
                --segment-threads 128 --segment-units 4 \
                --target-ctas-per-sm 2
            run "$trial" warp256_static_io 256 8 1 "${common[@]}" \
                --segment-cores homogeneous-warp256-static-io-radix2 \
                --segment-threads 256 --segment-units 8 \
                --target-ctas-per-sm 1
            run "$trial" warp256_static_half 128 4 3 "${common[@]}" \
                --segment-cores homogeneous-warp256-static-radix2 \
                --segment-threads 128 --segment-units 4 \
                --segment-data-space "$half_packet" \
                --target-ctas-per-sm 3
            run "$trial" warp256_static_half 256 8 1 "${common[@]}" \
                --segment-cores homogeneous-warp256-static-radix2 \
                --segment-threads 256 --segment-units 8 \
                --segment-data-space "$half_packet" \
                --target-ctas-per-sm 1
            run "$trial" warp256_static_io_half 128 4 3 "${common[@]}" \
                --segment-cores homogeneous-warp256-static-io-radix2 \
                --segment-threads 128 --segment-units 4 \
                --segment-data-space "$half_packet" \
                --segment-cta-weights "$io_half_weights" \
                --target-ctas-per-sm 3
            if [[ $bits == 32 ]]; then
                run "$trial" warp256_coefficient_reuse_static_io_half \
                    128 4 3 "${common[@]}" \
                    --segment-cores \
                        homogeneous-warp256-coefficient-reuse-static-io-radix2 \
                    --segment-threads 128 --segment-units 4 \
                    --segment-data-space 4 \
                    --segment-coefficient-reuse-stages 6 \
                    --segment-cta-weights "$io_half_weights" \
                    --target-ctas-per-sm 3
                for reuse_stages in 1 2 3 4 5; do
                    run "$trial" \
                        "warp256_coefficient_reuse${reuse_stages}_static_io_half" \
                        128 4 3 "${common[@]}" \
                        --segment-cores \
                            homogeneous-warp256-coefficient-reuse-static-io-radix2 \
                        --segment-threads 128 --segment-units 4 \
                        --segment-data-space 4 \
                        --segment-coefficient-reuse-stages "$reuse_stages" \
                        --segment-cta-weights "$io_half_weights" \
                        --target-ctas-per-sm 3
                done
            fi
            run "$trial" warp256_static_io_half 256 8 1 "${common[@]}" \
                --segment-cores homogeneous-warp256-static-io-radix2 \
                --segment-threads 256 --segment-units 8 \
                --segment-data-space "$half_packet" \
                --target-ctas-per-sm 1
            if [[ $bits == 32 ]]; then
                run "$trial" warp128_static_io_half 128 4 4 "${common[@]}" \
                    --segment-cores homogeneous-warp128-static-io-radix2 \
                    --segment-threads 128 --segment-units 4 \
                    --segment-data-space 4 \
                    --segment-cta-weights "$io_half_weights" \
                    --target-ctas-per-sm 4
                run "$trial" warp128_static_io_half_dual 256 8 2 \
                    "${common[@]}" \
                    --segment-cores homogeneous-warp128-static-io-radix2 \
                    --segment-threads 256 --segment-units 8 \
                    --segment-data-space 4 \
                    --segment-cta-weights "$dual_warp128_weights" \
                    --target-ctas-per-sm 2
                for reuse_stages in 1 2 3 4 5; do
                    run "$trial" \
                        "warp128_coefficient_reuse${reuse_stages}_static_io_half_dual" \
                        256 8 2 "${common[@]}" \
                        --segment-cores \
                            homogeneous-warp128-coefficient-reuse-static-io-radix2 \
                        --segment-threads 256 --segment-units 8 \
                        --segment-data-space 4 \
                        --segment-coefficient-reuse-stages "$reuse_stages" \
                        --segment-cta-weights "$dual_warp128_weights" \
                        --target-ctas-per-sm 2
                done
                run "$trial" warp128_coefficient_reuse_static_io_half_dual \
                    256 8 2 "${common[@]}" \
                    --segment-cores \
                        homogeneous-warp128-coefficient-reuse-static-io-radix2 \
                    --segment-threads 256 --segment-units 8 \
                    --segment-data-space 4 \
                    --segment-coefficient-reuse-stages 6 \
                    --segment-cta-weights "$dual_warp128_weights" \
                    --target-ctas-per-sm 2
                run "$trial" warp128_vector_radix4_static_io_half_dual \
                    256 8 2 "${common[@]}" \
                    --segment-cores \
                        homogeneous-warp128-vector-radix4-static-io \
                    --segment-threads 256 --segment-units 8 \
                    --segment-data-space 4 \
                    --segment-cta-weights "$dual_warp128_weights" \
                    --target-ctas-per-sm 2
                for reuse_stages in 3 4 5 6; do
                    run "$trial" \
                        "warp128_vector_radix4_reuse${reuse_stages}_static_io_half_dual" \
                        256 8 2 "${common[@]}" \
                        --segment-cores \
                            homogeneous-warp128-vector-radix4-static-io \
                        --segment-threads 256 --segment-units 8 \
                        --segment-data-space 4 \
                        --segment-coefficient-reuse-stages "$reuse_stages" \
                        --segment-cta-weights "$dual_warp128_weights" \
                        --target-ctas-per-sm 2
                done
                run "$trial" warp64_static_io_half 128 4 4 "${common[@]}" \
                    --segment-cores homogeneous-warp64-static-io-radix2 \
                    --segment-threads 128 --segment-units 4 \
                    --segment-data-space 4 \
                    --segment-cta-weights "$io_half_weights" \
                    --target-ctas-per-sm 4
                run "$trial" warp128_pipeline_static_io_half 128 4 4 \
                    "${common[@]}" \
                    --segment-cores \
                        homogeneous-warp128-pipeline-static-io-radix2 \
                    --segment-threads 128 --segment-units 4 \
                    --segment-data-space 4 --pipeline-buffers 1 \
                    --segment-cta-weights "$io_half_weights" \
                    --target-ctas-per-sm 4
                run "$trial" warp128_cooperative_static_io_half 128 4 4 \
                    "${common[@]}" \
                    --segment-cores \
                        homogeneous-warp128-cooperative-static-io-radix2 \
                    --segment-threads 128 --segment-units 4 \
                    --segment-data-space 4 \
                    --segment-cta-weights "$io_half_weights" \
                    --target-ctas-per-sm 4
                run "$trial" warp128_cooperative_static_io_half_occ3 128 4 3 \
                    "${common[@]}" \
                    --segment-cores \
                        homogeneous-warp128-cooperative-static-io-radix2 \
                    --segment-threads 128 --segment-units 4 \
                    --segment-data-space 4 \
                    --segment-cta-weights "$io_half_weights" \
                    --target-ctas-per-sm 3
            fi
        done
    done
done

python3 "$ROOT/scripts/summarize_homogeneous_warp_10x10.py" "$raw" \
    --csv "$OUTPUT_DIR/summary.csv" \
    --markdown "$OUTPUT_DIR/analysis.md"
printf 'Warp-granular 10+10 comparison: %s\n' "$OUTPUT_DIR/analysis.md"
