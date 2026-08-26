#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_packed_stage6_lane32_schedule"}
METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,lts__t_sector_hit_rate.pct,smsp__inst_executed.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,launch__registers_per_thread}

mkdir -p "$OUTPUT_DIR"
restore_owner() {
    if [[ ${EUID:-$(id -u)} -eq 0 && -n ${SUDO_USER:-} ]]; then
        chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$OUTPUT_DIR"
    fi
}
trap restore_owner EXIT

for executable in "$NCU" "$BIN"; do
    [[ -x "$executable" ]] || {
        echo "missing executable: $executable" >&2
        exit 1
    }
done

common=(--logN 20 --batch 16 --word-bits 32 --modulus 998244353
        --backend hierarchical-dataflow --stage-partition 10,10
        --boundary-storage full-scratch --segment-data-time 1,1
        --cross-twiddle fused --mod-multiply shoup)
lane32=(--segment-cores
        homogeneous-warp128-vector-radix4-packed-stage6-distributed-static-io
        --segment-threads 256 --segment-units 8 --segment-data-space 4
        --segment-coefficient-reuse-stages 6 --target-ctas-per-sm 2)

profile() {
    local label=$1
    shift
    local output="$OUTPUT_DIR/${label}.csv"
    local temporary="$output.tmp"
    rm -f "$temporary"
    "$NCU" --target-processes all --replay-mode kernel --cache-control all \
        --clock-control base --launch-count 1 --metrics "$METRICS" \
        --page raw --csv --force-overwrite --log-file "$temporary" \
        "$BIN" "$@" --warmup 0 --repeat 1 --csv
    grep -q '"Kernel Name"' "$temporary" || {
        cat "$temporary" >&2
        rm -f "$temporary"
        echo "NCU did not produce a raw CSV table for $label" >&2
        return 1
    }
    mv "$temporary" "$output"
}

echo "[preflight] lane32 86:74 forward/inverse"
"$BIN" "${common[@]}" "${lane32[@]}" --segment-cta-weights 86,74 \
    --warmup 0 --repeat 1 --verify >/dev/null
"$BIN" "${common[@]}" "${lane32[@]}" --segment-cta-weights 86,74 \
    --inverse --warmup 0 --repeat 1 --verify >/dev/null

profile v06 "${common[@]}" --segment-cores dataflow-radix4 \
    --segment-cta-weights 17,13 --target-ctas-per-sm 4
profile d7 "${common[@]}" \
    --segment-cores homogeneous-warp128-vector-radix4-static-io \
    --segment-threads 256 --segment-units 8 --segment-data-space 4 \
    --segment-coefficient-reuse-stages 7 --segment-cta-weights 85,75 \
    --target-ctas-per-sm 2
for producer in 84 85 86; do
    consumer=$((160 - producer))
    profile "lane32_${producer}_${consumer}" "${common[@]}" "${lane32[@]}" \
        --segment-cta-weights "$producer,$consumer"
done

python3 "$ROOT/scripts/summarize_ncu.py" \
    "$OUTPUT_DIR/v06.csv" "$OUTPUT_DIR/d7.csv" \
    "$OUTPUT_DIR/lane32_84_76.csv" "$OUTPUT_DIR/lane32_85_75.csv" \
    "$OUTPUT_DIR/lane32_86_74.csv" --output "$OUTPUT_DIR/summary.csv"
python3 "$ROOT/scripts/analyze_packed_stage6_lane32_schedule_ncu.py" \
    "$OUTPUT_DIR/summary.csv" --output "$OUTPUT_DIR/analysis.md"
printf 'wrote %s and %s\n' "$OUTPUT_DIR/summary.csv" "$OUTPUT_DIR/analysis.md"
