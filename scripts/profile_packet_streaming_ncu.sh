#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_packet_compute_layout"}
MIN_FREE_GPU_MIB=${MIN_FREE_GPU_MIB:-2048}
METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__inst_executed.sum,smsp__sass_inst_executed_op_global.sum,smsp__sass_inst_executed_op_shared.sum,smsp__inst_executed_pipe_cbu.sum,smsp__inst_executed_pipe_lsu.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block}

mkdir -p "$OUTPUT_DIR"
restore_owner() {
    if [[ ${EUID:-$(id -u)} -eq 0 && -n ${SUDO_USER:-} ]]; then
        chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$OUTPUT_DIR"
    fi
}
trap restore_owner EXIT

for executable in "$NCU" "$BIN"; do
    [[ -x "$executable" ]] || { echo "missing executable: $executable" >&2; exit 1; }
done

if command -v nvidia-smi >/dev/null; then
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits |
        head -n 1 | tr -d '[:space:]')
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib < MIN_FREE_GPU_MIB )); then
        echo "error: only ${free_mib} MiB GPU memory is free; need ${MIN_FREE_GPU_MIB} MiB" >&2
        nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
            --format=csv,noheader 2>/dev/null >&2 || true
        exit 1
    fi
fi

common=(--logN 20 --word-bits 32 --modulus 998244353
        --backend hierarchical-dataflow --stage-partition 10,10
        --boundary-storage full-scratch --segment-data-time 1,1
        --segment-cores homogeneous-warp128-packet-shared-radix4-static-io
        --segment-threads 128 --segment-units 4 --segment-data-space 4
        --target-ctas-per-sm 4 --cross-twiddle fused --mod-multiply shoup)

profile() {
    local label=$1
    shift
    local output="$OUTPUT_DIR/${label}.csv"
    local temporary="$output.tmp"
    echo "[ncu] $label"
    rm -f "$temporary"
    "$NCU" --target-processes all --replay-mode kernel --cache-control all \
        --clock-control base --launch-count 1 --metrics "$METRICS" \
        --page raw --csv --force-overwrite --log-file "$temporary" \
        "$BIN" "${common[@]}" "$@" --warmup 0 --repeat 1 --csv
    grep -q '"Kernel Name"' "$temporary" || {
        cat "$temporary" >&2
        rm -f "$temporary"
        echo "NCU did not produce a raw CSV table for $label" >&2
        return 1
    }
    mv "$temporary" "$output"
}

echo "[preflight] bitmap interleaved and warp-row compute layouts"
"$BIN" "${common[@]}" --batch 1 --segment-cta-weights 10,10 \
    --segment-token-interleave 0,16 --ready-window 2 \
    --packet-readiness wave-bitmap \
    --warmup 0 --repeat 1 --verify >/dev/null
"$BIN" "${common[@]}" --batch 1 --segment-cta-weights 10,10 \
    --segment-token-interleave 0,16 --ready-window 2 \
    --packet-readiness wave-bitmap --inverse \
    --warmup 0 --repeat 1 --verify >/dev/null
"$BIN" "${common[@]}" --batch 1 --segment-cta-weights 10,10 \
    --segment-token-interleave 0,16 --ready-window 2 \
    --packet-readiness wave-bitmap --packet-compute-layout warp-rows \
    --warmup 0 --repeat 1 --verify >/dev/null
"$BIN" "${common[@]}" --batch 1 --segment-cta-weights 10,10 \
    --segment-token-interleave 0,16 --ready-window 2 \
    --packet-readiness wave-bitmap --packet-compute-layout warp-rows \
    --inverse --warmup 0 --repeat 1 --verify >/dev/null

profile aggregate_b16 --batch 16 --segment-cta-weights 7,13 \
    --segment-token-interleave 0,0
profile bitmap_interleaved_b16 --batch 16 --segment-cta-weights 7,13 \
    --segment-token-interleave 0,16 --ready-window 16 \
    --packet-readiness wave-bitmap
profile bitmap_warp_rows_b16 --batch 16 --segment-cta-weights 7,13 \
    --segment-token-interleave 0,16 --ready-window 16 \
    --packet-readiness wave-bitmap --packet-compute-layout warp-rows
profile aggregate_b32 --batch 32 --segment-cta-weights 5,15 \
    --segment-token-interleave 0,0
profile bitmap_interleaved_b32 --batch 32 --segment-cta-weights 5,15 \
    --segment-token-interleave 0,16 --ready-window 8 \
    --packet-readiness wave-bitmap
profile bitmap_warp_rows_b32 --batch 32 --segment-cta-weights 5,15 \
    --segment-token-interleave 0,16 --ready-window 8 \
    --packet-readiness wave-bitmap --packet-compute-layout warp-rows
profile aggregate_b64 --batch 64 --segment-cta-weights 5,15 \
    --segment-token-interleave 0,0
profile bitmap_interleaved_b64 --batch 64 --segment-cta-weights 5,15 \
    --segment-token-interleave 0,16 --ready-window 8 \
    --packet-readiness wave-bitmap
profile bitmap_warp_rows_b64 --batch 64 --segment-cta-weights 5,15 \
    --segment-token-interleave 0,16 --ready-window 8 \
    --packet-readiness wave-bitmap --packet-compute-layout warp-rows

python3 "$ROOT/scripts/summarize_ncu.py" "$OUTPUT_DIR"/*.csv \
    --output "$OUTPUT_DIR/summary.csv"
python3 "$ROOT/scripts/analyze_packet_compute_layout_ncu.py" \
    "$OUTPUT_DIR/summary.csv" --output "$OUTPUT_DIR/analysis.md"
printf 'wrote %s and %s\n' "$OUTPUT_DIR/summary.csv" "$OUTPUT_DIR/analysis.md"
