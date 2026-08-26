#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}
BIN=${BIN:-"$ROOT/build-cuda118-gcc9/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_packet_shared_radix4"}
MIN_FREE_GPU_MIB=${MIN_FREE_GPU_MIB:-2048}
METRICS=${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__inst_executed.sum,smsp__sass_inst_executed_op_global.sum,smsp__sass_inst_executed_op_shared.sum,smsp__inst_executed_pipe_adu.sum,smsp__inst_executed_pipe_alu.sum,smsp__inst_executed_pipe_cbu.sum,smsp__inst_executed_pipe_lsu.sum,smsp__inst_executed_pipe_xu.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,smsp__sass_thread_inst_executed_op_memory_pred_on.sum,smsp__sass_thread_inst_executed_op_control_pred_on.sum,smsp__sass_thread_inst_executed_op_misc_pred_on.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,launch__registers_per_thread,launch__shared_mem_per_block}

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

check_gpu_memory() {
    command -v nvidia-smi >/dev/null || return 0
    local free_mib
    free_mib=$(nvidia-smi --query-gpu=memory.free \
        --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')
    [[ "$free_mib" =~ ^[0-9]+$ ]] || return 0
    if (( free_mib >= MIN_FREE_GPU_MIB )); then
        echo "[preflight] GPU free memory: ${free_mib} MiB"
        return 0
    fi
    echo "error: only ${free_mib} MiB GPU memory is free; this profile requires at least ${MIN_FREE_GPU_MIB} MiB" >&2
    echo "active GPU compute processes:" >&2
    while IFS=',' read -r pid process used_mib; do
        pid=${pid//[[:space:]]/}
        process=${process#${process%%[![:space:]]*}}
        used_mib=${used_mib//[[:space:]]/}
        local owner="unknown"
        [[ "$pid" =~ ^[0-9]+$ ]] && owner=$(ps -o user= -p "$pid" 2>/dev/null | xargs || true)
        printf '  pid=%s user=%s memory=%s MiB process=%s\n' \
            "$pid" "${owner:-unknown}" "$used_mib" "$process" >&2
    done < <(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null || true)
    echo "wait for those jobs to finish or ask their owner before terminating them" >&2
    return 1
}

check_gpu_memory

common=(--logN 20 --batch 16 --word-bits 32 --modulus 998244353
        --backend hierarchical-dataflow --stage-partition 10,10
        --boundary-storage full-scratch --segment-data-time 1,1
        --cross-twiddle fused --mod-multiply shoup)
packet=(--segment-cores homogeneous-warp128-packet-shared-radix4-static-io
        --segment-threads 128 --segment-units 4 --segment-data-space 4
        --segment-coefficient-reuse-stages 0 --segment-cta-weights 9,11
        --target-ctas-per-sm 4)

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
        "$BIN" "$@" --warmup 0 --repeat 1 --csv
    grep -q '"Kernel Name"' "$temporary" || {
        cat "$temporary" >&2
        rm -f "$temporary"
        echo "NCU did not produce a raw CSV table for $label" >&2
        return 1
    }
    mv "$temporary" "$output"
}

echo "[preflight] packet128 forward/inverse"
"$BIN" "${common[@]}" "${packet[@]}" --warmup 0 --repeat 1 --verify >/dev/null
"$BIN" "${common[@]}" "${packet[@]}" --inverse --warmup 0 --repeat 1 --verify >/dev/null

profile v06 "${common[@]}" --segment-cores dataflow-radix4 \
    --segment-cta-weights 17,13 --target-ctas-per-sm 4
profile d7 "${common[@]}" \
    --segment-cores homogeneous-warp128-vector-radix4-static-io \
    --segment-threads 256 --segment-units 8 --segment-data-space 4 \
    --segment-coefficient-reuse-stages 7 --segment-cta-weights 85,75 \
    --target-ctas-per-sm 2
profile lane32 "${common[@]}" \
    --segment-cores homogeneous-warp128-vector-radix4-packed-stage6-distributed-static-io \
    --segment-threads 256 --segment-units 8 --segment-data-space 4 \
    --segment-coefficient-reuse-stages 6 --segment-cta-weights 84,76 \
    --target-ctas-per-sm 2
profile packet128 "${common[@]}" "${packet[@]}"

python3 "$ROOT/scripts/summarize_ncu.py" "$OUTPUT_DIR"/*.csv \
    --output "$OUTPUT_DIR/summary.csv"
python3 "$ROOT/scripts/analyze_packet_shared_radix4_ncu.py" \
    "$OUTPUT_DIR/summary.csv" --output "$OUTPUT_DIR/analysis.md"
printf 'wrote %s and %s\n' "$OUTPUT_DIR/summary.csv" "$OUTPUT_DIR/analysis.md"
