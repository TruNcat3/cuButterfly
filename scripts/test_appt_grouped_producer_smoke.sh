#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-cuda118/cuntt_bench"}
for bits in 32 64; do
    if [[ $bits == 32 ]]; then
        modulus=998244353
        resident=3
        roles=6,14,2
    else
        modulus=576460756061519873
        resident=2
        roles=9,11,2
    fi
    for core in appt-online-register-tail-grouped \
                appt-online-register-tail-grouped-writer-final \
                appt-online-register-tail-grouped-writer-final-data-time; do
        for group in 8 16 32; do
            "$BIN" --word-bits "$bits" --modulus "$modulus" --logN 20 \
                --batch 1 --backend hierarchical-dataflow \
                --stage-partition 7,7,6 --segment-cores "$core" \
                --segment-units 8 --segment-data-space "$group",16,16 \
                --segment-data-time 4 --segment-role-stages 2 \
                --segment-token-interleave 1 --boundary-storage ring \
                --boundary-buffers 2 --target-ctas-per-sm "$resident" \
                --appt-role-weights "$roles" --appt-fragment-width 16 \
                --appt-writer-tiles 2 --output-order natural \
                --warmup 0 --repeat 1 --verify
        done
    done
done
