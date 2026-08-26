#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-11.8/cuntt_bench"}
TIMEOUT=${TIMEOUT:-20}

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }

run() {
    local label=$1
    shift
    printf '%s ... ' "$label"
    if timeout --foreground "$TIMEOUT" "$BIN" "$@" --warmup 0 --repeat 1 --verify >/dev/null; then
        echo PASS
    else
        status=$?
        if [[ $status -eq 124 ]]; then
            echo "TIMEOUT (possible kernel deadlock)" >&2
        else
            echo "FAIL (exit $status)" >&2
        fi
        return "$status"
    fi
}

common=(--backend hybrid-dataflow --logN 8 --batch 1 --word-bits 64 \
        --flow-tile-log 8 --stage-space 8 --data-space 16 --pipeline-buffers 2)
run "atomic handoff" "${common[@]}" --stage-handoff atomic
run "named-barrier handoff" "${common[@]}" --stage-handoff named-barrier
run "named-barrier batch" "${common[@]}" --batch 17 --stage-handoff named-barrier
run "uint32 inverse" --backend hybrid-dataflow --logN 10 --batch 2 --word-bits 32 \
    --modulus 1073479681 --inverse --flow-tile-log 8 --stage-space 8 \
    --data-space 32 --pipeline-buffers 2 --stage-handoff named-barrier
