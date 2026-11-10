#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/build-11.8/cuntt_bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/hierarchical_dataflow"}
MODE=${MODE:-screen}
WARMUP=${WARMUP:-20}
REPEAT=${REPEAT:-100}

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 1; }
case "$MODE" in
    screen|space|cores|all) ;;
    *) echo "MODE must be screen, space, cores, or all" >&2; exit 1 ;;
esac

RAW_DIR="$OUTPUT_DIR/raw/$MODE"
rm -rf "$RAW_DIR"
mkdir -p "$RAW_DIR"
case_index=0

run() {
    local label=$1
    shift
    case_index=$((case_index + 1))
    printf '[%03d] %s\n' "$case_index" "$label"
    "$BIN" "$@" --warmup "$WARMUP" --repeat "$REPEAT" --verify --csv \
        > "$RAW_DIR/${label}.csv"
}

screen() {
    for word_bits in 32 64; do
        if [[ $word_bits -eq 32 ]]; then modulus=998244353; else modulus=576460756061519873; fi
        for log_n in 12 14 16 18 20; do
            batch=$((1 << (22 - log_n)))
            common=(--word-bits "$word_bits" --modulus "$modulus" --logN "$log_n" --batch "$batch")
            run "hybrid2d_w${word_bits}_n${log_n}_b${batch}" "${common[@]}" \
                --backend hybrid2d --compute-unit radix4
            run "hierarchical_w${word_bits}_n${log_n}_b${batch}" "${common[@]}" \
                --backend hierarchical-barrier
        done
    done
}

space() {
    for word_bits in 32 64; do
        if [[ $word_bits -eq 32 ]]; then modulus=998244353; else modulus=576460756061519873; fi
        common=(--word-bits "$word_bits" --modulus "$modulus" --logN 20 --batch 4)
        run "hybrid2d_w${word_bits}_n20_b4" "${common[@]}" --backend hybrid2d --compute-unit radix4
        for rows in 1 2 4; do
            for data_time in 1 2 4 8; do
                for threads in 128 256 512; do
                    run "hierarchical_w${word_bits}_n20_b4_r${rows}_td${data_time}_t${threads}" \
                        "${common[@]}" --backend hierarchical-barrier --rows-per-block "$rows" \
                        --data-time "$data_time" --threads-per-block "$threads"
                done
            done
        done
    done
}

cores() {
    for word_bits in 32 64; do
        if [[ $word_bits -eq 32 ]]; then modulus=998244353; rows=4; threads=256
        else modulus=576460756061519873; rows=2; threads=512; fi
        for core in dataflow-radix4 hybrid2d-radix4; do
            run "hierarchical_w${word_bits}_n20_b4_${core}" \
                --word-bits "$word_bits" --modulus "$modulus" --logN 20 --batch 4 \
                --backend hierarchical-barrier --hierarchical-core "$core" \
                --rows-per-block "$rows" --data-time 1 --threads-per-block "$threads"
        done
    done
}

[[ $MODE == screen || $MODE == all ]] && screen
[[ $MODE == space || $MODE == all ]] && space
[[ $MODE == cores || $MODE == all ]] && cores

python3 "$ROOT/scripts/summarize_hierarchical_dataflow.py" "$RAW_DIR"/*.csv \
    --csv "$OUTPUT_DIR/summary_${MODE}.csv" --markdown "$OUTPUT_DIR/summary_${MODE}.md"
printf 'HierarchicalDataflow summary: %s\n' "$OUTPUT_DIR/summary_${MODE}.md"
