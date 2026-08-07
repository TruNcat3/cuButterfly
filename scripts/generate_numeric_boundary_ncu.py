#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import shlex

from run_comprehensive_suite import ntt_prime


METRICS = (
    "gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed", "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_hit_rate.pct", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", "smsp__inst_executed.sum",
    "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fp64_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_integer_pred_on.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "launch__registers_per_thread", "launch__shared_mem_per_block",
    "launch__waves_per_multiprocessor", "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_warps",
)


def _sanitize(value):
    return "".join(character if character.isalnum() else "_" for character in str(value)).strip("_")


def profile_label(row, mapping):
    return "_".join(map(_sanitize, (row["event_id"], row["status"], row["operator"],
                                       row["precision"], f"log{row['logN']}",
                                       f"b{row['batch']}", mapping)))


def generate(manifest, analysis):
    cases = {case["id"]: case for case in manifest["cases"]}
    by_shape_mapping = {}
    for case in cases.values():
        key = (case["operator"], case["precision"], case.get("accumulation", "native"),
               str(case["logN"]), str(case["batch"]), case["mapping_id"])
        by_shape_mapping[key] = case
    unstable = [row for row in analysis if row["status"] in ("confirmed-reversed", "not-confirmed")]
    commands = []
    for row in unstable:
        for mapping in (row["quick_from_mapping"], row["quick_to_mapping"]):
            key = (row["operator"], row["precision"], row["accumulation"],
                   row["logN"], row["batch"], mapping)
            case = by_shape_mapping.get(key)
            if case is None:
                raise ValueError(f"missing profiled case for {key}")
            binary = "$NTT_BIN" if case["runner"] == "ntt" else "$BUTTERFLY_BIN"
            arguments = [*map(str, case["args"]), "--logN", str(case["logN"]),
                         "--batch", str(case["batch"])]
            if case["runner"] == "ntt":
                arguments += ["--modulus", str(ntt_prime(int(case["modulus_bits"]), int(case["logN"])))]
            arguments += ["--warmup", "0", "--repeat", "1", "--csv"]
            label = profile_label(row, mapping)
            quoted = " ".join(shlex.quote(argument) for argument in arguments)
            commands.append(f"profile {shlex.quote(label)} \"{binary}\" {quoted}")
    lines = [
        "#!/usr/bin/env bash", "set -euo pipefail", "",
        'ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)',
        'NCU=${NCU:-/usr/local/cuda-11.8/bin/ncu}',
        'BUTTERFLY_BIN=${BUTTERFLY_BIN:-"$ROOT/build/cubutterfly_bench"}',
        'NTT_BIN=${NTT_BIN:-"$ROOT/build/cuntt_bench"}',
        'OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/ncu_numeric_boundaries"}',
        f'METRICS=${{METRICS:-{",".join(METRICS)}}}', "",
        'for executable in "$NCU" "$BUTTERFLY_BIN" "$NTT_BIN"; do',
        '    [[ -x "$executable" ]] || { echo "missing executable: $executable" >&2; exit 1; }',
        "done", 'mkdir -p "$OUTPUT_DIR"', "",
        "restore_owner() {",
        '    if [[ ${EUID:-$(id -u)} -eq 0 && -n ${SUDO_USER:-} ]]; then',
        '        chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$OUTPUT_DIR"',
        "    fi", "}",
        "trap restore_owner EXIT", "",
        "profile() {", "    local label=$1", "    shift",
        '    "$NCU" --target-processes all --replay-mode kernel --cache-control none --clock-control base \\',
        '        --launch-count 4 --metrics "$METRICS" --page raw --csv --force-overwrite \\',
        '        --log-file "$OUTPUT_DIR/${label}.csv" "$@"', "}", "",
        *commands, "",
        'python3 "$ROOT/scripts/summarize_ncu.py" "$OUTPUT_DIR"/*.csv --output "$OUTPUT_DIR/summary.csv"',
        'python3 "$ROOT/scripts/analyze_numeric_boundary_ncu.py" "$OUTPUT_DIR/summary.csv" \\',
        '    --output "$OUTPUT_DIR/analysis.csv" --report "$OUTPUT_DIR/analysis.md"',
        'printf "Numeric-boundary NCU attribution: %s\\n" "$OUTPUT_DIR/analysis.md"', "",
    ]
    return "\n".join(lines), len(unstable), len(commands)


def main():
    parser = argparse.ArgumentParser(description="Generate paired NCU commands for unstable numeric boundaries.")
    parser.add_argument("--manifest", type=pathlib.Path,
                        default=pathlib.Path("results/v100_numeric_confirmed_followup_suite.json"))
    parser.add_argument("--analysis", type=pathlib.Path,
                        default=pathlib.Path("results/v100_numeric_confirmed_followup_analysis.csv"))
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("scripts/profile_numeric_boundaries_ncu.sh"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    with args.analysis.open(newline="") as source:
        analysis = list(csv.DictReader(source))
    rendered, events, commands = generate(manifest, analysis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    args.output.chmod(0o755)
    print(f"events={events} profiles={commands} output={args.output}")


if __name__ == "__main__":
    main()
