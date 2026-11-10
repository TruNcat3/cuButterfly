#!/usr/bin/env python3
"""Summarize measured rows from the standalone core/dataflow ablation.

The input is the CSV emitted by ``cubutterfly_core_dataflow_ablation``.  The
tool deliberately refuses to infer missing cells or correctness: incomplete
and negative groups are retained in the report and are excluded from ratio
claims.  All ratios use the median steady-state kernel time per cell.
"""
from __future__ import print_function

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {
    "logN", "batch", "core", "organization", "trial", "kernel_ms", "correct",
    "max_error", "max_reference", "relative_error", "l2_error", "l2_reference",
    "relative_l2_error", "points"
}
VARIANTS = (
    ("native", "stagewise-hbm"),
    ("native", "resident-prefix-2"),
    ("cufftdx-thread", "stagewise-hbm"),
    ("cufftdx-thread", "resident-prefix-2"),
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def expected_groups(config):
    result = {}
    for case in config["cases"]:
        workload = case["workload"]
        log_n = int(workload["logN"])
        result[(log_n, int(workload["latency_batch"]))] = (case["id"], "latency")
        result[(log_n, int(workload["throughput_batch"]))] = (case["id"], "throughput")
    return result


def load_rows(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as source:
        # Multiple runbook commands may be concatenated with ``cat``; skip
        # repeated headers without weakening field validation.
        reader = csv.DictReader((line for line in source if line.strip()))
        if not reader.fieldnames or not REQUIRED.issubset(set(reader.fieldnames)):
            missing = sorted(REQUIRED - set(reader.fieldnames or []))
            raise ValueError("CSV is missing required fields: {}".format(", ".join(missing)))
        for raw in reader:
            if raw.get("device") == "device":
                continue
            try:
                row = dict(raw)
                row["logN"] = int(row["logN"])
                row["batch"] = int(row["batch"])
                row["trial"] = int(row["trial"])
                row["kernel_ms"] = float(row["kernel_ms"])
                row["correct"] = int(row["correct"])
                for field in ("max_error", "max_reference", "relative_error", "l2_error",
                              "l2_reference", "relative_l2_error"):
                    row[field] = float(row[field])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("invalid CSV row: {}".format(error))
            if row["core"] not in ("native", "cufftdx-thread"):
                raise ValueError("unknown core: {}".format(row["core"]))
            if row["organization"] not in ("stagewise-hbm", "resident-prefix-2"):
                raise ValueError("unknown organization: {}".format(row["organization"]))
            rows.append(row)
    if not rows:
        raise ValueError("CSV contains no measurement rows")
    return rows


def median_cell(rows):
    values = [row["kernel_ms"] for row in rows]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        return None
    return statistics.median(values)


def ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def geometric_mean(values):
    values = [value for value in values if value is not None]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def comparison_summary(values):
    values = [value for value in values if value is not None]
    if not values:
        return {
            "count": 0,
            "geomean": None,
            "faster": 0,
            "parity": 0,
            "slower": 0,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "geomean": geometric_mean(values),
        "faster": sum(value > 1.0 for value in values),
        "parity": sum(value == 1.0 for value in values),
        "slower": sum(value < 1.0 for value in values),
        "min": min(values),
        "max": max(values),
    }


def summarize(rows, config):
    expected = expected_groups(config)
    grouped = {}
    for row in rows:
        group_key = (row["logN"], row["batch"])
        if group_key not in expected:
            raise ValueError("measurement row is outside config: {}".format(group_key))
        key = group_key + (row["core"], row["organization"])
        grouped.setdefault(key, []).append(row)

    groups = []
    for group_key, (case_id, cohort) in sorted(expected.items()):
        cells = {}
        negative_cells = []
        for core, organization in VARIANTS:
            key = group_key + (core, organization)
            cell_rows = grouped.get(key, [])
            label = core + "/" + organization
            median = median_cell(cell_rows)
            all_correct = bool(cell_rows) and all(row["correct"] == 1 for row in cell_rows)
            expected_trials = int(config["protocol"]["trials"])
            trials = sorted(set(row["trial"] for row in cell_rows))
            accepted = bool(cell_rows) and len(trials) >= expected_trials and all_correct and median is not None
            cells[label] = {
                "rows": len(cell_rows),
                "trials": trials,
                "median_kernel_ms": median,
                "all_correct": all_correct,
                "accepted_for_ratio": accepted,
                "relative_max_error_max": max((row["relative_error"] for row in cell_rows), default=None),
                "relative_l2_error_max": max((row["relative_l2_error"] for row in cell_rows), default=None),
            }
            if not accepted:
                negative_cells.append({
                    "cell": label,
                    "reason": "missing rows/trials, correctness failure, or non-finite timing",
                    "rows": len(cell_rows),
                    "trials": trials,
                    "all_correct": all_correct,
                })

        native_stage = cells["native/stagewise-hbm"]
        native_resident = cells["native/resident-prefix-2"]
        dx_stage = cells["cufftdx-thread/stagewise-hbm"]
        dx_resident = cells["cufftdx-thread/resident-prefix-2"]
        resident_speedup = {
            "native": ratio(native_stage["median_kernel_ms"], native_resident["median_kernel_ms"])
            if native_stage["accepted_for_ratio"] and native_resident["accepted_for_ratio"] else None,
            "cufftdx-thread": ratio(dx_stage["median_kernel_ms"], dx_resident["median_kernel_ms"])
            if dx_stage["accepted_for_ratio"] and dx_resident["accepted_for_ratio"] else None,
        }
        core_speedup = {
            "stagewise-hbm": ratio(native_stage["median_kernel_ms"], dx_stage["median_kernel_ms"])
            if native_stage["accepted_for_ratio"] and dx_stage["accepted_for_ratio"] else None,
            "resident-prefix-2": ratio(native_resident["median_kernel_ms"], dx_resident["median_kernel_ms"])
            if native_resident["accepted_for_ratio"] and dx_resident["accepted_for_ratio"] else None,
        }
        interaction = None
        if core_speedup["stagewise-hbm"] is not None and core_speedup["resident-prefix-2"] is not None:
            interaction = {
                # >1 means the core replacement benefits more in the resident
                # organization; <1 means its relative benefit is reduced.
                "resident_over_stagewise_core_speedup": (
                    core_speedup["resident-prefix-2"] / core_speedup["stagewise-hbm"]),
                "log_ratio": math.log(core_speedup["resident-prefix-2"]) -
                             math.log(core_speedup["stagewise-hbm"]),
            }
        groups.append({
            "case": case_id,
            "cohort": cohort,
            "logN": group_key[0],
            "batch": group_key[1],
            "cells": cells,
            "resident_speedup_vs_stagewise": resident_speedup,
            "core_speedup_cufftdx_vs_native": core_speedup,
            "core_dataflow_interaction": interaction,
            "negative_cells": negative_cells,
            "complete_and_accepted": not negative_cells,
        })
    accepted_groups = [group for group in groups if group["complete_and_accepted"]]
    resident_values = {
        core: [group["resident_speedup_vs_stagewise"][core] for group in accepted_groups]
        for core in ("native", "cufftdx-thread")
    }
    core_values = {
        organization: [group["core_speedup_cufftdx_vs_native"][organization] for group in accepted_groups]
        for organization in ("stagewise-hbm", "resident-prefix-2")
    }
    interaction_values = [
        group["core_dataflow_interaction"]["resident_over_stagewise_core_speedup"]
        for group in accepted_groups if group["core_dataflow_interaction"] is not None
    ]
    comparisons = {
        "same_core_dataflow": {
            core: comparison_summary(values) for core, values in resident_values.items()
        },
        "same_dataflow_core": {
            organization: comparison_summary(values) for organization, values in core_values.items()
        },
        "core_dataflow_interaction": comparison_summary(interaction_values),
    }
    return {
        "schema": "cubutterfly-core-dataflow-ablation-analysis-v1",
        "status": "complete" if len(accepted_groups) == len(groups) else "incomplete",
        "input_rows": len(rows),
        "expected_groups": len(groups),
        "accepted_groups": len(accepted_groups),
        "groups": groups,
        "geomean": {
            "same_core_dataflow": {
                core: summary["geomean"] for core, summary in comparisons["same_core_dataflow"].items()
            },
            "same_dataflow_core": {
                organization: summary["geomean"]
                for organization, summary in comparisons["same_dataflow_core"].items()
            },
            "core_dataflow_interaction": comparisons["core_dataflow_interaction"]["geomean"],
        },
        "comparisons": comparisons,
        "interpretation": {
            "resident_speedup": "stagewise median kernel_ms / resident median kernel_ms; >1 is faster",
            "core_speedup": "native median kernel_ms / cuFFTDx Thread median kernel_ms; >1 is faster",
            "interaction": "ratio of resident to stagewise core speedup; no ratio is emitted for incomplete/incorrect groups",
            "geomean": "geometric mean across accepted latency and throughput groups; incomplete/incorrect groups are excluded",
        },
    }


def markdown(report):
    lines = [
        "# Strict Core x Dataflow Ablation Summary", "",
        "Status: `{}`; accepted groups: {}/{}; input rows: {}.".format(
            report["status"], report["accepted_groups"], report["expected_groups"], report["input_rows"]), "",
        "Ratios use per-cell median `kernel_ms`. Missing, incomplete, and incorrect cells are retained below and do not contribute to ratios.", "",
        "| Case | Cohort | Native stagewise | Native resident | cuFFTDx stagewise | cuFFTDx resident | Resident speedup native / dx | Core speedup stage / resident | Interaction |", 
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in report["groups"]:
        cells = group["cells"]
        value = lambda label: cells[label]["median_kernel_ms"]
        resident = group["resident_speedup_vs_stagewise"]
        core = group["core_speedup_cufftdx_vs_native"]
        interaction = group["core_dataflow_interaction"]
        fmt = lambda number: "-" if number is None else "{:.6g}".format(number)
        lines.append("| {} | {} | {} | {} | {} | {} | {} / {} | {} / {} | {} |".format(
            group["case"], group["cohort"], fmt(value("native/stagewise-hbm")),
            fmt(value("native/resident-prefix-2")), fmt(value("cufftdx-thread/stagewise-hbm")),
            fmt(value("cufftdx-thread/resident-prefix-2")), fmt(resident["native"]),
            fmt(resident["cufftdx-thread"]), fmt(core["stagewise-hbm"]),
            fmt(core["resident-prefix-2"]), fmt(interaction["resident_over_stagewise_core_speedup"] if interaction else None)))
    lines.extend(["", "## Negative or incomplete cells", ""])
    negatives = [(group["case"], group["cohort"], item)
                 for group in report["groups"] for item in group["negative_cells"]]
    if not negatives:
        lines.append("None.")
    else:
        for case, cohort, item in negatives:
            lines.append("- `{}/{} {}`: rows={}, trials={}, all_correct={}".format(
                case, cohort, item["cell"], item["rows"], item["trials"], item["all_correct"]))
    lines.extend(["", "## Excluded attempts", ""])
    excluded = report.get("excluded_attempts", [])
    if not excluded:
        lines.append("None.")
    else:
        lines.append("These telemetry attempts are retained as provenance and are not included in the 72-row analysis:")
        for item in excluded:
            lines.append("- `{}`: status={}, stdout_rows={}, pid_active={}; {}".format(
                item["metadata"], item["status"], item["stdout_rows"], item["pid_active"], item["reason"]))
    lines.extend(["", "## Geometric-mean effects", "",
                  "Geometric means use the accepted per-group median ratios; values above 1 indicate the named numerator effect.", "",
                  "| Comparison | Geomean | Faster | Parity | Slower |", "|---|---:|---:|---:|---:|"])
    comparison_labels = [
        ("same_core_dataflow", "native", "native: resident-prefix-2 vs stagewise-hbm"),
        ("same_core_dataflow", "cufftdx-thread", "cufftdx-thread: resident-prefix-2 vs stagewise-hbm"),
        ("same_dataflow_core", "stagewise-hbm", "stagewise-hbm: cuFFTDx Thread vs native"),
        ("same_dataflow_core", "resident-prefix-2", "resident-prefix-2: cuFFTDx Thread vs native"),
    ]
    for axis, key, label in comparison_labels:
        item = report["comparisons"][axis][key]
        lines.append("| {} | {} | {} | {} | {} |".format(
            label, "-" if item["geomean"] is None else "{:.6g}".format(item["geomean"]),
            item["faster"], item["parity"], item["slower"]))
    interaction = report["comparisons"]["core_dataflow_interaction"]
    lines.append("| core/dataflow interaction (resident / stagewise) | {} | {} | {} | {} |".format(
        "-" if interaction["geomean"] is None else "{:.6g}".format(interaction["geomean"]),
        interaction["faster"], interaction["parity"], interaction["slower"]))
    return "\n".join(lines) + "\n"


def excluded_attempts(output_dir):
    journal = Path(output_dir) / "journal"
    attempts = []
    if not journal.is_dir():
        return attempts
    for metadata in sorted(journal.glob("*.telemetry.json")):
        try:
            record = read_json(metadata)
        except (OSError, ValueError, TypeError):
            continue
        if record.get("status") == "completed":
            continue
        stdout = Path(record.get("files", {}).get("stdout", ""))
        try:
            stdout_rows = max(0, sum(1 for _ in stdout.open(encoding="utf-8")) - 1)
        except OSError:
            stdout_rows = 0
        pid = record.get("pid")
        attempts.append({
            "metadata": metadata.name,
            "status": record.get("status"),
            "pid": pid,
            "pid_active": bool(pid and Path("/proc").joinpath(str(pid)).exists()),
            "stdout_rows": stdout_rows,
            "reason": "non-completed telemetry is excluded from accepted measurements",
        })
    return attempts


def analyze(input_path, config_path, output_dir):
    config = read_json(config_path)
    output_dir = Path(output_dir).expanduser().resolve()
    report = summarize(load_rows(input_path), config)
    report["excluded_attempts"] = excluded_attempts(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    document = markdown(report)
    (output_dir / "SUMMARY.md").write_text(document, encoding="utf-8")
    (output_dir / "REPORT.md").write_text(document, encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/workloads/core_dataflow_ablation.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(args.input, args.config, args.output_dir)
    print(json.dumps({"status": report["status"], "input_rows": report["input_rows"],
                      "accepted_groups": report["accepted_groups"],
                      "expected_groups": report["expected_groups"],
                      "output": str(Path(args.output_dir).resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
