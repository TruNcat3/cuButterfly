#!/usr/bin/env python3
"""Render frozen A100 80GB acceptance evidence and CPU-only figures.

The default path reads the existing frozen snapshot.  It takes the median of
the three trials for each named measurement in each complete cell and computes
baseline-median / cuButterfly-median geometric means.  Unavailable cells are
retained as explicit pending records and are never encoded as zero-valued
measurements.  Refreshing the snapshot is an explicit opt-in operation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "results/comprehensive_20260917/a100-80gb-gpu1/acceptance.json"
DEFAULT_SNAPSHOT = ROOT / "paper/evidence/current_snapshot.json"
DEFAULT_OUTPUT_DIR = ROOT / "paper/generated"
DEFAULT_FIGURE_DIR = ROOT / "paper/figures"
OUTPUT_SCHEMA = "cubutterfly-current-acceptance-render-v1"
SNAPSHOT_SCHEMA = "cubutterfly-current-acceptance-snapshot-v1"
EXPECTED_TRIALS = 3

FACTORIAL_SOURCES = {
    "a100-80gb": ROOT / "results/paper_experiments_20260917/a100-80gb/fft_factorial_v2.json",
    "a100-40gb": ROOT / "results/paper_experiments_20260917/a100-40gb/fft_factorial_v2.json",
}
GENERATION_SOURCES = {
    "a100-80gb": ROOT / "results/paper_experiments_20260917/a100-80gb/generation_witness.json",
    "a100-40gb": ROOT / "results/paper_experiments_20260917/a100-40gb/generation_witness.json",
}
SCALAR_SOURCES = {
    "a100-80gb": ROOT / "results/paper_experiments_20260917/a100-80gb/scalar_organization.json",
    "a100-40gb": ROOT / "results/paper_experiments_20260917/a100-40gb/scalar_organization.json",
}
SEARCH_SUMMARY_SOURCE = ROOT / "results/paper_experiments_20260917/analysis/search_protocol_summary.json"
HOLDED_VALIDATION_SOURCE = ROOT / (
    "results/paper_experiments_20260917/protocols/a100-80gb-gpu1/covered-holdout/heldout_validation.json"
)
MIGRATION_SOURCE = ROOT / "results/paper_experiments_20260917/a100-40gb/acceptance/acceptance.json"
CONTROL_REPORT_SOURCE = ROOT / "results/paper_experiments_20260917/analysis/REPORT.md"

BASELINE_CLASS = {
    "cuFFT": "external",
    "VkFFT": "external",
    "GPU-NTT-natural": "external",
    "Dao-FWHT": "external",
    "in-tree-radix2": "in-tree-control",
}

BASELINE_LABEL = {
    "cuFFT": "cuFFT",
    "VkFFT": "VkFFT",
    "GPU-NTT-natural": "GPU-NTT",
    "Dao-FWHT": "Dao",
    "in-tree-radix2": "in-tree radix2",
}

OPERATOR_LABEL = {
    "fft": "FFT",
    "ntt": "NTT",
    "fwht": "FWHT",
    "subset-zeta": "subset-zeta",
    "superset-zeta": "superset-zeta",
    "structured-2x2": "structured-2x2",
    "xor-zeta": "xor-zeta",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, root: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = str(resolved.relative_to(root.resolve()))
    except ValueError:
        relative = None
    return {
        "path": str(resolved),
        "repo_relative": relative,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
        "role": role,
    }


def _strict_true(value: Any) -> bool:
    return value is True or value == 1 or (
        isinstance(value, str) and value.strip().lower() in {"1", "true"}
    )


def _positive_float(value: Any, where: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{where} is not numeric: {value!r}") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{where} is not positive and finite: {value!r}")
    return number


def _workload_row(cell: dict[str, Any]) -> dict[str, Any]:
    workload = cell.get("workload") or {}
    row = {
        "cell_id": cell.get("id"),
        "operator": workload.get("operator"),
        "precision": workload.get("precision"),
        "logN": workload.get("logN"),
        "batch": workload.get("batch"),
        "direction": workload.get("direction", "forward"),
        "normalization": workload.get("normalization", "none"),
        "placement": workload.get("placement", ""),
        "input_order": workload.get("input_order", ""),
        "output_order": workload.get("output_order", ""),
        "modulus": workload.get("modulus", ""),
        "element_stride": workload.get("element_stride", ""),
        "batch_stride": workload.get("batch_stride", ""),
        "stage_matrix": workload.get("stage_matrix", ""),
    }
    if row["operator"] is None or row["logN"] is None or row["batch"] is None:
        raise ValueError(f"workload identity is incomplete in cell {cell.get('id')}")
    return row


def _measurement_groups(cell: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, measurement in enumerate(cell.get("measurements") or []):
        name = measurement.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"missing measurement name in {cell.get('id')}[{index}]")
        sample = measurement.get("sample")
        if not isinstance(sample, dict):
            raise ValueError(f"missing sample in {cell.get('id')}[{name}]")
        if not _strict_true(measurement.get("correct")) or not _strict_true(sample.get("correct")):
            raise ValueError(f"incorrect measurement in {cell.get('id')}[{name}]")
        sample_ms = _positive_float(sample.get("kernel_ms"), f"{cell.get('id')}[{name}].kernel_ms")
        groups[name].append({
            "trial": measurement.get("trial"),
            "kernel_ms": sample_ms,
        })
    for name, rows in groups.items():
        if len(rows) != EXPECTED_TRIALS:
            raise ValueError(
                f"{cell.get('id')}[{name}] has {len(rows)} trials; expected {EXPECTED_TRIALS}"
            )
    return groups


def _trial_values(rows: list[dict[str, Any]]) -> list[float]:
    # Trial labels are metadata only; sorting makes the generated CSV stable.
    return [row["kernel_ms"] for row in sorted(rows, key=lambda row: (row["trial"] is None, row["trial"]))]


def _pending_cell(cell: dict[str, Any]) -> dict[str, Any]:
    entry = {
        "cell_id": cell.get("id"),
        "status": cell.get("status"),
        "workload": cell.get("workload", {}),
    }
    unavailable = cell.get("unavailability") or {}
    rejections = unavailable.get("capability_rejections") or []
    reasons = []
    for rejection in rejections:
        for error in rejection.get("errors") or []:
            if error not in reasons:
                reasons.append(error)
    entry["reason"] = reasons or ["cell has no complete accepted measurements"]
    entry["attempted_candidates"] = unavailable.get("attempted_candidates")
    entry["unavailability_scope"] = unavailable.get("scope")
    return entry


def _build_rows(document: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    coverage: dict[str, dict[str, int]] = defaultdict(
        lambda: {"expected_cells": 0, "complete_cells": 0, "pending_cells": 0}
    )
    for cell in document.get("cells") or []:
        workload = cell.get("workload") or {}
        operator = str(workload.get("operator", "unknown"))
        coverage[operator]["expected_cells"] += 1
        if cell.get("status") != "complete":
            coverage[operator]["pending_cells"] += 1
            pending.append(_pending_cell(cell))
            continue
        coverage[operator]["complete_cells"] += 1
        groups = _measurement_groups(cell)
        treatment_rows = groups.get("cuButterfly")
        if treatment_rows is None:
            raise ValueError(f"complete cell {cell.get('id')} has no cuButterfly measurement")
        treatment_trials = _trial_values(treatment_rows)
        treatment_median_us = statistics.median(treatment_trials) * 1000.0
        workload_row = _workload_row(cell)
        for baseline, baseline_rows in sorted(groups.items()):
            if baseline == "cuButterfly":
                continue
            baseline_trials = _trial_values(baseline_rows)
            baseline_median_us = statistics.median(baseline_trials) * 1000.0
            ratio = baseline_median_us / treatment_median_us
            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError(f"invalid ratio in {cell.get('id')}[{baseline}]")
            row = dict(workload_row)
            row.update({
                "baseline": baseline,
                "baseline_class": BASELINE_CLASS.get(baseline, "other"),
                "trial_count": EXPECTED_TRIALS,
                "cuButterfly_trial_0_us": treatment_trials[0] * 1000.0,
                "cuButterfly_trial_1_us": treatment_trials[1] * 1000.0,
                "cuButterfly_trial_2_us": treatment_trials[2] * 1000.0,
                "baseline_trial_0_us": baseline_trials[0] * 1000.0,
                "baseline_trial_1_us": baseline_trials[1] * 1000.0,
                "baseline_trial_2_us": baseline_trials[2] * 1000.0,
                "cuButterfly_median_us": treatment_median_us,
                "baseline_median_us": baseline_median_us,
                "baseline_over_cuButterfly": ratio,
                "correct": True,
            })
            rows.append(row)
    rows.sort(key=lambda row: (
        row["operator"], row["baseline"], int(row["logN"]), int(row["batch"]), row["cell_id"]
    ))
    return rows, pending, dict(sorted(coverage.items()))


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["baseline"], row["operator"])].append(row)
    summaries = []
    for (baseline, operator), group in sorted(grouped.items()):
        ratios = [row["baseline_over_cuButterfly"] for row in group]
        geometric_mean = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
        summaries.append({
            "baseline": baseline,
            "baseline_label": BASELINE_LABEL.get(baseline, baseline),
            "baseline_class": BASELINE_CLASS.get(baseline, "other"),
            "operator": operator,
            "operator_label": OPERATOR_LABEL.get(operator, operator),
            "paired_cells": len(group),
            "geometric_mean_speedup": geometric_mean,
            "median_speedup": statistics.median(ratios),
            "minimum_speedup": min(ratios),
            "maximum_speedup": max(ratios),
            "cuButterfly_faster_cells": sum(value > 1.0 for value in ratios),
            "cuButterfly_slower_cells": sum(value < 1.0 for value in ratios),
            "equal_cells": sum(value == 1.0 for value in ratios),
        })
    return summaries


def _ratio_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty ratio group")
    return {
        "comparisons": len(values),
        "geometric_mean_baseline_over_treatment": math.exp(
            sum(math.log(value) for value in values) / len(values)
        ),
        "minimum_baseline_over_treatment": min(values),
        "maximum_baseline_over_treatment": max(values),
        "treatment_faster_comparisons": sum(value > 1.0 for value in values),
        "baseline_faster_comparisons": sum(value < 1.0 for value in values),
        "equal_comparisons": sum(value == 1.0 for value in values),
    }


def _controlled_identity(identity: dict[str, Any]) -> dict[str, Any]:
    # Keep build and device provenance without copying candidate mappings.
    keys = (
        "device", "gpu_uuid", "compile_mode", "build_dir", "protocol",
        "source_sha256", "study_path", "study_sha256", "binaries",
    )
    return {key: identity[key] for key in keys if key in identity}


def _load_controlled_source(path: Path, role: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = path.read_bytes()
    document = json.loads(raw)
    if document.get("schema") != "cubutterfly-controlled-experiment-v1":
        raise ValueError(f"unexpected controlled-experiment schema in {path}")
    if document.get("status") != "complete" or document.get("completed") is not True:
        raise ValueError(f"controlled experiment is not complete in {path}")
    rows: list[dict[str, Any]] = []
    cases = document.get("cases") or []
    for case in cases:
        workload = case.get("workload") or {}
        case_id = case.get("id")
        for index, comparison in enumerate(case.get("comparisons") or []):
            axes = comparison.get("actual_changed_axes") or comparison.get("changed_axes")
            if not isinstance(axes, list) or not axes:
                raise ValueError(f"missing changed axes in {path}:{case_id}[{index}]")
            ratio = _positive_float(
                comparison.get("median_treatment_over_baseline"),
                f"{path}:{case_id}[{index}].median_treatment_over_baseline",
            )
            if comparison.get("status") != "measured":
                raise ValueError(f"unmeasured comparison in {path}:{case_id}[{index}]")
            if not _strict_true(comparison.get("performance_accepted")):
                raise ValueError(f"unaccepted comparison in {path}:{case_id}[{index}]")
            rows.append({
                "case_id": case_id,
                "operator": workload.get("operator"),
                "precision": workload.get("precision"),
                "logN": workload.get("logN"),
                "batch": workload.get("batch"),
                "baseline": comparison.get("baseline"),
                "treatment": comparison.get("treatment"),
                "changed_axes": list(axes),
                "axis_key": "+".join(str(axis) for axis in axes),
                # The controlled journal stores baseline/treatment speedup; >1 favors treatment.
                "baseline_over_treatment": ratio,
                "baseline_median_kernel_ms": comparison.get("baseline_median_kernel_ms"),
                "treatment_median_kernel_ms": comparison.get("treatment_median_kernel_ms"),
                "correct": True,
            })
    if not rows:
        raise ValueError(f"controlled experiment has no comparisons in {path}")

    by_axis: dict[str, list[float]] = defaultdict(list)
    by_operator: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_axis[row["axis_key"]].append(row["baseline_over_treatment"])
        by_operator[str(row["operator"])].append(row["baseline_over_treatment"])
    summary_rows = []
    for axis_key, values in sorted(by_axis.items()):
        summary_rows.append({"changed_axes": axis_key, **_ratio_summary(values)})
    operator_rows = []
    for operator, values in sorted(by_operator.items()):
        operator_rows.append({"operator": operator, **_ratio_summary(values)})
    source = _file_record(path, ROOT, role)
    report = {
        "source": source,
        "schema": document.get("schema"),
        "status": document.get("status"),
        "protocol": document.get("protocol"),
        "method": {
            "ratio": "baseline median / treatment median speedup",
            "source_ratio_field": "median_treatment_over_baseline",
            "favor_treatment_when": "ratio > 1",
            "aggregation": "geometric mean of per-comparison medians",
        },
        "identity": _controlled_identity(document.get("identity") or {}),
        "case_count": len(cases),
        "variant_count": sum(len(case.get("variants") or []) for case in cases),
        "comparison_count": len(rows),
        "valid_comparison_count": len(rows),
        "summary_rows": summary_rows,
        "operator_summary_rows": operator_rows,
        "limitations": document.get("limitations", []),
    }
    return report, rows


def _write_controlled_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "card", "case_id", "operator", "precision", "logN", "batch", "baseline", "treatment",
        "changed_axes", "axis_key", "baseline_over_treatment", "baseline_median_kernel_ms",
        "treatment_median_kernel_ms", "correct",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["changed_axes"] = "+".join(output["changed_axes"])
            for field in (
                "baseline_over_treatment", "baseline_median_kernel_ms", "treatment_median_kernel_ms",
            ):
                if output.get(field) is not None:
                    output[field] = f"{float(output[field]):.9f}"
            writer.writerow(output)


def _load_search_evidence(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    e6 = document.get("e6") or {}
    e9 = document.get("e9") or {}
    rows = []
    for row in e9.get("by_workload_policy_budget") or []:
        workload = row.get("workload") or {}
        rows.append({
            "workload": {
                key: workload.get(key)
                for key in ("operator", "precision", "logN", "batch", "input_order", "output_order")
                if key in workload
            },
            "policy": row.get("policy"),
            "budget": row.get("budget"),
            "seeds": row.get("seeds"),
            "regret_ratio_mean": row.get("regret_ratio_mean"),
            "regret_ratio_median": row.get("regret_ratio_median"),
            "regret_ratio_max": row.get("regret_ratio_max"),
            "oracle_hits": row.get("oracle_hits"),
        })
    return {
        "source": _file_record(path, ROOT, "A100 80GB search evidence summary"),
        "schema": document.get("schema"),
        "hardware_uuid": document.get("hardware_uuid"),
        "sources": document.get("sources", []),
        "e6": {
            "measured_candidates": e6.get("measured_candidates"),
            "covered_priors": e6.get("covered_priors"),
            "metrics": e6.get("metrics", {}),
        },
        "e9": {
            "measured_candidates": e9.get("measured_candidates"),
            "replay_cells": e9.get("replay_cells"),
            "covered_priors": e9.get("covered_priors"),
            "by_workload_policy_budget": rows,
        },
        "measured_invocation_wall_seconds": document.get("measured_invocation_wall_seconds", {}),
        "limitations": document.get("limitations", []),
    }


def _load_holdout_evidence(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    metrics = document.get("metrics") or {}
    identity = document.get("measurement_identity") or {}
    return {
        "source": _file_record(path, ROOT, "A100 80GB covered-holdout validation"),
        "schema": document.get("schema"),
        "status": document.get("status"),
        "protocol": document.get("protocol"),
        "protocol_sha256": document.get("protocol_sha256"),
        "training_scope": document.get("training_scope"),
        "workload_count": len(document.get("workloads") or []),
        "metrics": {
            key: metrics.get(key)
            for key in (
                "top1", "top1_count", "top1_eligible", "top2", "top2_count", "top2_eligible",
                "pairwise_correct", "pairwise_count", "relative_error_median", "relative_error_p90",
                "regret_ratio_median", "regret_ratio_p90", "service_gap_count",
            )
            if key in metrics
        },
        "measurement_identity": {
            key: identity.get(key)
            for key in ("uuid", "uuid_source", "uuid_verified", "uuid_preflight_verified", "build_verified", "protocol_verified", "checkpoint_verified")
            if key in identity
        },
    }


def _load_migration_subset(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    cells = document.get("cells") or []
    complete = sum(cell.get("status") == "complete" for cell in cells)
    identity = document.get("identity") or {}
    return {
        "source": _file_record(path, ROOT, "A100 40GB matched migration acceptance subset"),
        "schema": document.get("schema"),
        "status": document.get("status"),
        "scope": document.get("scope"),
        "expected_cells": len(cells),
        "complete_cells": complete,
        "pending_cells": len(cells) - complete,
        "full_migration_qualified": False,
        "identity": _controlled_identity(identity),
        "limitations": document.get("limitations", []),
    }


def _compact_acceptance(document: dict[str, Any]) -> dict[str, Any]:
    """Retain the fields needed for the numeric table and pending-cell audit."""
    compact = {
        "schema": document.get("schema"),
        "identity": document.get("identity", {}),
        "scope": document.get("scope"),
        "status": document.get("status"),
        "limitations": document.get("limitations", []),
        "code_revisions": document.get("code_revisions"),
        "comparison_identity": document.get("comparison_identity"),
        "cells": [],
    }
    for cell in document.get("cells") or []:
        compact_cell = {
            "id": cell.get("id"),
            "status": cell.get("status"),
            "workload": cell.get("workload", {}),
            "measurements": [],
        }
        for measurement in cell.get("measurements") or []:
            sample = measurement.get("sample") or {}
            compact_cell["measurements"].append({
                "name": measurement.get("name"),
                "trial": measurement.get("trial"),
                "correct": measurement.get("correct"),
                "sample": {
                    "kernel_ms": sample.get("kernel_ms"),
                    "correct": sample.get("correct"),
                },
            })
        if cell.get("unavailability") is not None:
            compact_cell["unavailability"] = cell.get("unavailability")
        compact["cells"].append(compact_cell)
    return compact


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "cell_id", "operator", "precision", "logN", "batch", "direction", "normalization",
        "placement", "input_order", "output_order", "modulus", "element_stride", "batch_stride",
        "stage_matrix", "baseline", "baseline_class", "trial_count",
        "cuButterfly_trial_0_us", "cuButterfly_trial_1_us", "cuButterfly_trial_2_us",
        "baseline_trial_0_us", "baseline_trial_1_us", "baseline_trial_2_us",
        "cuButterfly_median_us", "baseline_median_us", "baseline_over_cuButterfly", "correct",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for field in fields:
                if field.endswith("_us") or field == "baseline_over_cuButterfly":
                    if output.get(field) != "":
                        output[field] = f"{float(output[field]):.9f}"
            writer.writerow(output)


def _write_table(path: Path, summaries: list[dict[str, Any]], pending_count: int) -> None:
    lines = [
        "% Generated by paper/tools/render_experiment_figures.py; do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Current A100 80GB matched baseline comparisons. Ratios are baseline median divided by cuButterfly median; values above one favor cuButterfly.}",
        r"\label{tab:current-results}",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"Class & Baseline & Operator & Cells & Geo. mean & Ours faster \\",
        r"\hline",
    ]
    for summary in summaries:
        class_label = "external" if summary["baseline_class"] == "external" else "in-tree control"
        lines.append(
            "%s & %s & %s & %d & %.5f & %d \\\\" % (
                _latex_escape(class_label),
                _latex_escape(summary["baseline_label"]),
                _latex_escape(summary["operator_label"]),
                summary["paired_cells"],
                summary["geometric_mean_speedup"],
                summary["cuButterfly_faster_cells"],
            )
        )
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\parbox{0.98\linewidth}{\footnotesize A100 80GB PCIe, SM80; 100 warmups, 100 repetitions, three process trials. Each cell contributes the median of its three kernel-ms samples before the geometric mean is taken. The in-tree radix2 rows are ordinary-core controls, not external-library comparisons. One NTT natural-input/bit-reversed-output cell is unavailable and is omitted rather than encoded as zero.}",
        r"\end{table*}",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def _save_figure(fig: Any, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "render_experiment_figures.py", "Title": title}
    fig.savefig(path, bbox_inches="tight", metadata=metadata)


def _distribution_stats(values: list[float]) -> dict[str, Any]:
    """Compute inclusive quartiles and geometric means for a distribution."""
    values = sorted(float(value) for value in values)
    if not values:
        raise ValueError("cannot summarize an empty distribution")

    def _quantile(probability: float) -> float:
        index = (len(values) - 1) * probability
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        fraction = index - lower
        return values[lower] + fraction * (values[upper] - values[lower])

    q1, median, q3 = (_quantile(probability) for probability in (0.25, 0.5, 0.75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    inlier_values = [value for value in values if lower_fence <= value <= upper_fence]
    return {
        "count": len(values),
        "minimum": values[0],
        "q1": q1,
        "median": median,
        "q3": q3,
        "maximum": values[-1],
        "iqr": iqr,
        "lower_whisker": min(inlier_values),
        "upper_whisker": max(inlier_values),
        "geometric_mean": math.exp(sum(math.log(value) for value in values) / len(values)),
        "cubutterfly_faster": sum(value > 1.0 for value in values),
    }


def _plot_distribution_panel(
    axis: Any,
    groups: list[tuple[str, str, list[float]]],
    title: str,
    ylabel: str,
) -> None:
    """Draw one compact boxplot panel with complete-case dots and statistics."""
    box_values = [values for _, _, values in groups]
    positions = list(range(len(groups)))
    colors = [color for _, color, _ in groups]
    box = axis.boxplot(
        box_values,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202124", "linewidth": 1.05},
        whiskerprops={"color": "#263238", "linewidth": 0.8},
        capprops={"color": "#263238", "linewidth": 0.8},
        boxprops={"edgecolor": "#263238", "linewidth": 0.65},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)

    all_values: list[float] = []
    labels = [label for label, _, _ in groups]
    for position, (_, color, values) in enumerate(groups):
        all_values.extend(values)
        jitter = [((index % 9) - 4) * 0.018 for index in range(len(values))]
        axis.scatter(
            [position + offset for offset in jitter],
            values,
            s=18,
            alpha=0.78,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        stats = _distribution_stats(values)
        axis.scatter(
            [position],
            [stats["geometric_mean"]],
            marker="D",
            s=27,
            color="#202124",
            zorder=4,
        )
        labels[position] = f"{labels[position]}\n(n={stats['count']})"

    axis.axhline(1.0, color="#263238", linestyle="--", linewidth=0.85)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=7.5, ha="center")
    axis.set_ylabel(ylabel, fontsize=7.7)
    axis.set_title(title, fontsize=8.7)
    axis.grid(axis="y", alpha=0.22)
    if all_values:
        # Ratios use a logarithmic axis so gains and slowdowns around parity
        # stay visible alongside the larger ordinary-core improvements.
        axis.set_yscale("log", base=2)
        lower = min(min(all_values) / 1.18, 0.8)
        upper = max(max(all_values) * 1.18, 1.2)
        axis.set_ylim(lower, upper)
        ticks = [value for value in (0.5, 1, 2, 4, 8, 16) if lower <= value <= upper]
        axis.set_yticks(ticks)
        axis.set_yticklabels([f"{value:g}" for value in ticks], fontsize=7.5)
    axis.set_xlim(-0.5, len(groups) - 0.5)


def _plot_baseline_summary(rows: list[dict[str, Any]], figure_dir: Path, pending_count: int) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def select(operator: str, baseline: str, precision: str | None = None) -> list[float]:
        return [
            float(row["baseline_over_cuButterfly"])
            for row in rows
            if row["operator"] == operator
            and row["baseline"] == baseline
            and (precision is None or row["precision"] == precision)
        ]

    external_specs = [
        ("FFT / cuFFT\nFP32", "#28658c", select("fft", "cuFFT", "fp32")),
        ("FFT / cuFFT\nFP64", "#3c8d70", select("fft", "cuFFT", "fp64")),
        ("FFT / VkFFT\nFP32", "#a4512a", select("fft", "VkFFT", "fp32")),
        ("NTT / GPU-NTT\nword64", "#7a5a9e", select("ntt", "GPU-NTT-natural")),
        ("FWHT / Dao\nFP32", "#b7852a", select("fwht", "Dao-FWHT", "fp32")),
    ]
    internal_specs = [
        ("FFT", "#5e6a75", select("fft", "in-tree-radix2")),
        ("FWHT", "#687a89", select("fwht", "in-tree-radix2")),
        ("structured-2x2", "#718c7c", select("structured-2x2", "in-tree-radix2")),
        ("subset-zeta", "#8b7f6d", select("subset-zeta", "in-tree-radix2")),
        ("superset-zeta", "#8b6f78", select("superset-zeta", "in-tree-radix2")),
        ("xor-zeta", "#69708b", select("xor-zeta", "in-tree-radix2")),
    ]
    external_specs = [group for group in external_specs if group[2]]
    internal_specs = [group for group in internal_specs if group[2]]
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 3.85), squeeze=False)
    _plot_distribution_panel(
        axes[0, 0], external_specs, "(a) External libraries", "Relative throughput"
    )
    _plot_distribution_panel(
        axes[1, 0], internal_specs, "(b) In-tree radix2 controls", "Relative throughput"
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.105, hspace=0.82)
    paths = [figure_dir / "results_current_baseline_summary.svg", figure_dir / "results_current_baseline_summary.pdf"]
    for path in paths:
        _save_figure(fig, path, "A100 80GB matched baseline distributions")
    plt.close(fig)
    return paths


def _plot_fft_scaling(rows: list[dict[str, Any]], figure_dir: Path) -> list[Path]:
    """Plot only measured FFT cells in precision x direction facets."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Each complete cell appears once per paired baseline.  The cuButterfly
    # timing is identical across those rows, so de-duplicate by cell identity.
    cells: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["operator"] == "fft":
            cells[row["cell_id"]] = row

    def _default_stride(value: Any) -> bool:
        return value in (None, "", 1, "1")

    contiguous_cells = {
        cell_id: row
        for cell_id, row in cells.items()
        if row.get("placement") == "out-of-place"
        and _default_stride(row.get("element_stride"))
        and _default_stride(row.get("batch_stride"))
    }
    cells = contiguous_cells

    facets = [("fp32", "forward"), ("fp32", "inverse"), ("fp64", "forward"), ("fp64", "inverse")]
    all_batches = sorted({int(row["batch"]) for row in cells.values()})
    all_log_ns = sorted({int(row["logN"]) for row in cells.values()})
    all_times = [float(row["cuButterfly_median_us"]) for row in cells.values()]
    color_norm = None
    if all_times:
        from matplotlib.colors import LogNorm

        lower = min(all_times)
        upper = max(all_times)
        color_norm = LogNorm(vmin=lower, vmax=max(upper, lower * 1.01))
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.15), squeeze=False, sharex=True, sharey=True)
    scatter = None
    for axis, (precision, direction) in zip(axes.flat, facets):
        selected = sorted(
            (row for row in cells.values() if row["precision"] == precision and row["direction"] == direction),
            key=lambda row: (int(row["logN"]), int(row["batch"])),
        )
        if selected:
            scatter = axis.scatter(
                [int(row["logN"]) for row in selected],
                [int(row["batch"]) for row in selected],
                c=[float(row["cuButterfly_median_us"]) for row in selected],
                cmap="viridis",
                norm=color_norm,
                s=44,
                edgecolor="#202124",
                linewidth=0.35,
                zorder=3,
            )
            for row in selected:
                axis.annotate(
                    f"{float(row['cuButterfly_median_us']):.1f}",
                    (int(row["logN"]), int(row["batch"])),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.2,
                    color="#202124",
                )
        else:
            axis.text(0.5, 0.5, "no measured cells", transform=axis.transAxes, ha="center", va="center", fontsize=7)
        axis.set_title(f"{precision.upper()} / {direction} (n={len(selected)})", fontsize=8.3)
        axis.set_xticks(all_log_ns)
        axis.set_xticklabels([f"$2^{{{value}}}$" for value in all_log_ns], fontsize=7.5)
        axis.set_yscale("log", base=2)
        batch_ticks = sorted({min(all_batches), max(all_batches)} | {
            2 ** exponent for exponent in range(0, 32, 4)
            if min(all_batches) <= 2 ** exponent <= max(all_batches)
        })
        axis.set_yticks(batch_ticks)
        axis.set_yticklabels([f"{value}" for value in batch_ticks], fontsize=7.5)
        axis.grid(True, which="major", alpha=0.22)
        axis.set_xlim(min(all_log_ns) - 1.0, max(all_log_ns) + 1.0)
        axis.set_ylim(min(all_batches) / 1.6, max(all_batches) * 2.6)
    axes[0, 0].set_ylabel("Batch", fontsize=7.5)
    axes[1, 0].set_ylabel("Batch", fontsize=7.5)
    axes[1, 0].set_xlabel("Transform length N", fontsize=7.5)
    axes[1, 1].set_xlabel("Transform length N", fontsize=7.5)
    fig.subplots_adjust(left=0.09, right=0.86, top=0.93, bottom=0.13, hspace=0.35, wspace=0.2)
    if scatter is not None:
        # Use a dedicated axis after laying out all facets. A later
        # subplots_adjust would otherwise place data underneath the colorbar.
        color_axis = fig.add_axes([0.89, 0.22, 0.018, 0.60])
        colorbar = fig.colorbar(scatter, cax=color_axis)
        colorbar.ax.tick_params(labelsize=7.5)
        colorbar.set_label("Median kernel time (us)", fontsize=7.5)
    paths = [figure_dir / "results_current_fft_scaling.svg", figure_dir / "results_current_fft_scaling.pdf"]
    for path in paths:
        _save_figure(fig, path, "A100 80GB FFT length and batch scaling")
    plt.close(fig)
    return paths


def _plot_fft_distribution(rows: list[dict[str, Any]], figure_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("cuFFT", "#28658c"), ("VkFFT", "#a4512a")]
    fig, axis = plt.subplots(figsize=(3.45, 2.65))
    plotted_values = []
    geometric_means = []
    box_values = []
    for position, (baseline, color) in enumerate(groups):
        selected = [row for row in rows if row["operator"] == "fft" and row["baseline"] == baseline]
        selected.sort(key=lambda row: (int(row["logN"]), int(row["batch"]), row["cell_id"]))
        values = [row["baseline_over_cuButterfly"] for row in selected]
        plotted_values.extend(values)
        box_values.append(values)
    # The box is the distribution summary; the points retain the individual
    # cases so a favorable median cannot hide the small- and large-transform
    # tails.  The box/whisker convention is Tukey's 1.5 IQR rule.
    box = axis.boxplot(
        box_values,
        positions=list(range(len(groups))),
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202124", "linewidth": 1.1},
        whiskerprops={"color": "#263238", "linewidth": 0.8},
        capprops={"color": "#263238", "linewidth": 0.8},
        boxprops={"edgecolor": "#263238", "linewidth": 0.7},
    )
    for patch, (_, color) in zip(box["boxes"], groups):
        patch.set_facecolor(color)
        patch.set_alpha(0.27)
    for position, ((baseline, color), values) in enumerate(zip(groups, box_values)):
        jitter = [((index % 9) - 4) * 0.019 for index in range(len(values))]
        axis.scatter(
            [position + offset for offset in jitter], values, s=28, alpha=0.78,
            color=color, edgecolor="white", linewidth=0.4,
        )
        if values:
            geometric_mean = math.exp(sum(math.log(value) for value in values) / len(values))
            geometric_means.append((position, geometric_mean))
            axis.scatter([position], [geometric_mean], marker="D", s=52, color="#202124", zorder=4)
    axis.axhline(1.0, color="#263238", linestyle="--", linewidth=0.9)
    axis.set_xlim(-0.42, 1.42)
    if plotted_values:
        axis.set_ylim(min(plotted_values) - 0.07, max(plotted_values) + 0.09)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["cuFFT\n(n=29)", "VkFFT\n(n=18)"], fontsize=7.2)
    axis.set_ylabel("baseline / cuButterfly", fontsize=7.7)
    axis.set_title("A100 80GB FFT matched cases", fontsize=9)
    axis.grid(axis="y", alpha=0.24)
    if plotted_values:
        lower, upper = axis.get_ylim()
        for position, geometric_mean in geometric_means:
            axis.text(position, upper - 0.025 * (upper - lower), f"geo. {geometric_mean:.3f}",
                      ha="center", va="top", fontsize=7.0)
    fig.text(0.5, 0.005, "Boxes: median/IQR; whiskers: 1.5 IQR; dots: matched cases.", ha="center", fontsize=7.1)
    fig.tight_layout(rect=(0, 0.075, 1, 0.95))
    paths = [figure_dir / "results_current_fft_distribution.svg", figure_dir / "results_current_fft_distribution.pdf"]
    for path in paths:
        _save_figure(fig, path, "A100 80GB FFT case distribution")
    plt.close(fig)
    return paths


def _fft_distribution_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return reproducible distribution statistics for the FFT panel.

    This is deliberately based on the same complete paired rows as the
    figure.  It makes the boxplot claim auditable without asking readers to
    infer quartiles from a rendered PDF.
    """
    result: dict[str, dict[str, Any]] = {}
    for baseline in ("cuFFT", "VkFFT"):
        values = sorted(
            float(row["baseline_over_cuButterfly"])
            for row in rows
            if row["operator"] == "fft" and row["baseline"] == baseline
        )
        if not values:
            continue
        if len(values) == 1:
            quartiles = [values[0]] * 3
        else:
            # Python 3.7-compatible equivalent of the inclusive quantile
            # convention used by matplotlib's boxplot.
            def _quantile(probability: float) -> float:
                index = (len(values) - 1) * probability
                lower = int(index)
                upper = min(lower + 1, len(values) - 1)
                fraction = index - lower
                return values[lower] + fraction * (values[upper] - values[lower])

            quartiles = [_quantile(0.25), _quantile(0.5), _quantile(0.75)]
        result[baseline] = {
            "count": len(values),
            "minimum": values[0],
            "q1": quartiles[0],
            "median": statistics.median(values),
            "q3": quartiles[2],
            "maximum": values[-1],
            "geometric_mean": math.exp(sum(math.log(value) for value in values) / len(values)),
            "cubutterfly_faster": sum(value > 1.0 for value in values),
        }
    return result


def _plot_factorial_controls(
    factorial_controls: dict[str, dict[str, Any]], figure_dir: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [
        "data_tiles_per_cta", "prefetch_depth",
        "data_tiles_per_cta+prefetch_depth", "fft_core",
    ]
    labels = ["tiles", "prefetch", "tiles+prefetch", "FFT core"]
    cards = [("a100-80gb", "A100 80GB"), ("a100-40gb", "A100 40GB")]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.0), sharex="col", squeeze=False)
    colors = ["#28658c", "#a4512a", "#4b8063", "#7a5a9e"]
    for row_index, (card_key, card_label) in enumerate(cards):
        report = factorial_controls[card_key]
        by_axis = {item["changed_axes"]: item for item in report["summary_rows"]}
        means = [by_axis[key]["geometric_mean_baseline_over_treatment"] for key in order]
        minima = [by_axis[key]["minimum_baseline_over_treatment"] for key in order]
        maxima = [by_axis[key]["maximum_baseline_over_treatment"] for key in order]
        positions = list(range(len(order)))

        mean_axis = axes[row_index, 0]
        mean_axis.bar(positions, means, color=colors, width=0.68, edgecolor="#263238", linewidth=0.45)
        mean_axis.axhline(1.0, color="#263238", linestyle="--", linewidth=0.8)
        mean_axis.set_ylim(0.0, max(max(means) * 1.22, 1.25))
        mean_axis.set_ylabel(card_label + "\nspeedup", fontsize=7.5)
        mean_axis.grid(axis="y", alpha=0.22)
        for position, value in zip(positions, means):
            mean_axis.text(position, value + 0.025 * mean_axis.get_ylim()[1], f"{value:.2f}",
                           ha="center", va="bottom", fontsize=7.1)

        range_axis = axes[row_index, 1]
        for position, low, high, mean, color in zip(positions, minima, maxima, means, colors):
            range_axis.vlines(position, low, high, color=color, linewidth=2.0, alpha=0.9)
            range_axis.scatter([position], [low], color=color, marker="_", s=38, linewidth=0.9)
            range_axis.scatter([position], [high], color=color, marker="_", s=38, linewidth=0.9)
            range_axis.scatter([position], [mean], color="#202124", marker="D", s=22, zorder=3)
        range_axis.axhline(1.0, color="#263238", linestyle="--", linewidth=0.8)
        low_limit = min(minima) * 0.92
        high_limit = max(maxima) * 1.08
        range_axis.set_ylim(max(0.0, low_limit), high_limit)
        range_axis.set_ylabel("Speedup range", fontsize=7.5)
        range_axis.grid(axis="y", alpha=0.22)
        if row_index == 0:
            mean_axis.set_title("Geometric mean", fontsize=8.7)
            range_axis.set_title("Comparison range", fontsize=8.7)

    for axis in axes[-1, :]:
        axis.set_xticks(list(range(len(labels))))
        axis.set_xticklabels(labels, rotation=18, ha="right", fontsize=7.2)
    fig.suptitle("Matched FFT factorial controls; speedup > 1 favors treatment", fontsize=9.5)
    fig.text(0.5, 0.008, "Three trials per variant; 6 workloads, 36 variants and 54 pairwise comparisons per card.",
             ha="center", fontsize=7.2)
    fig.tight_layout(rect=(0, 0.075, 1, 0.94))
    paths = [
        figure_dir / "results_current_factorial_controls.svg",
        figure_dir / "results_current_factorial_controls.pdf",
    ]
    for path in paths:
        _save_figure(fig, path, "A100 matched FFT factorial controls")
    plt.close(fig)
    return paths


def _load_source(source_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = source_path.read_bytes()
    document = json.loads(raw)
    if document.get("schema") != "cubutterfly-research-acceptance-v1":
        raise ValueError(f"unexpected acceptance schema in {source_path}")
    source = _file_record(source_path, ROOT, "active A100 80GB acceptance source")
    return document, source


def _write_snapshot(
    snapshot_path: Path,
    document: dict[str, Any],
    source: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    envelope = {
        "schema": SNAPSHOT_SCHEMA,
        "source": source,
        "acceptance": _compact_acceptance(document),
        "evidence": evidence,
    }
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(envelope, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def _load_snapshot(snapshot_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    envelope = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if envelope.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError(f"unexpected acceptance snapshot schema in {snapshot_path}")
    document = envelope.get("acceptance")
    source = envelope.get("source")
    if not isinstance(document, dict) or document.get("schema") != "cubutterfly-research-acceptance-v1":
        raise ValueError(f"snapshot has no valid acceptance document in {snapshot_path}")
    if not isinstance(source, dict) or not source.get("sha256") or not source.get("path"):
        raise ValueError(f"snapshot has no source hash in {snapshot_path}")
    evidence = envelope.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError(f"snapshot has no frozen evidence bundle in {snapshot_path}; use --refresh-snapshot")
    return document, source, evidence


def _load_input(
    source_path: Path, snapshot_path: Path, refresh_snapshot: bool
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    if refresh_snapshot:
        document, source = _load_source(source_path)
        mode = "refreshed"
        snapshot = None
        evidence = None
    elif snapshot_path.exists():
        mode = "frozen"
        document, source, evidence = _load_snapshot(snapshot_path)
        snapshot = _file_record(snapshot_path, ROOT, "frozen paper evidence snapshot")
    else:
        raise FileNotFoundError(
            f"frozen snapshot is required by default: {snapshot_path}; "
            "use --refresh-snapshot only for an explicit evidence refresh"
        )
    return document, source, {"mode": mode, "snapshot": snapshot}, evidence


def render(
    source_path: Path,
    output_dir: Path,
    figure_dir: Path,
    make_figures: bool = True,
    snapshot_path: Path = DEFAULT_SNAPSHOT,
    refresh_snapshot: bool = False,
) -> dict[str, Any]:
    document, source, input_record, frozen_evidence = _load_input(
        source_path, snapshot_path, refresh_snapshot
    )
    rows, pending, coverage = _build_rows(document)
    summaries = _summary_rows(rows)
    if frozen_evidence is None:
        factorial_controls: dict[str, dict[str, Any]] = {}
        factorial_rows: list[dict[str, Any]] = []
        for card, path in FACTORIAL_SOURCES.items():
            report, detail_rows = _load_controlled_source(
                path, f"{card} FFT factorial v2 source"
            )
            factorial_controls[card] = report
            factorial_rows.extend([{**row, "card": card} for row in detail_rows])
        generation_witness: dict[str, dict[str, Any]] = {}
        for card, path in GENERATION_SOURCES.items():
            report, _ = _load_controlled_source(
                path, f"{card} generation witness source"
            )
            generation_witness[card] = report
        scalar_controls: dict[str, dict[str, Any]] = {}
        for card, path in SCALAR_SOURCES.items():
            report, _ = _load_controlled_source(
                path, f"{card} scalar organization control source"
            )
            scalar_controls[card] = report
        search_evidence = _load_search_evidence(SEARCH_SUMMARY_SOURCE)
        holdout_evidence = _load_holdout_evidence(HOLDED_VALIDATION_SOURCE)
        migration_subset = _load_migration_subset(MIGRATION_SOURCE)
        provenance = {
            "control_report": _file_record(CONTROL_REPORT_SOURCE, ROOT, "A100 control analysis report"),
        }
        frozen_evidence = {
            "factorial_controls": factorial_controls,
            "factorial_rows": factorial_rows,
            "generation_witness": generation_witness,
            "scalar_controls": scalar_controls,
            "search_evidence": search_evidence,
            "covered_holdout": holdout_evidence,
            "migration_subset": migration_subset,
            "provenance": provenance,
        }
        _write_snapshot(snapshot_path, document, source, frozen_evidence)
        input_record["snapshot"] = _file_record(snapshot_path, ROOT, "frozen paper evidence snapshot")
    else:
        factorial_controls = frozen_evidence["factorial_controls"]
        factorial_rows = frozen_evidence["factorial_rows"]
        generation_witness = frozen_evidence["generation_witness"]
        scalar_controls = frozen_evidence["scalar_controls"]
        search_evidence = frozen_evidence["search_evidence"]
        holdout_evidence = frozen_evidence["covered_holdout"]
        migration_subset = frozen_evidence["migration_subset"]
        provenance = frozen_evidence["provenance"]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "current_results_pairs.csv"
    table_path = output_dir / "current_results_table.tex"
    factorial_csv_path = output_dir / "current_factorial_pairs.csv"
    _write_csv(csv_path, rows)
    _write_table(table_path, summaries, len(pending))
    _write_controlled_csv(factorial_csv_path, factorial_rows)
    figure_paths = []
    fft_distribution = _fft_distribution_stats(rows)
    if make_figures:
        figure_paths.extend(_plot_baseline_summary(rows, figure_dir, len(pending)))
        figure_paths.extend(_plot_fft_scaling(rows, figure_dir))

    identity = document.get("identity") or {}
    result = {
        "schema": OUTPUT_SCHEMA,
        "source": source,
        "input": input_record,
        "snapshot": {
            "acceptance_schema": document.get("schema"),
            "status": document.get("status"),
            "scope": document.get("scope"),
            "expected_cells": len(document.get("cells") or []),
            "complete_cells": sum(1 for cell in document.get("cells") or [] if cell.get("status") == "complete"),
            "pending_cells": len(pending),
            "full_matrix_complete": len(pending) == 0,
        },
        "identity": {
            "device": identity.get("device"),
            "compile_mode": identity.get("compile_mode"),
            "protocol": identity.get("protocol"),
            "workloads_sha256": identity.get("workloads_sha256"),
            "profile_sha256": identity.get("profile_sha256"),
            "source_stage_sha256": identity.get("source_stage_sha256"),
            "cost_model_version": identity.get("cost_model_version"),
            "service_model_version": identity.get("service_model_version"),
            "binaries": identity.get("binaries"),
            "code_revisions": document.get("code_revisions"),
            "comparison_identity": document.get("comparison_identity"),
        },
        "method": {
            "trial_count": EXPECTED_TRIALS,
            "per_cell_aggregation": "median of the three kernel_ms samples for each named measurement",
            "ratio": "baseline_median_us / cuButterfly_median_us",
            "geometric_mean": "exp(mean(log(per_cell_ratio))) within each baseline/operator pair",
            "correctness_filter": "complete cells and measurements whose cell and sample correct fields are true",
            "unavailable_policy": "pending cells are explicit records and are omitted from numeric rows and figures",
            "normal_timing_scope": "acceptance sample kernel_ms; profiler timings are not substituted",
        },
        "coverage_by_operator": coverage,
        "summary_rows": summaries,
        "factorial_controls": factorial_controls,
        "generation_witness": generation_witness,
        "scalar_controls": scalar_controls,
        "search_evidence": search_evidence,
        "covered_holdout": holdout_evidence,
        "migration_subset": migration_subset,
        "provenance": provenance,
        "paired_row_count": len(rows),
        "paired_rows": rows,
        "fft_distribution": fft_distribution,
        "pending_cells": pending,
        "limitations": document.get("limitations", []),
        "artifacts": [
            _file_record(csv_path, ROOT, "generated current paired CSV"),
            _file_record(table_path, ROOT, "generated current LaTeX table"),
            _file_record(factorial_csv_path, ROOT, "generated factorial control CSV"),
            input_record["snapshot"],
        ] + [_file_record(path, ROOT, "generated current figure") for path in figure_paths],
    }
    summary_path = output_dir / "current_results_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps({
        "summary": str(summary_path),
        "csv": str(csv_path),
        "table": str(table_path),
        "figures": [str(path) for path in figure_paths],
        "input_mode": input_record["mode"],
        "complete_cells": result["snapshot"]["complete_cells"],
        "pending_cells": result["snapshot"]["pending_cells"],
        "paired_rows": len(rows),
        "geometric_means": {
            f"{row['baseline']}/{row['operator']}": row["geometric_mean_speedup"]
            for row in summaries
        },
        "factorial_geometric_means": {
            card: {
                row["changed_axes"]: row["geometric_mean_baseline_over_treatment"]
                for row in report["summary_rows"]
            }
            for card, report in factorial_controls.items()
        },
    }, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--refresh-snapshot", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    render(
        args.acceptance.resolve(),
        args.output_dir.resolve(),
        args.figure_dir.resolve(),
        not args.no_figures,
        args.snapshot.resolve(),
        args.refresh_snapshot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
