#!/usr/bin/env python3
"""Render absolute FFT throughput for the latest RC2 acceptance snapshot.

The renderer produces two complementary views.  The point plot keeps each
workload paired, while the boxplot gives the UnionSparse-style distribution
summary in absolute units.  Both report transforms/s and butterfly-pairs/s;
the boxplot is descriptive across the sampled workload shapes and is not a
cross-size score.  The source is a completed RC2 summary containing median
kernel time for each implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    ROOT.parent
    / "cuButterfly-v0.9.1-rc2/results/integrated_repairs_20261001_replay_fix1"
    / "a100-80gb/fft36/summary.json"
)
DEFAULT_FIGURE_DIR = ROOT / "experiments/figures"
DEFAULT_OUTPUT_DIR = ROOT / "experiments/snapshots"

SYSTEMS = (
    ("cuButterfly", "#1769aa", "o"),
    ("cuFFT", "#d95f02", "s"),
    ("VkFFT", "#2ca25f", "^"),
)


def _load_rows(source: Path) -> list[dict[str, Any]]:
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("status") != "complete-measured-matrix":
        raise ValueError(f"source is not a complete RC2 matrix: {source}")
    rows: list[dict[str, Any]] = []
    for cell in document.get("cells") or []:
        workload = cell.get("workload") or {}
        if workload.get("operator") != "fft":
            continue
        log_n = int(workload["logN"])
        batch = int(workload["batch"])
        n = 1 << log_n
        medians = cell.get("median_ms") or {}
        for system, _, _ in SYSTEMS:
            median_ms = medians.get(system)
            if median_ms is None:
                continue
            median_ms = float(median_ms)
            if not math.isfinite(median_ms) or median_ms <= 0:
                raise ValueError(f"invalid median for {cell.get('id')} / {system}")
            transforms_s = 1000.0 * batch / median_ms
            butterfly_pairs_s = transforms_s * (n * log_n / 2.0)
            rows.append({
                "cell_id": cell.get("id"),
                "system": system,
                "precision": workload.get("precision"),
                "logN": log_n,
                "N": n,
                "batch": batch,
                "direction": workload.get("direction", "forward"),
                "normalization": workload.get("normalization", "none"),
                "placement": workload.get("placement", "out-of-place"),
                "total_elements": n * batch,
                "median_ms": median_ms,
                "transforms_per_s": transforms_s,
                "butterfly_pairs_per_s": butterfly_pairs_s,
            })
    if not rows:
        raise ValueError(f"no FFT rows in {source}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig: Any, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"Creator": "render_rc2_throughput.py", "Title": title})


def _plot(rows: list[dict[str, Any]], figure_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    systems = {name: (color, marker) for name, color, marker in SYSTEMS}
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.75), sharex="col", squeeze=False)
    metrics = (
        ("transforms_per_s", "Transforms/s"),
        ("butterfly_pairs_per_s", "Butterfly pairs/s"),
    )
    precisions = (("fp32", "FP32"), ("fp64", "FP64"))
    for row_index, (metric, ylabel) in enumerate(metrics):
        for col_index, (precision, precision_label) in enumerate(precisions):
            axis = axes[row_index, col_index]
            subset = [item for item in rows if item["precision"] == precision]
            keys = sorted({(item["cell_id"], item["total_elements"]) for item in subset}, key=lambda x: x[1])
            x_by_cell = {cell_id: total for cell_id, total in keys}
            for system_index, (system, color, marker) in enumerate(SYSTEMS):
                values = [item for item in subset if item["system"] == system]
                if not values:
                    continue
                x_factor = (1.0, 1.015, 0.985)[system_index]
                axis.scatter(
                    [item["total_elements"] * x_factor for item in values],
                    [item[metric] for item in values],
                    s=25,
                    marker=marker,
                    color=color,
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=0.9,
                    label=system,
                    zorder=3,
                )
            # A thin segment makes the paired comparison visible without
            # pretending that adjacent sizes form one performance curve.
            for cell_id, total in keys:
                paired = [item for item in subset if item["cell_id"] == cell_id]
                if len(paired) >= 2:
                    axis.plot(
                        [item["total_elements"] for item in paired],
                        [item[metric] for item in paired],
                        color="#9aa0a6",
                        linewidth=0.35,
                        alpha=0.45,
                        zorder=1,
                    )
            axis.set_xscale("log", base=2)
            axis.set_yscale("log", base=10)
            axis.grid(True, which="major", axis="both", alpha=0.2, linewidth=0.55)
            axis.set_title(precision_label, fontsize=8.5)
            axis.set_ylabel(ylabel, fontsize=7.7)
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0e}"))
            axis.tick_params(labelsize=7.1)
            if row_index == 1:
                axis.set_xlabel("Total elements ($N\\times B$)", fontsize=7.7)
    handles = []
    labels = []
    for system, color, marker in SYSTEMS:
        handles.append(plt.Line2D([], [], marker=marker, linestyle="None", color=color, markersize=5))
        labels.append(system)
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7.2, frameon=False,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("A100 80GB RC2 absolute FFT throughput", fontsize=9.5, y=0.995)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.87, bottom=0.12, hspace=0.32, wspace=0.22)
    paths = [
        figure_dir / "rc2_fft_absolute_throughput.svg",
        figure_dir / "rc2_fft_absolute_throughput.pdf",
    ]
    for path in paths:
        _save_figure(fig, path, "A100 80GB RC2 absolute FFT throughput")
    plt.close(fig)
    return paths


def _plot_boxplot(rows: list[dict[str, Any]], figure_dir: Path) -> list[Path]:
    """Render absolute-rate distributions without hiding the sampled points."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    colors = {name: color for name, color, _ in SYSTEMS}
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.15), squeeze=False)
    metrics = (
        ("transforms_per_s", "Transforms/s"),
        ("butterfly_pairs_per_s", "Butterfly pairs/s"),
    )
    precisions = (("fp32", "FP32"), ("fp64", "FP64"))
    for row_index, (metric, ylabel) in enumerate(metrics):
        for col_index, (precision, precision_label) in enumerate(precisions):
            axis = axes[row_index, col_index]
            positions = []
            data = []
            labels = []
            for position, (system, _, _) in enumerate(SYSTEMS, start=1):
                values = [
                    float(item[metric])
                    for item in rows
                    if item["precision"] == precision and item["system"] == system
                ]
                if not values:
                    continue
                positions.append(position)
                data.append(values)
                labels.append(system)
            if data:
                artists = axis.boxplot(
                    data,
                    positions=positions,
                    widths=0.55,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "#202020", "linewidth": 1.0},
                    whiskerprops={"color": "#505050", "linewidth": 0.8},
                    capprops={"color": "#505050", "linewidth": 0.8},
                    boxprops={"linewidth": 0.8},
                )
                for patch, system in zip(artists["boxes"], labels):
                    patch.set_facecolor(colors[system])
                    patch.set_alpha(0.68)
                # Show every measured point with deterministic, small jitter.
                for position, values, system in zip(positions, data, labels):
                    jitter = [position + ((index % 9) - 4) * 0.018 for index in range(len(values))]
                    axis.scatter(
                        jitter,
                        values,
                        s=7,
                        color=colors[system],
                        edgecolor="white",
                        linewidth=0.25,
                        alpha=0.62,
                        zorder=3,
                    )
            axis.set_xlim(0.45, len(SYSTEMS) + 0.55)
            axis.set_xticks(range(1, len(SYSTEMS) + 1))
            axis.set_xticklabels([system for system, _, _ in SYSTEMS], fontsize=6.7)
            axis.set_yscale("log", base=10)
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0e}"))
            axis.grid(True, which="major", axis="y", alpha=0.22, linewidth=0.55)
            axis.set_title(precision_label, fontsize=8.5)
            axis.set_ylabel(ylabel, fontsize=7.7)
            axis.tick_params(axis="y", labelsize=7.0)
            axis.tick_params(axis="x", length=0, pad=2)
    fig.suptitle("A100 80GB RC2 absolute FFT throughput distributions", fontsize=9.5, y=0.995)
    fig.text(
        0.5,
        0.01,
        "Boxes summarize sampled workload shapes; dots are individual matched cells.  Log scale; no pooled score.",
        ha="center",
        fontsize=6.7,
        color="#505050",
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.87, bottom=0.16, hspace=0.34, wspace=0.22)
    paths = [
        figure_dir / "rc2_fft_absolute_boxplot.svg",
        figure_dir / "rc2_fft_absolute_boxplot.pdf",
    ]
    for path in paths:
        _save_figure(fig, path, "A100 80GB RC2 absolute FFT throughput distributions")
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    rows = _load_rows(args.source)
    csv_path = args.output_dir / "rc2_fft_absolute_throughput.csv"
    summary_path = args.output_dir / "rc2_fft_absolute_throughput_summary.json"
    _write_csv(csv_path, rows)
    figures = _plot_boxplot(rows, args.figure_dir) + _plot(rows, args.figure_dir)
    summary = {
        "schema": "cubutterfly-rc2-absolute-throughput-v2",
        "source": str(args.source.resolve()),
        "source_status": "complete-measured-matrix",
        "rows": len(rows),
        "cells": len({row["cell_id"] for row in rows}),
        "systems": [name for name, _, _ in SYSTEMS],
        "metrics": ["transforms_per_s", "butterfly_pairs_per_s"],
        "figure_paths": [str(path.resolve()) for path in figures],
        "csv_path": str(csv_path.resolve()),
        "note": "Absolute rates are paired by workload; boxplots summarize the sampled distribution and no cross-size aggregate is claimed.",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(f"wrote {csv_path}")
    for figure in figures:
        print(f"wrote {figure}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
