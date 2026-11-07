#!/usr/bin/env python3
"""Render two independent fixed-core cohorts without live artifact access."""
import csv
import json
import math
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from render_factor_organization import ROOT, SNAPSHOT, ORGANIZATIONS, IDS, sha, validate

STEM = "core_organization_compact"
CORE = ROOT / "results/paper_completion_20260917/core_dataflow_ablation"
GENERATED = ROOT / "paper/generated"
FIGURES = ROOT / "paper/figures"


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(actual, expected):
    if not math.isclose(actual, float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Point differs from original figure: {actual} != {expected}")


def fp32_data():
    provenance = json.loads((GENERATED / "principle_figure_metadata.json").read_text())
    hashes = {}
    for name in ("summary.json", "measurements.csv", "runtime_manifest.json"):
        path = CORE / name
        relative = str(path.relative_to(ROOT))
        hashes[relative] = sha(path)
        if hashes[relative] != provenance["source_hashes"][relative]:
            raise ValueError(f"Historical source hash changed: {relative}")
    document = json.loads((CORE / "summary.json").read_text())
    raw = read_csv(CORE / "measurements.csv")
    if len(document["groups"]) != 6 or len(raw) != 72 or any("80GB" not in r["device_name"] for r in raw):
        raise ValueError("Expected six complete A100 80GB requests")
    old = {(int(r["logN"]), int(r["B"]), r["core"]): r
           for r in read_csv(GENERATED / "principle_core_organization.csv")}
    points = []
    for group in document["groups"]:
        for core in ("native", "cufftdx-thread"):
            observed = {}
            for org in ("stagewise-hbm", "resident-prefix-2"):
                selected = [r for r in raw if int(r["logN"]) == group["logN"]
                            and int(r["batch"]) == group["batch"]
                            and r["core"] == core and r["organization"] == org]
                if len(selected) != 3 or any(r["correct"] != "1" for r in selected):
                    raise ValueError("Expected three correct trials per cell")
                observed[org] = {int(r["trial"]): float(r["kernel_ms"]) for r in selected}
                if len(observed[org]) != 3 or any(not math.isfinite(v) or v <= 0 for v in observed[org].values()):
                    raise ValueError("Invalid or duplicate timing trial")
                close(statistics.median(observed[org].values()),
                      group["cells"][core + "/" + org]["median_kernel_ms"])
            stage, resident = observed["stagewise-hbm"], observed["resident-prefix-2"]
            if set(stage) != set(resident):
                raise ValueError("Trial pairing differs")
            trials = [stage[t] / resident[t] for t in sorted(stage)]
            conventional = statistics.median(stage.values()) * 1000
            retained = statistics.median(resident.values()) * 1000
            ratio = conventional / retained
            entry = dict(logN=group["logN"], N=2**group["logN"], B=group["batch"], core=core,
                         conventional_us=conventional, resident_us=retained,
                         ratio=ratio, min_ratio=min(trials), max_ratio=max(trials),
                         trial_ratios=json.dumps(trials), n=3)
            reference = old[(entry["logN"], entry["B"], core)]
            for key in ("conventional_us", "resident_us", "ratio", "min_ratio", "max_ratio"):
                close(entry[key], reference[key])
            for actual, expected in zip(trials, json.loads(reference["trial_ratios"])):
                close(actual, expected)
            points.append(entry)
    if len(points) != len(old):
        raise ValueError("Original FP32 point coverage differs")
    return points, raw, hashes


def fp64_data():
    snapshot = json.loads(SNAPSHOT.read_text())
    _, rows = validate(snapshot)  # Checks frozen JSON only; never resolve live binaries.
    old = {r["case_id"]: r for r in read_csv(GENERATED / "factor_organization_20261025_latency.csv")}
    table = []
    for case_id in IDS:
        row = rows[case_id]
        entry = dict(case_id=case_id, median_us=row["median_ms"] * 1000,
                     min_us=row["min_ms"] * 1000, max_us=row["max_ms"] * 1000,
                     max_abs_cpu=row["max_abs_cpu"], relative_l2_cpu=row["relative_l2_cpu"])
        entry.update({f"trial_{i+1}_us": t * 1000 for i, t in enumerate(row["trial_ms"])})
        for key, value in entry.items():
            if key != "case_id":
                close(value, old[case_id][key])
        table.append(entry)
    if set(old) != set(IDS):
        raise ValueError("Original FP64 point coverage differs")
    return table, snapshot


def main():
    a, raw, hashes = fp32_data()
    b, snapshot = fp64_data()
    rows = {r["case_id"]: r for r in b}
    style = {"font.family": "DejaVu Sans", "font.size": 6.5,
             "axes.labelsize": 6.3, "xtick.labelsize": 5.6, "ytick.labelsize": 5.8,
             "legend.fontsize": 5.6, "axes.spines.top": False, "axes.spines.right": False,
             "axes.linewidth": .6, "pdf.fonttype": 42, "ps.fonttype": 42,
             "svg.fonttype": "none", "savefig.facecolor": "white"}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(3.45, 2.50))
        axes = [fig.add_axes((.105, .31, .365, .42)),
                fig.add_axes((.595, .31, .375, .42))]
        for ax in axes:
            ax.grid(axis="y", color=".88", linewidth=.45)
            ax.set_axisbelow(True)
        ax = axes[0]
        for core, label, color, marker, offset in (
                ("native", "register", "#0072B2", "o", -.07),
                ("cufftdx-thread", "cuFFTDx Thread", "#A34D00", "s", .07)):
            selected = [r for r in a if r["core"] == core]
            med = [r["ratio"] for r in selected]
            # Show every paired trial behind the cell summary.  This exposes
            # the 36 observations per core without changing the estimator.
            for i, row in enumerate(selected):
                trials = json.loads(row["trial_ratios"])
                jitter = np.linspace(-.026, .026, len(trials))
                ax.scatter([i + offset + x for x in jitter], trials,
                           s=3.5, color=color, alpha=.65, zorder=3,
                           linewidths=0)
            ax.errorbar([i+offset for i in range(6)], med,
                        yerr=[[r["ratio"]-r["min_ratio"] for r in selected],
                              [r["max_ratio"]-r["ratio"] for r in selected]],
                        fmt=marker, color=color, markersize=3.0, capsize=1.5,
                        linewidth=.8, label=label)
        ax.axhline(1, color=".35", linestyle=":", linewidth=.8)
        requests = a[::2]
        ax.set(xticks=range(6), xticklabels=[f'$2^{{{r["logN"]}}}$/B{r["B"]}' for r in requests],
               xlim=(-.4, 5.4), ylim=(.98, max(r["max_ratio"] for r in a)*1.07),
               ylabel="stagewise / resident", xlabel="N / batch")
        ax.tick_params(axis="x", pad=1)
        for tick in ax.get_xticklabels():
            tick.set_rotation(32)
            tick.set_ha("right")
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False,
                  borderaxespad=0, handlelength=1.2, labelspacing=.1)
        ax = axes[1]
        for core, label, color, marker, offset in (
                ("native", "register", "#0072B2", "o", -.045),
                ("cufftdx", "cuFFTDx Block", "#A34D00", "s", .045)):
            selected = [rows[f"{core}-{org}"] for org in ORGANIZATIONS]
            # Each FP64 cell contributes seven Graph timing rounds.
            for i, row in enumerate(selected):
                trials = [row[f"trial_{j}_us"] for j in range(1, 8)]
                jitter = np.linspace(-.018, .018, len(trials))
                ax.scatter([i + offset + x for x in jitter], trials,
                           s=3.0, color=color, alpha=.62, zorder=3,
                           linewidths=0)
            ax.errorbar([i+offset for i in range(4)], [r["median_us"] for r in selected],
                        yerr=[[r["median_us"]-r["min_us"] for r in selected],
                              [r["max_us"]-r["median_us"] for r in selected]],
                        color=color, marker=marker, markersize=3.0, linewidth=.75,
                        elinewidth=.8, capsize=2, label=label)
        comparator = rows["cufft"]
        ax.axhspan(comparator["min_us"], comparator["max_us"], color=".7", alpha=.5, linewidth=0)
        ax.axhline(comparator["median_us"], color=".25", linestyle="--", linewidth=.85,
                   label="cuFFT (complete library)")
        ax.set(xticks=range(4), xticklabels=["1 / 0", "4 / 0", "4 / 1", "4 / 2"],
               xlim=(-.3, 3.3), ylim=(0, 155), yticks=(0, 50, 100, 150),
               xlabel="Tile groups / prefetch depth", ylabel="latency (µs)")
        ax.tick_params(axis="x", pad=1)
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index(x) for x in ("register", "cuFFTDx Block", "cuFFT (complete library)")]
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False,
                  borderaxespad=0, handlelength=1.3, labelspacing=.1)
        fig.text(.105, .975, "(a) Stage retention", fontsize=7, va="top", weight="bold")
        fig.text(.105, .92, "FP32 · 16-point leaves", fontsize=6, va="top")
        fig.text(.595, .975, "(b) Tile supply", fontsize=7, va="top", weight="bold")
        fig.text(.595, .92, "FP64 · 256-point factors", fontsize=6, va="top")
        fig.text(.5, .045, "Markers show every paired trial (72 FP32, 63 FP64); bars/lines are cell summaries.\nPanels use independent FFT workloads; not a cross-precision comparison.",
                 ha="center",fontsize=6,linespacing=1.2)
        outputs = []
        for ext in ("pdf", "png", "svg"):
            path = FIGURES / f"{STEM}.{ext}"
            extra = {"metadata": {"CreationDate": None, "ModDate": None}} if ext == "pdf" else {}
            fig.savefig(path, dpi=600, facecolor="white", transparent=False, **extra)
            if ext == "png":
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                rgb.save(path, dpi=(600, 600))
            outputs.append(path)
        plt.close(fig)
    for suffix, table in (("fp32_points", a), ("fp32_measurements", raw), ("fp64_latency", b)):
        path = GENERATED / f"{STEM}_{suffix}.csv"
        write_csv(path, table)
        outputs.append(path)
    metadata = dict(schema="core-organization-compact-v2", figure_inches=[3.45, 2.50], png_dpi=600,
                    matplotlib=matplotlib.__version__, python=sys.version.split()[0],
                    renderer_sha256=sha(__file__),
                    source_hashes=hashes, snapshot=str(SNAPSHOT.relative_to(ROOT)), snapshot_sha256=sha(SNAPSHOT),
                    comparison_csv_hashes={str(p.relative_to(ROOT)): sha(p) for p in (
                        GENERATED / "principle_core_organization.csv",
                        GENERATED / "factor_organization_20261025_latency.csv")},
                    frozen_sources={k: {"path": v["path"], "sha256": v["sha256"]}
                                    for k, v in snapshot["sources"].items()},
                    panel_a={"estimator": "stagewise cell median / resident cell median",
                             "range": "min/max of 3 trial-paired ratios; markers show all 72 raw trial ratios",
                             "cells": 12, "trial_rows": 72,
                             "scope": "historical Sep17 matched-core FP32 control; A100 80GB"},
                    panel_b={"estimator": "median of 7 Graph timing rounds, ms * 1000",
                             "range": "min/max of 7 timing rounds; markers show all 63 timing rounds",
                             "cells": 9, "timing_rows": 63,
                             "scope": snapshot["scope"]},
                    cohorts="Independent experiments; no pooled statistics or cross-panel numerical comparisons",
                    exclusions=[], validation="All 12 FP32 cells and 72 trial ratios plus all 9 FP64 cells and 63 timings match original figure CSVs at 1e-12 tolerance",
                    publisher_status="General manuscript figure; no venue compliance claim",
                    outputs={str(p.relative_to(ROOT)): sha(p) for p in outputs})
    (GENERATED / f"{STEM}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Rendered 3.45 x 2.50 inches; 12 FP32 cells/72 trials and 9 FP64 cells/63 timings match original CSVs")


if __name__ == "__main__":
    main()
