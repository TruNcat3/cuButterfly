#!/usr/bin/env python3
"""Render the current measured snapshot; never fill missing performance cells."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from load_fft_l_comparison import load_fft_l_comparison
from load_fp16_acceptance import load_fp16_acceptance
from export_application_closeout import load as load_application_closeout

ROOT = Path(__file__).resolve().parents[2]
GROUPS = [("fft", "fp16"), ("fft", "bf16"), ("fft", "fp32"), ("fft", "fp64"), ("fwht", "fp32"),
          ("fwht", "fp64"), ("ntt", "word32"), ("ntt", "word64"),
          ("subset-zeta", "uint32"), ("subset-zeta", "word64"),
          ("superset-zeta", "uint32"), ("superset-zeta", "word64"),
          ("structured-2x2", "fp32"), ("structured-2x2", "fp64")]
LABELS = ["FFT-F16*", "FFT-BF16*", "FFT-F32", "FFT-F64", "FWHT-F32", "FWHT-F64",
          "NTT-w32", "NTT-w64", "SZ-u32", "SZ-w64",
          "PZ-u32", "PZ-w64", "2x2-F32", "2x2-F64"]
STYLES = {
    "cuButterfly": ("#0072B2", "o", ""),
    "cufft": ("#A34D00", "s", "//"),
    "vkfft": ("#007B59", "^", ".."),
    "daofht": ("#AA3377", "D", "\\\\"),
    "gpu-ntt": ("#555555", "P", "xx"),
}
NAMES = {"cufft": "cuFFT", "vkfft": "VkFFT", "daofht": "DaoFHT",
         "gpu-ntt": "GPU-NTT", "cuButterfly": "cuButterfly"}


def gm(values):
    return math.exp(statistics.mean(math.log(v) for v in values)) if values else None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def axis_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=.5)
    ax.set_axisbelow(True)


def save(fig, name, output, manifest, description):
    paths = []
    for ext in ("pdf", "svg", "png"):
        path = output / f"current_refresh_{name}.{ext}"
        fig.savefig(path, dpi=300, facecolor="white", transparent=False)
        paths.append(path)
    manifest["figures"][name] = {
        "description": description,
        "width_inches": float(fig.get_figwidth()),
        "height_inches": float(fig.get_figheight()),
        "files": {str(p.relative_to(ROOT)): digest(p) for p in paths},
    }
    plt.close(fig)


def main_pairs(data):
    return [p for p in data["operators"]["pairs"]
            if p["cohort"] == "main66" and p.get("speedup") is not None]


def shown_applications(data):
    return [p for p in data["applications"]["pairs"] if p.get("speedup") is not None]


def operator_plot(data, output, manifest):
    requests = [r for r in data["operators"]["requests"] if r["cohort"] == "main66"]
    pairs = main_pairs(data)
    low = data["fft_l_comparison"]
    public_default = low["identity"]["selection_kind"] == "public_DEFAULT"
    low_count = len(low["requests"])
    requests += low["requests"]
    pairs += low["pairs"]
    fig, ax = plt.subplots(figsize=(7, 2.55))
    fig.subplots_adjust(left=.075, right=.995, bottom=.30, top=.70)
    ax.axvspan(-.5, 1.5, color="#f3f3f3", zorder=0)
    all_values = []
    for i, group in enumerate(GROUPS):
        match = [r for r in pairs if (r["operator"], r["precision"]) == group
                 and r.get("speedup") is not None]
        series = {}
        for baseline in list(STYLES)[1:]:
            vals = [r["speedup"] for r in match if r["baseline"] == baseline]
            if vals:
                series[baseline] = vals
        offsets = np.linspace(-.25, .25, len(series)) if len(series) > 1 else [0]
        annotations = []
        for offset, (name, vals) in zip(offsets, series.items()):
            if not vals:
                continue
            color, marker, hatch = STYLES[name]
            all_values.extend(vals)
            if len(vals) > 1:
                box = ax.boxplot([vals], positions=[i + offset], widths=.15,
                             patch_artist=True, whis=(0, 100), showfliers=False,
                             manage_ticks=False,
                             boxprops={"edgecolor": color, "facecolor": "white", "linewidth": .8},
                             medianprops={"color": "black", "linewidth": .85},
                             whiskerprops={"color": color, "linewidth": .7},
                             capprops={"color": color, "linewidth": .7})
                box["boxes"][0].set_hatch(hatch)
            ax.scatter(i + offset + np.linspace(-.035, .035, len(vals)), vals,
                       s=10, marker=marker, color=color, zorder=3)
            short = {"cufft": "C", "vkfft": "V", "daofht": "D", "gpu-ntt": "G"}[name]
            annotations.append(f"{short} {gm(vals):.2f}x/{len(vals)}")
        note = "\n".join(annotations) if annotations else "No ext.\nresult"
        if group == ("fft", "fp16"):
            note += "\n" + ("DEFAULT" if public_default else "Replay")
        elif group == ("fft", "bf16"):
            note = "Deferred"
        ax.text(i, 1.035, note,
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.8)
    labels = [f"{label}\n(n={sum((r['operator'], r['precision']) == group for r in requests)})"
              for label, group in zip(["FP16*", "BF16*", "FP32", "FP64", "FP32", "FP64",
                                      "word32", "word64", "uint32", "word64",
                                      "uint32", "word64", "FP32", "FP64"], GROUPS)]
    ax.set_xticks(range(len(GROUPS)), labels, fontsize=6.2)
    for left, right, title in ((0, 3, "FFT"), (4, 5, "FWHT"), (6, 7, "NTT"),
                               (8, 9, "Subset zeta"), (10, 11, "Superset zeta"),
                               (12, 13, "Structured 2×2")):
        ax.text((left + right) / 2, -.30, title, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.5, fontweight="bold")
    if not all_values:
        raise ValueError("operator plot has no matched speedups")
    ax.set_yscale("log", base=10)
    ax.set_ylim(min(all_values) / 1.35, max(all_values) * 1.4)
    ax.set_xlim(-.55, len(GROUPS) - .45)
    ax.set_ylabel("Baseline / cuButterfly speedup (log10)", fontsize=7.5)
    axis_style(ax)
    for boundary in (3.5, 5.5, 7.5, 9.5, 11.5):
        ax.axvline(boundary, color="#AAAAAA", linestyle=":", linewidth=.7)
    # The plotted values are ratios, so cuButterfly is the denominator rather
    # than a plotted series.  Leaving its old marker in the legend would imply
    # an unpaired absolute-latency series.
    handles = [Line2D([], [], color=c, marker=m, linestyle="none", markersize=4,
                      label=NAMES[n]) for n, (c, m, _) in STYLES.items()
               if n != "cuButterfly"]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=7)
    fig.text(.5, .825, "Matched complete-call ratios (baseline / cuButterfly); C/V/D/G = cuFFT/VkFFT/DaoFHT/GPU-NTT",
             ha="center", fontsize=6.2)
    low_label = f"{low_count} FP16 DEFAULT requests" if public_default else "1 FFT-L replay"
    low_scope = ("*FP16: independent forward cohort; Further BF16 experiments deferred. Floating errors are reported."
                 if public_default else
                 "*FFT-L: public-winner replay only; Further BF16 experiments deferred. Floating errors are reported.")
    fig.text(.5, .012, f"Main 66 requests + {low_label}; boxes span workloads, not uncertainty.\n" + low_scope,
             ha="center", fontsize=6.1, linespacing=1.3)
    save(fig, "operators", output, manifest,
         f"Unified matched speedup view: main66 and {low_label}, kept as separate cohorts. Each box and point uses only a request with the same semantic case in cuButterfly and the named baseline. Further BF16 experiments are deferred. No fixed-control mappings enter the library comparison. Boxes span workloads, not uncertainty; missing results have no ratio.")
    manifest["figures"]["operators"].update(main_requests=66,
        fft_l_public_replays=0 if public_default else low_count,
        fft_l_public_defaults=low_count if public_default else 0,
        fft_l_bf16_status="deferred", fixed_controls_included=0,
        displayed_groups=[list(group) for group in GROUPS])


def application_plot(data, output, manifest):
    all_pairs = shown_applications(data)
    # Matched batching is an integration intervention, not an upstream API.
    # Keep those records in the snapshot and controlled-experiment table.
    controls = [r for r in all_pairs if r["baseline"] == "HEonGPU_native_stage_batch"]
    pairs = [r for r in all_pairs if r["baseline"] != "HEonGPU_native_stage_batch"]
    historical_path = ROOT / "experiments/snapshots/application_historical_complete_call.json"
    historical = json.loads(historical_path.read_text())
    if historical.get("status") != "historical-pinned-build" or len(historical.get("records", [])) != 4:
        raise ValueError("the four historical complete-call records are required explicitly")
    if any(type(r.get("performance_eligible")) is not bool for r in historical["records"]):
        raise ValueError("historical records require explicit numerical-evidence eligibility")
    eligible_history = [r for r in historical["records"] if r["performance_eligible"]]
    if [r["workload"] for r in eligible_history] != ["FNO 64² / B8", "FNO 128² / B4"]:
        raise ValueError("historical figure layout requires the two audited FNO records")
    fig, ax = plt.subplots(figsize=(3.45, 4.65))
    fig.subplots_adjust(left=.16, right=.985, bottom=.59, top=.86)
    fft = [r for r in pairs if r["cohort"] in ("poisson_graph", "fft_filter_graph")]
    order = [("poisson_graph", 1024, 4), ("poisson_graph", 2048, 4), ("poisson_graph", 2048, 16),
             ("fft_filter_graph", 1024, 16), ("fft_filter_graph", 2048, 16)]
    for idx, (cohort, n, batch) in enumerate(order):
        for r in fft:
            edge = r.get("N") or r.get("shape", [None])[0]
            if isinstance(edge, list):
                edge = edge[0]
            if (r["cohort"], int(edge), int(r["batch"])) != (cohort, n, batch):
                continue
            cupy = "cupy" in r["baseline"].lower()
            color, marker, offset = ("#007B59", "^", .13) if cupy else ("#A34D00", "s", -.13)
            ax.scatter(idx * .8 + offset, r["speedup"], color=color, marker=marker, s=12,
                       facecolors="none" if cohort == "fft_filter_graph" else color, zorder=4)
    ckks = [r for r in pairs if r["cohort"] == "ckks_auto36"]
    for i, batch in enumerate((1, 8, 32)):
        vals = [r["speedup"] for r in ckks if int(r["batch"]) == batch
                and r["baseline"] == "HEonGPU_native_loop"]
        if vals:
            center = 4.6 + i * 1.3
            offsets = []
            for value in vals:
                candidates = [0, -.08, .08, -.16, .16, -.21, .21]
                offset_x = min(candidates, key=lambda candidate: sum(
                    abs(candidate - previous_x) < .075 and abs(value - previous_y) < .04
                    for previous_x, previous_y in offsets))
                offsets.append((offset_x, value))
            ax.scatter([center + x for x, _ in offsets], vals, color="#A34D00",
                       s=7, marker="o", linewidths=.3, zorder=3)
            ax.scatter(center, gm(vals), s=19, color="#A34D00", edgecolors="black",
                       linewidth=.5, marker="D", zorder=5)
    bs = [r for r in pairs if r["cohort"] == "ckks_bootstrap"]
    for i, batch in enumerate((1, 8)):
        for r in bs:
            if int(r["batch"]) == batch:
                center = 8.7 + i * .8
                ax.scatter(center, r["speedup"], s=14, color="#0072B2", marker="s", zorder=3)
                ax.annotate(f"{r['speedup']:.3f}", (center, r["speedup"]),
                            xytext=(0, 7 + i * 7), textcoords="offset points",
                            ha="center", fontsize=6)
    # Historical complete-call records are a separate pinned-build cohort.
    historical_points = []
    for i, record in enumerate(eligible_history):
        center = 11.0 + i * 1.0
        workload = record["workload"]
        color, marker = (("#246a9e", "D") if workload.startswith("FNO")
                         else ("#30856d", "X"))
        value = float(record["speedup"])
        ax.scatter(center, value, s=18, color=color, marker=marker,
                   facecolors="none", linewidths=.75, zorder=4)
        historical_points.append(dict(workload=workload, speedup=value))
    ax.set_xlim(-.5, 12.65)
    ax.set_ylim(.8, 2.55)
    ax.set_xticks([0, .8, 1.6, 2.4, 3.2, 4.6, 5.9, 7.2, 8.7, 9.5,
                   11.0, 12.0],
                  ["P1\n4", "P2\n4", "P2\n16", "F1*\n16", "F2*\n16",
                   "1", "8", "32", "1", "8", "64²\nB8", "128²\nB4"], fontsize=5.4)
    for tick in ax.get_xticklabels():
        tick.set_rotation(18)
        tick.set_ha("right")
    ax.set_yticks([1, 1.5, 2, 2.5], ["1.0", "1.5", "2.0", "2.5"])
    ax.set_ylabel("Baseline / cuButterfly", fontsize=7)
    ax.set_xlabel("")
    for boundary in (3.85, 8.05, 10.35):
        ax.axvline(boundary, color="#BBBBBB", linewidth=.6)
    ax.axvspan(10.35, 12.65, color="#f4f7f8", zorder=0)
    for x, title in ((1.6, "Poisson / filter"), (5.9, "CKKS"), (9.1, "Bootstrap"),
                     (11.5, "Pinned build")):
        ax.text(x, 1.04, title, transform=ax.get_xaxis_transform(),
                ha="center", fontsize=6.2)
    axis_style(ax)
    ax.axhline(1, color="#555555", linestyle="--", linewidth=.8)
    ax.tick_params(axis="y", labelsize=6.5, pad=2)
    handles = [Line2D([], [], color=color, marker=marker, linestyle="none",
                      markersize=3, label=label)
               for color, marker, label in (
                   ("#A34D00", "s", "Torch/cuFFT"), ("#007B59", "^", "CuPy"),
                   ("#A34D00", "o", "Native loop"), ("#0072B2", "s", "Native bootstrap"),
                   ("#246a9e", "D", "FNO historical"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, 1.0),
               ncol=3, frameon=False, fontsize=5.7, handletextpad=.3,
               columnspacing=.7)
    fig.text(.5, .49, "P/F: Poisson/filter; 1/2: 1024/2048; *single process.\n"
             "FNO retains its historical build; invalid earlier QuaRot traces are excluded.",
             ha="center", fontsize=5.25, linespacing=1.25)
    supplement = data["application_closeout"]
    lower = fig.add_axes([.16, .12, .825, .25])
    fused = data.get("quarot_fused_followup", {}).get("records", [])
    fused_categories = [("QuaRot fused", 1), ("QuaRot fused", 4)] if fused else []
    categories = [("QuaRot", 1), ("QuaRot", 4), *fused_categories, ("filter", 1024), ("filter", 2048),
                  ("bootstrap", 1), ("bootstrap", 8), ("FNO", 64), ("FNO", 128)]
    for i, (kind, key) in enumerate(categories):
        selected = ([dict(row, role="external") for row in fused if row["batch"] == key]
                    if kind == "QuaRot fused" else [])
        for row in supplement["records"]:
            if row["timing_mode"] != "graph":
                continue
            scenario = row["scenario"]
            matches = ((kind == "QuaRot" and "Llama" in scenario and row["batch"] == key)
                       or (kind == "filter" and f"filter {key}^2" in scenario)
                       or (kind == "bootstrap" and "bootstrap" in scenario and row["batch"] == key)
                       or (kind == "FNO" and row.get("size") == key))
            if matches:
                selected.append(row)
        # Every process is shown. Offsets separate repeats; they are not uncertainty intervals.
        for j, row in enumerate(selected):
            diagnostic = row["role"] == "diagnostic"
            color = ("#777777" if diagnostic else "#007B59" if kind == "QuaRot fused"
                     else "#0072B2" if kind != "FNO" else "#AA3377")
            lower.scatter(i + (j - (len(selected) - 1) / 2) * .09, row["speedup"],
                          marker="D" if diagnostic else "o", s=15,
                          facecolors="none" if diagnostic else color, edgecolors=color, linewidths=.7)
    lower.axhline(1, color="#555555", linestyle="--", linewidth=.8)
    tail = len(categories) - .5
    lower.axvline(tail - 2, color="#BBBBBB", linewidth=.6)
    lower.axvspan(tail - 2, tail, color="#f4f7f8", zorder=0)
    if fused:
        lower.axvspan(1.5, 3.5, color="#edf7f2", zorder=0)
    lower.set_xlim(-.5, tail)
    lower.set_ylim(.93, 1.12)
    lower.set_yticks([.95, 1, 1.05, 1.10])
    lower.set_xticks(range(len(categories)), ["Q\nB1", "Q\nB4", *(["Qf\nB1", "Qf\nB4"] if fused else []),
                               "F1\nB16", "F2\nB16", "BS\nB1", "BS\nB8", "64²\nB4", "128²\nB4"], fontsize=5.7)
    lower.tick_params(axis="y", labelsize=6)
    lower.set_ylabel("Baseline / cuButterfly", fontsize=7)
    lower.set_title("Independent application cohorts", fontsize=6.3, pad=7)
    axis_style(lower)
    fig.text(.5, .015, "Bottom: dots are processes (FNO: median of 2); Q/Qf: composed/fused.\n"
             "Hollow diamonds: filter diagnostics; no pooling. FNO eager kept in table.",
             ha="center", fontsize=5.5, linespacing=1.25)
    save(fig, "applications", output, manifest,
         "Top: snapshot comparisons and two numerically checked historical FNO points. "
         "Two earlier QuaRot records are archived but excluded for invalid random-state preparation and absent output checks. "
         "Bottom: all 14 audited closure ratios (four QuaRot losses, four CuPy filter repeats, "
         "four separately marked same-padding filter diagnostics, two bootstrap repeats) and two historical "
         "FNO hotpath Graph aggregate ratios, plus separately marked fused QuaRot follow-up processes when available. "
         "FNO eager rows remain separate in frozen data/table. "
         "Panels use different labeled ratio ranges; no cohorts or builds are pooled.")
    manifest["figures"]["applications"].update(
        pair_count=len(pairs), ckks_pairs=len(ckks), ckks_requests_per_control=36,
        historical_records=len(historical_points), historical_source=str(historical_path.relative_to(ROOT)),
        historical_archived_records=len(historical["records"]),
        historical_exclusions=[dict(workload=r["workload"], reason=r["exclusion_reason"],
                                    source=r["source"], source_sha256=r["source_sha256"])
                               for r in historical["records"] if not r["performance_eligible"]],
        historical_source_sha256=digest(historical_path),
        matched_integration_pairs_in_controlled_table=len(controls),
        supplemental_graph_rows=16, supplemental_eager_rows_in_table=2,
        fused_quarot_processes=len(fused),
        supplement_source="experiments/snapshots/current_refresh/application_closeout.json",
        supplement_sources=supplement["sources"],
        statistic="Snapshot CKKS diamonds are GMs of 12 ratios. Supplemental closure dots retain every process; FNO dots are medians of two process medians. No confidence intervals or cross-cohort means.",
        layout="Two panels with separately labeled ratio ranges; historical and build-prior cohorts remain distinct.")


def mechanism_plot(data, output, manifest):
    # The earlier six-bar snapshot hid most of the controlled workload matrix.
    # This source is the audited A100-40GB M1 cohort: 18 workloads, three
    # organizations and three independent trials. We show local0/variant
    # ratios for every batch and retain the hardware cohort in the manifest.
    source = ROOT / "experiments/snapshots/motivation_organization_a100.csv"
    rows = list(csv.DictReader(source.open(newline="")))
    expected = {(op, str(n), str(b), variant)
                for op in ("fft", "fwht", "ntt")
                for n in (16, 20) for b in (1, 4, 16)
                for variant in ("local0", "local2", "local4")}
    seen = {(r["operator"], r["logN"], r["batch"], r["variant"]) for r in rows}
    if len(rows) != 162 or seen != expected:
        raise ValueError("stage-grouping source must contain 162 M1 trial rows")
    for row in rows:
        value = float(row["ratio_local0_over_variant"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("invalid stage-grouping ratio")

    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    fig.subplots_adjust(left=.14, right=.99, bottom=.26, top=.73)
    colors = {"local2": "#0072B2", "local4": "#A34D00"}
    labels = {"local2": "two-stage local groups", "local4": "up to four-stage groups"}
    groups = [(op, n) for op in ("fft", "fwht", "ntt") for n in (16, 20)]
    for gi, (op, logn) in enumerate(groups):
        for vi, variant in enumerate(("local2", "local4")):
            x = gi + (-.16 if vi == 0 else .16)
            values = [float(r["ratio_local0_over_variant"]) for r in rows
                      if r["operator"] == op and r["logN"] == str(logn)
                      and r["variant"] == variant]
            if len(values) != 9:
                raise ValueError("each stage-grouping box must contain 9 values")
            color = colors[variant]
            box = ax.boxplot([values], positions=[x], widths=.25,
                             patch_artist=True, whis=(0, 100), showfliers=False,
                             manage_ticks=False,
                             medianprops={"color": "black", "linewidth": .7},
                             boxprops={"edgecolor": color, "facecolor": "white", "linewidth": .7},
                             whiskerprops={"color": color, "linewidth": .6}, capprops={"color": color, "linewidth": .6})
            box["boxes"][0].set_hatch("//" if variant == "local2" else "..")
            jitter = np.linspace(-.075, .075, len(values))
            ax.scatter([x + delta for delta in jitter], values, s=3.2,
                       color=color, alpha=.63, zorder=3, linewidths=0)
    ax.axhline(1, color="#555555", linestyle="--", linewidth=.7)
    ax.set_xticks(range(len(groups)), [f"{op.upper()}\n$2^{{{logn}}}$" for op, logn in groups], fontsize=5.8)
    ax.set_xlim(-.55, len(groups) - .45)
    ax.set_ylim(.55, 1.55)
    ax.set_ylabel("local0 / grouped organization", fontsize=6.9)
    ax.set_xlabel("operator and transform length; B=1, 4, 16 pooled per box", fontsize=6.1, labelpad=2)
    axis_style(ax)
    handles = [Line2D([], [], color=colors[v], marker="s", linestyle="none",
                      markerfacecolor="white", markeredgewidth=.6, markersize=4,
                      label=labels[v]) for v in colors]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.02),
              ncol=2, frameon=False, fontsize=5.9, handletextpad=.25, columnspacing=.8)
    fig.text(.5, .015, "18 workloads × 3 organizations × 3 trials = 162 observations;\nboxes span batches and markers show every trial. A100-40GB M1 cohort.",
             ha="center", fontsize=5.9, linespacing=1.25)
    save(fig, "organization", output, manifest,
         "Cross-operator fixed-core stage-grouping controls from the audited A100-40GB M1 cohort. Six operator/length groups show all nine batch/trial ratios for each of two grouped organizations; ratios are local0 divided by the grouped variant, so values above one indicate an improvement over stagewise exchange.")
    manifest["figures"]["organization"].update(
        source=str(source.relative_to(ROOT)), source_sha256=digest(source),
        workloads=18, cells=54, trial_rows=162, batches=[1, 4, 16],
        statistic="local0/variant ratio; box min/max across 9 batch/trial observations; points are individual trials",
        variants=["local2", "local4"], hardware="NVIDIA A100 PCIe 40GB",
        resource_counters="Not plotted in this mechanism figure; matched NCU records appear in the scaling analysis figure")


def write_tables(data, generated):
    pairs = main_pairs(data)
    rows = []
    for op, precision in GROUPS:
        for baseline in list(STYLES)[1:]:
            selected = [r for r in pairs if (r["operator"], r["precision"], r["baseline"]) == (op, precision, baseline)]
            if not selected:
                continue
            a = [r["speedup"] for r in selected]
            rows.append({"cohort": "main66", "operator": op, "precision": precision, "baseline": baseline,
                         "n": len(a), "gm": gm(a), "faster": sum(v > 1 for v in a),
                         "min": min(a), "max": max(a)})
    for item in data["fft_l_comparison"]["summary"]:
        rows.append({"cohort": item["cohort"], "operator": item["operator"],
                     "precision": item["precision"], "baseline": item["baseline"],
                     "n": item["pairs"], "gm": item["gm"], "faster": item["wins"],
                     "min": item["min"], "max": item["max"]})
    cyclic_pairs = [r for r in data.get("cyclic", {}).get("pairs", [])
                    if r.get("speedup") is not None]
    for precision in ("word32", "word64"):
        a = [r["speedup"] for r in cyclic_pairs if r["precision"] == precision]
        if a:
            rows.append({"cohort": "cyclic90", "operator": "ntt", "precision": precision,
                         "baseline": "gpu-ntt", "n": len(a), "gm": gm(a),
                         "faster": sum(v > 1 for v in a), "min": min(a), "max": max(a)})
    tex = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
           r"Operator & Baseline & Pairs & GM & Faster & N/A/X\\\midrule"]
    cohort = None
    for r in rows:
        if r["cohort"] != cohort:
            if cohort is not None:
                tex.append(r"\midrule")
            title = {"main66": "Main 66-request study",
                     "fft_l_public_winner_replay": "FFT-L public-winner replay (rerun pending)",
                     "fp16_public_default9_20261105": "Independent FP16 public DEFAULT study",
                     "cyclic90": "Independent cyclic NTT study (90 requests)"}[r["cohort"]]
            tex.append(r"\multicolumn{6}{@{}l}{\emph{" + title + r"}}\\")
            cohort = r["cohort"]
        name = f"{r['operator'].upper()} {r['precision'].upper().replace('WORD', 'w')}"
        mix = "--"
        if r["cohort"] == "main66":
            selected = [p for p in pairs if (p["operator"], p["precision"], p["baseline"]) ==
                        (r["operator"], r["precision"], r["baseline"])]
            categories = ("native", "adapter", "external-axis-composition")
            if any(p.get("category") not in categories for p in selected):
                raise ValueError("Main operator table requires an explicit comparison category")
            mix = "/".join(str(sum(p["category"] == c for p in selected)) for c in categories)
        if r["n"]:
            tex.append(f"{name} & {NAMES[r['baseline']]} & {r['n']} & {r['gm']:.3f}$\\times$ & {r['faster']}/{r['n']} & {mix}\\\\")
        else:
            tex.append(f"{name} & {NAMES[r['baseline']]} & 0 & -- & -- & --\\\\")
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (generated / "operator_table.tex").write_text("\n".join(tex) + "\n")
    ckks = [r for r in shown_applications(data) if r["cohort"] == "ckks_auto36"]
    tex = [r"\begin{tabular}{@{}rrrrr@{}}", r"\toprule",
           r" & & External & \multicolumn{2}{c}{Integration controls}\\",
           r"$B$ & Requests & Loop & Batch & Faster native\\\midrule"]
    for batch in (1, 8, 32):
        selected = [r for r in ckks if int(r["batch"]) == batch]
        by_request = {}
        for r in selected:
            key = (r["scenario"], str(r.get("N", r.get("shape"))), str(r.get("method")))
            by_request.setdefault(key, []).append(r["speedup"])
        vals = [[r["speedup"] for r in selected if r["baseline"] == b]
                for b in ("HEonGPU_native_loop", "HEonGPU_native_stage_batch")]
        best = [min(v) for v in by_request.values()]
        tex.append(f"{batch} & {len(best)} & {gm(vals[0]):.3f} & {gm(vals[1]):.3f} & {gm(best):.3f}\\\\")
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (generated / "ckks_table.tex").write_text("\n".join(tex) + "\n")
    with (generated / "operator_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    macros = {}
    for precision in ("fp32", "fp64"):
        selected = [r["speedup"] for r in pairs if r["operator"] == "fft" and r["precision"] == precision
                    and r["baseline"] == "cufft"]
        macros["CurrentFFT" + ("Single" if precision == "fp32" else "Double")] = f"{gm(selected):.3f}"
    for op, baseline, macro in (("ntt", "gpu-ntt", "CurrentNTT"), ("fwht", "daofht", "CurrentFWHT")):
        macros[macro] = f"{gm([r['speedup'] for r in pairs if r['operator'] == op and r['baseline'] == baseline]):.3f}"
    for precision, suffix in (("word32", "WordThirtyTwo"), ("word64", "WordSixtyFour")):
        values = [r["speedup"] for r in cyclic_pairs if r["precision"] == precision]
        macros["CurrentCyclicNTT" + suffix] = f"{gm(values):.3f}" if values else "pending"
    for baseline, macro in (("HEonGPU_native_loop", "CurrentCKKSLoop"),
                            ("HEonGPU_native_stage_batch", "CurrentCKKSBatch")):
        macros[macro] = f"{gm([r['speedup'] for r in ckks if r['baseline'] == baseline]):.3f}"
    fastest = [r["speedup"] for r in data["applications"]["strongest_native_pairs"]
               if r.get("speedup") is not None]
    macros["CurrentCKKSFast"] = f"{gm(fastest):.3f}"
    (generated / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "experiments/snapshots/current_refresh/snapshot.json")
    parser.add_argument("--figures", type=Path, default=ROOT / "experiments/figures")
    parser.add_argument("--figures-only", action="store_true", help="Preserve existing tables and numeric macros")
    parser.add_argument("--fp16-acceptance", type=Path,
                        help="Use a completed, parent-committed FP16 acceptance cohort instead of the historical replay")
    parser.add_argument("--frozen-inputs", action="store_true",
                        help="Use repository-frozen FP16 and application data; no raw results required")
    args = parser.parse_args()
    data = json.loads(args.snapshot.read_text())
    if args.frozen_inputs and args.fp16_acceptance:
        parser.error("--frozen-inputs cannot import --fp16-acceptance")
    data["fft_l_comparison"] = (json.loads((args.snapshot.parent / "fp16_comparison.json").read_text())
                                if args.frozen_inputs else load_fp16_acceptance(args.fp16_acceptance)
                                if args.fp16_acceptance else load_fft_l_comparison())
    data["application_closeout"] = load_application_closeout(args.snapshot.parent / "application_closeout.json")
    fused_path = args.snapshot.parent / "quarot_fused_e2e.json"
    if fused_path.exists():
        data["quarot_fused_followup"] = json.loads(fused_path.read_text())
    unexpected = {(r["operator"], r["precision"]) for r in data["operators"]["requests"]
                  if r["cohort"] == "main66"} - set(GROUPS)
    if unexpected:
        raise ValueError(f"New groups need an explicit figure layout; refusing to omit {unexpected}")
    args.figures.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "current-paper-refresh-figures-v1", "snapshot": str(args.snapshot.resolve().relative_to(ROOT)),
                "snapshot_sha256": digest(args.snapshot), "renderer_sha256": digest(Path(__file__)),
                "matplotlib": matplotlib.__version__, "figures": {},
                "missing": "Absent measurements are not imputed; PRO 6000 and unfinished cohorts remain pending."}
    manifest["fft_l_evidence"] = data["fft_l_comparison"]["identity"]
    manifest["fft_l_loader_sha256"] = (None if args.frozen_inputs else digest(Path(__file__).with_name(
        "load_fp16_acceptance.py" if args.fp16_acceptance else "load_fft_l_comparison.py")))
    manifest["fft_l_acceptance"] = str(args.fp16_acceptance.resolve()) if args.fp16_acceptance else None
    manifest["frozen_inputs"] = args.frozen_inputs
    if args.frozen_inputs:
        manifest["frozen_fp16_sha256"] = digest(args.snapshot.parent / "fp16_comparison.json")
    manifest["application_closeout_sha256"] = digest(args.snapshot.parent / "application_closeout.json")
    manifest["application_closeout_exporter_sha256"] = digest(Path(__file__).with_name("export_application_closeout.py"))
    if fused_path.exists():
        manifest["quarot_fused_followup_sha256"] = digest(fused_path)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": .6}):
        operator_plot(data, args.figures, manifest)
        application_plot(data, args.figures, manifest)
        mechanism_plot(data, args.figures, manifest)
    if not args.figures_only:
        write_tables(data, args.snapshot.parent)
    (args.snapshot.parent / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"figures": list(manifest["figures"]), "snapshot": manifest["snapshot"]}))


if __name__ == "__main__":
    main()
