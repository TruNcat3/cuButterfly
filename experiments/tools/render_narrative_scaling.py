#!/usr/bin/env python3
"""Render all cells of the bounded, measured M1 organization cohort."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import io
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "results/narrative_gap_20261027/m1_gpu40_cases/collection.json"
STEM = "narrative_scaling_20261027"
OPERATORS = ("fft", "fwht", "ntt")
VARIANTS = ("local0", "local2", "local4")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_evidence(source):
    document = json.loads(source.read_text())
    if document.get("schema") != "cubutterfly-controlled-case-collection-v1":
        return document, []
    if document.get("status") != "complete":
        raise ValueError("the per-case collection is not complete")
    manifest_path = source.parent / "manifest.json"
    if sha(manifest_path) != document["manifest_sha256"]:
        raise ValueError("collection manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if document["parent_study_sha256"] != manifest["identity"]["study_sha256"]:
        raise ValueError("parent study identity changed")
    expected = {row["id"] for row in manifest["cases"]}
    if set(document["records"]) != expected:
        raise ValueError("collection does not cover every declared case")
    cases, provenance = [], []
    for key, record in document["records"].items():
        if record.get("status") != "complete":
            raise ValueError("incomplete case release")
        path = Path(record["release_path"])
        if sha(path) != record["release_sha256"]:
            raise ValueError("case release changed")
        journal = json.loads(path.read_text())
        if (journal.get("status") != "complete" or journal.get("completed") is not True
                or journal["identity"] != record["identity"]
                or journal["identity"]["device"] != document["device"]
                or journal["protocol"] != manifest["identity"]["protocol"]
                or len(journal["cases"]) != 1 or journal["cases"][0]["id"] != key):
            raise ValueError("case release contract changed")
        cases.extend(journal["cases"])
        provenance.append(dict(case=key, path=str(path.relative_to(ROOT)),
                               sha256=sha(path), identity=journal["identity"]))
    # A plotting view only. No merged measurement journal or new runtime identity
    # is written; each original guarded release remains the source of its data.
    view = dict(status="complete", completed=True, cases=cases,
                protocol=manifest["identity"]["protocol"], identity=dict(device=document["device"]))
    return view, provenance


def measured_rows(journal):
    if journal.get("status") != "complete" or journal.get("completed") is not True:
        raise ValueError("an incomplete journal cannot supply this complete-cohort figure")
    if journal["protocol"]["trials"] != 3 or journal["protocol"]["verify_batches"] != 0:
        raise ValueError("expected three independent trials and full-batch verification")
    expected = {(op, n, b) for op in OPERATORS for n in (16, 20) for b in (1, 4, 16)}
    seen, rows = set(), []
    for case in journal["cases"]:
        w = case["workload"]
        key = (w["operator"], w["logN"], w["batch"])
        if key not in expected or key in seen or case.get("status") != "complete":
            raise ValueError("duplicate, missing or unexpected M1 case")
        seen.add(key)
        if w["precision"] != ("word64" if w["operator"] == "ntt" else "fp64"):
            raise ValueError("M1 precision mismatch")
        if {v["id"] for v in case["variants"]} != set(VARIANTS):
            raise ValueError("incomplete organization controls")
        for variant in case["variants"]:
            if variant["preflight"].get("status") != "accepted":
                raise ValueError("missing recorded numerical sanity check")
            trials = variant["trials"]
            if len(trials) != 3 or {t["trial"] for t in trials} != {0, 1, 2}:
                raise ValueError("incomplete independent trials")
            for trial in trials:
                ms = float(trial["kernel_ms"])
                if not trial.get("performance_accepted") or not math.isfinite(ms) or ms <= 0:
                    raise ValueError("invalid measured time")
                rows.append(dict(case=case["id"], operator=w["operator"], precision=w["precision"],
                                 logN=w["logN"], batch=w["batch"], variant=variant["id"],
                                 trial=trial["trial"], kernel_ms=ms,
                                 billion_input_values_per_second=w["batch"] * (1 << w["logN"]) / (ms * 1e6)))
    if seen != expected or len(rows) != 162:
        raise ValueError("M1 requires 18 cases, 54 cells, 162 independent trial records")
    return rows


# Audited release bindings from NARRATIVE_SCALING_PROFILE_20261027.md.
PROFILE_HASHES = {
    "fft": ("25893b6a160c0208684231f1d40ae923596ab5d387f8318631da2098432253db",
            ("63f679d94cbe3e7e8e32b985943ca0abba82b9c77c5912fe29f7cd820a6e1833", "5c781a3b5fd897117770f99442b642c7fe6eb3b0b07d90140c78f787e6609672", "9e2ab7b52b4505a44fc21c88425fa6480762b5942e9bb54c1b731d49ea114533")),
    "fwht": ("d4221effbe2c42db5308dc5726510a942b48ff4f97fef44f7b355dbb87e7a13e",
             ("1bcb0885714a21695f94cde20b8f8bf82fc5dc236da709c92f48e445e10d5500", "2499b2d960c19ab5df05762c4b8ed28218a74bdf2c0659aede58c0d7940c4129", "334fd02a7a1504d50ee305d36a32b3b77f0ebb1821ba7e60255d3c7f91a2450c")),
    "ntt": ("b2e0a4c1df619f10f8185e315dd84c5a90d8c8dc1a6e232dc790589b517a2cb4",
            ("7b208795cafb332a0fa785b04e6edac254ddaff018163ca430d40a4af47bc6fb", "4a60e44c98d05f6acb3ec4b6d900035417e024da120482d319127799941180c4", "93b76a0c8f495879be1aa5412aa3ae4bceef28fbe5ae4f14f42be1774f81f02b")),
}
PROFILE_METRICS = (
    ("launch__registers_per_thread", "Registers / thread", 180),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "Active warps (%)", 105),
    ("smsp__warp_issue_stalled_barrier_per_warp_active.pct", "Barrier stall (%)", 60),
)


def load_profiles(journal, releases):
    """Read only hash-bound M1 kernels; never pool profiler and trial records."""
    profiles, provenance = [], []
    for op in OPERATORS:
        path = ROOT / f"results/narrative_gap_20261027/m1_profiles/{op}/release/profile/manifest.json"
        manifest_hash, csv_hashes = PROFILE_HASHES[op]
        if sha(path) != manifest_hash:
            raise ValueError("audited M1 profile manifest changed")
        manifest = json.loads(path.read_text())
        if (manifest["status"] != "complete" or manifest["failures"]
                or manifest["gpu_uuid"] != journal["identity"]["device"]["uuid"]
                or manifest["source_journal_sha256"] not in {r["sha256"] for r in releases}):
            raise ValueError("M1 profile hardware or ordinary-journal binding mismatch")
        case = f"{op}-{'word64' if op == 'ntt' else 'fp64'}-n20-b16"
        sources = []
        for variant, expected_hash in zip(VARIANTS, csv_hashes):
            record = next(v for v in manifest["variants"] if v["id"] == variant)
            if (record["case"] != case or record["status"] != "complete"
                    or not record["binary"]["matches_measured_identity"]):
                raise ValueError("M1 profile variant identity mismatch")
            csv_path = path.parent / case / variant / "ncu.csv"
            if sha(csv_path) != expected_hash:
                raise ValueError("audited M1 counter CSV changed")
            raw = csv_path.read_text()
            kernels = [r for r in csv.DictReader(io.StringIO(raw[raw.index('"ID"'):]))
                       if r["Kernel Name"]]
            family = "shared_iterative_kernel" if variant == "local0" else "shared_resident_kernel"
            if len(kernels) != 2 or any(family not in r["Kernel Name"] for r in kernels):
                raise ValueError("expected exactly two audited main kernels")
            for group, kernel in enumerate(kernels):
                profiles.append(dict(operator=op, variant=variant, group=group,
                    kernel=kernel["Kernel Name"], metrics={metric: float(kernel[metric].replace(',', ''))
                                                         for metric, _, _ in PROFILE_METRICS}))
            sources.append(dict(path=str(csv_path.relative_to(ROOT)), sha256=expected_hash))
        provenance.append(dict(manifest=str(path.relative_to(ROOT)), sha256=manifest_hash,
                               ordinary_journal_sha256=manifest["source_journal_sha256"], csv=sources))
    return profiles, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=DEFAULT)
    args = parser.parse_args()
    source = args.journal.resolve()
    journal, releases = load_evidence(source)
    rows = measured_rows(journal)
    profiles, profile_sources = load_profiles(journal, releases)
    out = ROOT / "experiments/figures"
    data = ROOT / "experiments/snapshots"
    out.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    csv_path = data / (STEM + ".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    colors = ("#555555", "#0072B2", "#A64B00")
    hatches = ("", "//", "..")
    labels = ("Stagewise shared exchange (local0)", "Register leaves: 2 stages (local2)", "Register leaves: up to 4 stages (local4)")
    settings = {"font.family": "DejaVu Sans", "font.size": 6.5, "axes.labelsize": 6.5,
                "axes.titlesize": 6.5, "legend.fontsize": 6.5, "pdf.fonttype": 42,
                "axes.spines.top": False, "axes.spines.right": False,
                "axes.linewidth": .65, "savefig.facecolor": "white"}
    outputs = [csv_path]
    with plt.rc_context(settings):
        fig = plt.figure(figsize=(3.45, 4.0))
        grid = fig.add_gridspec(3, 3, left=.13, right=.985, bottom=.105, top=.82,
                               hspace=.96, wspace=.52, height_ratios=[1, 1, .86])
        axes = [[fig.add_subplot(grid[r, c]) for c in range(3)] for r in range(2)]
        for col, op in enumerate(OPERATORS):
            limit = max(r["billion_input_values_per_second"] for r in rows if r["operator"] == op) * 1.12
            for row, logn in enumerate((16, 20)):
                ax = axes[row][col]
                for index, (variant, color, hatch, label) in enumerate(zip(VARIANTS, colors, hatches, labels)):
                    series = [[r["billion_input_values_per_second"] for r in rows
                               if (r["operator"], r["logN"], r["batch"], r["variant"]) == (op, logn, b, variant)]
                              for b in (1, 4, 16)]
                    medians = [statistics.median(values) for values in series]
                    errors = [[m - min(v) for m, v in zip(medians, series)],
                              [max(v) - m for m, v in zip(medians, series)]]
                    positions = [i + (index - 1) * .25 for i in range(3)]
                    ax.bar(positions, medians, width=.23, color="white", edgecolor=color,
                           hatch=hatch, linewidth=.7, yerr=errors, capsize=1.2,
                           error_kw={"elinewidth": .55, "ecolor": color, "capthick": .55}, label=label)
                    for center, values in zip(positions, series):
                        ax.scatter([center - .035, center, center + .035], values,
                                   s=3, color=color, zorder=4, linewidths=0)
                precision = "word64" if op == "ntt" else "FP64"
                ax.set_title(f"{op.upper()} {precision}\n$N=2^{{{logn}}}$", pad=4, linespacing=1.15)
                ax.set_xticks([0, 1, 2], ["1", "4", "16"])
                ax.set_xlim(-.55, 2.55)
                ax.set_ylim(0, limit)
                ax.minorticks_off()
                ax.grid(axis="y", linewidth=.4, color="#dddddd")
                ax.tick_params(length=2.5, pad=2)
                if col == 0:
                    ax.set_ylabel("Input values/s ($10^9$)", labelpad=2)
                if row == 1:
                    ax.set_xlabel("Batch", labelpad=2)
        handles, legend_labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", ncol=1, frameon=False,
                   bbox_to_anchor=(.51, 1), handlelength=1.8, labelspacing=.4)
        for col, (metric, title, limit) in enumerate(PROFILE_METRICS):
            ax = fig.add_subplot(grid[2, col])
            for index, (variant, color) in enumerate(zip(VARIANTS, colors)):
                for x, op in enumerate(OPERATORS):
                    values = [r["metrics"][metric] for r in profiles
                              if r["operator"] == op and r["variant"] == variant]
                    center = x + (index - 1) * .25
                    ax.plot([center, center], [min(values), max(values)], color=color, linewidth=.8)
                    # Both endpoints are actual main kernels, never trial errors.
                    ax.scatter([center] * 2, values, s=8, color=color, marker=("o", "s", "^")[index],
                               linewidths=.3, zorder=3)
            ax.set_title(title, pad=3)
            ax.set_xticks(range(3), ["F", "H", "N"])
            ax.set_xlim(-.55, 2.55); ax.set_ylim(0, limit)
            ax.grid(axis="y", linewidth=.4, color="#dddddd")
            ax.tick_params(length=2.5, pad=2)
        fig.text(.55, .285, "Profile: N=2²⁰, B=16; two main kernels / cell", ha="center", fontsize=6.5)
        fig.text(.55, .025, "F/H/N: FFT/FWHT/NTT; spans: two kernels, not trial error.\n○/□/△: local0/2/4. Single NCU capture per cell.",
                 ha="center", fontsize=6.5, linespacing=1.2)
        for extension in ("pdf", "png"):
            path = out / (STEM + "." + extension)
            fig.savefig(path, dpi=300, transparent=False)
            outputs.append(path)
        plt.close(fig)
    summary = []
    for case in journal["cases"]:
        summary.append(dict(case=case["id"], workload=case["workload"], comparisons=[
            {key: comparison[key] for key in ("baseline", "treatment", "baseline_median_kernel_ms",
                                             "treatment_median_kernel_ms", "median_treatment_over_baseline")}
            for comparison in case["comparisons"]]))
    manifest = dict(source=str(source.relative_to(ROOT)), source_sha256=sha(source),
                    script_sha256=sha(__file__), hardware=journal["identity"]["device"],
                    case_releases=releases,
                    protocol=journal["protocol"], rows=len(rows), cells=54,
                    statistic="median and full range of three independent process trials; not a confidence interval",
                    transformation="B*N/(kernel_ms*1e6) gives billions of input values per second; no observations excluded",
                    physical_groups="8+8 for logN16, 10+10 for logN20; scalar radix2 primitive fixed",
                    caveats=["Input values are complex for FFT and scalar for FWHT/NTT; this is not a common FLOP/s metric",
                             "Float timings are exploratory; preflight's rounded maximum error is not a complete error distribution",
                             "Changing local grouping also changes ownership, synchronization and resource use",
                             "Grouped bars show only observed batches; no interpolation or extrapolation is claimed",
                             "Fixed controls are not a tuned conventional baseline or full-HHD search"],
                    alt_text="A two-row, three-column grouped-bar figure shows throughput at batches 1,4,16 for FFT,FWHT,NTT at lengths 2^16 and 2^20. All 54 organization configurations show their median and full range of three trials; both timing rows share the same y limits within each operator. A bottom row shows registers per thread, active warp percentage, and barrier-stall percentage for the two main kernels in each of nine N2^20 B16 cells. Counter spans are kernel ranges, not trial uncertainty.",
                    publisher_status="Provisional single-column draft figure; final venue export requirements not certified",
                    size_inches=[3.45, 4.0], minimum_font_points=6.5,
                    profile_sources=profile_sources, profile_records=profiles,
                    profile_scope="Three N2^20 B16 workloads, nine organizations, 18 main kernel instances; one NCU capture each",
                    profile_statistic="Two individual main-kernel points joined by their observed range, not trial uncertainty; no profiler timing used",
                    layout="Two timing rows plus one three-metric profile row; grouped bars for three organizations at each measured batch; zero baseline and common per-operator y limits",
                    matplotlib_version=matplotlib.__version__,
                    outputs={str(p.relative_to(ROOT)): sha(p) for p in outputs}, comparisons=summary)
    (data / (STEM + ".json")).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "cells": 54, "source": str(source)}))


if __name__ == "__main__":
    main()
