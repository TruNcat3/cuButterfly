#!/usr/bin/env python3
"""CPU-only, synthetic evaluation layouts. Never reads acceptance evidence.

The source column and watermark are mandatory, including for --input-dir. This
renderer deliberately cannot promote an illustrative CSV into measured evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper/generated/draft"
FIGURES = ROOT / "paper/figures/draft"
SEED = 20261009
SOURCE = "synthetic_example_not_measured"
WARNING = "SYNTHETIC EXAMPLE \u2014 NOT MEASURED"
BLUE, GREEN, BROWN, GRAY = "#28658c", "#3c8d70", "#a4512a", "#5e6a75"
INK = "#202124"
ORGANIZATIONS = ["stagewise", "resident", "aligned", "overlap"]
SCHEMAS = {
    "mechanism": ["source", "organization", "hbm_bytes", "resident_bytes", "latency_us"],
    "scaling": ["source", "precision", "log2_n", "batch", "status", "baseline", "contract", "baseline_us", "ours_us"],
    "selection": ["source", "selector", "budget_candidates", "selected_us", "pool_best_us", "prepared_ms"],
    "application": ["source", "workload", "scale", "implementation", "butterfly_us", "adapter_us", "other_us", "prepared_ms"],
}


def examples():
    rng = random.Random(SEED)
    mechanism = [dict(source=SOURCE, organization=name, hbm_bytes=hbm,
                      resident_bytes=resident, latency_us=latency)
                 for name, hbm, resident, latency in zip(
                     ORGANIZATIONS, [1048576, 720896, 655360, 622592],
                     [0, 32768, 32768, 65536], [10, 8.6, 9.4, 10.4])]
    scaling = []
    for precision in ("fp32", "fp64"):
        for b, batch in enumerate((1, 16, 256)):
            for n, log_n in enumerate((10, 14, 18, 22)):
                missing = (precision, batch, log_n) in {("fp32", 1, 22), ("fp64", 256, 22)}
                ratio = round(rng.uniform(0.72, 1.38), 2)
                baseline = (n + 1) * (b + 1) * (2 if precision == "fp64" else 1)
                scaling.append(dict(source=SOURCE, precision=precision, log2_n=log_n,
                    batch=batch, status="missing" if missing else "synthetic",
                    baseline="cuFFT", contract="C2C; forward; natural I/O; out-of-place; same precision",
                    baseline_us="" if missing else baseline,
                    ours_us="" if missing else baseline / ratio))
    selection = []
    for selector, regrets, costs in (
        ("model", [1.23, 1.09, 1.04, 1.02], [3, 6, 13, 28]),
        ("heuristic", [1.16, 1.13, 1.02, 1.04], [2, 5, 10, 21]),
    ):
        for budget, regret, cost in zip((4, 8, 16, 32), regrets, costs):
            selection.append(dict(source=SOURCE, selector=selector, budget_candidates=budget,
                                  selected_us=10 * regret, pool_best_us=10, prepared_ms=cost))
    application = []
    ratios = [0.84, 1.06, 0.94, 0.76, 1.08, 0.89]
    for wi, workload in enumerate(("FNO / FFT", "QuaRot / FWHT", "HEonGPU / NTT")):
        for si, scale in enumerate(("small", "large")):
            total = (wi + 1) * (si + 1) * 100
            for implementation in ("baseline", "ours"):
                fraction = [0.57, 0.12, 0.31] if implementation == "baseline" else [0.48, 0.18, 0.34]
                factor = 1 if implementation == "baseline" else ratios[2 * wi + si]
                application.append(dict(source=SOURCE, workload=workload, scale=scale,
                    implementation=implementation, **{key + "_us": total * factor * value
                    for key, value in zip(("butterfly", "adapter", "other"), fraction)},
                    prepared_ms=(4 + wi * 2 + si * 5) * (1 if implementation == "baseline" else 1.5)))
    return dict(mechanism=mechanism, scaling=scaling, selection=selection, application=application)


def write_data(tables, overwrite):
    DATA.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        path = DATA / (name + ".csv")
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing overwrite: {path}; use --overwrite for this renderer's files")
        with path.open("w", newline="", encoding="ascii") as stream:
            writer = csv.DictWriter(stream, SCHEMAS[name])
            writer.writeheader()
            writer.writerows(rows)


def load_data(directory):
    tables = {}
    for name, columns in SCHEMAS.items():
        with (directory / (name + ".csv")).open(newline="", encoding="ascii") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != columns:
                raise ValueError(f"{name}: expected columns {columns}")
            tables[name] = list(reader)
        if not tables[name] or any(row["source"] != SOURCE for row in tables[name]):
            raise ValueError("Draft renderer accepts only explicitly synthetic rows")
    return tables


def number(row, column):
    value = float(row[column])
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {column}: {row[column]}")
    return value


def frame():
    fig = plt.figure(figsize=(7, 2.2), facecolor="white")
    fig.text(0.5, 0.975, WARNING, ha="center", va="top", color="#913715",
             fontsize=9, weight="bold")
    return fig


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, pad=2)
    ax.set_axisbelow(True)


def mechanism(rows):
    fig = frame()
    grid = fig.add_gridspec(1, 3, left=.09, right=.99, top=.77, bottom=.26,
                           width_ratios=[1.35, 1.5, 1], wspace=.55)
    # HBM and resident bytes use separate axes and explicit units.
    left = grid[0].subgridspec(1, 2, width_ratios=[1.35, .8], wspace=.2)
    hbm, resident = fig.add_subplot(left[0]), fig.add_subplot(left[1])
    ordered = [next(row for row in rows if row["organization"] == name) for name in ORGANIZATIONS]
    y = np.arange(4)
    colors = [GRAY, BLUE, GREEN, BROWN]
    hbm.barh(y, [number(row, "hbm_bytes") / number(ordered[0], "hbm_bytes") for row in ordered],
             color=colors, edgecolor=INK, linewidth=.4, height=.58)
    hbm.set(yticks=y, yticklabels=ORGANIZATIONS, xlim=(0, 1.12), xticks=[0, 1], xlabel="HBM / stagewise")
    hbm.invert_yaxis()
    hbm.set_title("(a) Dataflow resources", loc="left", pad=15)
    for i, row in enumerate(ordered):
        value = number(row, "resident_bytes") / 1024
        resident.barh(i, value, color=colors[i], edgecolor=INK, linewidth=.4, height=.58)
    resident.set(yticks=[], xlim=(0, 72), xticks=[0, 64], xlabel="Resident KiB")
    resident.invert_yaxis()
    for ax in (hbm, resident):
        clean(ax)
    timeline = fig.add_subplot(grid[1])
    timeline.set_title("(b) Ready / buffer reuse", loc="left", pad=15)
    for row in range(4):
        # Abstract events illustrate dependencies; positions are not trace times.
        shift = [0, .12, .06, 0][row]
        for start, duration, color, label in [(0, .8, GRAY, "load"),
                (.95 + shift, 1, BLUE, "compute"), (2.15, .75, GREEN, "reuse")]:
            timeline.add_patch(Rectangle((start, row - .29), duration, .58,
                                        facecolor=color, edgecolor=INK, linewidth=.4))
            timeline.text(start + duration / 2, row, label, color="white", ha="center", va="center", fontsize=7)
        timeline.annotate("", xy=(.95 + shift, row), xytext=(.8, row),
                          arrowprops=dict(arrowstyle="->", color=INK, lw=.6))
    timeline.text(.9, -.72, "ready", ha="center", fontsize=7)
    timeline.set(xlim=(-.05, 3), ylim=(3.5, -.5), yticks=[], xticks=[], xlabel="Dependency order (not time)")
    timeline.spines[:].set_visible(False)
    latency = fig.add_subplot(grid[2])
    latency.set_title("(c) Warm latency", loc="left", pad=15)
    latency.barh(y, [number(row, "latency_us") for row in ordered], height=.58,
                 color=colors, edgecolor=INK, linewidth=.4)
    latency.set(yticks=y, yticklabels=[], xlim=(0, 12), xticks=[0, 5, 10], xlabel="Latency (us)")
    latency.invert_yaxis()
    clean(latency)
    fig.text(.5, .035, "Fixed core: stagewise / resident / aligned / overlap; schematic is not a kernel trace. No profiler-time latency.",
             ha="center", fontsize=7)
    return fig


def scaling(tables):
    fig = frame()
    grid = fig.add_gridspec(1, 4, left=.065, right=.985, top=.75, bottom=.38,
                           width_ratios=[1.1, 1.1, 1.25, 1.1], wspace=.65)
    cmap = LinearSegmentedColormap.from_list("paper_ratio", [BROWN, "#ffffff", BLUE])
    norm = TwoSlopeNorm(vmin=.7, vcenter=1, vmax=1.4)
    heat_axes = []
    for pi, precision in enumerate(("fp32", "fp64")):
        ax = fig.add_subplot(grid[pi])
        heat_axes.append(ax)
        selected = [row for row in tables["scaling"] if row["precision"] == precision]
        values = np.full((3, 4), np.nan)
        for row in selected:
            i = [1, 16, 256].index(int(row["batch"]))
            j = [10, 14, 18, 22].index(int(row["log2_n"]))
            if row["status"] != "missing":
                values[i, j] = number(row, "baseline_us") / number(row, "ours_us")
        # pcolormesh keeps all cells vector in PDF/SVG.
        image = ax.pcolormesh(np.ma.masked_invalid(values), cmap=cmap, norm=norm,
                             edgecolors="white", linewidth=.7)
        for i in range(3):
            for j in range(4):
                if np.isnan(values[i, j]):
                    ax.add_patch(Rectangle((j, i), 1, 1, facecolor="#eeeeee", edgecolor=GRAY, hatch="////", linewidth=.4))
                else:
                    ax.text(j + .5, i + .5, f"{values[i,j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set(xlim=(0, 4), ylim=(3, 0), xticks=np.arange(4) + .5,
               xticklabels=["10", "14", "18", "22"], yticks=np.arange(3) + .5,
               yticklabels=["1", "16", "256"], xlabel="log2 N")
        ax.set_title(f"(a{pi+1}) {precision.upper()}", loc="left")
        if pi == 0:
            ax.set_ylabel("Batch")
        ax.tick_params(length=0)
    regret, cost = fig.add_subplot(grid[2]), fig.add_subplot(grid[3])
    for selector, color, marker, linestyle in (("model", BLUE, "o", "-"), ("heuristic", BROWN, "s", "--")):
        rows = [row for row in tables["selection"] if row["selector"] == selector]
        x = [number(row, "budget_candidates") for row in rows]
        ratios = [number(row, "selected_us") / number(row, "pool_best_us") for row in rows]
        if min(ratios) < 1:
            raise ValueError("Selected / measured pool best must be >= 1")
        for ax, values in ((regret, ratios), (cost, [number(row, "prepared_ms") for row in rows])):
            ax.plot(x, values, color=color, marker=marker, linestyle=linestyle,
                    markersize=3, linewidth=1, label=selector)
    regret.axhline(1, color=GRAY, linestyle=":", linewidth=.7)
    regret.set(title="(b) Selection regret", ylim=(.98, 1.26), yticks=[1, 1.1, 1.2],
               ylabel="selected / pool best")
    cost.set(title="(c) Preparation", ylim=(0, 32), yticks=[0, 15, 30], ylabel="Prepared cost (ms)")
    for ax in (regret, cost):
        ax.set(xlabel="Budget (candidates)", xticks=[4, 16, 32])
        clean(ax)
    regret.legend(loc="upper right", frameon=False, fontsize=7, handlelength=1.4)
    cax = fig.add_axes([.095, .12, .30, .035])
    bar = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=[.7, 1, 1.4])
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor("face")
    bar.ax.tick_params(labelsize=7, length=2, pad=1)
    fig.text(.245, .205, "cuFFT / ours; hatch = missing", ha="center", fontsize=7)
    fig.text(.70, .15, "Pool: fixed measured candidates; no global optimum claim", ha="center", fontsize=7)
    fig.text(.5, .025, "Matched C2C, forward, natural I/O, out-of-place, same precision; warm execution. Preparation reported separately.",
             ha="center", fontsize=7)
    return fig


def application(rows):
    fig = frame()
    axes = fig.subplots(1, 3, sharey=True, gridspec_kw=dict(left=.07, right=.99, top=.73, bottom=.39, wspace=.18))
    components = [("butterfly", BLUE, ""), ("adapter", BROWN, "///"), ("other", GRAY, "..")]
    for wi, (ax, workload) in enumerate(zip(axes, ("FNO / FFT", "QuaRot / FWHT", "HEonGPU / NTT"))):
        prep = []
        for si, scale in enumerate(("small", "large")):
            selected = [row for row in rows if row["workload"] == workload and row["scale"] == scale]
            baseline = next(row for row in selected if row["implementation"] == "baseline")
            denominator = sum(number(baseline, key + "_us") for key, _, _ in components)
            pair_prep = []
            for ii, implementation in enumerate(("baseline", "ours")):
                row = next(row for row in selected if row["implementation"] == implementation)
                x = si * 2.5 + ii * .75
                bottom = 0
                for key, color, hatch in components:
                    value = number(row, key + "_us") / denominator
                    ax.bar(x, value, bottom=bottom, width=.57, color=color,
                           hatch=hatch, edgecolor=INK, linewidth=.4)
                    bottom += value
                if implementation == "ours":
                    ax.plot(x, bottom, marker="D", color=INK, markersize=3)
                    ax.text(x, bottom + .055, f"{1 / bottom:.2f}x", ha="center", fontsize=7)
                pair_prep.append(number(row, "prepared_ms"))
            prep.append(f"{pair_prep[0]:g}/{pair_prep[1]:g}")
        ax.axhline(1, color=GRAY, linestyle=":", linewidth=.65)
        ax.set(title=f"({chr(97+wi)}) {workload}", xticks=[0, .75, 2.5, 3.25],
               xticklabels=["B", "O", "B", "O"], ylim=(0, 1.27), yticks=[0, .5, 1])
        ax.text(.19, -.38, "small", transform=ax.transAxes, ha="center", fontsize=7)
        ax.text(.79, -.38, "large", transform=ax.transAxes, ha="center", fontsize=7)
        ax.text(.5, -.57, "Prep B/O (ms): " + "; ".join(prep), transform=ax.transAxes, ha="center", fontsize=7)
        clean(ax)
    axes[0].set_ylabel("Latency / baseline total")
    fig.legend(handles=[Patch(facecolor=color, edgecolor=INK, hatch=hatch, label=key)
                        for key, color, hatch in components], ncol=3, loc="upper center",
               bbox_to_anchor=(.5, .915), frameon=False, fontsize=7, handlelength=1.4, columnspacing=1.5)
    fig.text(.5, .025, "B = matched application baseline; O = ours. Diamond: total speedup. Warm execution excludes compilation; no uncertainty inferred.",
             ha="center", fontsize=7)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, help="Replacement synthetic CSV directory, same schema")
    parser.add_argument("--overwrite", action="store_true", help="Replace only named draft outputs")
    args = parser.parse_args()
    tables = load_data(args.input_dir) if args.input_dir else examples()
    input_dir = args.input_dir or DATA
    targets = [FIGURES / (name + "." + extension) for name in
               ("draft_mechanism", "draft_scaling_selection", "draft_application_results")
               for extension in ("pdf", "svg", "png")]
    manifest_path = DATA / "manifest.json"
    for path in targets + [manifest_path]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing overwrite: {path}; pass --overwrite explicitly")
    if not args.input_dir:
        write_data(tables, args.overwrite)
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8,
             "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
             "axes.labelcolor": INK, "text.color": INK, "axes.linewidth": .5,
             "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
    with plt.rc_context(style):
        figures = {"draft_mechanism": mechanism(tables["mechanism"]),
                   "draft_scaling_selection": scaling(tables),
                   "draft_application_results": application(tables["application"])}
        for name, fig in figures.items():
            for extension in ("pdf", "svg", "png"):
                metadata = {"Title": name, "Subject": SOURCE, "Creator": "render_draft_evaluation.py"} if extension == "pdf" else {
                    "Title": name, "Description": SOURCE, "Creator": "render_draft_evaluation.py"} if extension == "svg" else {
                    "Title": name, "Description": SOURCE}
                fig.savefig(FIGURES / (name + "." + extension), dpi=200, facecolor="white", metadata=metadata)
            plt.close(fig)
    manifest = {
        "schema": "cubutterfly-synthetic-draft-layouts-v1", "source": SOURCE, "seed": SEED,
        "warning": WARNING, "purpose": "Layout only; values are invented and cannot support performance claims.",
        "renderer": "paper/tools/render_draft_evaluation.py", "matplotlib_version": matplotlib.__version__,
        "figure_size_inches": [7, 2.2], "minimum_font_pt": 7, "uncertainty": "none; no measured replicates",
        "replacement_entry": "--input-dir DIR accepts same synthetic CSV schemas; source must remain synthetic_example_not_measured. Measured promotion requires a separately reviewed evidence renderer, actual provenance and caption; never relabel these examples.",
        "input_files": [{"path": str(input_dir / (name + ".csv")),
                         "sha256": hashlib.sha256((input_dir / (name + ".csv")).read_bytes()).hexdigest(),
                         "source": SOURCE} for name in SCHEMAS],
        "columns": SCHEMAS,
        "units": {"*_us": "microseconds", "*_ms": "milliseconds", "*_bytes": "bytes",
                  "resident_display": "KiB = bytes / 1024", "budget_candidates": "count", "log2_n": "log2 transform length"},
        "real_data_required": {
            "common": ["source artifact and hash", "build fingerprint", "GPU UUID/model/memory", "compile policy", "timing protocol", "trial_id", "correctness status", "semantic contract", "availability reason"],
            "mechanism": ["organization", "same workload/precision/N/batch", "hbm_bytes (separate profiler run)", "resident_bytes (specified allocation scope)", "latency_us (unprofiled warm run)"],
            "scaling": ["precision", "log2_n", "batch", "baseline", "full matched semantic contract", "status", "baseline_us", "ours_us"],
            "selection": ["selector", "budget_candidates", "selected_us", "pool_best_us", "pool_id/hash", "candidate availability", "prepared_ms", "heldout/training split"],
            "application": ["workload/version", "concrete scale/shape", "implementation", "butterfly_us", "adapter_us", "other_us", "prepared_ms", "independently timed total_us"]},
        "statistics": {
            "draft": "one synthetic value per cell, not an estimator; no confidence intervals",
            "real_replacement": "retain paired trials; predeclare independent replication unit, median estimator and timing boundaries; normalize medians only for matched cases",
            "mechanism": "HBM / same stagewise bytes; resident bytes separate; normal latency excludes profiling, preparation and compilation; schematic is dependency order, never a timing trace",
            "scaling": "baseline median latency / ours median latency; >1 favors ours; fixed range 0.7..1.4 centered at 1, no aggregation across cells",
            "selection": "selected measured latency / best measured latency in the declared fixed pool; >=1; never global optimality; preparation shown on separate ms axis",
            "application": "each component / matched baseline total; baseline sum=1; diamond=baseline total / ours total; additive components must use same execution scope and reconcile with independent total; preparation excluded and reported separately"},
        "missing_policy": "Scaling status=missing retains empty numeric fields and visible hatch. Do not replace with zero, impute, silently bridge missing budgets, or include missing cells in any statistic. Other missing rows require explicit pending-layout adaptation before measured replacement.",
        "layout_descriptions": {
            "draft_mechanism": "Three panels: separate resource axes, abstract dependency schematic, unprofiled warm latency; four fixed organizations.",
            "draft_scaling_selection": "FP32/FP64 N-by-batch ratios with missing hatches; bounded measured-pool selection regret; separate preparation cost.",
            "draft_application_results": "Three application panels, two scales each, matched baseline/ours component stacks and total-ratio diamonds; preparation values below."},
        "outputs": [{"path": str(path.relative_to(ROOT)), "source": SOURCE,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in targets],
        "publisher_status": "provisional working-draft layout, no venue compliance claim",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print("Rendered 3 explicitly synthetic layouts (PDF/SVG/PNG) and draft-only provenance.")


if __name__ == "__main__":
    main()
