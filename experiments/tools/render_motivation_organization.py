#!/usr/bin/env python3
"""Render the complete, released A10040 M1 organization-control cohort."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from render_narrative_scaling import DEFAULT, OPERATORS, ROOT, load_evidence, measured_rows, sha

STEM = "motivation_organization_a100"
GROUPS = (("local2", "2-stage leaves", "#0072B2", "o"),
          ("local4", "Up to 4-stage leaves", "#A64B00", "^"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=DEFAULT)
    args = parser.parse_args()
    source = args.journal.resolve()
    journal, releases = load_evidence(source)
    rows = measured_rows(journal)
    if len(releases) != 18:
        raise ValueError("expected the hash-validated 18-release M1 collection")
    provenance = {item["case"]: item for item in releases}
    cells = {}
    for row in rows:
        cells.setdefault((row["case"], row["variant"]), []).append(row["kernel_ms"])
    medians = {key: statistics.median(values) for key, values in cells.items()}
    ratios = []
    for op in OPERATORS:
        for logn in (16, 20):
            for batch in (1, 4, 16):
                case_rows = [r for r in rows if (r["operator"], r["logN"], r["batch"]) == (op, logn, batch)]
                case = case_rows[0]["case"]
                baseline = medians[case, "local0"]
                for variant, _, _, _ in GROUPS:
                    ratios.append(dict(case=case, operator=op, logN=logn, batch=batch,
                                       variant=variant, baseline_median_ms=baseline,
                                       treatment_median_ms=medians[case, variant],
                                       ratio=baseline / medians[case, variant]))
    for row in rows:
        case, variant = row["case"], row["variant"]
        row.update(cell_median_ms=medians[case, variant],
                   local0_median_ms=medians[case, "local0"],
                   ratio_local0_over_variant=medians[case, "local0"] / medians[case, variant],
                   release_path=provenance[case]["path"],
                   release_sha256=provenance[case]["sha256"], collection_sha256=sha(source))
    out = ROOT / "experiments/figures"
    data = ROOT / "experiments/snapshots"
    out.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    csv_path = data / f"{STEM}.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    style = {"font.family": "DejaVu Sans", "font.size": 8,
             "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7.5,
             "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": .6,
             "axes.spines.top": False, "axes.spines.right": False,
             "savefig.facecolor": "white"}
    outputs = [csv_path]
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(3.5, 1.8))
        fig.subplots_adjust(left=.17, right=.975, bottom=.25, top=.70)
        fig.suptitle("(b) A100 40GB organization controls", x=.51, y=.99, fontsize=8)
        ax.axhline(1, color="#555555", linestyle="--", linewidth=.8, zorder=1)
        for index, op in enumerate(OPERATORS):
            for offset, (variant, label, color, marker) in zip((-.18, .18), GROUPS):
                values = [r["ratio"] for r in ratios if r["operator"] == op and r["variant"] == variant]
                position = index + offset
                ax.boxplot([values], positions=[position], widths=.25, whis=(0, 100),
                           patch_artist=True, showfliers=False, manage_ticks=False,
                           boxprops=dict(facecolor="white", edgecolor=color, linewidth=.9,
                                         hatch="///" if variant == "local4" else None),
                           whiskerprops=dict(color=color, linewidth=.8),
                           capprops=dict(color=color, linewidth=.8),
                           medianprops=dict(color=color, linewidth=1.2))
                # Fixed horizontal offsets keep every workload visible without altering ratios.
                jitter = (-.075, -.045, -.015, .015, .045, .075)
                ax.scatter([position + d for d in jitter], values, s=15, marker=marker,
                           facecolor=color, edgecolor="white", linewidth=.35, zorder=3)
                summaries.append(dict(operator=op, variant=variant, workloads=6,
                                      geometric_mean=math.exp(statistics.mean(math.log(v) for v in values)),
                                      minimum=min(values), maximum=max(values)))
        handles = [Line2D([], [], color=color, marker=marker, linestyle="none", markersize=4,
                          label=label) for _, label, color, marker in GROUPS]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .88), ncol=2,
                   frameon=False, handletextpad=.35, columnspacing=.8)
        ax.set_xticks(range(3), ["FFT\nFP64", "FWHT\nFP64", "NTT\nword64"])
        ax.set_ylabel("Stagewise /\ngrouped time", labelpad=3)
        ax.set_xlim(-.55, 2.55)
        ax.set_ylim(.55, 1.55)
        ax.set_yticks((.6, 1, 1.4))
        ax.tick_params(length=2.5, pad=3)
        ax.grid(axis="y", color="#dddddd", linewidth=.4, zorder=0)
        for extension in ("pdf", "png", "svg"):
            target = out / f"{STEM}.{extension}"
            fig.savefig(target, dpi=300, transparent=False)
            outputs.append(target)
        plt.close(fig)
    manifest = dict(
        schema="cubutterfly-m1-organization-figure-v1", source=str(source.relative_to(ROOT)),
        source_sha256=sha(source), script_sha256=sha(__file__),
        hardware=journal["identity"]["device"], protocol=journal["protocol"],
        case_releases=releases, trial_rows=len(rows), organization_cells=54,
        workloads=18, plotted_ratios=36, exclusions=[],
        transformation="For each workload, median of three process trial times per organization; ratio = median(local0) / median(local2 or local4). Greater than one favors grouped leaves.",
        distribution="Six workloads per operator/organization; box=25th/75th percentiles (Matplotlib linear quantiles), line=median, whiskers=min/max. This is workload spread, not trial uncertainty.",
        jitter="Fixed x offsets [-.075,-.045,-.015,.015,.045,.075] in logN16 B1/4/16, then logN20 B1/4/16 order; y values unchanged.",
        physical_groups="8+8 for logN16; 10+10 for logN20; scalar radix2 fixed, local shared/register publication varied.",
        caveats=["Unchanged logical HBM boundaries do not imply equal measured traffic.",
                 "Ratios compare organization controls within each arithmetic family, not equal FLOP counts across families.",
                 "local0 is a fixed shared-publication control, not a searched strong conventional baseline or full-HHD comparison.",
                 "All ordinary trial ranges remain in the CSV; substantial trial variation limits inference.",
                 "Floating-point preflights do not supply complete numerical-error distributions.",
                 "Same-architecture A10040 diagnostic data do not establish cross-architecture transfer."],
        alt_text="Grouped box-and-dot plot of all 18 A10040 M1 workloads. Each FFT, FWHT and NTT family has two six-point distributions of local0 time divided by grouped-leaf time. Two-stage leaves mostly improve medians; up-to-four-stage leaves include substantial FFT and NTT losses. The dashed line is equal time.",
        publisher_status="Provisional manuscript panel; exact publisher and submission-phase requirements not certified.",
        size_inches=[3.5, 1.8], minimum_font_pt=7.5, matplotlib_version=matplotlib.__version__,
        summaries=summaries, ratios=ratios,
        outputs={str(path.relative_to(ROOT)): sha(path) for path in outputs})
    (data / f"{STEM}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(trial_rows=len(rows), plotted_ratios=len(ratios), summaries=summaries), indent=2))


if __name__ == "__main__":
    main()
