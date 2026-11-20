#!/usr/bin/env python3
"""Render the Figure 9 CPU-normalized comparison on two GPU platforms.

The A100 panel is the frozen compact comparison.  The PRO 6000 panel reads
the independently packaged, CPU-confirmed records and keeps its cohorts
separate.  No values are pooled across platforms or cohorts.
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullLocator
import numpy as np

from paper_plot_style import RC, BLUE, ORANGE, GREEN, PURPLE

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PRO_BACKENDS = DATA / "pro6000_backends.csv"

STYLES = {
    "cuButterfly": (BLUE, "o", ""),
    "cufft": (ORANGE, "s", "//"),
    "vkfft": (GREEN, "^", ".."),
    "daofht": (PURPLE, "D", "\\\\"),
    "gpu-ntt": ("#555555", "P", "xx"),
    "gpu-fft": ("#A65E00", "v", "xx"),
    "hadacore": ("#8C564B", "X", "++"),
}
NAMES = {
    "cuButterfly": "cuButterfly", "cufft": "cuFFT", "vkfft": "VkFFT",
    "daofht": "DaoFHT", "gpu-ntt": "GPU-NTT", "gpu-fft": "GPU-FFT",
    "hadacore": "HadaCore",
}

FAMILY = {
    "fft": "FFT", "ntt": "NTT", "fwht": "FWHT",
    "subset-zeta": "Subset zeta", "superset-zeta": "Superset zeta",
    "structured-2x2": "2×2",
}
PREC_LABEL = {"fp16": "F16", "fp32": "F32", "fp64": "F64",
              "word32": "w32", "word64": "w64", "uint32": "u32"}


def finite_pos(value: object) -> bool:
    try:
        value = float(value)
        return math.isfinite(value) and value > 0
    except (TypeError, ValueError):
        return False


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def top_groups() -> tuple[list[tuple[tuple[str, str, str], list[dict]]], list[dict]]:
    """Recompute the published A100 intersection from paired CPU/GPU times."""
    records = []
    for cohort in ("main66", "cyclic90", "fft8", "fwht9-cpu-adapted"):
        records.extend(json.loads((DATA / (cohort + ".json")).read_text())["records"])
    manifest = json.loads((DATA / "a100_order.json").read_text())
    original = json.loads((DATA / "a100_population.json").read_text())
    order = [tuple(g["key"]) for g in manifest["groups"]]
    groups = []
    for key in order:
        rows = [r for r in records if (r["cohort"], r["operator"], r["precision"]) == key
                and (r["operator"] != "fft" or r["cpu_backend"] == "fftw-cpu-t8")]
        by_backend = defaultdict(dict)
        for r in rows:
            if r["case_id"] in by_backend[r["backend"]]:
                raise ValueError(f"duplicate A100 measurement: {key}/{r['case_id']}")
            by_backend[r["backend"]][r["case_id"]] = r
        if not by_backend or "cuButterfly" not in by_backend:
            raise ValueError(f"missing A100 population: {key}")
        common = set.intersection(*(set(v) for v in by_backend.values()))
        if not common:
            raise ValueError(f"empty A100 intersection: {key}")
        if key[0] in ("main66", "cyclic90"):
            prior = next(g for g in original["groups"] if tuple(g["group"]) == key)
            if common != set(prior["plotted_case_ids"]):
                raise ValueError(f"published A100 population changed: {key}")
        selected = []
        for case in sorted(common):
            denoms = {(v[case]["cpu_backend"], float(v[case]["cpu_us"])) for v in by_backend.values()}
            if len(denoms) != 1:
                raise ValueError(f"CPU denominator differs across backends: {key}/{case}")
            for backend, values in by_backend.items():
                r = values[case]
                if not finite_pos(r["cpu_us"]) or not finite_pos(r["backend_us"]):
                    raise ValueError(f"invalid measured timing: {key}/{case}")
                ratio = float(r["cpu_us"]) / float(r["backend_us"])
                if "speedup" in r and not math.isclose(float(r["speedup"]), ratio, rel_tol=1e-10):
                    raise ValueError(f"incorrect normalization: {key}/{case}")
                selected.append(dict(r, backend=backend, value=ratio, normalization="CPU/GPU"))
        groups.append((key, selected))
    return groups, original


def pro_groups() -> list[tuple[tuple[str, str, str], list[dict]]]:
    rows = read_csv(PRO_BACKENDS)
    # Use the final FFT17 records and the Full66 population for other
    # operators.  MMA9 remains a separate FP16 FWHT cohort.
    selected_cohort = {
        ("fft", "fp32"): "fft17-final", ("fft", "fp64"): "fft17-final",
        ("fwht", "fp16"): "mma9-original",
    }
    keys = [
        ("fft", "fp32"), ("fft", "fp64"),
        ("ntt", "word32"), ("ntt", "word64"),
        ("subset-zeta", "word32"), ("subset-zeta", "word64"),
        ("superset-zeta", "word32"), ("superset-zeta", "word64"),
        ("structured-2x2", "fp32"), ("structured-2x2", "fp64"),
        ("fwht", "fp32"), ("fwht", "fp64"), ("fwht", "fp16"),
    ]
    out = []
    for op, precision in keys:
        cohort = selected_cohort.get((op, precision), "full66")
        selected = []
        for row in rows:
            if row["cohort"] != cohort:
                continue
            if row["operator"].lower() != op or row["precision"].lower() != precision:
                continue
            if str(row.get("backend_usable", "")).lower() != "true":
                continue
            if not finite_pos(row.get("cpu_normalized")):
                continue
            if not finite_pos(row["cpu_reference_us"]) or not finite_pos(row["latency_us"]):
                raise ValueError("PRO 6000 record lacks measured positive CPU/GPU time")
            ratio = float(row["cpu_reference_us"]) / float(row["latency_us"])
            if not math.isclose(float(row["cpu_normalized"]), ratio, rel_tol=1e-10):
                raise ValueError("PRO 6000 CPU normalization disagrees with measured times")
            selected.append({
                "cohort": cohort, "operator": op, "precision": precision,
                "backend": {"cubutterfly": "cuButterfly"}.get(row["backend"], row["backend"]),
                "case_id": row["case_id"], "value": ratio,
                "normalization": "CPU/GPU",
            })
        if selected:
            out.append(((cohort, op, precision), selected))
    return out


def draw_panel(ax, groups, title, *, show_family_labels=True, inside_title=False):
    exported = []
    cursor = 0.0
    family_spans = []
    layout = []
    for index, (key, selected) in enumerate(groups):
        backends = [b for b in STYLES if any(r["backend"] == b for r in selected)]
        width = 0.76
        center = cursor + width / 2
        family = FAMILY[key[1]]
        if not family_spans or family_spans[-1][2] != family:
            if index:
                ax.axvline(cursor - 0.08, color="#AAAAAA", lw=.45)
            family_spans.append([cursor, cursor + width, family])
        else:
            family_spans[-1][1] = cursor + width
        for backend_index, backend in enumerate(backends):
            values = sorted(float(r["value"]) for r in selected if r["backend"] == backend)
            color, marker, hatch = STYLES[backend]
            position = center + (backend_index - (len(backends) - 1) / 2) * .18
            box = ax.boxplot(
                values, positions=[position], widths=.14, patch_artist=True,
                manage_ticks=False, showfliers=False, whis=(0, 100),
                boxprops=dict(facecolor="white", edgecolor=color, lw=.6),
                medianprops=dict(color="black", lw=.6),
                whiskerprops=dict(color=color, lw=.5), capprops=dict(color=color, lw=.5),
            )
            box["boxes"][0].set_hatch(hatch)
            ax.scatter(position + np.linspace(-.025, .025, len(values)), values,
                       s=4.5, marker=marker, color=color, edgecolors="white",
                       linewidths=.12, zorder=3)
            exported.extend({
                "platform": title, "cohort": key[0], "operator": key[1],
                "precision": key[2], "backend": backend,
                "case_id": r["case_id"], "value": float(r["value"]),
                "normalization": "CPU/GPU",
            } for r in selected if r["backend"] == backend)
        layout.append((center, PREC_LABEL[key[2]], key))
        cursor += width + .08
    ax.set_xlim(-.06, cursor - .02)
    ax.set_yscale("log")
    # Shared limits are set from BOTH platform populations in main().
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks([c for c, _, _ in layout], [label for _, label, _ in layout], fontsize=5.7)
    ax.axhline(1, color="#444444", ls="--", lw=.6)
    ax.grid(axis="y", alpha=.55)
    ax.tick_params(labelsize=5.8, pad=1.3)
    if inside_title:
        ax.text(.01, .96, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=6.6, bbox=dict(facecolor="white", edgecolor="none", pad=0.4),
                zorder=5)
    else:
        ax.set_title(title, loc="left", y=1.14, fontsize=7.1, pad=0.0)
    if show_family_labels:
        for left, right, family in family_spans:
            ax.text((left + right) / 2, 1.015, family, ha="center", va="bottom",
                    transform=ax.get_xaxis_transform(), fontsize=5.9)
    return exported, layout


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "complete_comparison_platforms"
    top, top_manifest = top_groups()
    bottom = pro_groups()
    if not top or not bottom:
        raise RuntimeError("empty Figure 9 input")
    fig, axes = plt.subplots(2, 1, figsize=(7.05, 2.95), sharey=True,
                             gridspec_kw={"hspace": .26})
    top_exported, top_layout = draw_panel(axes[0], top, "(a) A100")
    bottom_exported, bottom_layout = draw_panel(axes[1], bottom, "(b) RTX PRO 6000",
                                                inside_title=True)
    all_rows = top_exported + bottom_exported
    expected = read_csv(DATA / "expected_points.csv")
    fields = ("platform", "cohort", "operator", "precision", "backend", "case_id")
    def keyed(rows):
        values = {tuple(r[k] for k in fields): float(r["value"]) for r in rows}
        if len(values) != len(rows):
            raise ValueError("duplicate plotted measurements")
        return values
    measured, reference = keyed(all_rows), keyed(expected)
    if measured.keys() != reference.keys() or any(
            not math.isclose(v, reference[k], rel_tol=1e-10) for k, v in measured.items()):
        raise ValueError("export differs from the frozen paper numerical population")
    limits = (min(measured.values()) / 1.2, max(measured.values()) * 1.2)
    ticks = [10.0 ** i for i in range(math.floor(math.log10(limits[0])),
                                    math.ceil(math.log10(limits[1])) + 1)
             if limits[0] <= 10.0 ** i <= limits[1]]
    for ax in axes:
        ax.set_ylim(*limits)
        ax.set_yticks(ticks)
    axes[0].set_ylabel("CPU / GPU time (×, log)", fontsize=6.5, labelpad=1.5)
    axes[1].set_ylabel("CPU / GPU time (×, log)", fontsize=6.5, labelpad=1.5)
    axes[1].set_xlabel("Operator and precision group", fontsize=6.5, labelpad=1.5)
    handles = [Line2D([], [], color=color, marker=marker, ls="none", markersize=3,
                      label=NAMES[name]) for name, (color, marker, _) in STYLES.items()
               if any(r["backend"] == name for r in top_exported + bottom_exported)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .995),
               ncol=7, fontsize=6.0, columnspacing=.8, handlelength=.6)
    fig.subplots_adjust(left=.068, right=.995, top=.875, bottom=.15)
    outputs = {}
    for ext in ("pdf", "svg", "png"):
        path = out.with_suffix("." + ext)
        fig.savefig(path, dpi=300)
        outputs[ext] = digest(path)
    with out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(all_rows)
    outputs["csv"] = digest(out.with_suffix(".csv"))
    manifest = {
        "dimensions_inches": [7.05, 2.95], "minimum_font_pt": 5.8,
        "platform_panels": {"A100": len(top_exported), "PRO6000": len(bottom_exported)},
        "top_input": "data/{main66,cyclic90,fft8,fwht9-cpu-adapted}.json",
        "top_manifest_sha256": digest(DATA / "a100_population.json"),
        "pro_input": str(PRO_BACKENDS.relative_to(ROOT)), "pro_input_sha256": digest(PRO_BACKENDS),
        "normalization": "CPU/GPU complete-call ratio; no pooling across platforms or cohorts",
        "pro_cohorts": "FFT uses fft17-final; other operators use full66; FP16 FWHT uses mma9-original",
        "ntt8_note": "The separate PRO6000 cyclic NTT8 GPU/GPU follow-up is reported in text and is not pooled into these CPU-normalized boxes.",
        "outputs": outputs, "groups_top": top_layout, "groups_bottom": bottom_layout,
        "inputs": {p.name: digest(p) for p in sorted(DATA.iterdir()) if p.is_file()},
        "axis_limits": limits, "all_observations_visible": True,
        "numerical_equivalence": "411 unique points match the frozen paper export within 1e-10 relative tolerance",
        "comparison_scope": "A100 boxes use the common backend intersection within each subgroup. PRO6000 boxes retain every usable measured backend row; backend populations can differ. GPU/GPU claims require pairwise matching.",
        "cpu_qualification": "Separately measured CPU/GPU; shared-host CPU timings; FP16 FWHT CPU baseline uses FP16 storage and FP32 arithmetic including conversions/scaling.",
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Figure 9: {len(all_rows)} verified points; all within shared axis limits {limits}")
    plt.close(fig)


if __name__ == "__main__":
    with plt.rc_context(RC):
        main()
