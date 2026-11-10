#!/usr/bin/env python3
"""Render a provenance-preserving NCU profile comparison.

The figure uses existing profiler exports rather than the pending E14 campaign.
The first panel compares the historical stagewise reference, resident control,
and APPT-style hybrid variants at batch one.  The second panel is a separate
fixed-core/organization control from the same V100-era structured 2x2 capture.
The cohorts are kept separate and the output metadata records their hashes.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/figures"
GENERATED = ROOT / "experiments/snapshots"
V100_CSV = ROOT / "results/ncu_appt_pipeline/summary.csv"
STRUCTURED_CSV = ROOT / "results/ncu_structured_2x2_register/summary.csv"
BLUE, ORANGE, GREEN, PURPLE = "#246a9e", "#c77922", "#30856d", "#87679d"
INK, GRAY = "#243341", "#68747e"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def num(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", "None", "nan"} else float("nan")


def v100_profile(rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    selected = [r for r in rows if r["label"].endswith("_b1")]
    names = {
        "barrier_b1": "stagewise reference",
        "resident1010_b1": "resident 10+10",
        "appt_online_b1": "hybrid online",
        "appt_us7_rs2_rep2_ti2_b1": "hybrid 7+7+6",
        "appt_us8_rs2_td8_b1": "hybrid 8+8",
        "appt_us8_td8_b1": "hybrid 8+8 temporal",
    }
    base = next(r for r in selected if r["label"] == "barrier_b1")
    base_traffic = num(base, "dram_read_mib") + num(base, "dram_write_mib")
    output = []
    for row in selected:
        if row["label"] not in names:
            continue
        traffic = num(row, "dram_read_mib") + num(row, "dram_write_mib")
        output.append({
            "label": names[row["label"]],
            "time_ratio": num(row, "time_us") / num(base, "time_us"),
            "traffic_ratio": traffic / base_traffic,
            "registers": num(row, "registers_per_thread"),
            "active": num(row, "active_warps_pct"),
            "barrier_stall": num(row, "barrier_stall_pct"),
            "scoreboard_stall": num(row, "long_scoreboard_stall_pct"),
        })
    return output


def aggregate(rows: list[dict[str, str]], prefix: str) -> dict[str, float]:
    chosen = [r for r in rows if r["label"].startswith(prefix)]
    total_time = sum(num(r, "time_us") for r in chosen)
    def weighted(key: str) -> float:
        values = [(num(r, key), num(r, "time_us")) for r in chosen]
        return sum(v * t for v, t in values if np.isfinite(v)) / total_time
    return {"time": total_time, "active": weighted("active_warps_pct"),
            "registers": max(num(r, "registers_per_thread") for r in chosen)}


def structured_control(rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    prefixes = [
        ("fwht12_hierarchical_r8", "FWHT hierarchical"),
        ("fwht12_warp_register", "FWHT warp-register"),
        ("structured12_hierarchical_r8", "2x2 hierarchical"),
        ("structured12_warp_register", "2x2 warp-register"),
    ]
    records = [(aggregate(rows, prefix), label) for prefix, label in prefixes]
    base = records[0][0]["time"]
    return [{"label": label, "time_ratio": record["time"] / base,
             "active": record["active"], "registers": record["registers"]}
            for record, label in records]


def _bar(ax, labels, values, colors, ylabel, title, log=False):
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, alpha=.78, edgecolor="#4b5963", linewidth=.45)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.1)
    ax.set_ylabel(ylabel, fontsize=6.7)
    ax.set_title(title, fontsize=7.5, fontweight="bold", pad=4)
    ax.tick_params(axis="y", labelsize=6.1)
    ax.grid(axis="y", alpha=.2, linewidth=.45)
    if log:
        ax.set_yscale("log")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def render(v100: list[dict[str, float | str]], structured: list[dict[str, float | str]], figure: Path):
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.30))
    labels = [str(row["label"]) for row in v100]
    colors = [BLUE, GREEN, PURPLE, ORANGE, "#a64d79", "#69737c"][:len(labels)]
    _bar(axes[0, 0], labels, [float(r["time_ratio"]) for r in v100], colors,
         "NCU kernel time / reference", "V100 NCU time", log=True)
    _bar(axes[0, 1], labels, [float(r["traffic_ratio"]) for r in v100], colors,
         "DRAM traffic / reference", "V100 traffic", log=True)
    _bar(axes[0, 2], labels, [float(r["registers"]) for r in v100], colors,
         "registers / thread", "V100 register pressure")
    _bar(axes[1, 0], labels, [float(r["active"]) for r in v100], colors,
         "active warps (%)", "V100 active warps")
    x = np.arange(len(labels))
    width = .34
    axes[1, 1].bar(x - width / 2, [float(r["barrier_stall"]) for r in v100], width,
                    color=ORANGE, label="barrier")
    axes[1, 1].bar(x + width / 2, [float(r["scoreboard_stall"]) for r in v100], width,
                    color=PURPLE, label="long scoreboard")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=35, ha="right", fontsize=6.1)
    axes[1, 1].set_ylabel("stall cycles (%)", fontsize=6.7)
    axes[1, 1].set_title("V100 stall composition", fontsize=7.5, fontweight="bold", pad=4)
    axes[1, 1].legend(fontsize=5.8, frameon=False, loc="upper right")
    axes[1, 1].tick_params(axis="y", labelsize=6.1)
    axes[1, 1].grid(axis="y", alpha=.2, linewidth=.45)
    for spine in ("top", "right"):
        axes[1, 1].spines[spine].set_visible(False)
    a_labels = [str(row["label"]) for row in structured]
    a_colors = [BLUE, PURPLE, GREEN, ORANGE]
    x = np.arange(len(structured))
    axes[1, 2].bar(x - width / 2, [float(r["time_ratio"]) for r in structured], width,
                    color=a_colors, alpha=.78, label="NCU time / FWHT hierarchical")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(a_labels, rotation=35, ha="right", fontsize=6.1)
    axes[1, 2].set_ylabel("latency ratio", fontsize=6.7)
    axes[1, 2].set_title("V100 fixed-control NCU time", fontsize=7.5, fontweight="bold", pad=4)
    axes[1, 2].tick_params(axis="y", labelsize=6.1)
    axes[1, 2].grid(axis="y", alpha=.2, linewidth=.45)
    for spine in ("top", "right"):
        axes[1, 2].spines[spine].set_visible(False)
    fig.suptitle("Historical NCU profile evidence: resource and stall costs", fontsize=9.2,
                 fontweight="bold", y=.995)
    fig.text(.01, .965, "Separate V100 captures; NCU replay time is mechanism evidence, not CUDA-event latency",
             fontsize=6.4, color=GRAY, va="top")
    fig.subplots_adjust(left=.08, right=.99, bottom=.19, top=.89, hspace=.64, wspace=.34)
    figure.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "render_profile_comparison.py",
                "Title": "Historical NCU profile evidence"}
    fig.savefig(figure, metadata=metadata)
    fig.savefig(figure.with_suffix(".svg"), metadata=metadata)
    plt.close(fig)


def main() -> int:
    v100_rows, structured_rows = read_csv(V100_CSV), read_csv(STRUCTURED_CSV)
    v100, structured = v100_profile(v100_rows), structured_control(structured_rows)
    figure = OUT / "profile_comparison_historical.pdf"
    render(v100, structured, figure)
    provenance = {
        "schema": "cubutterfly-profile-comparison-v1",
        "status": "historical-profiler-evidence",
        "sources": [],
        "cohorts": {"v100_appt": "results/ncu_appt_pipeline/summary.csv batch-1",
                    "v100_structured": "results/ncu_structured_2x2_register/summary.csv"},
        "warning": "Not the pending E14 RC2 counter campaign; profiler replay time is not warm CUDA-event latency.",
    }
    for path in (V100_CSV, STRUCTURED_CSV):
        provenance["sources"].append({"path": str(path),
                                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / "profile_comparison_historical.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
