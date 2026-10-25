#!/usr/bin/env python3
"""Render the complete-call application records already reported in the paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures"
GENERATED = ROOT / "paper/generated"

# These values are the historical pinned-build records in evaluation.tex.
# Keeping the source table here makes the plot reproducible and auditable.
RECORDS = [
    ("FNO 64² / B8", 3.007, 3.152),
    ("FNO 128² / B4", 2.990, 3.140),
    ("QuaRot 512 / 32", 2668.600, 2647.400),
    ("QuaRot 2048 / 128", 10573.719, 10544.185),
]


def render(path: Path) -> None:
    labels = [r[0] for r in RECORDS]
    baseline = np.asarray([r[1] for r in RECORDS])
    ours = np.asarray([r[2] for r in RECORDS])
    speedup = baseline / ours
    colors = ["#246a9e", "#246a9e", "#30856d", "#30856d"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.05))
    x = np.arange(len(labels))
    width = .36
    axes[0].bar(x - width / 2, baseline, width, color="#a9b8c3",
                edgecolor="#52616b", linewidth=.45, label="baseline")
    axes[0].bar(x + width / 2, ours, width, color=colors,
                edgecolor="#52616b", linewidth=.45, label="cuButterfly")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("complete-call latency (ms)", fontsize=6.8)
    axes[0].set_title("Absolute application latency", fontsize=7.8, fontweight="bold", pad=4)
    axes[0].legend(fontsize=5.9, frameon=False, loc="upper left")
    axes[1].axhline(1.0, color="#555e65", linestyle="--", linewidth=.7)
    axes[1].bar(x, speedup, color=colors, edgecolor="#52616b", linewidth=.45)
    axes[1].set_ylim(.90, 1.06)
    axes[1].set_ylabel("baseline / cuButterfly", fontsize=6.8)
    axes[1].set_title("End-to-end speedup", fontsize=7.8, fontweight="bold", pad=4)
    for i, value in enumerate(speedup):
        axes[1].text(i, value + .008, f"{value:.3f}×", ha="center", fontsize=6.0)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=6.0)
        ax.tick_params(axis="y", labelsize=6.1)
        ax.grid(axis="y", alpha=.2, linewidth=.45)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Historical pinned-build complete-call records", fontsize=8.8,
                 fontweight="bold", y=.995)
    fig.text(.01, .945, "FNO and QuaRot only; HEonGPU remains unmeasured",
             fontsize=6.2, color="#68747e", va="top")
    fig.subplots_adjust(left=.08, right=.99, bottom=.27, top=.79, wspace=.30)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "render_application_historical.py",
                "Title": "Historical pinned-build application records"}
    fig.savefig(path, metadata=metadata)
    fig.savefig(path.with_suffix(".svg"), metadata=metadata)
    plt.close(fig)


def main() -> int:
    path = OUT / "application_historical_complete_call.pdf"
    render(path)
    provenance = {
        "schema": "cubutterfly-application-historical-v1",
        "status": "historical-pinned-build",
        "records": [{"workload": name, "baseline_ms": base, "cubutterfly_ms": ours,
                     "speedup": base / ours} for name, base, ours in RECORDS],
        "missing": ["HEonGPU CKKS bootstrap"],
        "note": "Historical application records are separate from RC2 operator and pending E13 cohorts.",
    }
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / "application_historical_complete_call.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
