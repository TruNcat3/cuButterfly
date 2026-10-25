#!/usr/bin/env python3
"""Render the matched A100 hardware-profile evidence figure.

Input is the normalized long CSV emitted by the profile queue.  The script
does not accept raw NCU/NSYS output because source identity and metric units
must be normalized before the paper consumes them.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {"case_id", "variant", "role", "metric", "value", "unit"}
RESOURCE_METRICS = (
    "registers_per_thread", "active_warps_pct", "dram_peak_pct",
    "l2_hit_pct", "barrier_stall_pct", "long_scoreboard_stall_pct",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not REQUIRED.issubset(rows[0]):
        raise ValueError(f"normalized profile CSV must contain {sorted(REQUIRED)}")
    for row in rows:
        if row["role"] not in {"baseline", "treatment"}:
            raise ValueError(f"invalid role {row['role']!r}")
        float(row["value"])
    return rows


def _plot(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cases = []
    for row in rows:
        if row["case_id"] not in cases:
            cases.append(row["case_id"])
    colors = {"baseline": "#9aa0a6", "treatment": "#1769aa"}
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.8), sharex=True)
    latency = []
    for case in cases:
        values = {row["role"]: float(row["value"]) for row in rows if row["case_id"] == case and row["metric"] == "kernel_time_us"}
        if set(values) == {"baseline", "treatment"}:
            latency.append((values["baseline"] / values["treatment"], case))
    if latency:
        axes[0].bar(np.arange(len(latency)), [value for value, _ in latency], color="#1769aa", alpha=0.82)
        axes[0].axhline(1.0, color="#505050", linewidth=0.8, linestyle="--")
        axes[0].set_ylabel("Baseline / treatment\nlatency")
        axes[0].set_title("A100 80GB RC2 hardware-profile cases", fontsize=9.5)
        axes[0].set_ylim(bottom=0)
        axes[0].grid(axis="y", alpha=0.2)
    matrix = np.full((len(RESOURCE_METRICS), len(cases)), np.nan)
    for column, case in enumerate(cases):
        for row_index, metric in enumerate(RESOURCE_METRICS):
            values = {row["role"]: float(row["value"]) for row in rows if row["case_id"] == case and row["metric"] == metric}
            if set(values) == {"baseline", "treatment"}:
                matrix[row_index, column] = values["treatment"] - values["baseline"]
    image = axes[1].imshow(matrix, aspect="auto", cmap="coolwarm", interpolation="nearest")
    axes[1].set_yticks(np.arange(len(RESOURCE_METRICS)))
    axes[1].set_yticklabels([metric.replace("_", " ") for metric in RESOURCE_METRICS], fontsize=6.8)
    axes[1].set_xticks(np.arange(len(cases)))
    axes[1].set_xticklabels(cases, rotation=28, ha="right", fontsize=6.7)
    axes[1].set_title("Treatment minus baseline resource counters", fontsize=8.5)
    fig.colorbar(image, ax=axes[1], fraction=0.025, pad=0.02, label="counter delta")
    fig.text(0.5, 0.01, "Positive latency bars favor the treatment. Resource deltas retain metric units from the normalized CSV.", ha="center", fontsize=6.5, color="#505050")
    fig.subplots_adjust(left=0.20, right=0.95, top=0.89, bottom=0.19, hspace=0.38)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", metadata={"Creator": "render_profile_evidence.py", "Title": "A100 RC2 hardware profile evidence"})
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", metadata={"Creator": "render_profile_evidence.py", "Title": "A100 RC2 hardware profile evidence"})
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "paper/figures/rc2_hardware_profile.pdf")
    args = parser.parse_args()
    rows = _read(args.input)
    _plot(rows, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
