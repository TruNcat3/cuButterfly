#!/usr/bin/env python3
"""Render measured E13 application results; reject pending or incomplete rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {
    "application", "payload", "variant", "latency_ms", "butterfly_fraction",
    "adapter_ms", "peak_memory_mib", "quality_error", "status",
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not REQUIRED.issubset(rows[0]):
        raise ValueError(f"application result CSV must contain {sorted(REQUIRED)}")
    for row in rows:
        if row["status"] != "complete":
            raise ValueError(f"cannot render incomplete application row {row['application']}/{row['payload']}")
        if row["variant"] not in {"baseline", "cuButterfly"}:
            raise ValueError(f"invalid variant {row['variant']!r}")
        for field in REQUIRED - {"application", "payload", "variant", "status"}:
            float(row[field])
    return rows


def _plot(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    keys = []
    for row in rows:
        key = (row["application"], row["payload"])
        if key not in keys:
            keys.append(key)
    variants = ("baseline", "cuButterfly")
    x = np.arange(len(keys))
    width = 0.34
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.5), sharex=True)
    colors = {"baseline": "#9aa0a6", "cuButterfly": "#1769aa"}
    for index, variant in enumerate(variants):
        values = []
        for application, payload in keys:
            row = next(row for row in rows if (row["application"], row["payload"], row["variant"]) == (application, payload, variant))
            values.append(float(row["latency_ms"]))
        axes[0].bar(x + (index - 0.5) * width, values, width, label=variant, color=colors[variant], alpha=0.82)
    axes[0].set_ylabel("Application latency (ms)")
    axes[0].set_title("End-to-end application comparison", fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    axes[0].grid(axis="y", alpha=0.2)
    fractions = []
    overheads = []
    for application, payload in keys:
        treatment = next(row for row in rows if (row["application"], row["payload"], row["variant"]) == (application, payload, "cuButterfly"))
        fractions.append(float(treatment["butterfly_fraction"]))
        overheads.append(float(treatment["adapter_ms"]))
    axes[1].bar(x - width / 2, fractions, width, label="Butterfly fraction", color="#2ca25f", alpha=0.82)
    axes[1].bar(x + width / 2, overheads, width, label="cuButterfly adapter (ms)", color="#d95f02", alpha=0.82)
    axes[1].set_ylabel("Fraction / milliseconds")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{application}\n{payload}" for application, payload in keys], fontsize=6.7)
    axes[1].tick_params(axis="x", length=0)
    fig.text(0.5, 0.01, "Each application and payload is reported separately; no cross-application average.", ha="center", fontsize=6.5, color="#505050")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.18, hspace=0.24)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", metadata={"Creator": "render_end_to_end_results.py", "Title": "End-to-end application comparison"})
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", metadata={"Creator": "render_end_to_end_results.py", "Title": "End-to-end application comparison"})
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/figures/end_to_end_results.pdf")
    args = parser.parse_args()
    rows = _read(args.input)
    _plot(rows, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
