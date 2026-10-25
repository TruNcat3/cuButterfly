#!/usr/bin/env python3
"""Render the frozen end-to-end measurement contract as a vector schematic."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "paper/figures"


def _box(axis, x, y, width, height, text, face, edge="#46515c", fontsize=7.0):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=0.7,
        edgecolor=edge,
        facecolor=face,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2.0, y + height / 2.0, text, ha="center", va="center", fontsize=fontsize)


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.16, 2.25))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 3.55)
    axis.axis("off")

    columns = [
        (0.25, 1.45, "Frozen input\nweights / stream"),
        (2.05, 1.55, "Application\ncontext"),
        (4.15, 2.05, "Backend replacement\nbaseline | cuButterfly"),
        (6.85, 1.65, "Same output\nquality check"),
    ]
    axis.text(5.15, 3.38, "End-to-end contract: only the butterfly backend changes", ha="center", va="center", fontsize=9.2)
    axis.text(5.15, 3.12, "The shaded interval is timed; reusable setup and data acquisition are reported separately.",
              ha="center", va="center", fontsize=6.8, color="#505050")

    for x, width, label in columns:
        axis.text(x + width / 2, 2.78, label, ha="center", va="center", fontsize=6.8, color="#263238")

    rows = [
        ("FNO / FFT", "rfftn / irfftn", "#d9edf7"),
        ("QuaRot / FWHT", "int4 prefill / decode", "#e8def8"),
        ("HEonGPU / NTT", "CKKS bootstrap", "#dff0d8"),
    ]
    ys = [1.86, 1.16, 0.46]
    from matplotlib.patches import Rectangle

    axis.add_patch(Rectangle((1.98, 0.27), 6.60, 2.20, facecolor="#d9ecf8", edgecolor="none", alpha=0.20, zorder=0))
    for (name, operation, color), y in zip(rows, ys):
        _box(axis, 0.25, y, 1.45, 0.46, f"{name}\nsame inputs", "#f3f5f6", fontsize=6.2)
        _box(axis, 2.05, y, 1.55, 0.46, operation, color)
        _box(axis, 4.15, y + 0.03, 0.95, 0.40, "baseline", "#f5d6c6", edge="#b35d32", fontsize=6.5)
        _box(axis, 5.25, y + 0.03, 0.95, 0.40, "cuButterfly", "#c9e3f6", edge="#1769aa", fontsize=6.5)
        _box(axis, 6.85, y, 1.65, 0.46, "same result", "#f3f5f6")
        for x0, x1 in ((1.70, 2.05), (3.60, 4.15), (6.20, 6.85)):
            axis.annotate("", xy=(x1, y + 0.23), xytext=(x0, y + 0.23),
                          arrowprops={"arrowstyle": "->", "linewidth": 0.7, "color": "#68737d"})
    axis.annotate("", xy=(5.25, 0.18), xytext=(4.15, 0.18),
                  arrowprops={"arrowstyle": "<->", "linewidth": 1.0, "color": "#1769aa"})
    axis.text(4.70, 0.04, "paired replacement", ha="center", va="bottom", fontsize=6.4, color="#1769aa")
    axis.plot([2.0, 8.55], [2.42, 2.42], color="#1769aa", linewidth=1.2, alpha=0.65)
    axis.text(5.28, 2.48, "timed application interval", ha="center", va="bottom", fontsize=6.6, color="#1769aa")
    axis.text(8.82, 2.98, "outside timed interval", ha="center", va="center", fontsize=6.4, color="#666666")
    axis.text(8.82, 2.76, "compile / setup / I/O", ha="center", va="center", fontsize=6.4, color="#666666")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        fig.savefig(FIGURE_DIR / f"end_to_end_contract.{suffix}", bbox_inches="tight",
                    metadata={"Creator": "render_end_to_end_contract.py", "Title": "End-to-end measurement contract"})
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
