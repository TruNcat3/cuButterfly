#!/usr/bin/env python3
"""Render the historical matched-core CTA-organization crossover on the CPU."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, __version__ as pillow_version


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/homogeneous_warp_dual_groups/analysis.md"
STEM = "motivation_crossover_20261027"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_first_table(text: str) -> tuple[list[str], list[dict]]:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("|"))
    table = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        table.append(line)
    cells = lambda line: [cell.strip() for cell in line.strip("|").split("|")]
    header = cells(table[0])
    expected = ["batch", "v0.6 ms", "single ms", "dual ms",
                "dual/single throughput", "dual/v0.6 throughput"]
    if header != expected or len(table) != 5:
        raise ValueError("Expected the original first table with exactly three data rows")
    rows = []
    for line in table[2:]:
        values = cells(line)
        if len(values) != len(header):
            raise ValueError("Unexpected table row width")
        raw = dict(zip(header, values))
        row = {
            "batch": int(raw["batch"]),
            "single_ms_rounded": float(raw["single ms"]),
            "dual_ms_rounded": float(raw["dual ms"]),
            "dual_over_single_throughput": float(raw["dual/single throughput"].removesuffix("x")),
            "reported_ratio_text": raw["dual/single throughput"],
        }
        if not all(math.isfinite(value) and value > 0 for value in row.values()
                   if isinstance(value, (int, float))):
            raise ValueError("All observations must be finite and positive")
        rows.append(row)
    if [row["batch"] for row in rows] != [1, 4, 16]:
        raise ValueError("Expected batches 1, 4, and 16 in source order")
    return table, rows


def main() -> None:
    table, rows = read_first_table(SOURCE.read_text(encoding="utf-8"))
    figure_dir = ROOT / "paper/figures"
    generated_dir = ROOT / "paper/generated"
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    csv_path = generated_dir / f"{STEM}.csv"
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    csv_path.write_text(stream.getvalue(), encoding="utf-8")

    style = {
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "path",
        "svg.hashsalt": STEM, "savefig.facecolor": "white",
    }
    outputs = [csv_path]
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(3.4, 1.7))
        fig.subplots_adjust(left=0.185, right=0.965, bottom=0.29, top=0.94)
        x = [row["batch"] for row in rows]
        y = [row["dual_over_single_throughput"] for row in rows]
        ax.axhline(1, color="#666666", linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
        ax.plot(x, y, color="#0072B2", marker="o", markersize=4.5,
                linewidth=1.0, zorder=3)
        for batch, ratio in zip(x, y):
            ax.annotate(f"{ratio:.3f}x", (batch, ratio), xytext=(0, 7),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [str(batch) for batch in x])
        ax.minorticks_off()
        ax.set_xlim(0.65, 24.5)
        ax.set_ylim(0.85, 1.24)
        ax.set_yticks([0.9, 1.0, 1.1, 1.2], ["0.9", "1.0", "1.1", "1.2"])
        ax.set_xlabel("Batch size (log2 scale)", labelpad=3)
        ax.set_ylabel("Dual / single\nthroughput (x)", labelpad=4)
        ax.tick_params(length=3, pad=3)
        for extension in ("pdf", "svg", "png"):
            path = figure_dir / f"{STEM}.{extension}"
            metadata = {"Creator": "render_motivation_crossover.py"}
            if extension == "pdf":
                metadata.update({"CreationDate": None, "ModDate": None})
            elif extension == "svg":
                metadata = {"Date": None, "Creator": "render_motivation_crossover.py"}
            fig.savefig(path, dpi=300, metadata=metadata, transparent=False)
            outputs.append(path)
        plt.close(fig)

    with Image.open(figure_dir / f"{STEM}.png") as preview:
        raster = {"size_pixels": list(preview.size), "mode": preview.mode,
                  "dpi": list(preview.info.get("dpi", [])),
                  "opaque": preview.convert("RGBA").getextrema()[3] == (255, 255)}
    manifest = {
        "figure": STEM,
        "source": {"path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE),
                   "selection": "First Markdown table; all three rows retained",
                   "original_table": table},
        "script": {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                   "sha256": sha256(Path(__file__).resolve())},
        "context": {"device": "V100", "operation": "uint32 Shoup NTT",
                    "N": 2**20, "factorization": "10+10", "io": "four-row static IO",
                    "single": "one synchronous four-warp row group; CTA128; target four CTAs/SM",
                    "dual": "two independent synchronous four-warp row groups; CTA256; target two CTAs/SM",
                    "matched": "same register-resident stage-7/8/9 continuation; theoretical capacity of 16 resident warps per SM"},
        "estimator": "Source reports trial medians from three confirmation trials, each with five warmups and thirty repetitions.",
        "units": {"batch": "NTTs", "single_ms_rounded": "ms", "dual_ms_rounded": "ms",
                  "dual_over_single_throughput": "dimensionless ratio (x)"},
        "normalization": "dual throughput / single throughput = single time / dual time for equal work; plotted values read directly from the reported throughput column, never recomputed from rounded ms",
        "selection_reason": "Compare CTA organization at fixed computational shape and register continuation, including its changed launch and resource configuration; this is not a comparison against the v0.6 baseline.",
        "transformations": ["Parse the first Markdown table", "Select batch, rounded single/dual latency, and reported dual/single throughput", "Use base-2 logarithmic batch axis; no smoothing, extrapolation, exclusions, or re-estimation"],
        "limitations": ["Underlying per-trial measurements were not read; no error bars or uncertainty estimates are available", "Only three measured batches; connecting segments guide the eye and do not imply interpolation or a general crossover threshold"],
        "encoding": {"observations": "blue (#0072B2) connected circle markers with direct numeric labels",
                     "reference": "gray dashed horizontal 1x baseline",
                     "size_inches": [3.4, 1.7], "y_limits": [0.85, 1.24],
                     "png_preview": "Same figure artists and geometry as the SVG; rasterized by Matplotlib at 300 dpi",
                     "pdf_fonts": "Type 42 embedded TrueType", "svg_fonts": "glyph paths for portable appearance"},
        "alt_description": "Point-and-line plot of dual-group relative to single-group throughput for a matched V100 uint32 Shoup NTT at N=2^20. On a base-2 log batch axis, batches 1, 4, and 16 have reported ratios 0.999x, 0.906x, and 1.154x. The first two lie below the dashed 1x reference; batch 16 lies above it. Lines only guide the eye; uncertainty is unavailable.",
        "versions": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                     "pillow": pillow_version, "backend": matplotlib.get_backend()},
        "raster": raster,
        "outputs": {str(path.relative_to(ROOT)): {"sha256": sha256(path), "bytes": path.stat().st_size}
                    for path in outputs},
        "hash_note": "Output hashes cover CSV and figure files; the JSON manifest does not hash itself.",
        "publisher_requirements": "Provisional single-column manuscript figure; venue-specific rules and final manuscript layout remain to be checked.",
    }
    manifest_path = generated_dir / f"{STEM}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows, "outputs": [str(p.relative_to(ROOT)) for p in outputs] +
                      [str(manifest_path.relative_to(ROOT))], "raster": raster}, indent=2))


if __name__ == "__main__":
    main()
