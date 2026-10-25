#!/usr/bin/env python3
"""Render analytic construction diagrams, never performance observations.

Provenance: radix-two unnormalized FWHT, N=16, stage bits (0,1,2,3).
Each band reads and writes N values: transfers=2*N*number_of_bands.
Each stage contains N/2 pairs: pair work=(N/2)*log2(N)=32.
Closed components vary precisely the bits of that band; their local row
ordering changes between bands to keep every component visibly enclosed.
The second figure uses three-bit index reversal and invertible scalars w and 2.
No measurements, random inputs, GPU, smoothing, or inferred hardware behavior.
Reproduce: /home/wt/yes/envs/cubutterfly/bin/python this_script.py
Outputs preserve stated physical dimensions; PDF/SVG contain vector artwork.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "figures"
BLUE, ORANGE, GREEN, PURPLE = "#246a9e", "#c77922", "#30856d", "#87679d"
INK, GRAY = "#243341", "#68747e"
PALE = ["#dcebf5", "#f8e8ce", "#dcefe8", "#eae1f0"]
STYLE = {"font.family": "DejaVu Sans", "font.size": 7,
         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}


def label(ax, x, y, s, **kwargs):
    kwargs.setdefault("ha", "center")
    kwargs.setdefault("va", "center")
    kwargs.setdefault("color", INK)
    return ax.text(x, y, s, **kwargs)


def arrow(ax, start, end, color=GRAY):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                mutation_scale=6, lw=.7, color=color))


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        metadata = {"Title": stem, "Description": "Analytic illustration; no performance measurements."}
        if suffix == "pdf":
            metadata = {"Title": stem, "Subject": metadata["Description"],
                        "Creator": "render_construction_figures.py / Matplotlib " + matplotlib.__version__}
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=300, bbox_inches=None,
                    facecolor="white", metadata=metadata)
    plt.close(fig)


def components(bits):
    mask = sum(1 << q for q in bits)
    groups = [[i for i in range(16) if i & ~mask == fixed]
              for fixed in range(16) if fixed & mask == 0]
    assert len(groups) == 16 // (1 << len(bits))
    assert all(len(g) == 1 << len(bits) for g in groups)
    return groups


def organization_tradeoffs():
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.48))
    fig.subplots_adjust(left=.018, right=.99, top=.98, bottom=.02, wspace=.12)
    organizations = [("(a) Stage-wise", [1, 1, 1, 1], BLUE, PALE[0]),
                     ("(b) Two bands", [2, 2], GREEN, PALE[2]),
                     ("(c) Whole transform", [4], PURPLE, PALE[3])]
    for ax, (title, bands, color, fill) in zip(axes, organizations):
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
        label(ax, .5, .96, title, fontsize=8, fontweight="bold")
        label(ax, .5, .89, "bands " + str(bands), fontsize=7)
        cursor = 0
        band_x = .075
        unit = .202
        top, step = .79, .030
        for depth in bands:
            bits = tuple(range(cursor, cursor + depth))
            groups = components(bits)
            rows = [i for g in groups for i in g]
            y = {i: top - r * step for r, i in enumerate(rows)}
            x0 = band_x + .019
            x1 = band_x + unit * depth - .023
            xs = [x0 + (x1 - x0) * k / depth for k in range(depth + 1)]
            for group in groups:
                lo, hi = min(y[i] for i in group), max(y[i] for i in group)
                ax.add_patch(Rectangle((x0 - .012, lo - .012),
                                       x1 - x0 + .024, hi - lo + .024,
                                       facecolor=fill, edgecolor=color, lw=.55,
                                       zorder=0))
            for k, q in enumerate(bits):
                label(ax, (xs[k] + xs[k + 1]) / 2, .83, str(q), fontsize=7)
                for i in rows:
                    ax.plot(xs[k:k+2], [y[i], y[i]], color=GRAY, lw=.32)
                    ax.plot(xs[k:k+2], [y[i], y[i ^ (1 << q)]], color=color, lw=.48)
            for x in xs:
                ax.scatter([x] * 16, [y[i] for i in rows], s=2.3, color=INK, zorder=3)
            # Thick gray cuts include input/output, and only band boundaries.
            for x in (band_x, band_x + unit * depth - .004):
                ax.plot([x, x], [.325, .805], color=GRAY, lw=1.5, zorder=4)
            band_x += unit * depth
            cursor += depth
        assert cursor == 4
        label(ax, .03, .83, "q", fontsize=7)
        label(ax, .5, .275, f"Reuse depth: {bands[0]} stage" + ("s" if bands[0] != 1 else ""))
        label(ax, .5, .208, f"{2**bands[0]} values / closed component")
        count = 16 // 2**bands[0]
        label(ax, .5, .141, f"{count} independent component" + ("s" if count != 1 else "") + " / band")
        label(ax, .5, .074, f"32 pairs; {32 * len(bands)} value transfers")
    fig.text(.5, .006, "Analytic N = 16 FWHT  |  boxes: closed components  |  gray cuts: materialized boundaries  |  components scale with B",
             ha="center", va="bottom", fontsize=7, color=INK)
    save(fig, "organization_tradeoffs")


def reverse3(i):
    return int(f"{i:03b}"[::-1], 2)


def semantic_transport():
    fig, ax = plt.subplots(figsize=(3.45, 2.78))
    fig.subplots_adjust(left=.02, right=.98, bottom=.015, top=.99)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    label(ax, .5, .965, "Three-bit index reversal", fontsize=8, fontweight="bold")
    label(ax, .245, .899, r"$q=0$ pairs", color=BLUE)
    label(ax, .775, .899, r"$q=2$ pairs", color=GREEN)
    ys = [.835 - .046 * i for i in range(8)]
    for i in range(8):
        j = reverse3(i)
        label(ax, .245, ys[i], f"{i} : {i:03b}", fontsize=7)
        label(ax, .775, ys[i], f"{j} : {j:03b}", fontsize=7)
        arrow(ax, (.37, ys[i]), (.645, ys[i]))
    for first in range(0, 8, 2):
        assert reverse3(first) ^ reverse3(first + 1) == 4
        for x, color, fill in ((.095, BLUE, PALE[0]), (.625, GREEN, PALE[2])):
            ax.add_patch(Rectangle((x, ys[first + 1] - .02), .30, .086,
                                   facecolor=fill, edgecolor=color, lw=.7, zorder=-1))
            ax.plot([x + .025, x + .025], [ys[first+1], ys[first]],
                    lw=1.0, color=color)
    label(ax, .5, .459, r"Stage order: $(0,1,2)\ \mapsto\ (2,1,0)$", fontsize=7)
    label(ax, .5, .407, "Same indices; axis labels for shape (4, 2)", fontsize=7)
    label(ax, .5, .355, r"$i=2r+c:\quad q_0=c_0,\ q_1=r_0,\ q_2=r_1$", fontsize=7)
    for x, title, formula, color, fill in [
        (.025, "CT local core", r"$(a+wb,\ a-wb)$", BLUE, PALE[0]),
        (.515, "Inverse local GS", r"$((u+v)/2,\ (u-v)/(2w))$", ORANGE, PALE[1])]:
        ax.add_patch(Rectangle((x, .12), .46, .155, facecolor=fill,
                               edgecolor=color, lw=.8))
        label(ax, x + .23, .238, title, fontsize=7, fontweight="bold")
        label(ax, x + .23, .170, formula, fontsize=7)
    label(ax, .5, .066, r"Local inverse requires invertible $w$ and $2$", fontsize=7)
    label(ax, .5, .021, "Analytic correspondence", fontsize=7, color=GRAY)
    save(fig, "semantic_transport")


if __name__ == "__main__":
    with plt.rc_context(STYLE):
        organization_tradeoffs()
        semantic_transport()
    print(f"Wrote two analytic PDF/SVG/PNG figures to {OUT}")
