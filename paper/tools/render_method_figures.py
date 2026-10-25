#!/usr/bin/env python3
"""Render analytical, editable paper diagrams; no GPU or performance samples.

The stage/data and bank/offset presentations extend the APPT26 illustrations.
PDF/SVG outputs are retained so a paper build never needs plotting packages.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "figures"
BLUE, ORANGE, GREEN, PURPLE = "#246a9e", "#c77922", "#30856d", "#87679d"
INK, GRAY = "#243341", "#68747e"
PALE = ["#dcebf5", "#f8e8ce", "#dcefe8", "#eae1f0"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "savefig.pad_inches": 0.04,
})


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)
    return fig, ax


def text(ax, x, y, label, **kw):
    kw.setdefault("color", INK)
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    return ax.text(x, y, label, **kw)


def box(ax, xy, w, h, label="", color=BLUE, fill="#eef5fa", size=7):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.006,rounding_size=.012",
                               edgecolor=color, facecolor=fill, linewidth=.8))
    if label:
        text(ax, xy[0]+w/2, xy[1]+h/2, label, fontsize=size)


def arrow(ax, a, b, color=GRAY, dashed=False, curve=0, **kw):
    mutation_scale = kw.pop("mutation_scale", 8)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=mutation_scale,
                                linewidth=.9, color=color,
                                linestyle="--" if dashed else "-",
                                connectionstyle=f"arc3,rad={curve}", **kw))


def save(fig, name, tight=True):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        fig.savefig(OUT / f"{name}.{suffix}",
                    bbox_inches="tight" if tight else None)
    plt.close(fig)


def butterfly():
    fig, ax = canvas(3.45, 2.4)
    xs = np.linspace(.12, .69, 4)
    ys = np.linspace(.84, .27, 8)
    for stage in range(3):
        for i in range(8):
            partner = i ^ (1 << stage)
            ax.plot(xs[stage:stage+2], [ys[i], ys[i]], color="#b3bdc5", lw=.55)
            ax.plot(xs[stage:stage+2], [ys[i], ys[partner]], color=BLUE, lw=.65, alpha=.8)
        text(ax, (xs[stage]+xs[stage+1])/2, .94, f"bit {stage}", fontsize=7)
    for x in xs:
        ax.scatter([x]*8, ys, s=6, color=INK, zorder=3)
    for i, y in enumerate(ys):
        text(ax, .05, y, str(i), fontsize=6.5)
    text(ax, .825, .87, "Local update", fontsize=7, fontweight="bold")
    for y, lab in zip([.74,.61,.48,.35], ["FFT / NTT\ntwiddled pair", "FWHT\nsum / difference", "Zeta\none-sided sum", "Structured\n2 × 2 matrix"]):
        text(ax, .84, y, lab, fontsize=6.1)
    arrow(ax, (.68,.60), (.74,.60))
    text(ax, .5, .15, r"Shared pairing: $i \leftrightarrow i\oplus 2^q$", fontsize=8)
    text(ax, .5, .055, "Arithmetic, order and scaling remain explicit.", fontsize=6.8)
    save(fig, "butterfly")


def roofline():
    """Symbolic resource bounds and separate conditions for reaching them."""
    fig, ax = canvas(3.45, 2.75)
    text(ax, .5, .955, "(a) Ceiling at a selected storage boundary",
         fontsize=8, fontweight="bold")
    arrow(ax, (.15, .59), (.94, .59), color=INK)
    arrow(ax, (.15, .59), (.15, .88), color=INK)
    ax.plot([.15, .49, .91], [.59, .81, .81], color=BLUE, lw=1.5)
    ax.plot([.49, .63], [.81, .90], color=GRAY, ls="--", lw=.8)
    text(ax, .73, .845, r"core ceiling $P_{\rm core}$", fontsize=7)
    text(ax, .30, .785, r"$\mathrm{BW}_h I_h$", color=BLUE,
         fontsize=7, rotation=28)
    text(ax, .07, .735, "Pair throughput", rotation=90, fontsize=7)
    text(ax, .55, .535, r"Boundary intensity $I_h$ (pair work / byte)",
         fontsize=7)
    text(ax, .5, .465, r"$P\leq\min(P_{\rm core},\mathrm{BW}_h I_h)$",
         fontsize=8)
    text(ax, .5, .37, "(b) Reaching the ceiling also requires", fontsize=8,
         fontweight="bold")
    for x, label, color, fill in [
            (.02, "Ready\nsubgraphs", BLUE, PALE[0]),
            (.355, "Resident\nresources", GREEN, PALE[2]),
            (.69, "Valid reads /\nsynchronization", PURPLE, PALE[3])]:
        box(ax, (x, .205), .29, .105, label, color=color, fill=fill, size=7)
        arrow(ax, (x+.145, .20), (.5, .145), color=color)
    text(ax, .5, .12, "Limited readiness or occupancy can leave roofs unused.",
         fontsize=7)
    text(ax, .5, .035, "Schematic axes; no samples or quantitative attainment model.",
         fontsize=7, color=GRAY)
    save(fig, "roofline", tight=False)


def hybrid_dataflow():
    """Render a generic, aligned counterpart to the inherited APPT sketch.

    Both panels share the same stage and tile coordinates.  The left panel
    materializes every stage boundary; the right panel retains bounded stage
    bands in a tile and exchanges only the group boundary.  The graph is an
    abstract butterfly envelope: no NTT-specific coefficient, ordering, or
    butterfly-unit inventory is implied.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.48), sharex=True, sharey=True)
    fig.subplots_adjust(left=.055, right=.995, bottom=.16, top=.84, wspace=.09)
    ys = np.array([.76, .58, .40, .22])
    xs = np.array([.16, .34, .52, .70, .88])
    stage_labels = ["input", "S0", "S1", "S2", "S3"]
    pairings = [(0, 1), (2, 3)]

    def stage_edges(ax, left, right, color=INK, alpha=.72, lw=.62,
                    dashed=False):
        for a, b in pairings:
            ax.plot([left, right], [ys[a], ys[b]], color=color, lw=lw,
                    alpha=alpha, ls="--" if dashed else "-", zorder=2)
            ax.plot([left, right], [ys[b], ys[a]], color=color, lw=lw,
                    alpha=alpha, ls="--" if dashed else "-", zorder=2)

    def base(ax, title, subtitle):
        ax.set_xlim(.02, .98)
        ax.set_ylim(.08, .94)
        ax.axis("off")
        ax.text(.50, 1.03, title, ha="center", va="bottom", fontsize=8.0,
                fontweight="bold", color=INK, transform=ax.transAxes)
        ax.text(.50, .97, subtitle, ha="center", va="bottom", fontsize=6.5,
                color=GRAY, transform=ax.transAxes)
        for i, y in enumerate(ys):
            ax.text(.045, y, f"tile {i}", ha="right", va="center",
                    fontsize=6.2, color=INK)
        for x, label in zip(xs, stage_labels):
            ax.text(x, .09, label, ha="center", va="top", fontsize=6.1,
                    color=BLUE if label != "input" else GRAY)
        for y in ys:
            ax.plot([.105, .92], [y, y], color="#b8c1c7", lw=.4, zorder=0)
        for x in xs:
            ax.plot([x, x], [.14, .83], color="#e1e5e8", lw=.35, zorder=0)

    # Stage-at-a-time: all four rows pass through every boundary, with an
    # explicit materialization stripe after every stage.
    ax = axes[0]
    base(ax, "Boundary-materialized", "one stage group at a time")
    for idx in range(4):
        stage_edges(ax, xs[idx], xs[idx + 1], color=BLUE, alpha=.72)
        for y in ys:
            ax.add_patch(Rectangle((xs[idx + 1] - .018, y - .035), .036, .07,
                                   facecolor="#f4f5f6", edgecolor=GRAY,
                                   lw=.35, zorder=3))
        ax.text(xs[idx + 1], .85, "write / reload", ha="center", va="bottom",
                fontsize=5.6, rotation=90, color=GRAY)
    for x in xs[1:]:
        ax.annotate("", xy=(x + .014, .80), xytext=(x + .014, .85),
                    arrowprops={"arrowstyle": "->", "color": GRAY, "lw": .55})

    # Hybrid: same graph coordinates, but stage bands are bounded resident
    # tiles.  Internal edges are solid; group handoff is the only dashed path.
    ax = axes[1]
    base(ax, "Hierarchical hybrid dataflow", "resident tiles + explicit group exchange")
    ax.add_patch(Rectangle((xs[0] - .05, .14), xs[2] - xs[0] + .10, .70,
                           facecolor=PALE[0], edgecolor=BLUE, lw=1.0,
                           linestyle="-", zorder=1))
    ax.add_patch(Rectangle((xs[2] - .05, .14), xs[4] - xs[2] + .10, .70,
                           facecolor=PALE[1], edgecolor=ORANGE, lw=1.0,
                           linestyle="-", zorder=1))
    ax.text((xs[0] + xs[2]) / 2, .81, "resident subgraph A\n(register / shared)",
            ha="center", va="top", fontsize=6.0, color=BLUE, zorder=4)
    ax.text((xs[2] + xs[4]) / 2, .81, "resident subgraph B\n(register / shared)",
            ha="center", va="top", fontsize=6.0, color=ORANGE, zorder=4)
    stage_edges(ax, xs[0], xs[1], color=BLUE, alpha=.84, lw=.7)
    stage_edges(ax, xs[1], xs[2], color=BLUE, alpha=.84, lw=.7)
    stage_edges(ax, xs[2], xs[3], color=ORANGE, alpha=.84, lw=.7)
    stage_edges(ax, xs[3], xs[4], color=ORANGE, alpha=.84, lw=.7)
    for a, b in pairings:
        ax.plot([xs[2], xs[2] + .07], [ys[a], ys[b]], color=GREEN, lw=.8,
                ls="--", zorder=3)
        ax.plot([xs[2], xs[2] + .07], [ys[b], ys[a]], color=GREEN, lw=.8,
                ls="--", zorder=3)
        ax.text(xs[2] + .015, .48, "aligned chunk\nexchange", ha="left", va="center",
            fontsize=5.5, color=GREEN, rotation=90, zorder=5)
    # Compact legend, common to the two aligned panels.
    fig.lines.extend([])
    fig.text(.50, .02, "solid: resident dependency    dashed: inter-group exchange",
             ha="center", va="bottom", fontsize=6.1, color=INK)
    save(fig, "hybrid_dataflow")


def component_grid(ax, x, y, transpose=False):
    step = .073
    for row in range(4):
        for col in range(4):
            logical = 4*col+row if transpose else 4*row+col
            cell_y = y+(3-row)*step
            ax.add_patch(Rectangle((x+col*step, cell_y),step,step,
                                   facecolor=PALE[logical//4], edgecolor="white", lw=.8))
            text(ax,x+(col+.5)*step,cell_y+step/2,str(logical),fontsize=6.5)


def hierarchy():
    fig, ax = canvas(3.45, 2.6)
    text(ax,.235,.95,"Producer rows",fontweight="bold")
    text(ax,.765,.95,"Consumer columns",fontweight="bold")
    text(ax,.235,.86,r"$P=\{0,1\}$")
    text(ax,.765,.86,r"$Q=\{2,3\}$")
    component_grid(ax,.09,.51)
    component_grid(ax,.62,.51,transpose=True)
    arrow(ax,(.405,.67),(.59,.67),color=GREEN)
    text(ax,.5,.77,"write",fontsize=6.7)
    text(ax,.5,.59,r"$4r+c$",fontsize=7)
    text(ax,.5,.50,r"$\downarrow\ 4c+r$",fontsize=7)
    text(ax,.5,.40,"Each consumer needs all four producer colors.",fontsize=6.5)
    ax.plot([.04,.96],[.335,.335],color="#ced5da",lw=.8)
    text(ax,.5,.28,r"Larger domain: $L=24$, leaf limit $k_{max}=8$",fontsize=7)
    for j in range(3):
        x=.055+j*.323
        box(ax,(x,.105),.245,.10,f"8-stage group {j}",fill=PALE[j],size=6.5)
        if j<2:
            arrow(ax,(x+.252,.155),(x+.314,.155))
    text(ax,.5,.03,"Bounded local state; explicit exchanges between groups.",fontsize=6.4)
    save(fig,"hierarchy")


def bank_layout():
    fig, ax = canvas(3.45, 2.1)
    for start, swizzle, title in [(.085,False,"(a) Linear"),(.60,True,"(b) Writer-aligned")]:
        text(ax,start+.146,.94,title,fontweight="bold")
        for bank in range(4):
            text(ax,start+(bank+.5)*.073,.80,str(bank),fontsize=6.5)
        occupied=[]
        for r in range(4):
            for c in range(4):
                bank = (c^r) if swizzle else c
                logical=4*r+c
                x=start+bank*.073; y=.37+(3-r)*.088
                selected=c==1
                ax.add_patch(Rectangle((x,y),.073,.088,facecolor=PALE[0] if selected else "#f3f4f5",
                                       edgecolor=BLUE if selected else "white",lw=1 if selected else .7))
                text(ax,x+.0365,y+.044,str(logical),fontsize=6.5)
                if selected: occupied.append(bank)
        assert len(set(occupied)) == (4 if swizzle else 1)
        text(ax,start+.146,.27,"4 reads, 1 bank" if not swizzle else "4 reads, 4 banks",fontsize=7)
    text(ax,.48,.80,"bank",fontsize=6.2)
    arrow(ax,(.415,.55),(.555,.55),color=GREEN)
    text(ax,.5,.135,r"Address: $4r+c\ \longrightarrow\ 4r+(c\oplus r)$",fontsize=7.5)
    text(ax,.5,.035,"Highlight: fixed logical column c = 1; one word/value.",fontsize=6.3)
    save(fig,"bank_layout")


def pipeline():
    """Three independent transforms, full-row barriers, and two slot lifetimes."""
    fig, ax = canvas(3.45, 2.75)
    x0, dx = .16, .05
    text(ax, .5, .955, "Three independent 16-point Hadamard transforms",
         fontsize=8, fontweight="bold")
    text(ax, .5, .88, "Each: 4 row producers -> 4 column consumers", fontsize=7)
    rows = {"Rows": .73, "Cols": .52, "Slot 0": .36, "Slot 1": .23}
    for label, y in rows.items():
        text(ax, .075, y+.035, label, fontsize=7)
        ax.plot([x0, x0+16*dx], [y-.015, y-.015], color="#c8ced3", lw=.6)
    for name, start, color, fill in [
            ("A", 0, BLUE, PALE[0]), ("B", 4, GREEN, PALE[2]),
            ("C", 8, PURPLE, PALE[3])]:
        for lane, offset in [("Rows", 0), ("Cols", 4)]:
            y = rows[lane]
            for k in range(4):
                x = x0+(start+offset+k)*dx
                ax.add_patch(Rectangle((x, y), dx, .075,
                                      facecolor=fill, edgecolor=color, lw=.7))
                text(ax, x+dx/2, y+.0375, f"{name}{k}", fontsize=7)
        barrier = x0+(start+4)*dx
        ax.plot([barrier, barrier], [.605, .81], color=color, lw=1,
                linestyle="--")
        ax.scatter([barrier], [.73], marker="v", s=18, color=color, zorder=4)
        arrow(ax, (barrier, .66), (barrier, .605), color=color)
        text(ax, barrier+.085, .65, f"{name}: all 4", fontsize=7, color=color)
    # A slot is held from its first producer write through its last consumer.
    for lane, lo, hi, name, color, fill in [
            ("Slot 0", 0, 8, "A live", BLUE, PALE[0]),
            ("Slot 0", 8, 16, "C live", PURPLE, PALE[3]),
            ("Slot 1", 4, 12, "B live", GREEN, PALE[2])]:
        y = rows[lane]
        ax.add_patch(Rectangle((x0+lo*dx, y), (hi-lo)*dx, .075,
                              facecolor=fill, edgecolor=color, lw=.8))
        text(ax, x0+(lo+hi)*dx/2, y+.0375, name, fontsize=7)
        ax.scatter([x0+hi*dx], [y+.0375], marker="D", s=15,
                   color=color, zorder=4)
    arrow(ax, (x0+8*dx, .51), (x0+8*dx, .41), color=ORANGE)
    text(ax, .5, .17, "Slot 0: C starts after A's final column read.", fontsize=7)
    arrow(ax, (x0, .105), (x0+16*dx, .105), color=INK)
    for t in [0, 4, 8, 12, 16]:
        text(ax, x0+t*dx, .07, str(t), fontsize=7)
    text(ax, .5, .025, "Illustrative units; overlap requires available resources.",
         fontsize=7, color=GRAY)
    save(fig, "pipeline", tight=False)


def framework():
    fig, ax = canvas(3.45, 3.05)
    for x,title in [(.02,"Operator\nlabels / order"),(.36,"Workload\nN / batch"),(.70,"Hardware\nscopes / cores")]:
        box(ax,(x,.845),.28,.115,title,size=6.6)
        arrow(ax,(x+.14,.835),(.5,.765))
    box(ax,(.12,.625),.76,.14,"Construct legal organizations\nstage bands · ownership · layouts · readiness",size=6.6)
    arrow(ax,(.5,.62),(.5,.555))
    box(ax,(.12,.42),.76,.13,"Instantiate compatible cores\ncompile · resource checks · numerical check",size=6.6)
    arrow(ax,(.50,.415),(.50,.35))
    box(ax,(.12,.215),.76,.13,"Rank and measure within a budget\ncalibrated services + structural neighbors",fill=PALE[2],color=GREEN,size=6.6)
    # Feedback denotes a repeated search round, outside the hot execution path.
    arrow(ax,(.885,.28),(.92,.70),color=GREEN,curve=.5)
    text(ax,.96,.49,"refine",rotation=90,fontsize=6.4)
    ax.plot([.025,.975],[.175,.175],color=GRAY,lw=.8,ls="--")
    arrow(ax,(.5,.208),(.5,.14))
    box(ax,(.18,.035),.64,.10,"Selected kernel: time execution only",fill=PALE[1],color=ORANGE,size=6.5)
    save(fig,"framework")


def _pair_icon(ax, x, y, label, color, scale=.045):
    """Draw a small operator-specific pair update without hiding its label."""
    ax.plot([x - scale, x + scale], [y - scale * .7, y + scale * .7],
            color=color, lw=.9, zorder=3)
    ax.scatter([x - scale, x + scale], [y - scale * .7, y + scale * .7],
               s=8, color=color, edgecolor="white", linewidth=.35, zorder=4)
    text(ax, x, y - scale * 1.95, label, fontsize=6.6, color=INK)


def _draw_butterfly(ax, x_positions, y_positions, stage_bits,
                    colors=None, lw=.55, node_size=8, zorder=3):
    """Draw the complete two-output edge set for an N=8 staged graph."""
    n = len(y_positions)
    if colors is None:
        colors = [BLUE] * len(stage_bits)
    for stage, bit in enumerate(stage_bits):
        left, right = x_positions[stage], x_positions[stage + 1]
        for i in range(n):
            partner = i ^ (1 << bit)
            # A pair update has two outputs.  Drawing i -> i and i -> partner
            # keeps both straight and crossing dependencies visible.
            ax.plot([left, right], [y_positions[i], y_positions[i]],
                    color=colors[stage], lw=lw, alpha=.78, zorder=zorder)
            ax.plot([left, right], [y_positions[i], y_positions[partner]],
                    color=colors[stage], lw=lw, alpha=.78, zorder=zorder)
    for x in x_positions:
        ax.scatter([x] * n, y_positions, s=node_size, color=INK,
                   edgecolor="white", linewidth=.35, zorder=zorder + 1)


def overview():
    """Render the operator-independent construction and mapping overview."""
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.86),
                             gridspec_kw={"width_ratios": [1.02, 1.08, 1.0]})
    fig.subplots_adjust(left=.018, right=.988, bottom=.065, top=.89,
                        wspace=.075)

    # Panel (a) keeps arithmetic labels separate from the shared dependency
    # construction.  Each source has a distinct local glyph and color.
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    text(ax, .50, 1.035, "(a) Shared dependency construction",
         fontsize=8.0, fontweight="bold")
    text(ax, .50, .965, "operator labels stay explicit",
         fontsize=7.0, color=GRAY)
    # Compact labels feed the complete graph.  The labels name distinct local
    # arithmetic contracts and do not imply equal semantics or cost.
    sources = [
        (.16, .85, "FFT", BLUE), (.50, .85, "NTT", ORANGE),
        (.84, .85, "FWHT", GREEN), (.30, .70, "zeta", PURPLE),
        (.70, .70, "structured 2x2", "#a34f62"),
    ]
    for x, y, label, color in sources:
        ax.scatter([x - .045], [y], s=16, color=color, zorder=4)
        text(ax, x - .025, y, label, ha="left", fontsize=7.0, color=INK)
        if y > .8 and x != .5:
            ax.plot([x, x, .50], [y - .035, .765, .765],
                    color=color, lw=.5)
        else:
            arrow(ax, (x, y - .035), (.50, .555), color=color, lw=.5,
                  mutation_scale=5)
    # A complete N=8 dependency graph supplies the construction witness.
    graph_x = np.array([.27, .42, .57, .72])
    graph_y = np.linspace(.49, .25, 8)
    _draw_butterfly(ax, graph_x, graph_y, [0, 1, 2],
                    colors=[BLUE, BLUE, BLUE], lw=.38, node_size=7)
    for x, label in zip(graph_x, ["in", "S0", "S1", "out"]):
        text(ax, x, .535, label, fontsize=7.0,
             color=GRAY if label == "in" else BLUE)
    ax.add_patch(Rectangle((.235, .205), .53, .315, facecolor="none",
                           edgecolor=BLUE, linewidth=.75, linestyle="-",
                           zorder=1))
    text(ax, .50, .125, "same dependency rule",
         fontsize=7.0, color=BLUE, fontweight="bold")
    text(ax, .50, .073, "operator-specific order and scaling",
         fontsize=7.0, color=GRAY)

    # Panel (b) uses one aligned stage/data plane.  The first two stages form
    # two N=4 closed components; stage 2 consumes one producer from each.
    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    text(ax, .50, 1.035, "(b) Stage and data organization",
         fontsize=8.0, fontweight="bold")
    text(ax, .50, .965, "closed bands with explicit fan-in",
         fontsize=7.0, color=GRAY)
    stage_x = np.array([.10, .23, .36])
    ys = np.array([.80, .73, .66, .59, .49, .42, .35, .28])
    # These boxes are the dependency-closed N=4 components for stage bits 0,1.
    ax.add_patch(Rectangle((.055, .545), .345, .315, facecolor=PALE[0],
                           edgecolor=BLUE, linewidth=.85, zorder=0))
    ax.add_patch(Rectangle((.055, .225), .345, .315, facecolor=PALE[1],
                           edgecolor=ORANGE, linewidth=.85, zorder=0))
    text(ax, .075, .842, "4 values", ha="left", fontsize=7.0,
         color=BLUE, fontweight="bold")
    text(ax, .075, .522, "4 values", ha="left", fontsize=7.0,
         color=ORANGE, fontweight="bold")
    _draw_butterfly(ax, stage_x, ys, [0, 1],
                    colors=[BLUE, BLUE], lw=.36, node_size=6)
    for x, label in zip(stage_x, ["in", "S0", "S1"]):
        text(ax, x, .885, label, fontsize=7.0,
             color=GRAY if label == "in" else BLUE)
    # Stage 2 is shown as four separate N=2 consumer groups.  Each receives
    # both outputs of the pair (top producer and bottom producer), so the
    # fan-in is visible rather than represented by independent arrows.
    target_centers = np.array([.78, .62, .46, .30])
    ax.plot([.44, .44], [.225, .86], color=GREEN, lw=.8, ls="--", zorder=2)
    text(ax, .447, .56, "stage 2\nhandoff", ha="left", fontsize=7.0,
         color=GREEN, rotation=90, zorder=5,
         bbox={"facecolor": "white", "edgecolor": "none", "pad": .45})
    text(ax, .755, .885, "2-value subgraphs", fontsize=7.0,
         color=ORANGE, fontweight="bold")
    for group, center in enumerate(target_centers):
        ax.add_patch(Rectangle((.68, center - .037), .24, .074,
                               facecolor=PALE[1], edgecolor=ORANGE,
                               linewidth=.65, zorder=1))
        target_y = (center + .016, center - .016)
        src_i = (group, group + 4)
        for i, ty in zip(src_i, target_y):
            ax.plot([.36, .70], [ys[i], ty], color=GREEN, lw=.48,
                    ls="--", alpha=.9, zorder=2)
        _draw_butterfly(ax, [.70, .87], target_y, [0],
                        colors=[ORANGE], lw=.45, node_size=6)
    # Aligned boundary strips compare explicit HBM materialization with a
    # bounded resident band on the same stage coordinate axis.
    text(ax, .40, .155, "stagewise", ha="right", fontsize=7.0,
         color=GRAY)
    ax.add_patch(Rectangle((.44, .138), .17, .032, facecolor="#f1f3f4",
                           edgecolor=GRAY, linewidth=.45))
    ax.add_patch(Rectangle((.61, .138), .17, .032, facecolor="#f1f3f4",
                           edgecolor=GRAY, linewidth=.45))
    ax.add_patch(Rectangle((.78, .138), .17, .032, facecolor="#f1f3f4",
                           edgecolor=GRAY, linewidth=.45))
    text(ax, .40, .088, "resident", ha="right", fontsize=7.0,
         color=BLUE)
    ax.add_patch(Rectangle((.44, .071), .34, .032, facecolor=PALE[0],
                           edgecolor=BLUE, linewidth=.55))
    ax.add_patch(Rectangle((.78, .071), .17, .032, facecolor=PALE[1],
                           edgecolor=ORANGE, linewidth=.55))

    # Panel (c) gives the hierarchy as graphical storage scopes and makes the
    # selection conditional on both workload live state and hardware capacity.
    ax = axes[2]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    text(ax, .50, 1.035, "(c) Hierarchical physical mapping",
         fontsize=8.0, fontweight="bold")
    text(ax, .50, .965, "storage and synchronization scopes",
         fontsize=7.0, color=GRAY)
    # Separate scope entries avoid implying that one physical address space
    # contains all other scopes.  The small glyph at right is illustrative.
    scopes = [
        ("register", .70, BLUE), ("warp", .59, GREEN),
        ("shared", .48, ORANGE), ("global", .37, PURPLE),
    ]
    for label, y, color in scopes:
        ax.add_patch(Rectangle((.50, y), .43, .085, facecolor="#fbfcfc",
                               edgecolor=color, linewidth=.8, zorder=1))
        text(ax, .525, y + .0425, label, ha="left", fontsize=7.0,
             color=color, fontweight="bold")
    # Four compact motifs keep the mapping graphical without nesting scopes.
    for j in range(3):
        xx = .80 + j * .035
        ax.plot([xx, xx], [.715, .77], color=BLUE, lw=.8, zorder=3)
    for j in range(2):
        for k in range(2):
            ax.add_patch(Rectangle((.80 + j * .035, .605 + k * .030),
                                   .025, .021, facecolor="#ffffff",
                                   edgecolor=GREEN, linewidth=.4, zorder=3))
    for j in range(4):
        ax.add_patch(Rectangle((.80 + (j % 2) * .035,
                                .495 + (j // 2) * .030), .025, .021,
                               facecolor="#ffffff", edgecolor=ORANGE,
                               linewidth=.4, zorder=3))
    ax.add_patch(Rectangle((.80, .385), .07, .036, facecolor=PALE[3],
                           edgecolor=PURPLE, linewidth=.5, zorder=3))
    ax.add_patch(Rectangle((.885, .385), .035, .036, facecolor=PALE[3],
                           edgecolor=PURPLE, linewidth=.5, zorder=3))
    # Live state and capacity feed a selected scope.  The dashed arrows
    # express configuration and selection, not simultaneous residency.
    box(ax, (.035, .15), .28, .105, "live state", color=BLUE,
        fill=PALE[0], size=7.0)
    box(ax, (.40, .15), .28, .105, "capacity", color=ORANGE,
        fill=PALE[1], size=7.0)
    box(ax, (.035, .51), .28, .105, "scope", color=GREEN,
        fill=PALE[2], size=7.0)
    arrow(ax, (.175, .26), (.175, .50), color=BLUE, dashed=True,
          lw=.65, mutation_scale=6)
    arrow(ax, (.54, .26), (.32, .51), color=ORANGE, dashed=True,
          lw=.65, mutation_scale=6)
    ax.plot([.47, .445, .445, .47], [.785, .785, .37, .37],
            color=GRAY, lw=.6)
    arrow(ax, (.32, .562), (.44, .562), color=GREEN, dashed=True,
          lw=.65, mutation_scale=6)
    save(fig, "overview", tight=False)


def _architecture_node(ax, x, y, w, h, label, color=BLUE, fill="#eef5fa",
                       size=7.0, radius=.010):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.005,rounding_size={radius}",
                                edgecolor=color, facecolor=fill, linewidth=.75,
                                zorder=2))
    text(ax, x + w / 2, y + h / 2, label, fontsize=size, zorder=3)


def _mini_subgraph(ax, x, y, w, h, color=BLUE):
    """Draw a candidate subgraph glyph inside an architecture node."""
    left = x + w * .22
    right = x + w * .78
    for yy in (y + h * .36, y + h * .64):
        ax.plot([left, right], [yy, yy], color=color, lw=.65, zorder=4)
        ax.scatter([left, right], [yy, yy], s=5, color=color, zorder=5)
    ax.plot([left, right], [y + h * .36, y + h * .64], color=color,
            lw=.65, zorder=4)


def system_architecture():
    """Render a four-lane realization with an explicit timing boundary."""
    fig, ax = canvas(7.16, 3.18)
    lanes = [(.76, "1  contracts", BLUE, PALE[0]),
             (.54, "2  lowering", GREEN, PALE[2]),
             (.32, "3  search and validation", PURPLE, PALE[3]),
             (.07, "4  cached runtime", ORANGE, PALE[1])]
    for y, label, color, fill in lanes:
        ax.add_patch(Rectangle((.018, y), .964, .18 if y > .1 else .19,
                               facecolor=fill, edgecolor="#d7dee3", linewidth=.65))
        text(ax, .045, y + .153, label, ha="left", fontsize=7.2,
             color=color, fontweight="bold")
    text(ax, .50, .955, "semantic / workload / hardware inputs", fontsize=7.4,
         color=INK, fontweight="bold")
    text(ax, .50, .015, "warm timing begins at device input and ends after the selected plan completes",
         fontsize=6.5, color=GRAY)

    # Contract lane: the three orthogonal inputs remain separate until lowering.
    for x, label, color, fill in [(.12, "operator\nsemantics", BLUE, PALE[0]),
                                  (.35, "N, batch,\nshape, order", GREEN, PALE[2]),
                                  (.58, "GPU, memory,\ncore catalog", ORANGE, PALE[1])]:
        _architecture_node(ax, x, .785, .17, .085, label, color=color, fill=fill, size=6.5)
    _architecture_node(ax, .80, .785, .15, .085, "calibrated\nservices", PURPLE, PALE[3], size=6.5)

    # Lowering lane: make the semantic transformation explicit.
    steps = [(.10, "dependency\nclosure"), (.31, "boundary\nintersections"),
             (.52, "aligned\nwriteback"), (.73, "core + tile\nlowering")]
    for x, label in steps:
        _architecture_node(ax, x, .565, .16, .085, label, color=GREEN, fill="#ffffff", size=6.3)
    for x in (.27, .48, .69):
        arrow(ax, (x, .607), (x + .035, .607), color=GREEN, lw=.75, mutation_scale=6)
    for x in (.205, .435, .665, .875):
        arrow(ax, (x, .785), (x - .02, .65), color=PURPLE, dashed=True, lw=.55, mutation_scale=5)
    text(ax, .89, .68, "legal candidate", fontsize=5.7, color=PURPLE, rotation=90)

    # Search lane: rejected candidates become explicit gaps; selected plan is cached.
    for x, label, color in [(.10, "compile +\nstatic checks", ORANGE),
                            (.31, "correctness\ncontract", BLUE),
                            (.52, "estimate /\nmeasure", PURPLE),
                            (.73, "rank +\nconfirm", GREEN)]:
        _architecture_node(ax, x, .345, .16, .085, label, color=color, fill="#ffffff", size=6.3)
    for x in (.27, .48, .69):
        arrow(ax, (x, .387), (x + .035, .387), color=PURPLE, lw=.75, mutation_scale=6)
    ax.plot([.23, .23], [.345, .31], color=ORANGE, lw=.6, ls="--")
    text(ax, .23, .295, "gap", fontsize=5.8, color=ORANGE)
    _architecture_node(ax, .86, .345, .10, .085, "plan\ncache", color=GREEN, fill="#ffffff", size=6.1)
    arrow(ax, (.89, .43), (.89, .345), color=GREEN, dashed=True, lw=.65, mutation_scale=6)

    # Runtime lane: show conversion, subgraph execution and synchronization as
    # timed work.  Cache lookup is outside the warm kernel interval.
    for x, label, color, fill in [(.08, "input /\nadapt", BLUE, "#ffffff"),
                                  (.29, "resident\nsubgraphs", GREEN, "#ffffff"),
                                  (.50, "streaming\nhandoffs", PURPLE, "#ffffff"),
                                  (.71, "sync +\noutput", ORANGE, "#ffffff")]:
        _architecture_node(ax, x, .105, .16, .09, label, color=color, fill=fill, size=6.3)
    for x in (.25, .46, .67):
        arrow(ax, (x, .15), (x + .035, .15), color=ORANGE, lw=.85, mutation_scale=7)
    arrow(ax, (.91, .43), (.80, .205), color=PURPLE, dashed=True, lw=.7, mutation_scale=6, curve=.06)
    text(ax, .92, .27, "cache hit", ha="left", fontsize=5.8, color=PURPLE, rotation=90)
    ax.plot([.065, .885], [.095, .095], color=ORANGE, lw=.9)
    ax.plot([.065, .065], [.085, .105], color=ORANGE, lw=.9)
    ax.plot([.885, .885], [.085, .105], color=ORANGE, lw=.9)
    text(ax, .475, .075, "timed device path", fontsize=6.0, color=ORANGE, fontweight="bold")
    text(ax, .50, .515, "compile, calibration, search, and cache fill are preparation costs",
         fontsize=6.0, color=GRAY, fontstyle="italic")
    save(fig, "system_architecture", tight=False)


if __name__ == "__main__":
    for render in (butterfly,roofline,hybrid_dataflow,hierarchy,bank_layout,
                   pipeline,framework,overview,system_architecture):
        render()
    print(f"Wrote 9 analytical PDF/SVG figures to {OUT}")
