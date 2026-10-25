#!/usr/bin/env python3
"""Render one unified absolute-throughput boxplot for the RC2 operator suites.

The plot deliberately uses effective radix-two butterfly-pairs/s as its common
work unit.  Operator, precision, direction, and source identity remain in the
CSV, while the figure pools only direction and transform shape within each
operator/precision group.  It never computes a cross-operator score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT.parent / "cuButterfly-v0.9.1-rc2/results/integrated_repairs_20261001_replay_fix1/a100-80gb"
DEFAULT_OUT = ROOT / "paper/generated"
DEFAULT_FIGURES = ROOT / "paper/figures"

SYSTEM_ORDER = (
    "cuButterfly",
    "cuFFT",
    "VkFFT",
    "GPU-NTT Merge",
    "DAO",
    "shared-iterative",
)
SYSTEM_STYLE = {
    "cuButterfly": ("#1769aa", "o"),
    "cuFFT": ("#d95f02", "s"),
    "VkFFT": ("#2ca25f", "^"),
    "GPU-NTT Merge": ("#7570b3", "D"),
    "DAO": ("#e7298a", "P"),
    "shared-iterative": ("#666666", "X"),
}
GROUP_ORDER = ("FFT-FP32", "FFT-FP64", "NTT-word32", "NTT-word64", "FWHT-FP16", "FWHT-BF16")


def _speedup_summary(rows: list[dict[str, Any]], group: str) -> dict[str, Any] | None:
    """Return matched-cell cuButterfly speedups for one plot group."""
    by_cell = {
        (row["cell_id"], row["system"]): float(row["butterfly_pairs_per_s"])
        for row in rows
        if row["group"] == group
    }
    speedups: list[float] = []
    cells = sorted({row["cell_id"] for row in rows if row["group"] == group})
    for cell_id in cells:
        ours = by_cell.get((cell_id, "cuButterfly"))
        if ours is None:
            continue
        for baseline in SYSTEM_ORDER:
            if baseline == "cuButterfly":
                continue
            reference = by_cell.get((cell_id, baseline))
            if reference is not None and reference > 0:
                speedups.append(ours / reference)
    if not speedups:
        return None
    return {
        "matched_pairs": len(speedups),
        "geometric_mean_speedup": math.exp(sum(math.log(value) for value in speedups) / len(speedups)),
        "minimum_speedup": min(speedups),
        "maximum_speedup": max(speedups),
    }


def _positive(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"expected positive finite value, got {value!r}")
    return number


def _row(*, group: str, operator: str, precision: str, direction: str, log_n: int,
         batch: int, channels: int, system: str, median_ms: float, source: str,
         cell_id: str) -> dict[str, Any]:
    n = 1 << int(log_n)
    transforms_s = 1000.0 * int(batch) * int(channels) / _positive(median_ms)
    pairs_s = transforms_s * (n * int(log_n) / 2.0)
    return {
        "group": group,
        "operator": operator,
        "precision": precision,
        "direction": direction,
        "logN": int(log_n),
        "N": n,
        "batch": int(batch),
        "channels": int(channels),
        "system": system,
        "median_ms": _positive(median_ms),
        "transforms_per_s": transforms_s,
        "butterfly_pairs_per_s": pairs_s,
        "cell_id": cell_id,
        "source": source,
    }


def _load_fft(root: Path) -> list[dict[str, Any]]:
    document = json.loads((root / "fft36/summary.json").read_text(encoding="utf-8"))
    if document.get("status") != "complete-measured-matrix":
        raise ValueError("FFT36 is not a complete measured matrix")
    rows: list[dict[str, Any]] = []
    for cell in document["cells"]:
        workload = cell["workload"]
        group = f"FFT-{str(workload['precision']).upper()}"
        if group not in GROUP_ORDER:
            continue
        for system, median in cell.get("median_ms", {}).items():
            if system == "cuButterfly":
                label = "cuButterfly"
            elif system in {"cuFFT", "VkFFT"}:
                label = system
            else:
                continue
            rows.append(_row(
                group=group,
                operator="fft",
                precision=str(workload["precision"]),
                direction=str(workload.get("direction", "forward")),
                log_n=int(workload["logN"]),
                batch=int(workload["batch"]),
                channels=1,
                system=label,
                median_ms=median,
                source="fft36",
                cell_id=str(cell["id"]),
            ))
    return rows


def _load_ntt(root: Path) -> list[dict[str, Any]]:
    journal_path = root / "ntt/journal.json"
    document = json.loads(journal_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for cell in document.get("cells", []):
        if cell.get("status") != "complete" or not isinstance(cell.get("median_ms"), dict):
            continue
        workload = cell["workload"]
        word_bits = int(workload["word_bits"])
        group = f"NTT-word{word_bits}"
        if group not in GROUP_ORDER:
            continue
        channels = max(1, len(workload.get("moduli", [])))
        for source_name, label in (("cubutterfly", "cuButterfly"), ("gpuntt", "GPU-NTT Merge")):
            median = cell["median_ms"].get(source_name)
            if median is None:
                continue
            rows.append(_row(
                group=group,
                operator="ntt",
                precision=f"word{word_bits}",
                direction=str(workload.get("direction", "forward")),
                log_n=int(workload["logN"]),
                batch=int(workload["batch"]),
                channels=channels,
                system=label,
                median_ms=median,
                source="ntt-journal",
                cell_id=str(workload["id"]),
            ))
    return rows


def _load_fwht(root: Path) -> list[dict[str, Any]]:
    document = json.loads((root / "fwht/summary.json").read_text(encoding="utf-8"))
    if document.get("status") != "complete":
        raise ValueError("FWHT is not a complete measured matrix")
    rows: list[dict[str, Any]] = []
    for cell in document["cells"]:
        semantic = cell["semantic"]
        precision = str(semantic["precision"])
        group = f"FWHT-{precision.upper()}"
        if group not in GROUP_ORDER:
            continue
        for source_name, label in (("warp-register", "cuButterfly"), ("dao", "DAO"), ("shared-iterative", "shared-iterative")):
            implementation = cell.get("implementations", {}).get(source_name, {})
            if implementation.get("status") != "complete":
                continue
            rows.append(_row(
                group=group,
                operator="fwht",
                precision=precision,
                direction=str(semantic.get("direction", "forward")),
                log_n=int(semantic["logN"]),
                batch=int(semantic["batch"]),
                channels=1,
                system=label,
                median_ms=implementation["kernel_ms_median"],
                source="fwht",
                cell_id=str(cell["cell_id"]),
            ))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    metrics = (("butterfly_pairs_per_s", "Pairs/s"), ("transforms_per_s", "Transforms/s"))
    # Keep the two metric rows compact enough for a two-column paper while
    # leaving room for the per-group speedup labels in the upper row.
    # The unified plot carries six workload groups and per-group statistics;
    # preserve vertical room for readable annotations at paper width.
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 2.90), sharex=True, squeeze=False)
    group_centers: list[float] = []
    group_bounds: list[tuple[float, float]] = []
    separator_positions: list[float] = []
    current_position = 1.0
    for group_index, group in enumerate(GROUP_ORDER):
        systems = [system for system in SYSTEM_ORDER if any(row["group"] == group and row["system"] == system for row in rows)]
        start = current_position
        positions = {system: current_position + index for index, system in enumerate(systems)}
        end = start + max(1, len(systems)) - 1
        group_bounds.append((start, end))
        group_centers.append((start + end) / 2.0)
        current_position = end + 2.5
        if group_index < len(GROUP_ORDER) - 1:
            separator_positions.append(current_position - 0.75)
        for metric_index, (metric, ylabel) in enumerate(metrics):
            axis = axes[metric_index, 0]
            data = []
            box_positions = []
            labels = []
            for system in systems:
                values = [float(row[metric]) for row in rows if row["group"] == group and row["system"] == system]
                data.append(values)
                box_positions.append(positions[system])
                labels.append(system)
            if not data:
                continue
            artists = axis.boxplot(
                data,
                positions=box_positions,
                widths=0.62,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#202020", "linewidth": 0.95},
                whiskerprops={"color": "#505050", "linewidth": 0.75},
                capprops={"color": "#505050", "linewidth": 0.75},
                boxprops={"linewidth": 0.7},
            )
            for patch, system in zip(artists["boxes"], labels):
                patch.set_facecolor(SYSTEM_STYLE[system][0])
                patch.set_alpha(0.67)
                values = [float(row[metric]) for row in rows if row["group"] == group and row["system"] == system]
                jitter = [positions[system] + ((index % 9) - 4) * 0.018 for index in range(len(values))]
                axis.scatter(jitter, values, s=5.5, color=SYSTEM_STYLE[system][0], edgecolor="white", linewidth=0.2, alpha=0.48, zorder=3)
            axis.set_ylabel(ylabel, fontsize=8)
            axis.set_yscale("log")
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0e}"))
            axis.tick_params(axis="y", labelsize=7.0)
            axis.grid(True, which="major", axis="y", alpha=0.2, linewidth=0.55)
            for position in separator_positions:
                axis.axvline(position, color="#555555", linewidth=0.7, linestyle=(0, (3, 3)), alpha=0.7)
        # The common annotation is computed once per group from matched cells,
        # then placed in the upper-right corner of that group's top panel.
        speedup = _speedup_summary(rows, group)
        if speedup:
            sample_count = len({row["cell_id"] for row in rows
                                if row["group"] == group and row["system"] == "cuButterfly"})
            annotation = (f"n={sample_count}\n"
                          f"GM {speedup['geometric_mean_speedup']:.2f}x\n"
                          f"{speedup['minimum_speedup']:.2f}–{speedup['maximum_speedup']:.2f}x")
            _, end = group_bounds[-1]
            axes[0, 0].text(
                end + 0.36,
                0.965,
                annotation,
                transform=axes[0, 0].get_xaxis_transform(),
                ha="right",
                va="top",
                fontsize=5.2,
                linespacing=0.9,
                color="#303030",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
                zorder=5,
            )
    fig.suptitle("A100 80GB RC2 unified absolute butterfly throughput", fontsize=8.8, y=0.925)
    axes[1, 0].set_xticks(group_centers)
    axes[1, 0].set_xticklabels(GROUP_ORDER, fontsize=7.2)
    axes[1, 0].tick_params(axis="x", length=0, pad=3)
    axes[1, 0].set_xlim(0, current_position - 0.1)
    handles = [Patch(facecolor=SYSTEM_STYLE[system][0], edgecolor="#505050", alpha=0.67, label=system) for system in SYSTEM_ORDER if any(row["system"] == system for row in rows)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=6, fontsize=6.1, frameon=False)
    fig.subplots_adjust(left=0.095, right=0.99, top=0.80, bottom=0.14, hspace=0.42)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "render_rc2_unified_boxplot.py", "Title": "Unified RC2 absolute butterfly throughput"}
    fig.savefig(path, metadata=metadata)
    fig.savefig(path.with_suffix(".svg"), metadata=metadata)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()
    rows = _load_fft(args.source_root) + _load_ntt(args.source_root) + _load_fwht(args.source_root)
    if not rows:
        raise ValueError("no complete RC2 rows found")
    missing = sorted(set(GROUP_ORDER) - {row["group"] for row in rows})
    if missing:
        raise ValueError(f"missing operator groups: {missing}")
    csv_path = args.output_dir / "rc2_unified_absolute_throughput.csv"
    figure_path = args.figure_dir / "rc2_unified_absolute_boxplot.pdf"
    _write_csv(csv_path, rows)
    _plot(rows, figure_path)
    summary = {
        "schema": "cubutterfly-rc2-unified-absolute-throughput-v1",
        "source_root": str(args.source_root.resolve()),
        "device": "NVIDIA A100 80GB PCIe",
        "groups": {
            group: {
                system: sum(1 for row in rows if row["group"] == group and row["system"] == system)
                for system in SYSTEM_ORDER
                if any(row["group"] == group and row["system"] == system for row in rows)
            }
            for group in GROUP_ORDER
        },
        "group_rows": {group: len({row["cell_id"] for row in rows if row["group"] == group}) for group in GROUP_ORDER},
        "speedup_summary": {group: _speedup_summary(rows, group) for group in GROUP_ORDER},
        "rows": len(rows),
        "metrics": ["effective_butterfly_pairs_per_s", "transforms_per_s"],
        "csv_path": str(csv_path.resolve()),
        "figure_path": str(figure_path.resolve()),
        "note": "Boxes are descriptive distributions within operator/precision groups; no cross-operator score is reported.",
    }
    (args.output_dir / "rc2_unified_absolute_throughput_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(f"wrote {csv_path}")
    print(f"wrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
