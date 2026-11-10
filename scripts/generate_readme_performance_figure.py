#!/usr/bin/env python3
"""Generate the compact V100 external-library comparison figure."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


LABELS = {
    "fft": ("FFT", "cuFFT"),
    "fwht": ("FWHT", "Dao FHT"),
    "ntt": ("NTT", "GPU-NTT"),
}


def geomean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("throughput ratios must be positive")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def load_ratios(path: Path) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            operator = row["operator"].strip().lower()
            if operator not in LABELS:
                continue
            grouped[operator].append(float(row["searched_throughput_vs_library"]))
    missing = sorted(set(LABELS) - set(grouped))
    if missing:
        raise ValueError(f"missing operator rows: {', '.join(missing)}")
    return {operator: geomean(values) for operator, values in grouped.items()}


def esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def render(ratios: dict[str, float]) -> str:
    width, height = 960, 520
    left, right, top, bottom = 90, 38, 76, 92
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = 0.0, 1.45

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">V100 selected-matrix performance versus specialist libraries</title>',
        '<desc id="desc">Geometric mean throughput ratio for searched cuButterfly or cuNTT '
        'relative to cuFFT, Dao FHT, and GPU-NTT. A value of one is parity.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#17202a} '
        '.small{font-size:14px}.label{font-size:16px;font-weight:600} '
        '.value{font-size:15px;font-weight:700}</style>',
        '<text x="90" y="32" font-size="22" font-weight="700">'
        'V100 selected-matrix throughput vs specialist libraries</text>',
        '<text x="90" y="55" class="small">Geometric mean; 1.0 = external library parity</text>',
    ]

    for tick in (0.0, 0.5, 1.0, 1.5):
        if tick > y_max:
            continue
        yy = y(tick)
        stroke = "#b45309" if math.isclose(tick, 1.0) else "#d8dee8"
        dash = ' stroke-dasharray="6 5"' if math.isclose(tick, 1.0) else ""
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" '
                     f'stroke="{stroke}" stroke-width="{2 if tick == 1.0 else 1}"{dash}/>')
        parts.append(f'<text x="{left-14}" y="{yy+5:.1f}" text-anchor="end" class="small">'
                     f'{tick:.1f}</text>')

    parts.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#596579"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#596579"/>',
        f'<text x="22" y="{top + plot_h/2:.1f}" transform="rotate(-90 22 {top + plot_h/2:.1f})" '
        'text-anchor="middle" class="small">Throughput ratio</text>',
    ])

    categories = list(LABELS)
    group_w = plot_w / len(categories)
    bar_w, gap = 84, 18
    for index, operator in enumerate(categories):
        center = left + group_w * (index + 0.5)
        x_library = center - bar_w - gap / 2
        x_cub = center + gap / 2
        for x_pos, value, color in ((x_library, 1.0, "#9ca3af"),
                                    (x_cub, ratios[operator], "#2563eb")):
            yy = y(value)
            parts.append(f'<rect x="{x_pos:.1f}" y="{yy:.1f}" width="{bar_w}" '
                         f'height="{height-bottom-yy:.1f}" rx="3" fill="{color}"/>')
            parts.append(f'<text x="{x_pos + bar_w/2:.1f}" y="{yy-8:.1f}" '
                         f'text-anchor="middle" class="value">{value:.3f}x</text>')
        label, library = LABELS[operator]
        parts.append(f'<text x="{center:.1f}" y="{height-bottom+28}" text-anchor="middle" class="label">'
                     f'{esc(label)}</text>')
        parts.append(f'<text x="{center:.1f}" y="{height-bottom+49}" text-anchor="middle" class="small">'
                     f'vs {esc(library)}</text>')

    legend_y = height - 18
    parts.extend([
        f'<rect x="{width-300}" y="{legend_y-12}" width="14" height="14" fill="#2563eb"/>',
        f'<text x="{width-278}" y="{legend_y}" class="small">searched cuButterfly/cuNTT</text>',
        f'<rect x="{width-116}" y="{legend_y-12}" width="14" height="14" fill="#9ca3af"/>',
        f'<text x="{width-94}" y="{legend_y}" class="small">library</text>',
        '</svg>',
    ])
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/v100_three_way_comparison.csv"))
    parser.add_argument("--output", type=Path, default=Path("figures/v100_library_comparison.svg"))
    args = parser.parse_args()
    ratios = load_ratios(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(ratios), encoding="utf-8")
    print(f"wrote {args.output} ({', '.join(f'{key}={value:.3f}x' for key, value in ratios.items())})")


if __name__ == "__main__":
    main()
