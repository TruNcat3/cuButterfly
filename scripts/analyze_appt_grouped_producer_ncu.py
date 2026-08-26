#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def value(row, field):
    return float(row.get(field) or 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(newline="") as source:
        rows = {row["label"]: row for row in csv.DictReader(source)}

    lines = [
        "# APPT Grouped-Producer NCU Attribution",
        "",
        "> **Superseded correctness-negative capture.** The profiled producer state was `[c][b][a]` while the tail expected `[b][c][a]`. Recollect corrected selected points before using these counters.",
        "",
        "V100 base-clock kernel replay. Sector and instruction ratios use the matched register-tail radix-4 point as the denominator. The three grouped widths share one grouped-core role mapping at each numeric point; radix-4 retains its own calibrated mapping.",
        "",
        "| bits | batch | variant | load/radix4 | store/radix4 | L1 hit | L2 hit | warp-inst/radix4 | active warps | barrier | scoreboard | MIO | regs |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bits in (32, 64):
        for batch in (1, 4, 16):
            radix4 = rows[f"radix4_u{bits}_b{batch}"]
            for variant in ("v06", "radix4", "group8", "group16", "group32"):
                row = rows[f"{variant}_u{bits}_b{batch}"]
                lines.append(
                    f"| {bits} | {batch} | `{variant}` | "
                    f"{value(row, 'global_load_sectors') / value(radix4, 'global_load_sectors'):.3f}x | "
                    f"{value(row, 'global_store_sectors') / value(radix4, 'global_store_sectors'):.3f}x | "
                    f"{value(row, 'l1_hit_pct'):.1f}% | {value(row, 'l2_hit_pct'):.1f}% | "
                    f"{value(row, 'warp_instructions') / value(radix4, 'warp_instructions'):.3f}x | "
                    f"{value(row, 'active_warps_pct'):.1f}% | "
                    f"{value(row, 'barrier_stall_pct'):.1f}% | "
                    f"{value(row, 'long_scoreboard_stall_pct'):.1f}% | "
                    f"{value(row, 'mio_throttle_stall_pct'):.1f}% | "
                    f"{value(row, 'registers_per_thread'):.0f} |")
    lines.extend([
        "",
        "Use CUDA-event timing from `results/appt_grouped_producer/analysis.md` for ranking. This report tests whether grouped `a` loads reduce request sectors and restore L1 locality.",
    ])
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
