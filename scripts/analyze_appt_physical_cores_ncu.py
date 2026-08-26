#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def number(row, field):
    value = row.get(field, "")
    return float(value) if value else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(newline="") as source:
        rows = {row["label"]: row for row in csv.DictReader(source)}

    lines = [
        "# APPT Physical-Core NCU Attribution",
        "",
        "V100 base-clock kernel replay. Load/store ratios are normalized per transform and use v0.6 as the denominator.",
        "",
        "| bits | batch | variant | us | throughput/v0.6 | load/v0.6 | store/v0.6 | L1 hit | L2 hit | active warps | barrier | scoreboard | MIO | regs |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bits in (32, 64):
        for batch in (1, 4, 16):
            baseline = rows[f"v06_u{bits}_b{batch}"]
            baseline_time = number(baseline, "time_us")
            baseline_load = number(baseline, "global_load_sectors")
            baseline_store = number(baseline, "global_store_sectors")
            for variant in ("v06", "radix4_default", "radix4_matched", "radix8_matched"):
                row = rows[f"{variant}_u{bits}_b{batch}"]
                time = number(row, "time_us")
                lines.append(
                    f"| {bits} | {batch} | `{variant}` | {time:.1f} | "
                    f"{baseline_time / time:.3f}x | "
                    f"{number(row, 'global_load_sectors') / baseline_load:.3f}x | "
                    f"{number(row, 'global_store_sectors') / baseline_store:.3f}x | "
                    f"{number(row, 'l1_hit_pct'):.1f}% | {number(row, 'l2_hit_pct'):.1f}% | "
                    f"{number(row, 'active_warps_pct'):.1f}% | "
                    f"{number(row, 'barrier_stall_pct'):.1f}% | "
                    f"{number(row, 'long_scoreboard_stall_pct'):.1f}% | "
                    f"{number(row, 'mio_throttle_stall_pct'):.1f}% | "
                    f"{number(row, 'registers_per_thread'):.0f} |")

    lines.extend([
        "",
        "## Controlled Deltas",
        "",
        "| bits | batch | role rebalance | radix8/radix4 | radix8 load delta | radix8 warp-inst delta | radix8 integer-inst delta | scoreboard delta | MIO delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for bits in (32, 64):
        for batch in (1, 4, 16):
            default = rows[f"radix4_default_u{bits}_b{batch}"]
            radix4 = rows[f"radix4_matched_u{bits}_b{batch}"]
            radix8 = rows[f"radix8_matched_u{bits}_b{batch}"]
            def delta(field):
                base = number(radix4, field)
                return number(radix8, field) / base - 1.0 if base else 0.0
            lines.append(
                f"| {bits} | {batch} | "
                f"{number(default, 'time_us') / number(radix4, 'time_us'):.3f}x | "
                f"{number(radix4, 'time_us') / number(radix8, 'time_us'):.3f}x | "
                f"{100 * delta('global_load_sectors'):+.2f}% | "
                f"{100 * delta('warp_instructions'):+.2f}% | "
                f"{100 * delta('integer_thread_instructions'):+.2f}% | "
                f"{number(radix8, 'long_scoreboard_stall_pct') - number(radix4, 'long_scoreboard_stall_pct'):+.1f} pp | "
                f"{number(radix8, 'mio_throttle_stall_pct') - number(radix4, 'mio_throttle_stall_pct'):+.1f} pp |")

    lines.extend([
        "",
        "NCU replay time is diagnostic rather than the performance authority for this persistent cooperative pipeline. Use the matched CUDA-event table for final ranking.",
    ])
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
