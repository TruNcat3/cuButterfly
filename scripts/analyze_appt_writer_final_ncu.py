#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def number(row, field):
    return float(row.get(field) or 0.0)


def ratio(row, baseline, field):
    denominator = number(baseline, field)
    return number(row, field) / denominator if denominator else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(newline="") as source:
        rows = {row["label"]: row for row in csv.DictReader(source)}

    lines = [
        "# APPT Writer-Final NCU Attribution",
        "",
        "V100 base-clock kernel replay. Ratios use the corrected grouped producer with its independently selected mapping.",
        "",
        "| bits | batch | variant | time us | load/grouped | store/grouped | DRAM read MiB | DRAM write MiB | L2 hit | warp-inst/grouped | integer/grouped | active warps | barrier | scoreboard | MIO | regs | waves/SM |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bits in (32, 64):
        for batch in (1, 4, 16):
            grouped = rows[f"grouped_u{bits}_b{batch}"]
            for variant in ("v06", "grouped", "writer_final"):
                row = rows[f"{variant}_u{bits}_b{batch}"]
                lines.append(
                    f"| {bits} | {batch} | `{variant}` | "
                    f"{number(row, 'time_us'):.1f} | "
                    f"{ratio(row, grouped, 'global_load_sectors'):.3f}x | "
                    f"{ratio(row, grouped, 'global_store_sectors'):.3f}x | "
                    f"{number(row, 'dram_read_mib'):.1f} | "
                    f"{number(row, 'dram_write_mib'):.1f} | "
                    f"{number(row, 'l2_hit_pct'):.1f}% | "
                    f"{ratio(row, grouped, 'warp_instructions'):.3f}x | "
                    f"{ratio(row, grouped, 'integer_thread_instructions'):.3f}x | "
                    f"{number(row, 'active_warps_pct'):.1f}% | "
                    f"{number(row, 'barrier_stall_pct'):.1f}% | "
                    f"{number(row, 'long_scoreboard_stall_pct'):.1f}% | "
                    f"{number(row, 'mio_throttle_stall_pct'):.1f}% | "
                    f"{number(row, 'registers_per_thread'):.0f} | "
                    f"{number(row, 'waves_per_sm'):.2f} |")
    lines.extend([
        "",
        "Writer-final reduces global-load sectors versus corrected grouped by 21.6%--23.4% for uint32 and 16.4%--17.3% for uint64. Warp and integer instruction counts change by at most 1.8%; uint64 remains at the same two-CTA/SM shared-memory limit and essentially the same 127/128 registers per thread. The numeric crossover is therefore not caused by extra butterfly work, register spill, or lower theoretical CTA residency.",
        "",
        "The saved sectors do not translate uniformly into DRAM traffic. Uint32 DRAM reads fall by 6.6%--7.6% despite a 3.2--5.4 point L2-hit loss, which is sufficient for the measured event gain. Uint64 L2 hit rate falls by 6.1--8.6 points and DRAM reads rise by 0.7%--6.4%; this cancels the sector reduction and explains its neutral event result. The transform-major writer alternates pre-final low/high streams separated by `N/2` and does not reuse stage-19 coefficients across batch transforms.",
        "",
        "Relative to v0.6, writer-final still issues about 1.50x/1.15x load sectors for uint32/uint64. Uint64 also retains exactly 1.50x store sectors because APPT materializes producer state, static tail publication, and natural output, whereas the `10+10` v0.6 path has two coalesced value stores. The next controlled mapping should traverse the batch data-time coordinate inside a fixed spatial subgraph so homogeneous transforms reuse coefficients and cache lines; another arithmetic-only radix change is not indicated.",
        "",
        "CUDA-event timing in `results/appt_writer_final/confirmed/analysis.md` remains the ranking authority. This report determines whether stage-19 writer fusion reduces requests and whether writer occupancy or arithmetic latency offsets that reduction.",
    ])
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
