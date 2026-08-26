#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def number(row, key):
    value = row.get(key, "")
    return float(value) if value else 0.0


def ratio(row, baseline, key):
    denominator = number(baseline, key)
    return number(row, key) / denominator if denominator else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Compare APPT physical cores with the v0.6 NTT kernel")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    rows = {row["label"]: row for row in csv.DictReader(args.summary.open())}
    records = []
    for bits in (32, 64):
        for batch in (1, 4):
            v06 = rows[f"v06_u{bits}_b{batch}"]
            warp = rows[f"warp_u{bits}_b{batch}"]
            cta = rows[f"cta-radix4_u{bits}_b{batch}"]
            for core, row in (("warp-radix2", warp), ("cta-radix4", cta)):
                records.append({
                    "word_bits": bits,
                    "batch": batch,
                    "core": core,
                    "time_us": number(row, "time_us"),
                    "time_vs_v06": ratio(row, v06, "time_us"),
                    "dram_read_vs_v06": ratio(row, v06, "dram_read_mib"),
                    "dram_write_vs_v06": ratio(row, v06, "dram_write_mib"),
                    "load_sectors_vs_v06": ratio(row, v06, "global_load_sectors"),
                    "store_sectors_vs_v06": ratio(row, v06, "global_store_sectors"),
                    "warp_instructions_vs_v06": ratio(row, v06, "warp_instructions"),
                    "integer_instructions_vs_v06": ratio(
                        row, v06, "integer_thread_instructions"),
                    "active_warps_pct": number(row, "active_warps_pct"),
                    "barrier_stall_pct": number(row, "barrier_stall_pct"),
                    "long_scoreboard_stall_pct": number(
                        row, "long_scoreboard_stall_pct"),
                    "dram_peak_pct": number(row, "dram_peak_pct"),
                    "l1_hit_pct": number(row, "l1_hit_pct"),
                    "l2_hit_pct": number(row, "l2_hit_pct"),
                    "registers_per_thread": number(row, "registers_per_thread"),
                    "v06_time_us": number(v06, "time_us"),
                    "v06_active_warps_pct": number(v06, "active_warps_pct"),
                    "v06_barrier_stall_pct": number(v06, "barrier_stall_pct"),
                    "v06_dram_peak_pct": number(v06, "dram_peak_pct"),
                    "v06_l1_hit_pct": number(v06, "l1_hit_pct"),
                })

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# APPT Physical-Core NCU Attribution", "",
        "Matched V100 base-clock NCU replay. Ratios below use the v0.6 resident 10+10 radix-4 kernel at the same precision and batch.", "",
        "| bits | batch | core | time us | time/v0.6 | DRAM R/W | load/store sectors | warp/int inst | active warps | barrier stall | DRAM peak | L1 hit |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['word_bits']} | {row['batch']} | {row['core']} | "
            f"{row['time_us']:.1f} | {row['time_vs_v06']:.2f}x | "
            f"{row['dram_read_vs_v06']:.2f}/{row['dram_write_vs_v06']:.2f}x | "
            f"{row['load_sectors_vs_v06']:.2f}/{row['store_sectors_vs_v06']:.2f}x | "
            f"{row['warp_instructions_vs_v06']:.2f}/{row['integer_instructions_vs_v06']:.2f}x | "
            f"{row['active_warps_pct']:.1f}% | {row['barrier_stall_pct']:.1f}% | "
            f"{row['dram_peak_pct']:.1f}% | {row['l1_hit_pct']:.1f}% |")

    lines += [
        "", "## Findings", "",
        "1. The uint64 CTA radix-4 core closes the processing-unit instruction gap: it executes 0.83x/0.93x the v0.6 warp instructions and 0.95x/0.97x the integer thread instructions at batch 1/4. Arithmetic instruction count is no longer the reason for its 1.94x/2.29x runtime gap.",
        "2. The 7+7+6 online graph materializes two intermediate states, while v0.6 10+10 materializes one. The minimum state traffic therefore grows from two to three read/write passes. Measured DRAM writes grow 1.56x--1.87x, close to or above that 1.5x structural floor.",
        "3. Online layouts amplify request traffic beyond the extra pass: CTA radix-4 load sectors are 2.10x--3.13x and store sectors 4.25x--4.50x v0.6. High L2 hit rates hide much of this from DRAM, but not from L1/L2 request handling and dependency latency.",
        "4. CTA radix-4 barrier stalls are 25.7%--46.3%, versus 2.8%--5.1% for v0.6. This combines shared-core synchronization with the current one-thread readiness wait followed by a CTA-wide barrier. Static roles leave the other warps unable to execute ready work.",
        "5. APPT reaches only 13.0%--31.9% of sustained DRAM throughput, below the matched v0.6 points. The kernel is latency, synchronization, and issue limited; it is not saturating memory bandwidth.",
        "6. Shared bank conflicts are not the primary regression: CTA radix-4 shared-load conflicts are only about 1.2x--1.3x v0.6 and shared-store conflicts are lower. Removing bank conflicts alone cannot explain a roughly 2x runtime gap.",
        "", "The stall percentages are scheduler-state samples and are not additive time fractions. They identify bottlenecks but must not be summed into a percentage attribution.",
    ]
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
