#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


LABELS = ("v06", "vector_d6", "vector_d7", "vector_packed_stage6",
          "vector_packed_stage6_distributed")
METRICS = (
    ("time_us", "time (us)"),
    ("dram_read_mib", "DRAM read (MiB)"),
    ("dram_write_mib", "DRAM write (MiB)"),
    ("global_load_sectors", "global-load sectors"),
    ("global_store_sectors", "global-store sectors"),
    ("l2_hit_pct", "L2 hit (%)"),
    ("l1_hit_pct", "L1/TEX hit (%)"),
    ("warp_instructions", "warp instructions"),
    ("local_load_sectors", "local-load sectors"),
    ("local_store_sectors", "local-store sectors"),
    ("registers_per_thread", "registers/thread"),
    ("active_warps_pct", "active warps (%)"),
    ("long_scoreboard_stall_pct", "long scoreboard stall (%)"),
    ("mio_throttle_stall_pct", "MIO throttle stall (%)"),
)


def read_summary(path):
    with path.open(newline="") as handle:
        rows = {row["label"]: row for row in csv.DictReader(handle)}
    missing = [label for label in LABELS if label not in rows]
    if missing:
        raise ValueError(f"missing NCU records: {', '.join(missing)}")
    return rows


def number(row, field):
    value = row.get(field, "")
    return None if value == "" else float(value)


def ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def formatted(value):
    if value is None:
        return "-"
    if abs(value) >= 100000:
        return f"{value:,.0f}"
    return f"{value:.3f}"


def make_report(rows):
    d6 = rows["vector_d6"]
    d7 = rows["vector_d7"]
    packed = rows["vector_packed_stage6"]
    distributed = rows["vector_packed_stage6_distributed"]
    packed_time = number(packed, "time_us")
    distributed_time = number(distributed, "time_us")
    d6_time = number(d6, "time_us")
    d7_time = number(d7, "time_us")
    packed_sectors = number(packed, "global_load_sectors")
    distributed_sectors = number(distributed, "global_load_sectors")
    d6_sectors = number(d6, "global_load_sectors")
    d7_sectors = number(d7, "global_load_sectors")
    packed_instructions = number(packed, "warp_instructions")
    distributed_instructions = number(distributed, "warp_instructions")
    d7_instructions = number(d7, "warp_instructions")

    lines = [
        "# Packed Stage-6 Vector Load",
        "",
        "| metric | v0.6 | vector d6 | vector d7 | packed vector16 | packed lane32 | lane32/vector16 | lane32/d7 |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for field, title in METRICS:
        values = [number(rows[label], field) for label in LABELS]
        lines.append(
            f"| {title} | {formatted(values[0])} | {formatted(values[1])} | "
            f"{formatted(values[2])} | {formatted(values[3])} | "
            f"{formatted(values[4])} | "
            f"{formatted(ratio(values[4], values[3]))}x | "
            f"{formatted(ratio(values[4], values[2]))}x |")

    lines += ["", "## Decision", ""]
    required = (packed_time, distributed_time, d6_time, d7_time,
                packed_sectors, distributed_sectors, d6_sectors, d7_sectors,
                packed_instructions, distributed_instructions, d7_instructions)
    if any(value is None for value in required):
        lines.append("The capture is incomplete; do not select the packed core.")
    elif distributed_sectors > d7_sectors * 1.05:
        lines.append(
            "The 32-lane packed table did not preserve d7's sector reduction. Inspect the "
            "generated LDG.64 addresses before changing the schedule.")
    elif distributed_time < min(packed_time, d7_time):
        lines.append(
            "The 32-lane packed core combines aligned sectors with recovered memory-level "
            "parallelism. Promote it to the next v0.7 schedule search while retaining the "
            "other physical cores as ablations.")
    elif distributed_time < packed_time:
        lines.append(
            "Distributing the packed pair across all lanes recovers part of the vector16 "
            "scoreboard loss, but the physical core still does not beat d7.")
    else:
        lines.append(
            "The 32-lane load does not recover the vector16 loss. The remaining issue is not "
            "lane-level memory parallelism; inspect cache residency and dependency placement.")
    if not any(value is None for value in required):
        lines.append(
            f"Lane32/vector16 time is {ratio(distributed_time, packed_time):.3f}x, "
            f"lane32/d7 time is {ratio(distributed_time, d7_time):.3f}x, and "
            f"lane32/d7 warp instructions are "
            f"{ratio(distributed_instructions, d7_instructions):.3f}x.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    args = parser.parse_args()
    report = make_report(read_summary(args.summary))
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
