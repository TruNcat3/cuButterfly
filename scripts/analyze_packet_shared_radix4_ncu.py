#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


LABELS = ("v06", "d7", "lane32", "packet128")
METRICS = (
    ("time_us", "time (us)"),
    ("warp_instructions", "warp instructions"),
    ("adu_warp_instructions", "ADU pipe instructions"),
    ("alu_warp_instructions", "ALU pipe instructions"),
    ("cbu_warp_instructions", "CBU pipe instructions"),
    ("lsu_warp_instructions", "LSU pipe instructions"),
    ("xu_warp_instructions", "XU pipe instructions"),
    ("global_warp_instructions", "global-memory instructions"),
    ("shared_warp_instructions", "shared-memory instructions"),
    ("global_load_sectors", "global-load sectors"),
    ("active_warps_pct", "active warps (%)"),
    ("barrier_stall_pct", "barrier stall (%)"),
    ("long_scoreboard_stall_pct", "long scoreboard stall (%)"),
    ("mio_throttle_stall_pct", "MIO throttle stall (%)"),
    ("registers_per_thread", "registers/thread"),
    ("shared_mem_bytes", "shared bytes/CTA"),
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


def formatted(value):
    if value is None:
        return "-"
    if abs(value) >= 100000:
        return f"{value:,.0f}"
    return f"{value:.3f}"


def make_report(rows):
    lines = [
        "# Packet-Shared Radix-4 Fixed-Clock Confirmation", "",
        "| metric | v0.6 | d7 | lane32 | packet128 |",
        "|:--|--:|--:|--:|--:|",
    ]
    for field, title in METRICS:
        lines.append("| " + title + " | " + " | ".join(
            formatted(number(rows[label], field)) for label in LABELS) + " |")
    lines.extend(["", "## Decision", ""])
    times = {label: number(rows[label], "time_us") for label in LABELS}
    instructions = {label: number(rows[label], "warp_instructions")
                    for label in LABELS}
    if any(value is None for value in (*times.values(), *instructions.values())):
        lines.append("The capture is incomplete; do not update the selector.")
        return "\n".join(lines) + "\n"
    packet_time = times["packet128"]
    packet_instructions = instructions["packet128"]
    lines.append(
        f"Packet128 throughput/v0.6={times['v06'] / packet_time:.3f}x, "
        f"throughput/lane32={times['lane32'] / packet_time:.3f}x. "
        f"Its dynamic instruction count is {packet_instructions / instructions['v06']:.3f}x "
        f"v0.6 and {packet_instructions / instructions['lane32']:.3f}x lane32.")
    if packet_time < times["v06"] and packet_instructions <= instructions["v06"] * 1.05:
        lines.append(
            "The packet core closes the physical-codelet instruction gap and beats the "
            "fixed-clock v0.6 control. It is eligible for broader batch/length screening.")
    elif packet_time < times["lane32"]:
        lines.append(
            "The packet core improves v0.7 but does not yet close v0.6; retain it as the "
            "next optimization base without changing the public selector.")
    else:
        lines.append(
            "The packet core does not improve lane32 under fixed clock; do not promote it.")
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
