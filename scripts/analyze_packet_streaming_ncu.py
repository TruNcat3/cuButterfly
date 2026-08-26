#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


PAIRS = (("aggregate_b16", "online_b16"),
         ("aggregate_b32", "online_b32"),
         ("aggregate_b64", "online_b64"))
METRICS = (
    ("time_us", "time (us)"),
    ("dram_read_mib", "DRAM read (MiB)"),
    ("warp_instructions", "warp instructions"),
    ("cbu_warp_instructions", "CBU pipe instructions"),
    ("lsu_warp_instructions", "LSU pipe instructions"),
    ("global_warp_instructions", "global-memory instructions"),
    ("shared_warp_instructions", "shared-memory instructions"),
    ("global_load_sectors", "global-load sectors"),
    ("global_store_sectors", "global-store sectors"),
    ("shared_load_bank_conflicts", "shared-load bank conflicts"),
    ("shared_store_bank_conflicts", "shared-store bank conflicts"),
    ("active_warps_pct", "active warps (%)"),
    ("barrier_stall_pct", "barrier stall (%)"),
    ("long_scoreboard_stall_pct", "long scoreboard stall (%)"),
    ("mio_throttle_stall_pct", "MIO throttle stall (%)"),
    ("short_scoreboard_stall_pct", "short scoreboard stall (%)"),
    ("registers_per_thread", "registers/thread"),
    ("shared_mem_bytes", "shared bytes/CTA"),
)


def read_rows(path):
    with path.open(newline="") as handle:
        rows = {row["label"]: row for row in csv.DictReader(handle)}
    missing = [label for pair in PAIRS for label in pair if label not in rows]
    if missing:
        raise ValueError(f"missing NCU records: {', '.join(missing)}")
    return rows


def number(row, field):
    value = row.get(field, "")
    return None if value == "" else float(value)


def display(value):
    if value is None:
        return "-"
    if abs(value) >= 100000:
        return f"{value:,.0f}"
    return f"{value:.3f}"


def report(rows):
    labels = tuple(label for pair in PAIRS for label in pair)
    lines = [
        "# Packet Streaming Fixed-Clock Attribution", "",
        "| metric | aggregate b16 | online b16 | aggregate b32 | online b32 | aggregate b64 | online b64 |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for field, title in METRICS:
        lines.append("| " + title + " | " + " | ".join(
            display(number(rows[label], field)) for label in labels) + " |")
    lines.extend(["", "## Deltas", ""])
    for aggregate, online in PAIRS:
        batch = aggregate.removeprefix("aggregate_b")
        aggregate_time = number(rows[aggregate], "time_us")
        online_time = number(rows[online], "time_us")
        aggregate_inst = number(rows[aggregate], "warp_instructions")
        online_inst = number(rows[online], "warp_instructions")
        if None in (aggregate_time, online_time, aggregate_inst, online_inst):
            lines.append(f"Batch {batch}: capture incomplete.")
        else:
            detail = (
                f"Batch {batch}: online/aggregate time={online_time / aggregate_time:.3f}x; "
                f"instructions={online_inst / aggregate_inst:.3f}x"
            )
            for field, title in (
                ("global_load_sectors", "load sectors"),
                ("shared_load_bank_conflicts", "shared-load conflicts"),
                ("cbu_warp_instructions", "CBU instructions"),
                ("long_scoreboard_stall_pct", "long-scoreboard"),
            ):
                aggregate_value = number(rows[aggregate], field)
                online_value = number(rows[online], field)
                if aggregate_value not in (None, 0.0) and online_value is not None:
                    detail += f"; {title}={online_value / aggregate_value:.3f}x"
            lines.append(detail + ".")
    lines.append(
        "Interpret time together with barrier/CBU/LSU deltas: online publication is "
        "useful only when producer-consumer overlap repays its repeated readiness and "
        "wave synchronization cost.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    args = parser.parse_args()
    output = report(read_rows(args.summary))
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
