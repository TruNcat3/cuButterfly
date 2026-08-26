#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


BATCHES = (16, 32, 64)
MODES = ("aggregate", "per_packet", "wave_bitmap")
METRICS = (
    ("time_us", "time (us)"),
    ("dram_read_mib", "DRAM read (MiB)"),
    ("warp_instructions", "warp instructions"),
    ("cbu_warp_instructions", "CBU instructions"),
    ("lsu_warp_instructions", "LSU instructions"),
    ("global_load_sectors", "global-load sectors"),
    ("global_store_sectors", "global-store sectors"),
    ("shared_load_bank_conflicts", "shared-load conflicts"),
    ("barrier_stall_pct", "barrier stall (%)"),
    ("long_scoreboard_stall_pct", "long scoreboard (%)"),
    ("registers_per_thread", "registers/thread"),
)


def read_rows(path):
    with path.open(newline="") as handle:
        rows = {row["label"]: row for row in csv.DictReader(handle)}
    expected = [f"{mode}_b{batch}" for batch in BATCHES for mode in MODES]
    missing = [label for label in expected if label not in rows]
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
    lines = ["# Packet Readiness Fixed-Clock Attribution", ""]
    for batch in BATCHES:
        labels = [f"{mode}_b{batch}" for mode in MODES]
        lines.extend([
            f"## Batch {batch}", "",
            "| metric | aggregate | per-packet | wave-bitmap |",
            "|:--|--:|--:|--:|",
        ])
        for field, title in METRICS:
            lines.append(
                f"| {title} | "
                + " | ".join(display(number(rows[label], field)) for label in labels)
                + " |"
            )
        aggregate = rows[labels[0]]
        per_packet = rows[labels[1]]
        bitmap = rows[labels[2]]
        deltas = []
        for field, title in (
            ("time_us", "time"),
            ("warp_instructions", "instructions"),
            ("global_load_sectors", "load sectors"),
            ("shared_load_bank_conflicts", "shared-load conflicts"),
            ("cbu_warp_instructions", "CBU instructions"),
        ):
            packet_value = number(per_packet, field)
            bitmap_value = number(bitmap, field)
            if packet_value not in (None, 0.0) and bitmap_value is not None:
                deltas.append(f"{title}={bitmap_value / packet_value:.3f}x")
        aggregate_time = number(aggregate, "time_us")
        bitmap_time = number(bitmap, "time_us")
        if aggregate_time not in (None, 0.0) and bitmap_time is not None:
            deltas.append(f"bitmap/aggregate time={bitmap_time / aggregate_time:.3f}x")
        lines.extend(["", "Bitmap/per-packet: " + "; ".join(deltas) + ".", ""])
    lines.append(
        "CUDA-event timing remains the ranking authority. This replay capture "
        "tests whether identity-preserving bitmap publication removes polling "
        "sectors without transferring the cost to CBU serialization."
    )
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
