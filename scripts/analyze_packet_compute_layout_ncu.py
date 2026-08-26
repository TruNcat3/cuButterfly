#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


BATCHES = (16, 32, 64)
MODES = ("aggregate", "bitmap_interleaved", "bitmap_warp_rows")
METRICS = (
    ("time_us", "time (us)"),
    ("warp_instructions", "warp instructions"),
    ("cbu_warp_instructions", "CBU instructions"),
    ("lsu_warp_instructions", "LSU instructions"),
    ("global_load_sectors", "global-load sectors"),
    ("shared_load_bank_conflicts", "shared-load conflicts"),
    ("shared_store_bank_conflicts", "shared-store conflicts"),
    ("barrier_stall_pct", "barrier stall (%)"),
    ("long_scoreboard_stall_pct", "long scoreboard (%)"),
    ("short_scoreboard_stall_pct", "short scoreboard (%)"),
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
    return f"{value:,.0f}" if abs(value) >= 100000 else f"{value:.3f}"


def report(rows):
    lines = ["# Packet Compute Layout Fixed-Clock Attribution", ""]
    for batch in BATCHES:
        labels = [f"{mode}_b{batch}" for mode in MODES]
        lines.extend([
            f"## Batch {batch}", "",
            "| metric | aggregate | bitmap interleaved | bitmap warp-row |",
            "|:--|--:|--:|--:|",
        ])
        for field, title in METRICS:
            lines.append(
                f"| {title} | "
                + " | ".join(display(number(rows[label], field)) for label in labels)
                + " |"
            )
        interleaved = rows[labels[1]]
        warp_rows = rows[labels[2]]
        deltas = []
        for field, title in (
            ("time_us", "time"),
            ("warp_instructions", "instructions"),
            ("global_load_sectors", "load sectors"),
            ("shared_load_bank_conflicts", "shared-load conflicts"),
            ("cbu_warp_instructions", "CBU instructions"),
            ("lsu_warp_instructions", "LSU instructions"),
        ):
            before = number(interleaved, field)
            after = number(warp_rows, field)
            if before not in (None, 0.0) and after is not None:
                deltas.append(f"{title}={after / before:.3f}x")
        lines.extend(["", "Warp-row/interleaved: " + "; ".join(deltas) + ".", ""])
    lines.append(
        "CUDA-event timing remains the ranking authority. Promote warp-row only "
        "if its predicted conflict/sector reduction is visible and is not "
        "offset by instruction or scoreboard growth."
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
