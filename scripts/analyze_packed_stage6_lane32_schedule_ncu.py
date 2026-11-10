#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


LABELS = ("v06", "d7", "lane32_84_76", "lane32_85_75",
          "lane32_86_74")
METRICS = (
    ("time_us", "time (us)"),
    ("global_load_sectors", "global-load sectors"),
    ("dram_read_mib", "DRAM read (MiB)"),
    ("l2_hit_pct", "L2 hit (%)"),
    ("warp_instructions", "warp instructions"),
    ("long_scoreboard_stall_pct", "long scoreboard stall (%)"),
    ("active_warps_pct", "active warps (%)"),
    ("registers_per_thread", "registers/thread"),
)


def read_summary(path):
    with path.open(newline="") as source:
        rows = {row["label"]: row for row in csv.DictReader(source)}
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
    candidates = LABELS[2:]
    complete = all(number(rows[label], "time_us") is not None
                   for label in LABELS)
    lines = [
        "# Packed Lane32 Fixed-Clock Schedule Confirmation", "",
        "| metric | v0.6 | d7 85:75 | lane32 84:76 | lane32 85:75 | lane32 86:74 |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for field, title in METRICS:
        values = [number(rows[label], field) for label in LABELS]
        lines.append("| " + title + " | " + " | ".join(
            formatted(value) for value in values) + " |")
    lines.extend(["", "## Decision", ""])
    if not complete:
        lines.append("The capture is incomplete; do not update the selector.")
        return "\n".join(lines) + "\n"
    best_label = min(candidates, key=lambda label: number(rows[label], "time_us"))
    best_time = number(rows[best_label], "time_us")
    v06_time = number(rows["v06"], "time_us")
    d7_time = number(rows["d7"], "time_us")
    weights = best_label.removeprefix("lane32_").replace("_", ":")
    if best_time < v06_time:
        lines.append(
            "The tuned lane32 point exceeds both fixed-clock controls. It is eligible for "
            "the V100 logN20/batch16 static selector entry.")
    elif best_time < d7_time:
        lines.append(
            "The tuned lane32 point is the best v0.7 core but remains behind v0.6; keep "
            "v0.6 as the public default while retaining lane32 for schedule research.")
    else:
        lines.append(
            "The tuned lane32 point does not beat d7 under fixed clock; do not promote it.")
    lines.append(
        f"Best lane32 weights={weights}, time={best_time:.3f} us, "
        f"throughput/v0.6={v06_time / best_time:.3f}x, "
        f"throughput/d7={d7_time / best_time:.3f}x.")
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
