#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


LABELS = ("v06", "vector_d6", "vector_d7")
ROWS_PER_RUN = 2 * 16 * 1024
PREDICTED_STAGE6_REDUCTION = 256 * ROWS_PER_RUN
METRICS = (
    ("time_us", "time (us)"),
    ("dram_read_mib", "DRAM read (MiB)"),
    ("global_load_sectors", "global-load sectors"),
    ("global_store_sectors", "global-store sectors"),
    ("warp_instructions", "warp instructions"),
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
    time_ratio = ratio(number(d7, "time_us"), number(d6, "time_us"))
    sector_ratio = ratio(number(d7, "global_load_sectors"),
                         number(d6, "global_load_sectors"))
    instruction_ratio = ratio(number(d7, "warp_instructions"),
                              number(d6, "warp_instructions"))
    v06_sectors = number(rows["v06"], "global_load_sectors")
    d6_sectors = number(d6, "global_load_sectors")
    d7_sectors = number(d7, "global_load_sectors")
    measured_reduction = (d6_sectors - d7_sectors
                          if None not in (d6_sectors, d7_sectors) else None)
    excess_removed = ratio(measured_reduction, d6_sectors - v06_sectors)
    model_closure = ratio(measured_reduction, PREDICTED_STAGE6_REDUCTION)

    lines = [
        "# Vector Radix-4 Stage-6 Distribution",
        "",
        "| metric | v0.6 | vector d6 | vector d7 | d7/d6 |",
        "|:--|--:|--:|--:|--:|",
    ]
    for field, title in METRICS:
        values = [number(rows[label], field) for label in LABELS]
        lines.append(
            f"| {title} | {formatted(values[0])} | {formatted(values[1])} | "
            f"{formatted(values[2])} | {formatted(ratio(values[2], values[1]))}x |")

    lines += ["", "## Stage-6 Model Closure", ""]
    if measured_reduction is not None:
        lines += [
            f"- measured d6 -> d7 reduction: {measured_reduction:,.0f} sectors",
            f"- even/odd dual-bank prediction: {PREDICTED_STAGE6_REDUCTION:,.0f} sectors",
            f"- measured/predicted: {model_closure:.3f}x",
            f"- d6 excess over v0.6 removed by d7: {100.0 * excess_removed:.1f}%",
        ]

    lines += ["", "## Decision", ""]
    if time_ratio is None or sector_ratio is None:
        lines.append("The capture is incomplete; do not select d7 from this run.")
    elif sector_ratio >= 0.95:
        lines.append(
            "d7 does not remove enough load sectors to validate the stage-6 "
            "distribution model. Inspect the generated SASS before tuning the schedule.")
    elif time_ratio <= 1.0:
        lines.append(
            "d7 reduces global-load sectors and kernel time. It is a viable physical-core "
            "candidate for the next schedule search.")
    else:
        lines.append(
            "d7 reduces global-load sectors but does not reduce kernel time. Keep d6 as "
            "the performance default. The next physical-core candidate should use an aligned, "
            "prepacked stage-6 coefficient table so each active lane can issue LDG.128 without "
            "the d7 shuffle path.")
    if instruction_ratio is not None:
        lines.append(
            f"The measured d7/d6 warp-instruction ratio is {instruction_ratio:.3f}x; "
            "use it with the stall deltas to distinguish instruction overhead from memory latency.")
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
