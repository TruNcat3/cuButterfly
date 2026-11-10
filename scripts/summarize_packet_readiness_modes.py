#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
from pathlib import Path


LABEL = re.compile(
    r"(aggregate|per_packet|wave_bitmap)_b(\d+)(?:_rw(\d+))?_t(\d+)$"
)


def summarize(rows):
    samples = {}
    for row in rows:
        match = LABEL.fullmatch(row["label"])
        if not match:
            continue
        mode, batch, ready_window, _ = match.groups()
        key = (int(batch), mode, int(ready_window or 0))
        samples.setdefault(key, []).append(float(row["kernel_ms"]))
    result = []
    for (batch, mode, ready_window), values in sorted(samples.items()):
        result.append({
            "batch": batch,
            "mode": mode,
            "ready_window": ready_window,
            "sleep_cycles": ready_window * 32,
            "kernel_ms": statistics.median(values),
            "trials": len(values),
        })
    if not result:
        raise ValueError("no packet readiness records found")
    for batch in sorted({row["batch"] for row in result}):
        modes = {row["mode"] for row in result if row["batch"] == batch}
        if not {"aggregate", "per_packet", "wave_bitmap"} <= modes:
            raise ValueError(f"incomplete batch {batch}")
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_markdown(rows):
    lines = [
        "# Packet Readiness Mode Screen", "",
        "CUDA-event medians from independent processes.", "",
        "| batch | mode | ready window | sleep cycles | kernel (ms) |",
        "|---:|:--|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch']} | {row['mode']} | {row['ready_window']} | "
            f"{row['sleep_cycles']} | {row['kernel_ms']:.6f} |"
        )
    lines.extend(["", "## Selected Bitmap Points", ""])
    ratios = []
    for batch in sorted({row["batch"] for row in rows}):
        batch_rows = [row for row in rows if row["batch"] == batch]
        aggregate = next(row for row in batch_rows if row["mode"] == "aggregate")
        per_packet = next(row for row in batch_rows if row["mode"] == "per_packet")
        bitmap = min(
            (row for row in batch_rows if row["mode"] == "wave_bitmap"),
            key=lambda row: row["kernel_ms"],
        )
        bitmap_over_packet = per_packet["kernel_ms"] / bitmap["kernel_ms"]
        bitmap_over_aggregate = aggregate["kernel_ms"] / bitmap["kernel_ms"]
        ratios.append(bitmap_over_packet)
        lines.append(
            f"Batch {batch}: bitmap ready_window={bitmap['ready_window']}, "
            f"bitmap/per-packet throughput={bitmap_over_packet:.3f}x, "
            f"bitmap/aggregate={bitmap_over_aggregate:.3f}x."
        )
    geomean = statistics.geometric_mean(ratios)
    lines.extend(["", f"Bitmap/per-packet geomean throughput: {geomean:.3f}x."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()
    with args.raw.open(newline="") as handle:
        rows = summarize(csv.DictReader(handle))
    write_csv(args.csv, rows)
    args.markdown.write_text(make_markdown(rows))


if __name__ == "__main__":
    main()
