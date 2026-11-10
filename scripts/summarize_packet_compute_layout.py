#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
from pathlib import Path


LABEL = re.compile(r"(.+)_b(\d+)_t(\d+)$")
MODES = (
    "aggregate",
    "per_packet_interleaved_rows",
    "per_packet_warp_rows",
    "wave_bitmap_interleaved_rows",
    "wave_bitmap_warp_rows",
)


def summarize(rows):
    samples = {}
    for row in rows:
        match = LABEL.fullmatch(row["label"])
        if not match or match.group(1) not in MODES:
            continue
        mode, batch, _ = match.groups()
        samples.setdefault((int(batch), mode), []).append(float(row["kernel_ms"]))
    result = []
    for (batch, mode), values in sorted(samples.items()):
        result.append({
            "batch": batch,
            "mode": mode,
            "kernel_ms": statistics.median(values),
            "trials": len(values),
        })
    for batch in sorted({row["batch"] for row in result}):
        present = {row["mode"] for row in result if row["batch"] == batch}
        if set(MODES) != present:
            raise ValueError(f"incomplete batch {batch}")
    if not result:
        raise ValueError("no packet compute-layout records found")
    return result


def make_markdown(rows):
    lines = [
        "# Packet Compute Layout Screen", "",
        "CUDA-event medians from independent processes.", "",
        "| batch | mode | kernel (ms) |",
        "|---:|:--|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['batch']} | {row['mode']} | {row['kernel_ms']:.6f} |")
    lines.extend(["", "## Deltas", ""])
    for batch in sorted({row["batch"] for row in rows}):
        points = {row["mode"]: row["kernel_ms"] for row in rows if row["batch"] == batch}
        packet_gain = points["per_packet_interleaved_rows"] / points["per_packet_warp_rows"]
        bitmap_gain = points["wave_bitmap_interleaved_rows"] / points["wave_bitmap_warp_rows"]
        best = min(points, key=points.get)
        aggregate_ratio = points["aggregate"] / points[best]
        lines.append(
            f"Batch {batch}: warp-row per-packet={packet_gain:.3f}x; "
            f"warp-row bitmap={bitmap_gain:.3f}x; best={best}; "
            f"best/aggregate throughput={aggregate_ratio:.3f}x."
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()
    with args.raw.open(newline="") as handle:
        rows = summarize(csv.DictReader(handle))
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    args.markdown.write_text(make_markdown(rows))


if __name__ == "__main__":
    main()
