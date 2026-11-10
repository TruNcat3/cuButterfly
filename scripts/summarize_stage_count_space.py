#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
from pathlib import Path


LABEL = re.compile(r"^m(\d+)_p([0-9-]+)_b(\d+)_t(\d+)$")


def summarize(rows):
    grouped = {}
    for row in rows:
        match = LABEL.match(row["label"])
        if not match:
            raise ValueError(f"invalid stage-count label: {row['label']}")
        segments = int(match.group(1))
        partition = tuple(int(value) for value in match.group(2).split('-'))
        batch = int(match.group(3))
        if len(partition) != segments:
            raise ValueError("label segment count disagrees with partition")
        grouped.setdefault((batch, segments, partition), []).append(
            float(row["kernel_ms"]))
    records = []
    for (batch, segments, partition), values in grouped.items():
        records.append({
            "batch": batch,
            "segment_count": segments,
            "partition": "+".join(map(str, partition)),
            "trials": len(values),
            "kernel_ms": statistics.median(values),
            "min_kernel_ms": min(values),
            "max_kernel_ms": max(values),
        })
    records.sort(key=lambda row: (row["batch"], row["segment_count"]))
    return records


def render(records):
    best = {}
    for row in records:
        best[row["batch"]] = min(best.get(row["batch"], float("inf")),
                                 row["kernel_ms"])
    lines = [
        "# Stage-Count Design-Space Screen", "",
        "The generic radix-4 core, full-scratch boundaries, thread count, and "
        "CTA residency are held fixed. Each row is the static shortlist winner "
        "within one value of M; CUDA-event timing remains the ranking authority.",
        "", "| batch | M | partition | kernel (ms) | relative to batch winner |",
        "|---:|---:|:--:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['batch']} | {row['segment_count']} | `{row['partition']}` | "
            f"{row['kernel_ms']:.6f} | {row['kernel_ms'] / best[row['batch']]:.3f}x |"
        )
    lines.extend(["", "This is a full-scratch stress test: every logical edge "
                  "is materialized, so G=M. It does not establish that a "
                  "smaller logical M is preferable; resident lowering is "
                  "measured separately.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    with args.raw.open(newline="") as handle:
        records = summarize(csv.DictReader(handle))
    if not records:
        raise ValueError("no stage-count records")
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    args.markdown.write_text(render(records))


if __name__ == "__main__":
    main()
