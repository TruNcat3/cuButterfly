#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
from pathlib import Path


LABEL = re.compile(r"^p(\d+)-(\d+)-(\d+)_b(\d+)_t(\d+)$")


def summarize(rows):
    samples = {}
    correctness = {}
    for row in rows:
        match = LABEL.match(row["label"])
        if not match:
            raise ValueError(f"invalid three-level label: {row['label']}")
        partition = tuple(int(match.group(index)) for index in range(1, 4))
        batch = int(match.group(4))
        key = (batch, partition)
        samples.setdefault(key, []).append(float(row["kernel_ms"]))
        correctness.setdefault(key, []).append(int(row["correct"]))
    records = []
    for (batch, partition), values in samples.items():
        records.append({
            "batch": batch,
            "partition": "+".join(map(str, partition)),
            "trials": len(values),
            "kernel_ms": statistics.median(values),
            "min_kernel_ms": min(values),
            "max_kernel_ms": max(values),
            "correct": int(all(value == 1 for value in correctness[(batch, partition)])),
        })
    records.sort(key=lambda row: (row["batch"], row["kernel_ms"]))
    return records


def render(records):
    lines = [
        "# Parameterized Three-Level Partition Screen", "",
        "All points use the same uint32 arithmetic, 32-row resident radix-4 "
        "unit, two target CTAs/SM, and stage-proportional service weights. "
        "Only the position of the 6/7/8-stage subgraph changes.", "",
        "| batch | partition | kernel (ms) | relative to batch winner | correct |",
        "|---:|:--:|---:|---:|:--:|",
    ]
    winners = {}
    for row in records:
        winners.setdefault(row["batch"], row["kernel_ms"])
        winners[row["batch"]] = min(winners[row["batch"]], row["kernel_ms"])
    for row in records:
        ratio = row["kernel_ms"] / winners[row["batch"]]
        lines.append(
            f"| {row['batch']} | `{row['partition']}` | {row['kernel_ms']:.6f} | "
            f"{ratio:.3f}x | {'yes' if row['correct'] else 'no'} |"
        )
    lines.extend(["", "The winner is a partition-order result, not yet a "
                  "fully searched design point; CTA weights and row granularity "
                  "remain fixed in this screen.", ""])
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
        raise ValueError("no three-level records")
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    args.markdown.write_text(render(records))


if __name__ == "__main__":
    main()
