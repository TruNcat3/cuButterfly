#!/usr/bin/env python3
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(input_path):
    samples = defaultdict(list)
    with input_path.open(newline="") as source:
        for row in csv.DictReader(source):
            key = (row["variant"], int(row["producer_dt"]),
                   int(row["consumer_dt"]), int(row["producer_blocks"]),
                   int(row["consumer_blocks"]))
            samples[key].append(float(row["kernel_ms"]))
    rows = []
    for key, values in samples.items():
        variant, producer_dt, consumer_dt, producer_blocks, consumer_blocks = key
        rows.append({
            "variant": variant,
            "producer_dt": producer_dt,
            "consumer_dt": consumer_dt,
            "producer_blocks": producer_blocks,
            "consumer_blocks": consumer_blocks,
            "kernel_ms": statistics.median(values),
            "trials": len(values),
            "relative_range": ((max(values) - min(values)) /
                               statistics.median(values)),
        })
    v06_ms = min(row["kernel_ms"] for row in rows if row["variant"] == "v06")
    d7_ms = min(row["kernel_ms"] for row in rows if row["variant"] == "d7")
    for row in rows:
        row["throughput_vs_v06"] = v06_ms / row["kernel_ms"]
        row["throughput_vs_d7"] = d7_ms / row["kernel_ms"]
    rows.sort(key=lambda row: (row["variant"], row["kernel_ms"]))
    return rows, v06_ms, d7_ms


def report(rows, v06_ms, d7_ms):
    candidates = sorted((row for row in rows if row["variant"] == "lane32"),
                        key=lambda row: row["kernel_ms"])
    best = candidates[0]
    v06 = next(row for row in rows if row["variant"] == "v06")
    d7 = next(row for row in rows if row["variant"] == "d7")
    controls_stable = max(v06["relative_range"], d7["relative_range"]) <= 0.03
    lines = [
        "# Packed Stage-6 Lane32 Schedule Search", "",
        f"v0.6 bracket median: {v06_ms:.6f} ms (range "
        f"{v06['relative_range']:.1%}); d7 control: {d7_ms:.6f} ms "
        f"(range {d7['relative_range']:.1%}).",
        "",
        "| rank | D_t producer:consumer | producer:consumer CTAs | kernel ms | vs v0.6 | vs d7 |",
        "|---:|:---:|:---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(candidates[:12], 1):
        lines.append(
            f"| {rank} | `{row['producer_dt']}:{row['consumer_dt']}` | "
            f"`{row['producer_blocks']}:{row['consumer_blocks']}` | "
            f"{row['kernel_ms']:.6f} | {row['throughput_vs_v06']:.3f}x | "
            f"{row['throughput_vs_d7']:.3f}x |")
    lines.extend(["", "## Decision", ""])
    if not controls_stable:
        lines.append(
            "Cross-core event controls exceed the 3% stability limit. Use this screen only "
            "to select the lane32-local schedule; fixed-clock NCU is required for v0.6/d7 "
            "claims.")
    elif best["kernel_ms"] < v06_ms:
        lines.append("The lane32 schedule screen exceeds the bracketed v0.6 control.")
    elif best["kernel_ms"] < d7_ms:
        lines.append(
            "The lane32 schedule remains the best v0.7 physical point, but does not yet "
            "close the v0.6 instruction/occupancy gap.")
    else:
        lines.append(
            "No lane32 schedule beats d7; keep the physical core experimental and inspect "
            "the role trace before expanding the search.")
    best_text = (
        f"Best point: D_t={best['producer_dt']}:{best['consumer_dt']}, "
        f"CTAs={best['producer_blocks']}:{best['consumer_blocks']}, "
        f"time={best['kernel_ms']:.6f} ms")
    if controls_stable:
        best_text += f", throughput/v0.6={best['throughput_vs_v06']:.3f}x"
    lines.append(best_text + ".")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    rows, v06_ms, d7_ms = summarize(args.input)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "producer_dt", "consumer_dt", "producer_blocks",
              "consumer_blocks", "kernel_ms", "trials", "relative_range",
              "throughput_vs_v06", "throughput_vs_d7"]
    with args.csv.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    args.markdown.write_text(report(rows, v06_ms, d7_ms))


if __name__ == "__main__":
    main()
