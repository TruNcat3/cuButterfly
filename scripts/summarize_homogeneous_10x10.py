#!/usr/bin/env python3
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    samples = defaultdict(list)
    with args.input.open(newline="") as source:
        for row in csv.DictReader(source):
            key = (int(row["word_bits"]), int(row["batch"]), row["variant"],
                   int(row["producer_dt"]), int(row["consumer_dt"]),
                   row["weights"], int(row["resident_ctas"]))
            samples[key].append(float(row["kernel_ms"]))

    rows = []
    for key, values in samples.items():
        bits, batch, variant, producer_dt, consumer_dt, weights, resident = key
        rows.append({
            "word_bits": bits, "batch": batch, "variant": variant,
            "producer_dt": producer_dt, "consumer_dt": consumer_dt,
            "weights": weights, "resident_ctas": resident,
            "kernel_ms": statistics.median(values), "trials": len(values),
        })

    baselines = {}
    for row in rows:
        if row["variant"] == "v06":
            key = (row["word_bits"], row["batch"])
            if key not in baselines or row["kernel_ms"] < baselines[key]:
                baselines[key] = row["kernel_ms"]
    for row in rows:
        row["throughput_vs_v06"] = (
            baselines[(row["word_bits"], row["batch"])] / row["kernel_ms"])
    rows.sort(key=lambda row: (row["word_bits"], row["batch"], row["variant"],
                               row["producer_dt"], row["consumer_dt"],
                               row["weights"]))

    best = {}
    equivalent = {}
    temporal = {}
    for row in rows:
        if row["variant"] != "homogeneous":
            continue
        key = (row["word_bits"], row["batch"])
        if key not in best or row["kernel_ms"] < best[key]["kernel_ms"]:
            best[key] = row
        if row["producer_dt"] == 1 and row["consumer_dt"] == 1:
            if key not in equivalent or row["kernel_ms"] < equivalent[key]["kernel_ms"]:
                equivalent[key] = row
        elif key not in temporal or row["kernel_ms"] < temporal[key]["kernel_ms"]:
            temporal[key] = row

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["word_bits", "batch", "variant", "producer_dt", "consumer_dt",
              "weights", "resident_ctas", "kernel_ms", "trials",
              "throughput_vs_v06"]
    with args.csv.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Homogeneous 10+10 Subgraph Screen", "",
        "Median CUDA-event time after warmup. v0.6 and the generated template use the same resident radix-4 arithmetic core, full-scratch boundary, and cooperative launch.",
        "",
        "| bits | batch | v0.6 ms | template D_t=1 ms | equivalence | best non-degenerate D_t | weights | temporal ms | temporal/v0.6 |",
        "|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for key in sorted(best):
        bits, batch = key
        candidate = temporal[key]
        control = equivalent[key]
        lines.append(
            f"| {bits} | {batch} | {baselines[key]:.6f} | "
            f"{control['kernel_ms']:.6f} | {control['throughput_vs_v06']:.3f}x | "
            f"`{candidate['producer_dt']}:{candidate['consumer_dt']}` | "
            f"`{candidate['weights']}` | {candidate['kernel_ms']:.6f} | "
            f"{candidate['throughput_vs_v06']:.3f}x |")

    ratios_by_bits = defaultdict(list)
    for key, row in temporal.items():
        ratios_by_bits[key[0]].append((key[1], row["throughput_vs_v06"]))
    lines.extend(["", "## Interpretation", ""])
    for bits in sorted(ratios_by_bits):
        ordered = sorted(ratios_by_bits[bits])
        trend = ", ".join(f"batch {batch}: {ratio:.3f}x"
                          for batch, ratio in ordered)
        lines.append(f"- {bits}-bit best/v0.6 by batch: {trend}.")
    lines.extend([
        "",
        "`D_t=1:1` isolates template/selector overhead. Non-degenerate rows test role-local temporal traversal; the GPU warp scheduler selects among ready resident warps but does not overlap load/compute/store inside one CTA subgraph.",
    ])
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
