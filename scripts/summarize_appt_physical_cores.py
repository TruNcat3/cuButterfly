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

    groups = defaultdict(list)
    with args.input.open(newline="") as source:
        for row in csv.DictReader(source):
            key = (int(row["word_bits"]), int(row["batch"]), row["variant"])
            groups[key].append(float(row["kernel_ms"]))

    records = []
    for (bits, batch, variant), samples in groups.items():
        records.append({
            "word_bits": bits,
            "batch": batch,
            "variant": variant,
            "kernel_ms": statistics.median(samples),
            "trials": len(samples),
        })
    records.sort(key=lambda row: (row["word_bits"], row["batch"], row["variant"]))
    v06 = {(row["word_bits"], row["batch"]): row["kernel_ms"]
           for row in records if row["variant"] == "v06"}
    radix4 = {(row["word_bits"], row["batch"]): row["kernel_ms"]
              for row in records if row["variant"] == "radix4-matched"}
    for row in records:
        key = (row["word_bits"], row["batch"])
        row["throughput_vs_v06"] = v06[key] / row["kernel_ms"]
        row["throughput_vs_matched_radix4"] = (
            radix4[key] / row["kernel_ms"] if row["variant"] == "radix8-matched" else 0.0)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["word_bits", "batch", "variant", "kernel_ms", "trials",
              "throughput_vs_v06", "throughput_vs_matched_radix4"]
    with args.csv.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# APPT Physical-Core Comparison",
        "",
        "Median CUDA-event time. `radix4-matched` and `radix8-matched` use identical role and layout mappings.",
        "",
        "| bits | batch | variant | kernel ms | throughput/v0.6 | radix8/radix4 |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for row in records:
        relative = (f"{row['throughput_vs_matched_radix4']:.3f}x"
                    if row["variant"] == "radix8-matched" else "-")
        lines.append(
            f"| {row['word_bits']} | {row['batch']} | `{row['variant']}` | "
            f"{row['kernel_ms']:.6f} | {row['throughput_vs_v06']:.3f}x | {relative} |")
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
