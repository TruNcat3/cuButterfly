#!/usr/bin/env python3
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Summarize APPT tail-core timings")
    parser.add_argument("raw", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    groups = defaultdict(list)
    for row in csv.DictReader(args.raw.open(newline="")):
        groups[(int(row["word_bits"]), int(row["batch"]),
                row["physical_core"], row["weights"])].append(
                    float(row["kernel_ms"]))

    records = []
    baselines = {}
    for (bits, batch, core, weights), values in sorted(groups.items()):
        median_ms = statistics.median(values)
        record = {
            "word_bits": bits, "batch": batch, "physical_core": core,
            "weights": weights, "kernel_ms": median_ms,
            "min_ms": min(values), "trials": len(values),
        }
        records.append(record)
        if core == "v06":
            baselines[(bits, batch)] = median_ms
    for record in records:
        record["throughput_vs_v06"] = (
            baselines[(record["word_bits"], record["batch"])] /
            record["kernel_ms"])

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# APPT Tail Physical Cores", "",
        "V100 logN=20 timings at independently calibrated role weights.", "",
        "| bits | batch | physical core | weights | kernel ms | throughput/v0.6 |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['word_bits']} | {row['batch']} | `{row['physical_core']}` | "
            f"`{row['weights']}` | {row['kernel_ms']:.6f} | "
            f"{row['throughput_vs_v06']:.3f}x |")
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
