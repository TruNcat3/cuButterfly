#!/usr/bin/env python3
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    with args.raw.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["word_bits"]), int(row["batch"]), row["variant"])
            groups[key].append(float(row["kernel_ms"]))
    records = []
    baselines = {}
    for (bits, batch, variant), values in sorted(groups.items()):
        record = {"word_bits": bits, "batch": batch, "variant": variant,
                  "kernel_ms": statistics.median(values), "trials": len(values)}
        records.append(record)
        if variant == "v06": baselines[(bits, batch)] = record["kernel_ms"]
    for record in records:
        record["throughput_vs_v06"] = baselines[(record["word_bits"], record["batch"])] / record["kernel_ms"]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    best = {}
    for record in records:
        if record["variant"] == "v06": continue
        kind = "static" if record["variant"].startswith("static-") else "natural"
        key = (record["word_bits"], record["batch"], kind)
        if key not in best or record["kernel_ms"] < best[key]["kernel_ms"]:
            best[key] = record
    lines = ["# APPT Static Layout Search", "",
             "Median V100 logN=20 timings. Natural includes the online writer; static is the chaining ceiling.", "",
             "| bits | batch | contract | best point | kernel ms | throughput/v0.6 |",
             "|---:|---:|---|---|---:|---:|"]
    for (bits, batch, kind), row in sorted(best.items()):
        lines.append(f"| {bits} | {batch} | {kind} | `{row['variant']}` | {row['kernel_ms']:.6f} | {row['throughput_vs_v06']:.3f}x |")
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
