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
                   int(row["a_group"]), row["roles"])
            samples[key].append(float(row["kernel_ms"]))

    rows = []
    for (bits, batch, variant, group, roles), values in samples.items():
        rows.append({
            "word_bits": bits,
            "batch": batch,
            "variant": variant,
            "a_group": group,
            "roles": roles,
            "kernel_ms": statistics.median(values),
            "trials": len(values),
        })
    rows.sort(key=lambda row: (row["word_bits"], row["batch"],
                               row["variant"], row["a_group"], row["roles"]))
    baselines = {(row["word_bits"], row["batch"], row["variant"]): row["kernel_ms"]
                 for row in rows if row["variant"] != "grouped"}
    for row in rows:
        key = (row["word_bits"], row["batch"])
        row["throughput_vs_v06"] = (
            baselines[key + ("v06",)] / row["kernel_ms"])
        row["throughput_vs_radix4"] = (
            baselines[key + ("radix4-matched",)] / row["kernel_ms"])

    best = {}
    for row in rows:
        if row["variant"] != "grouped":
            continue
        key = (row["word_bits"], row["batch"], row["a_group"])
        if key not in best or row["kernel_ms"] < best[key]["kernel_ms"]:
            best[key] = row

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["word_bits", "batch", "variant", "a_group", "roles",
              "kernel_ms", "trials", "throughput_vs_v06",
              "throughput_vs_radix4"]
    with args.csv.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# APPT Grouped-Producer Screen",
        "",
        "> **Superseded correctness-negative capture.** The grouped producer published `[c][b][a]` while the tail consumed `[b][c][a]`. These timings are retained for diagnosis only.",
        "",
        "Median CUDA-event time. Each grouped row is the best role mapping for that `a`-space group.",
        "",
        "| bits | batch | a group | best roles (P/T/W) | kernel ms | throughput/v0.6 | throughput/radix4 |",
        "|---:|---:|---:|---|---:|---:|---:|",
    ]
    for key in sorted(best):
        row = best[key]
        lines.append(
            f"| {row['word_bits']} | {row['batch']} | {row['a_group']} | "
            f"`{row['roles']}` | {row['kernel_ms']:.6f} | "
            f"{row['throughput_vs_v06']:.3f}x | "
            f"{row['throughput_vs_radix4']:.3f}x |")
    lines.extend([
        "",
        "The grouped core changes the physical request layout only: the logical `7+7+6` APPT graph and butterfly arithmetic remain fixed.",
    ])
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
