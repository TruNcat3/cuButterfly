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
            key = (int(row["word_bits"]), int(row["batch"]),
                   row["variant"], row["scan_roles"])
            samples[key].append(float(row["kernel_ms"]))

    rows = []
    for (bits, batch, variant, roles), values in samples.items():
        rows.append({
            "word_bits": bits, "batch": batch, "variant": variant,
            "roles": roles, "kernel_ms": statistics.median(values),
            "trials": len(values),
        })
    baselines = {
        (row["word_bits"], row["batch"], row["variant"]): row["kernel_ms"]
        for row in rows if row["variant"] != "resident-2d"
    }
    for row in rows:
        key = (row["word_bits"], row["batch"])
        row["throughput_vs_writer_final"] = (
            baselines[key + ("writer-final",)] / row["kernel_ms"])
        row["throughput_vs_v06"] = (
            baselines[key + ("v06",)] / row["kernel_ms"])
    rows.sort(key=lambda row: (row["word_bits"], row["batch"],
                               row["variant"], row["roles"]))

    best = {}
    for row in rows:
        if row["variant"] != "resident-2d":
            continue
        key = (row["word_bits"], row["batch"])
        if key not in best or row["kernel_ms"] < best[key]["kernel_ms"]:
            best[key] = row

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["word_bits", "batch", "variant", "roles", "kernel_ms",
              "trials", "throughput_vs_writer_final", "throughput_vs_v06"]
    with args.csv.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# APPT Resident 2D Screen", "",
        "Median CUDA-event time. The resident core fixes `a`, executes the first two 7-stage dimensions in one 128x128 subgraph, then streams stage-13 state to the existing register tail and natural writer.",
        "",
        "| bits | batch | best P/T/W | resident ms | throughput/writer-final | throughput/v0.6 |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for key in sorted(best):
        row = best[key]
        lines.append(
            f"| {row['word_bits']} | {row['batch']} | `{row['roles']}` | "
            f"{row['kernel_ms']:.6f} | "
            f"{row['throughput_vs_writer_final']:.3f}x | "
            f"{row['throughput_vs_v06']:.3f}x |")
    lines.extend([
        "", "## Interpretation", "",
        "This is the first mathematically dependency-closed resident 2D point. A fixed producer `c` tile is not closed because the online `[b][c][a] -> [c][d][a]` reorder makes one tail tile consume columns from 128 producer tiles.",
        "",
        "The implementation uses 96 KiB dynamic shared memory, so every physical role in the unified cooperative kernel is limited to one CTA/SM. Event timing determines whether subgraph locality compensates for that loss of role concurrency.",
    ])
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
