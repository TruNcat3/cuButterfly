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
    reference = {
        (row["word_bits"], row["batch"], row["variant"]): row["kernel_ms"]
        for row in rows if row["variant"] != "resident-quarter"
    }
    for row in rows:
        key = (row["word_bits"], row["batch"])
        row["throughput_vs_resident_2d"] = (
            reference[key + ("resident-2d",)] / row["kernel_ms"])
        row["throughput_vs_writer_final"] = (
            reference[key + ("writer-final",)] / row["kernel_ms"])
        row["throughput_vs_v06"] = (
            reference[key + ("v06",)] / row["kernel_ms"])
    rows.sort(key=lambda row: (row["word_bits"], row["batch"],
                               row["variant"], row["roles"]))

    best = {}
    for row in rows:
        if row["variant"] != "resident-quarter":
            continue
        key = (row["word_bits"], row["batch"])
        if key not in best or row["kernel_ms"] < best[key]["kernel_ms"]:
            best[key] = row

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["word_bits", "batch", "variant", "roles", "kernel_ms",
              "trials", "throughput_vs_resident_2d",
              "throughput_vs_writer_final", "throughput_vs_v06"]
    with args.csv.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# APPT Resident Quarter Screen", "",
        "Median CUDA-event time. The quarter core executes four 32x128 tiles, joins quarter pairs at stage 12, and joins the two halves at stage 13.",
        "",
        "| bits | batch | best P/T/W | quarter ms | throughput/96KiB | throughput/writer-final | throughput/v0.6 |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for key in sorted(best):
        row = best[key]
        lines.append(
            f"| {row['word_bits']} | {row['batch']} | `{row['roles']}` | "
            f"{row['kernel_ms']:.6f} | "
            f"{row['throughput_vs_resident_2d']:.3f}x | "
            f"{row['throughput_vs_writer_final']:.3f}x | "
            f"{row['throughput_vs_v06']:.3f}x |")
    lines.extend([
        "", "## Resource Interpretation", "",
        "The 48 KiB specialization restores two CTA/SM, but launch-bounds cap it at 128 registers/thread. The current compiler allocation spills retained state to the thread stack, so event timing must be read together with cuobjdump resource usage.",
    ])
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
