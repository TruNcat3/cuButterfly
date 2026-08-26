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
                   int(row["fragment_width"]), int(row["writer_tiles"]),
                   row["roles"])
            samples[key].append(float(row["kernel_ms"]))

    rows = []
    for key, values in samples.items():
        bits, batch, variant, fragment, writer_tiles, roles = key
        rows.append({
            "word_bits": bits,
            "batch": batch,
            "variant": variant,
            "fragment_width": fragment,
            "writer_tiles": writer_tiles,
            "roles": roles,
            "kernel_ms": statistics.median(values),
            "trials": len(values),
        })
    baselines = {
        (row["word_bits"], row["batch"], row["variant"]): row["kernel_ms"]
        for row in rows if row["variant"] != "writer-final"
    }
    for row in rows:
        key = (row["word_bits"], row["batch"])
        row["throughput_vs_v06"] = baselines[key + ("v06",)] / row["kernel_ms"]
        row["throughput_vs_grouped"] = (
            baselines[key + ("grouped",)] / row["kernel_ms"])
    rows.sort(key=lambda row: (
        row["word_bits"], row["batch"], row["variant"],
        row["fragment_width"], row["writer_tiles"], row["roles"]))

    best_by_fragment = {}
    best_overall = {}
    for row in rows:
        if row["variant"] != "writer-final":
            continue
        fragment_key = (row["word_bits"], row["batch"],
                        row["fragment_width"])
        overall_key = (row["word_bits"], row["batch"])
        if (fragment_key not in best_by_fragment or
                row["kernel_ms"] < best_by_fragment[fragment_key]["kernel_ms"]):
            best_by_fragment[fragment_key] = row
        if (overall_key not in best_overall or
                row["kernel_ms"] < best_overall[overall_key]["kernel_ms"]):
            best_overall[overall_key] = row

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["word_bits", "batch", "variant", "fragment_width",
              "writer_tiles", "roles", "kernel_ms", "trials",
              "throughput_vs_v06", "throughput_vs_grouped"]
    with args.csv.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# APPT Writer-Final Screen",
        "",
        "Median CUDA-event time. The grouped baseline uses the corrected group-32 layout with the prior selected mapping.",
        "",
        "## Best Overall",
        "",
        "| bits | batch | fragment | writer tiles | roles (P/T/W) | kernel ms | throughput/v0.6 | throughput/grouped |",
        "|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for key in sorted(best_overall):
        row = best_overall[key]
        lines.append(
            f"| {row['word_bits']} | {row['batch']} | "
            f"{row['fragment_width']} | {row['writer_tiles']} | "
            f"`{row['roles']}` | {row['kernel_ms']:.6f} | "
            f"{row['throughput_vs_v06']:.3f}x | "
            f"{row['throughput_vs_grouped']:.3f}x |")
    lines.extend([
        "",
        "## Best Per Fragment",
        "",
        "| bits | batch | fragment | writer tiles | roles (P/T/W) | kernel ms | throughput/grouped |",
        "|---:|---:|---:|---:|---|---:|---:|",
    ])
    for key in sorted(best_by_fragment):
        row = best_by_fragment[key]
        lines.append(
            f"| {row['word_bits']} | {row['batch']} | "
            f"{row['fragment_width']} | {row['writer_tiles']} | "
            f"`{row['roles']}` | {row['kernel_ms']:.6f} | "
            f"{row['throughput_vs_grouped']:.3f}x |")
    lines.extend([
        "",
        "Writer-final preserves the `7+7+6` logical graph. It moves stage 19 into the existing static-to-natural writer so the final coefficient axis is lane-coalesced.",
    ])
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
