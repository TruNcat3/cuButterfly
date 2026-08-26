#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
from pathlib import Path


LABEL = re.compile(
    r"^m(\d+)_g(\d+)_p([0-9-]+)_e([0-9-]+)_b(\d+)_t(\d+)$")


def summarize(rows):
    grouped = {}
    for row in rows:
        match = LABEL.match(row["label"])
        if not match:
            raise ValueError(f"invalid resident-group label: {row['label']}")
        m, g = int(match.group(1)), int(match.group(2))
        logical = match.group(3).replace('-', '+')
        execution = match.group(4).replace('-', '+')
        batch = int(match.group(5))
        grouped.setdefault((batch, m, g, logical, execution), []).append(
            float(row["kernel_ms"]))
    records = []
    for (batch, m, g, logical, execution), values in grouped.items():
        records.append({
            "batch": batch,
            "logical_subgraphs": m,
            "execution_groups": g,
            "logical_partition": logical,
            "execution_partition": execution,
            "materialized_boundaries": g - 1,
            "trials": len(values),
            "kernel_ms": statistics.median(values),
            "min_kernel_ms": min(values),
            "max_kernel_ms": max(values),
        })
    records.sort(key=lambda row: (
        row["batch"], row["logical_subgraphs"], row["execution_groups"]))
    return records


def render(records):
    full = {(row["batch"], row["logical_subgraphs"],
             row["logical_partition"]): row["kernel_ms"]
            for row in records
            if row["logical_subgraphs"] == row["execution_groups"]}
    lines = [
        "# Resident Execution-Group Screen", "",
        "M is the logical homogeneous-subgraph count. G is the physical "
        "execution-group count after resident boundary lowering. The arithmetic "
        "core is fixed. When present, a resident row has a matched G=M control with the "
        "same logical partition; different (M,G) cells may select different "
        "logical partitions.",
        "", "| batch | M | G | logical | execution | boundaries | kernel (ms) | vs G=M |",
        "|---:|---:|---:|:--:|:--:|---:|---:|---:|",
    ]
    for row in records:
        baseline = full.get((row["batch"], row["logical_subgraphs"],
                             row["logical_partition"]))
        relative = (f"{baseline / row['kernel_ms']:.3f}x"
                    if baseline is not None else "n/a")
        lines.append(
            f"| {row['batch']} | {row['logical_subgraphs']} | "
            f"{row['execution_groups']} | `{row['logical_partition']}` | "
            f"`{row['execution_partition']}` | "
            f"{row['materialized_boundaries']} | {row['kernel_ms']:.6f} | "
            f"{relative} |")
    equivalence = {}
    for row in records:
        key = (row["batch"], row["execution_groups"],
               row["execution_partition"])
        equivalence.setdefault(key, []).append(row["kernel_ms"])
    repeated = [(key, values) for key, values in equivalence.items()
                if len(values) > 1]
    if repeated:
        lines.extend(["", "## Physical-Equivalence Check", "",
                      "Rows below lower different logical M values to the same physical schedule.",
                      "", "| batch | G | execution | logical variants | min (ms) | max (ms) | spread |",
                      "|---:|---:|:--:|---:|---:|---:|---:|"])
        for (batch, g, execution), values in sorted(repeated):
            lines.append(
                f"| {batch} | {g} | `{execution}` | {len(values)} | "
                f"{min(values):.6f} | {max(values):.6f} | "
                f"{(max(values) / min(values) - 1.0) * 100.0:.2f}% |")
    lines.extend(["", "This screen measures boundary realization. It does not "
                  "rank logical M independently of the physical core.", ""])
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
        raise ValueError("no resident execution-group records")
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    args.markdown.write_text(render(records))


if __name__ == "__main__":
    main()
