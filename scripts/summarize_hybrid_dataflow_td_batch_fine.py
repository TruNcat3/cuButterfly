#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def compress_regions(winners):
    regions = []
    start = previous = winners[0][0]
    td = winners[0][1]
    for batch, winner in winners[1:]:
        if winner != td:
            regions.append((start, previous, td))
            start, td = batch, winner
        previous = batch
    regions.append((start, previous, td))
    return ", ".join(f"{start}{'' if start == end else '-' + str(end)}:Td{td}"
                     for start, end, td in regions)


def main():
    parser = argparse.ArgumentParser(description="Summarize fine batch/Td boundary scans")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    groups = defaultdict(list)
    for path in args.inputs:
        with path.open(newline="") as handle:
            source = next(csv.DictReader(handle))
        if int(source["correct"]) != 1:
            raise ValueError(f"incorrect batch-boundary point: {path}")
        row = {"word_bits": int(source["word_bits"]), "logN": int(source["logN"]),
               "batch": int(source["batch"]), "data_time": int(source["data_time"]),
               "kernel_ms": float(source["kernel_ms"]), "case": path.stem}
        rows.append(row)
        groups[(row["word_bits"], row["logN"], row["batch"])].append(row)
    for group in groups.values():
        best = min(row["kernel_ms"] for row in group)
        for row in group:
            row["regret"] = row["kernel_ms"] / best
    rows.sort(key=lambda row: (row["word_bits"], row["logN"], row["batch"], row["data_time"]))

    fields = ["case", "word_bits", "logN", "batch", "data_time", "kernel_ms", "regret"]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    shape_groups = defaultdict(list)
    for key, group in groups.items():
        best = min(group, key=lambda row: row["kernel_ms"])
        plateau = ",".join(str(row["data_time"]) for row in group if row["regret"] <= 1.02)
        shape_groups[key[:2]].append((key[2], best["data_time"], best["kernel_ms"], plateau))
    lines = ["# HybridDataflow fine batch/Td boundaries", "",
             "| bits | logN | compressed winner regions |", "|---:|---:|---|"]
    for shape, values in sorted(shape_groups.items()):
        winners = [(batch, td) for batch, td, _, _ in sorted(values)]
        lines.append(f"| {shape[0]} | {shape[1]} | {compress_regions(winners)} |")
    lines += ["", "## Per-batch winner", "",
              "| bits | logN | batch | best Td | kernel ms | <=2% candidates |",
              "|---:|---:|---:|---:|---:|---|"]
    for shape, values in sorted(shape_groups.items()):
        for batch, td, kernel_ms, plateau in sorted(values):
            lines.append(f"| {shape[0]} | {shape[1]} | {batch} | {td} | {kernel_ms:.6f} | {plateau} |")
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
