#!/usr/bin/env python3
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Summarize repeated HybridDataflow Td confirmation scans")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    samples = defaultdict(list)
    for path in args.inputs:
        with path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        if int(row["correct"]) != 1:
            raise ValueError(f"incorrect confirmation point: {path}")
        key = (int(row["word_bits"]), int(row["logN"]), int(row["batch"]),
               int(row["role_stages"]), int(row["data_time"]))
        samples[key].append(float(row["kernel_ms"]))

    rows = []
    for key, values in sorted(samples.items()):
        if len(values) != 3:
            raise ValueError(f"expected three trials for {key}, got {len(values)}")
        median = statistics.median(values)
        rows.append({
            "word_bits": key[0], "logN": key[1], "batch": key[2], "role_stages": key[3],
            "data_time": key[4], "median_ms": median, "min_ms": min(values), "max_ms": max(values),
            "cv_pct": 100.0 * statistics.pstdev(values) / statistics.mean(values),
        })

    groups = defaultdict(list)
    for row in rows:
        groups[(row["word_bits"], row["logN"], row["batch"])].append(row)
    for group in groups.values():
        best = min(row["median_ms"] for row in group)
        for row in group:
            row["regret"] = row["median_ms"] / best

    fields = ["word_bits", "logN", "batch", "role_stages", "data_time", "median_ms",
              "min_ms", "max_ms", "cv_pct", "regret"]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# HybridDataflow repeated Td confirmation", "",
             "Each candidate has three trials of 30 warmups and 500 timed repetitions.", "",
             "| bits | logN | batch | best Td | median ms | <=2% candidates | winner CV |",
             "|---:|---:|---:|---:|---:|---|---:|"]
    for key in sorted(groups):
        group = groups[key]
        best = min(group, key=lambda row: row["median_ms"])
        plateau = ",".join(str(row["data_time"]) for row in group if row["regret"] <= 1.02)
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {best['data_time']} | "
                     f"{best['median_ms']:.6f} | {plateau} | {best['cv_pct']:.2f}% |")
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
