#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pathlib
import statistics


ROLES = {
    "v0.6-fixed-baseline": "v06",
    "v0.6-fixed-hybrid2d-radix2": "v06",
    "v0.6-search-tile256": "v06_search",
    "v0.6-search-hybrid2d-radix4": "v06_search",
    "v0.7-base-hybrid-dataflow-Ur1-Td4": "v07",
    "v0.7-search-hybrid-dataflow-static-table": "v07_search",
}


def read_rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def build_rows(rows):
    groups = {}
    for row in rows:
        role = ROLES.get(row["implementation"])
        if role:
            groups.setdefault(row["group"], {})[role] = row
    output = []
    for group, entries in sorted(groups.items()):
        missing = set(ROLES.values()) - set(entries)
        if missing:
            raise ValueError(f"{group} is missing roles: {sorted(missing)}")
        times = {role: float(row["median_kernel_ms"]) for role, row in entries.items()}
        first = entries["v06"]
        output.append({
            "group": group,
            "precision": first["precision"],
            "logN": int(first["logN"]),
            "batch": int(first["batch"]),
            "v06_ms": times["v06"],
            "v06_search_ms": times["v06_search"],
            "v07_ms": times["v07"],
            "v07_search_ms": times["v07_search"],
            "v06_search_speedup": times["v06"] / times["v06_search"],
            "v07_search_speedup": times["v07"] / times["v07_search"],
            "v07_search_vs_v06_search": times["v06_search"] / times["v07_search"],
            "fastest": min(times, key=times.get),
            "stability": "/".join(entries[role]["stability_class"]
                                  for role in ("v06", "v06_search", "v07", "v07_search")),
            "correct": int(all(int(row["correct"]) == 1 for row in entries.values())),
        })
    return output


def geomean(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def aggregate(rows):
    grouped = {"overall": rows}
    for precision in sorted({row["precision"] for row in rows}):
        grouped[precision] = [row for row in rows if row["precision"] == precision]
    for precision, log_n in sorted({(row["precision"], row["logN"]) for row in rows}):
        grouped[f"{precision}/logN{log_n}"] = [
            row for row in rows if row["precision"] == precision and row["logN"] == log_n]
    output = []
    for numeric, samples in grouped.items():
        output.append({
            "numeric": numeric,
            "shapes": len(samples),
            "v06_search_geomean": geomean([row["v06_search_speedup"] for row in samples]),
            "v07_search_geomean": geomean([row["v07_search_speedup"] for row in samples]),
            "v07_search_vs_v06_search_geomean": geomean(
                [row["v07_search_vs_v06_search"] for row in samples]),
            "v07_search_wins": sum(row["v07_search_vs_v06_search"] > 1.03 for row in samples),
            "parity": sum(0.97 <= row["v07_search_vs_v06_search"] <= 1.03 for row in samples),
            "v06_search_wins": sum(row["v07_search_vs_v06_search"] < 0.97 for row in samples),
        })
    return output


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, aggregates):
    lines = [
        "# V100 v0.6/v0.7 NTT Comparison", "",
        "All paths implement the same forward, natural-order NTT contract. Each point uses 30 warmups,",
        "500 timed repetitions, three randomized process trials, CUDA-event kernel time, and correctness checks.",
        "Ratios above one favor the searched configuration named in the numerator.", "",
        "`v0.7-base` is the first generated HybridDataflow point (`Ur=1,Td=4`); `v0.7-search` uses",
        "the generated V100 static table. HybridDataflow is currently an on-chip whole-transform backend",
        "for `logN=10/12`, so this table does not directly join the archived GPU-NTT `logN=16/20` rows.",
        "At `logN=10`, the v0.6 pair is generic baseline/Tile256; at `logN=12`, it is Hybrid2D radix-2/radix-4.", "",
        "| Numeric | logN | Batch | v0.6 ms | v0.6 search ms | v0.7 ms | v0.7 search ms | v0.6 search/base | v0.7 search/base | v0.7 search / v0.6 search throughput | Fastest | Stability |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|:--|",
    ]
    for row in rows:
        lines.append(
            f"| {row['precision']} | {row['logN']} | {row['batch']:,} | {row['v06_ms']:.6f} | "
            f"{row['v06_search_ms']:.6f} | {row['v07_ms']:.6f} | {row['v07_search_ms']:.6f} | "
            f"{row['v06_search_speedup']:.3f}x | {row['v07_search_speedup']:.3f}x | "
            f"{row['v07_search_vs_v06_search']:.3f}x | {row['fastest']} | {row['stability']} |")
    lines += ["", "## Aggregate", "",
              "| Numeric | Shapes | v0.6 search/base | v0.7 search/base | v0.7 search / v0.6 search | v0.7 wins | Parity | v0.6 wins |",
              "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for row in aggregates:
        lines.append(f"| {row['numeric']} | {row['shapes']} | {row['v06_search_geomean']:.3f}x | "
                     f"{row['v07_search_geomean']:.3f}x | {row['v07_search_vs_v06_search_geomean']:.3f}x | "
                     f"{row['v07_search_wins']} | {row['parity']} | {row['v06_search_wins']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize matched v0.6/v0.7 NTT measurements.")
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--metrics", required=True, type=pathlib.Path)
    parser.add_argument("--markdown", required=True, type=pathlib.Path)
    args = parser.parse_args()
    rows = build_rows(read_rows(args.input))
    aggregates = aggregate(rows)
    if not rows or not all(row["correct"] for row in rows):
        raise ValueError("comparison contains no rows or a correctness failure")
    write_csv(args.output, rows)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(aggregates, indent=2) + "\n")
    write_markdown(args.markdown, rows, aggregates)
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
