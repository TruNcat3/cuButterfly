#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pathlib
import statistics


def read_rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def geomean(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def build_rows(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    output = []
    for group, entries in sorted(groups.items()):
        mature = [row for row in entries if row["implementation"].startswith("v0.6-")]
        pipeline = next(row for row in entries if row["implementation"] == "v0.7-pipeline-radix2")
        resident = next(row for row in entries if row["implementation"] == "v0.7-resident-radix4")
        if not mature:
            raise ValueError(f"{group} has no v0.6 comparison")
        mature_best = min(mature, key=lambda row: float(row["median_kernel_ms"]))
        mature_ms = float(mature_best["median_kernel_ms"])
        pipeline_ms = float(pipeline["median_kernel_ms"])
        resident_ms = float(resident["median_kernel_ms"])
        output.append({
            "group": group, "precision": resident["precision"], "logN": int(resident["logN"]),
            "batch": int(resident["batch"]), "v06_best": mature_best["implementation"],
            "v06_best_ms": mature_ms, "v07_pipeline_ms": pipeline_ms,
            "v07_resident_radix4_ms": resident_ms,
            "radix4_speedup_vs_pipeline": pipeline_ms / resident_ms,
            "radix4_throughput_vs_v06": mature_ms / resident_ms,
            "winner": "v0.7-resident-radix4" if resident_ms < mature_ms else mature_best["implementation"],
            "correct": int(all(row["correct"] == "1" for row in entries)),
        })
    return output


def aggregate(rows):
    groups = {"overall": rows}
    for precision, log_n in sorted({(row["precision"], row["logN"]) for row in rows}):
        groups[f"{precision}/logN{log_n}"] = [row for row in rows
                                               if row["precision"] == precision and row["logN"] == log_n]
    return [{
        "numeric": name, "shapes": len(samples),
        "radix4_speedup_vs_pipeline": geomean([row["radix4_speedup_vs_pipeline"] for row in samples]),
        "radix4_throughput_vs_v06": geomean([row["radix4_throughput_vs_v06"] for row in samples]),
        "radix4_wins": sum(row["radix4_throughput_vs_v06"] > 1.03 for row in samples),
        "parity": sum(0.97 <= row["radix4_throughput_vs_v06"] <= 1.03 for row in samples),
        "v06_wins": sum(row["radix4_throughput_vs_v06"] < 0.97 for row in samples),
    } for name, samples in groups.items()]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, metrics):
    lines = ["# V100 HybridDataflow Resident Radix-4 Comparison", "",
             "All rows use 30 warmups, 500 timed iterations, three randomized process trials, and correctness checks.",
             "The v0.6 column is the measured minimum of the legal mature cores in that group.", "",
             "| Numeric | logN | Batch | v0.6 best | v0.6 ms | v0.7 pipeline ms | v0.7 resident R4 ms | R4/pipeline | R4/v0.6 throughput | Winner |",
             "|:--|--:|--:|:--|--:|--:|--:|--:|--:|:--|"]
    for row in rows:
        lines.append(f"| {row['precision']} | {row['logN']} | {row['batch']:,} | {row['v06_best']} | "
                     f"{row['v06_best_ms']:.6f} | {row['v07_pipeline_ms']:.6f} | "
                     f"{row['v07_resident_radix4_ms']:.6f} | {row['radix4_speedup_vs_pipeline']:.3f}x | "
                     f"{row['radix4_throughput_vs_v06']:.3f}x | {row['winner']} |")
    lines += ["", "## Aggregate", "",
              "| Numeric | Shapes | R4/pipeline | R4/v0.6 throughput | R4 wins | Parity | v0.6 wins |",
              "|:--|--:|--:|--:|--:|--:|--:|"]
    for row in metrics:
        lines.append(f"| {row['numeric']} | {row['shapes']} | {row['radix4_speedup_vs_pipeline']:.3f}x | "
                     f"{row['radix4_throughput_vs_v06']:.3f}x | {row['radix4_wins']} | "
                     f"{row['parity']} | {row['v06_wins']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--metrics", required=True, type=pathlib.Path)
    parser.add_argument("--markdown", required=True, type=pathlib.Path)
    args = parser.parse_args()
    rows = build_rows(read_rows(args.input))
    metrics = aggregate(rows)
    if not rows or not all(row["correct"] for row in rows):
        raise ValueError("comparison contains no rows or a correctness failure")
    write_csv(args.output, rows)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    write_markdown(args.markdown, rows, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
