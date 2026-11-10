#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


FIELDS = (
    "word_bits", "logN", "batch", "backend", "n1_log", "rows_per_block",
    "threads_per_block", "data_time", "target_ctas_per_sm", "hierarchical_core",
    "kernel_ms", "correct",
)


def read_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open(newline="") as stream:
            for row in csv.DictReader(stream):
                if not row.get("backend"):
                    continue
                rows.append({field: row.get(field, "") for field in FIELDS})
    return rows


def number(row, field):
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return math.inf


def main():
    parser = argparse.ArgumentParser(description="Summarize HierarchicalDataflow scans")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    rows = read_rows(args.inputs)
    if not rows:
        raise ValueError("no benchmark CSV records found")

    output_csv = Path(args.csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (int(r["word_bits"]), int(r["logN"]), int(r["batch"]), r["backend"], number(r, "kernel_ms"))))

    groups = {}
    for row in rows:
        groups.setdefault((row["word_bits"], row["logN"], row["batch"]), []).append(row)
    lines = [
        "# HierarchicalDataflow comparison", "",
        "| bits | logN | batch | Hybrid2D ms | best hierarchical ms | speedup | core | rows | Td | threads | Rb | correct |",
        "|---:|---:|---:|---:|---:|---:|:--|---:|---:|---:|---:|:---:|",
    ]
    for key in sorted(groups, key=lambda k: tuple(map(int, k))):
        group = groups[key]
        baselines = [r for r in group if r["backend"] == "hybrid2d"]
        candidates = [r for r in group if r["backend"] in ("hierarchical-barrier", "hierarchical-dataflow")]
        if not candidates:
            continue
        best = min(candidates, key=lambda r: number(r, "kernel_ms"))
        base_ms = min((number(r, "kernel_ms") for r in baselines), default=math.inf)
        best_ms = number(best, "kernel_ms")
        base_text = f"{base_ms:.6f}" if math.isfinite(base_ms) else "n/a"
        speedup = f"{base_ms / best_ms:.3f}x" if math.isfinite(base_ms) and best_ms > 0 else "n/a"
        correct = "yes" if best["correct"] == "1" else "no"
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {base_text} | {best_ms:.6f} | {speedup} | "
            f"{best['hierarchical_core'] or 'dataflow-radix4'} | {best['rows_per_block']} | {best['data_time']} | {best['threads_per_block']} | "
            f"{best['target_ctas_per_sm']} | {correct} |"
        )
    Path(args.markdown).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
