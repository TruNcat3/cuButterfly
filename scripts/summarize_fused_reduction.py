#!/usr/bin/env python3
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fused Shoup versus fused Barrett cuNTT runs")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/fused_reduction_summary.csv"))
    args = parser.parse_args()

    groups = defaultdict(list)
    metadata = {}
    with args.input.open(newline="") as source:
        for row in csv.DictReader(source):
            multiply = row.get("mod_multiply")
            if not multiply:
                multiply = "barrett" if row["cross_twiddle"] == "fused-barrett" else "shoup"
            key = (int(row["logN"]), multiply)
            groups[key].append(float(row["kernel_ms"]))
            metadata[int(row["logN"])] = row

    fields = [
        "logN",
        "batch",
        "n1_log",
        "rows_per_block",
        "threads_per_block",
        "fused_shoup_median_ms",
        "fused_barrett_median_ms",
        "barrett_latency_change_pct",
        "shoup_fused_table_mib",
        "barrett_fused_table_mib",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for log_n in sorted(metadata):
            shoup = statistics.median(groups[(log_n, "shoup")])
            barrett = statistics.median(groups[(log_n, "barrett")])
            row = metadata[log_n]
            n1 = 1 << int(row["n1_log"])
            n2 = 1 << (log_n - int(row["n1_log"]))
            word_bytes = int(row["word_bits"]) // 8
            root_bytes = n2 * (n1 - 1) * word_bytes
            writer.writerow(
                {
                    "logN": log_n,
                    "batch": row["batch"],
                    "n1_log": row["n1_log"],
                    "rows_per_block": row["rows_per_block"],
                    "threads_per_block": row["threads_per_block"],
                    "fused_shoup_median_ms": f"{shoup:.6f}",
                    "fused_barrett_median_ms": f"{barrett:.6f}",
                    "barrett_latency_change_pct": f"{(barrett / shoup - 1) * 100:.3f}",
                    "shoup_fused_table_mib": f"{2 * root_bytes / (1 << 20):.3f}",
                    "barrett_fused_table_mib": f"{(root_bytes + word_bytes) / (1 << 20):.3f}",
                }
            )

    print(f"Summary written to {args.output}")


if __name__ == "__main__":
    main()
