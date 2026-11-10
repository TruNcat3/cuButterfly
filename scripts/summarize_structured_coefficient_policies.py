#!/usr/bin/env python3
import argparse
import csv
import pathlib
import statistics


def main():
    parser = argparse.ArgumentParser(description="Compare equivalent Structured2x2 coefficient policies.")
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    with args.input.open() as source:
        rows = list(csv.DictReader(source))
    groups = {}
    for row in rows:
        groups.setdefault((int(row["logN"]), row["coefficient_policy"]), []).append(row)

    records = []
    for log_n in sorted({key[0] for key in groups}):
        broadcast = groups[(log_n, "broadcast")]
        per_stage = groups[(log_n, "per-stage")]
        broadcast_ms = statistics.median(float(row["kernel_ms"]) for row in broadcast)
        per_stage_ms = statistics.median(float(row["kernel_ms"]) for row in per_stage)
        records.append({
            "logN": log_n,
            "N": 1 << log_n,
            "batch": int(broadcast[0]["batch"]),
            "trials": len(broadcast),
            "broadcast_ms": f"{broadcast_ms:.6f}",
            "per_stage_ms": f"{per_stage_ms:.6f}",
            "broadcast_Gbutterfly_s": f"{statistics.median(float(row['Gbutterfly_s']) for row in broadcast):.6f}",
            "per_stage_Gbutterfly_s": f"{statistics.median(float(row['Gbutterfly_s']) for row in per_stage):.6f}",
            "broadcast_speedup": f"{per_stage_ms / broadcast_ms:.6f}",
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=records[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
