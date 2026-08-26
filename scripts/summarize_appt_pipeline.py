#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


POINT = re.compile(
    r"appt-us(?P<us>\d+)-rs(?P<role_stages>\d+)"
    r"(?:-rep(?P<replicas>\d+))?-td(?P<td>\d+)"
    r"(?:-ti(?P<token_interleave>\d+))?-buf(?P<buffers>\d+)-cta(?P<ctas>\d+)"
)


def main():
    parser = argparse.ArgumentParser(description="Summarize the four-axis APPT pipeline scan")
    parser.add_argument("raw", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    with args.raw.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped = defaultdict(list)
    metadata = {}
    for row in rows:
        key = (int(row["word_bits"]), int(row["logN"]), int(row["batch"]), row["label"])
        grouped[key].append(float(row["kernel_ms"]))
        metadata[key] = row

    summary = []
    for key, samples in grouped.items():
        bits, logn, batch, label = key
        row = metadata[key]
        point = POINT.fullmatch(label)
        folds = len(row["stage_partition"].split("+"))
        record = {
            "word_bits": bits, "logN": logn, "batch": batch,
            "family": row["family"], "label": label,
            "kernel_ms": statistics.median(samples), "trials": len(samples),
            "stage_partition": row["stage_partition"],
            "us": int(point.group("us")) if point else "",
            "role_stages": int(point.group("role_stages")) if point else "",
            "replicas": int(point.group("replicas") or 1) if point else "",
            "td": int(point.group("td")) if point else "",
            "token_interleave": int(point.group("token_interleave") or 1) if point else "",
            "buffers": int(point.group("buffers")) if point else "",
            "ctas_per_sm": int(point.group("ctas")) if point else "",
        }
        points = (1 << logn) * batch
        state_bytes = 2 * (2 * folds - 1) * points * (bits // 8) if point else 0
        record["state_traffic_gbps"] = state_bytes / (record["kernel_ms"] * 1.0e6) if point else ""
        summary.append(record)

    fields = list(summary[0])
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(summary, key=lambda row: (
            row["word_bits"], row["logN"], row["batch"], row["kernel_ms"])))

    indexed = defaultdict(list)
    for row in summary:
        indexed[(row["word_bits"], row["logN"], row["batch"])].append(row)
    lines = [
        "# APPT Four-Axis Pipeline Scan", "",
        "Each APPT row is one fixed physical subgraph reused across stage-time folds. "
        "The reported state bandwidth counts token reads/writes and online fold reorders, "
        "but excludes twiddle traffic.", "",
        "| bits | logN | batch | best staged APPT point | staged ms | online ms | v0.6 ms | online/staged | online/v0.6 |",
        "|---:|---:|---:|:--|---:|---:|---:|---:|---:|",
    ]
    for shape in sorted(indexed):
        candidates = indexed[shape]
        appt = min((row for row in candidates if row["family"] == "appt"),
                   key=lambda row: row["kernel_ms"])
        v06 = next((row for row in candidates if row["label"] == "resident1010"), None)
        online = min((row for row in candidates if row["family"] == "appt-online"),
                     key=lambda row: row["kernel_ms"], default=None)
        v06_ms = v06["kernel_ms"] if v06 else None
        point = (f"({appt['us']},{appt['role_stages']},{appt['replicas']},"
                 f"{appt['td']},{appt['token_interleave']},{appt['buffers']},"
                 f"{appt['ctas_per_sm']})")
        online_ms = online["kernel_ms"] if online else None
        lines.append(
            f"| {shape[0]} | {shape[1]} | {shape[2]} | {point} | "
            f"{appt['kernel_ms']:.4f} | "
            f"{online_ms:.4f} | " if online_ms is not None else
            f"| {shape[0]} | {shape[1]} | {shape[2]} | {point} | "
            f"{appt['kernel_ms']:.4f} | n/a | "
        )
        lines[-1] += f"{v06_ms:.4f} | " if v06_ms is not None else "n/a | "
        lines[-1] += (f"{appt['kernel_ms'] / online_ms:.3f}x | "
                      f"{v06_ms / online_ms:.3f}x |"
                      if online_ms is not None and v06_ms is not None else
                      "n/a | n/a |")
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
