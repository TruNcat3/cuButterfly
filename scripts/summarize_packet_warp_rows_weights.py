#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
from pathlib import Path


AGGREGATE = re.compile(r"aggregate_b(\d+)_t(\d+)$")
CONTROL = re.compile(r"interleaved_b(\d+)_p(\d+)_t(\d+)$")
WARP_ROWS = re.compile(r"warp_rows_b(\d+)_p(\d+)_t(\d+)$")


def summarize(rows):
    samples = {}
    for row in rows:
        label = row["label"]
        match = AGGREGATE.fullmatch(label)
        if match:
            key = (int(match.group(1)), "aggregate", 0)
        else:
            match = CONTROL.fullmatch(label)
            if match:
                key = (int(match.group(1)), "interleaved", int(match.group(2)))
            else:
                match = WARP_ROWS.fullmatch(label)
                if not match:
                    continue
                key = (int(match.group(1)), "warp_rows", int(match.group(2)))
        samples.setdefault(key, []).append(float(row["kernel_ms"]))
    result = []
    for (batch, mode, producer_weight), values in sorted(samples.items()):
        result.append({
            "batch": batch,
            "mode": mode,
            "producer_weight": producer_weight,
            "consumer_weight": 0 if mode == "aggregate" else 20 - producer_weight,
            "kernel_ms": statistics.median(values),
            "trials": len(values),
        })
    if not result:
        raise ValueError("no warp-row weight records found")
    for batch in sorted({row["batch"] for row in result}):
        modes = {row["mode"] for row in result if row["batch"] == batch}
        if not {"aggregate", "interleaved", "warp_rows"} <= modes:
            raise ValueError(f"incomplete batch {batch}")
    return result


def make_markdown(rows):
    lines = [
        "# Packet Warp-Row Service-Weight Screen", "",
        "CUDA-event medians from independent processes.", "",
        "| batch | mode | producer:consumer | kernel (ms) |",
        "|---:|:--|:--:|---:|",
    ]
    for row in rows:
        weights = "-" if row["mode"] == "aggregate" else f"{row['producer_weight']}:{row['consumer_weight']}"
        lines.append(f"| {row['batch']} | {row['mode']} | {weights} | {row['kernel_ms']:.6f} |")
    lines.extend(["", "## Selected Points", ""])
    for batch in sorted({row["batch"] for row in rows}):
        batch_rows = [row for row in rows if row["batch"] == batch]
        aggregate = next(row for row in batch_rows if row["mode"] == "aggregate")
        control = next(row for row in batch_rows if row["mode"] == "interleaved")
        best = min((row for row in batch_rows if row["mode"] == "warp_rows"), key=lambda row: row["kernel_ms"])
        lines.append(
            f"Batch {batch}: best={best['producer_weight']}:{best['consumer_weight']}; "
            f"warp-row/interleaved throughput={control['kernel_ms'] / best['kernel_ms']:.3f}x; "
            f"warp-row/aggregate={aggregate['kernel_ms'] / best['kernel_ms']:.3f}x."
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()
    with args.raw.open(newline="") as handle:
        rows = summarize(csv.DictReader(handle))
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    args.markdown.write_text(make_markdown(rows))


if __name__ == "__main__":
    main()
