#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
from pathlib import Path


LABEL = re.compile(r"(.+)_b(\d+)_t(\d+)$")
MODES = (
    "aggregate",
    "interleaved",
    "interleaved_folded",
    "warp_rows",
    "warp_rows_folded",
)


def summarize(rows):
    samples = {}
    for row in rows:
        match = LABEL.fullmatch(row["label"])
        if not match or match.group(1) not in MODES:
            continue
        mode, batch, _ = match.groups()
        samples.setdefault((int(batch), mode), []).append(float(row["kernel_ms"]))
    result = []
    for (batch, mode), values in sorted(samples.items()):
        result.append({
            "batch": batch,
            "mode": mode,
            "kernel_ms": statistics.median(values),
            "trials": len(values),
        })
    for batch in sorted({row["batch"] for row in result}):
        present = {row["mode"] for row in result if row["batch"] == batch}
        if set(MODES) != present:
            raise ValueError(f"incomplete batch {batch}")
    if not result:
        raise ValueError("no folded-barrier records found")
    return result


def make_markdown(rows):
    lines = [
        "# Packet Wave-Barrier Folding", "",
        "CUDA-event medians from independent processes.", "",
        "| batch | mode | kernel (ms) |",
        "|---:|:--|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['batch']} | {row['mode']} | {row['kernel_ms']:.6f} |")
    lines.extend(["", "## Deltas", ""])
    for batch in sorted({row["batch"] for row in rows}):
        points = {row["mode"]: row["kernel_ms"] for row in rows if row["batch"] == batch}
        interleaved_gain = points["interleaved"] / points["interleaved_folded"]
        warp_gain = points["warp_rows"] / points["warp_rows_folded"]
        best_mode = min((mode for mode in MODES if mode != "aggregate"), key=lambda mode: points[mode])
        ratio = points["aggregate"] / points[best_mode]
        lines.append(
            f"Batch {batch}: interleaved folding={interleaved_gain:.3f}x; "
            f"warp-row folding={warp_gain:.3f}x; best={best_mode}; "
            f"best/aggregate throughput={ratio:.3f}x."
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
