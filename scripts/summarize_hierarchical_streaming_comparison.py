#!/usr/bin/env python3
import argparse
import csv
import math
import re
import statistics
from pathlib import Path


LABEL = re.compile(r"^(barrier|resident1010|generic776|resident776)-w(32|64)-b([0-9]+)$")


def summarize(rows):
    samples = {}
    correct = {}
    barriers = {}
    for row in rows:
        match = LABEL.match(row["label"])
        if not match:
            raise ValueError(f"invalid comparison label: {row['label']}")
        variant, bits, batch = match.group(1), int(match.group(2)), int(match.group(3))
        key = (variant, bits, batch)
        samples.setdefault(key, []).append(float(row["kernel_ms"]))
        correct.setdefault(key, []).append(int(row["correct"]))
    parsed = []
    for (variant, bits, batch), values in samples.items():
        median = statistics.median(values)
        record = {
            "variant": variant, "word_bits": bits, "batch": batch,
            "trials": len(values), "kernel_ms": median,
            "min_kernel_ms": min(values), "max_kernel_ms": max(values),
            "spread_pct": (max(values) / min(values) - 1.0) * 100.0,
            "correct": 0 if 0 in correct[(variant, bits, batch)] else correct[(variant, bits, batch)][0],
        }
        parsed.append(record)
        if variant == "barrier":
            barriers[(bits, batch)] = median
    for record in parsed:
        base = barriers.get((record["word_bits"], record["batch"]))
        if base is None:
            raise ValueError(f"missing barrier for uint{record['word_bits']} batch {record['batch']}")
        record["speedup_vs_barrier"] = base / record["kernel_ms"]
    return sorted(parsed, key=lambda row: (row["word_bits"], row["batch"], row["variant"]))


def geomean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main():
    parser = argparse.ArgumentParser(description="Summarize hierarchical streaming batch scaling")
    parser.add_argument("raw")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    with Path(args.raw).open(newline="") as stream:
        records = summarize(list(csv.DictReader(stream)))
    with Path(args.csv).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    variants = ("barrier", "resident1010", "generic776", "resident776")
    lines = [
        "# Hierarchical Streaming Batch Comparison", "",
        "CUDA-event times use identical V100 binaries and measurement settings. "
        "Rows are medians of independent process trials; speedup above one is "
        "better than the v0.6 barrier implementation.", "",
    ]
    for bits in (32, 64):
        subset = [row for row in records if row["word_bits"] == bits]
        batches = sorted({row["batch"] for row in subset})
        indexed = {(row["batch"], row["variant"]): row for row in subset}
        lines.extend([
            f"## uint{bits}", "",
            "| batch | barrier ms | resident 10+10 ms/speedup | generic 7+7+6 ms/speedup | resident 7+7+6 ms/speedup |",
            "|---:|---:|:--|:--|:--|",
        ])
        for batch in batches:
            cells = []
            for variant in variants[1:]:
                row = indexed[(batch, variant)]
                cells.append(f"{row['kernel_ms']:.6f} / {row['speedup_vs_barrier']:.3f}x")
            lines.append(
                f"| {batch} | {indexed[(batch, 'barrier')]['kernel_ms']:.6f} | "
                f"{cells[0]} | {cells[1]} | {cells[2]} |"
            )
        lines.extend(["", "Geomean speedup versus barrier:", ""])
        for variant in variants[1:]:
            values = [row["speedup_vs_barrier"] for row in subset if row["variant"] == variant]
            lines.append(f"- `{variant}`: `{geomean(values):.3f}x`")
        max_spread = max(row["spread_pct"] for row in subset)
        lines.extend(["", f"Maximum trial spread: `{max_spread:.2f}%`.", ""])
    Path(args.markdown).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
