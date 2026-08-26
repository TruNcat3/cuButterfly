#!/usr/bin/env python3
import argparse
import csv
import math
import re
import statistics
from pathlib import Path


LABEL = re.compile(r"(v06|aggregate|online)_b(\d+)_t(\d+)$")


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows):
    samples = {}
    for row in rows:
        match = LABEL.fullmatch(row["label"])
        if not match:
            continue
        implementation, batch, _ = match.groups()
        samples.setdefault(int(batch), {}).setdefault(implementation, []).append(
            float(row["kernel_ms"])
        )
    result = []
    for batch in sorted(samples):
        point = samples[batch]
        missing = {"v06", "aggregate", "online"} - point.keys()
        if missing:
            raise ValueError(
                f"batch {batch} is missing: {', '.join(sorted(missing))}"
            )
        medians = {name: statistics.median(values) for name, values in point.items()}
        result.append({
            "batch": batch,
            "v06_ms": medians["v06"],
            "aggregate_ms": medians["aggregate"],
            "online_ms": medians["online"],
            "aggregate_speedup_v06": medians["v06"] / medians["aggregate"],
            "online_speedup_v06": medians["v06"] / medians["online"],
            "online_over_aggregate": medians["aggregate"] / medians["online"],
            "trials": min(len(values) for values in point.values()),
        })
    if not result:
        raise ValueError("no comparison records found")
    return result


def geometric_mean(values):
    values = list(values)
    return math.exp(sum(math.log(value) for value in values) / len(values))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_markdown(rows):
    lines = [
        "# Packet Streaming versus v0.6", "",
        "CUDA-event medians from three order-rotated independent processes per "
        "implementation. Forward and inverse correctness are checked before timing.", "",
        "| batch | v0.6 (ms) | v0.7 aggregate (ms) | v0.7 online (ms) | aggregate/v0.6 | online/v0.6 | online/aggregate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch']} | {row['v06_ms']:.6f} | "
            f"{row['aggregate_ms']:.6f} | {row['online_ms']:.6f} | "
            f"{row['aggregate_speedup_v06']:.3f}x | "
            f"{row['online_speedup_v06']:.3f}x | "
            f"{row['online_over_aggregate']:.3f}x |"
        )
    aggregate_geomean = geometric_mean(
        row["aggregate_speedup_v06"] for row in rows
    )
    online_geomean = geometric_mean(row["online_speedup_v06"] for row in rows)
    online_aggregate_geomean = geometric_mean(
        row["online_over_aggregate"] for row in rows
    )
    aggregate_wins = sum(row["aggregate_speedup_v06"] > 1.0 for row in rows)
    online_wins = sum(row["online_speedup_v06"] > 1.0 for row in rows)
    lines.extend([
        "", "## Result", "",
        f"v0.7 aggregate beats v0.6 at {aggregate_wins}/{len(rows)} points with "
        f"{aggregate_geomean:.3f}x geomean throughput. v0.7 online beats v0.6 "
        f"at {online_wins}/{len(rows)} points with {online_geomean:.3f}x geomean "
        f"throughput. Online reaches {online_aggregate_geomean:.3f}x of the "
        "aggregate path on this batch set.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()
    rows = summarize(read_rows(args.raw))
    write_csv(args.csv, rows)
    args.markdown.write_text(make_markdown(rows))


if __name__ == "__main__":
    main()
