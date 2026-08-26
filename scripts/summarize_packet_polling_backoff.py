#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
from pathlib import Path


AGGREGATE = re.compile(r"aggregate_b(\d+)_t(\d+)$")
ONLINE = re.compile(r"online_b(\d+)_rw(\d+)_t(\d+)$")


def summarize(rows):
    batches = {}
    for row in rows:
        aggregate = AGGREGATE.fullmatch(row["label"])
        online = ONLINE.fullmatch(row["label"])
        if aggregate:
            batch = int(aggregate.group(1))
            batches.setdefault(batch, {"aggregate": [], "online": {}})[
                "aggregate"
            ].append(float(row["kernel_ms"]))
        elif online:
            batch, ready_window, _ = map(int, online.groups())
            batches.setdefault(batch, {"aggregate": [], "online": {}})[
                "online"
            ].setdefault(ready_window, []).append(float(row["kernel_ms"]))
    result = []
    for batch in sorted(batches):
        point = batches[batch]
        if not point["aggregate"] or not point["online"]:
            raise ValueError(f"incomplete batch {batch}")
        aggregate_ms = statistics.median(point["aggregate"])
        for ready_window in sorted(point["online"]):
            values = point["online"][ready_window]
            online_ms = statistics.median(values)
            result.append({
                "batch": batch,
                "ready_window": ready_window,
                "poll_sleep_cycles": ready_window * 32,
                "aggregate_ms": aggregate_ms,
                "online_ms": online_ms,
                "online_over_aggregate": aggregate_ms / online_ms,
                "trials": min(len(point["aggregate"]), len(values)),
            })
    if not result:
        raise ValueError("no polling records found")
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_markdown(rows):
    lines = [
        "# Packet Readiness Polling Backoff", "",
        "CUDA-event medians from independent processes. `ready_window` is the "
        "number of 32-cycle sleep quanta between acquire loads.", "",
        "| batch | ready window | sleep cycles | aggregate (ms) | online (ms) | online/aggregate |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch']} | {row['ready_window']} | "
            f"{row['poll_sleep_cycles']} | {row['aggregate_ms']:.6f} | "
            f"{row['online_ms']:.6f} | {row['online_over_aggregate']:.3f}x |"
        )
    lines.extend(["", "## Selected Points", ""])
    for batch in sorted({row["batch"] for row in rows}):
        candidates = [row for row in rows if row["batch"] == batch]
        best = max(candidates, key=lambda row: row["online_over_aggregate"])
        lines.append(
            f"Batch {batch}: ready_window={best['ready_window']} "
            f"({best['poll_sleep_cycles']} cycles), online/aggregate="
            f"{best['online_over_aggregate']:.3f}x."
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
    write_csv(args.csv, rows)
    args.markdown.write_text(make_markdown(rows))


if __name__ == "__main__":
    main()
