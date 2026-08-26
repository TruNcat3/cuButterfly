#!/usr/bin/env python3
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def median(rows, field):
    return statistics.median(float(row[field]) for row in rows)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize the joint APPT physical-core and role-weight scan")
    parser.add_argument("raw", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.raw.open(newline="")))
    groups = defaultdict(list)
    baselines = defaultdict(list)
    for row in rows:
        key = (int(row["word_bits"]), int(row["batch"]))
        if row["family"] == "v06":
            baselines[key].append(row)
        else:
            groups[key + (row["physical_core"], row["weights"])].append(row)

    records = []
    for (bits, batch, core, weights), samples in sorted(groups.items()):
        v06_ms = median(baselines[(bits, batch)], "kernel_ms")
        kernel_ms = median(samples, "kernel_ms")
        records.append({
            "word_bits": bits,
            "batch": batch,
            "physical_core": core,
            "weights": weights,
            "kernel_ms": kernel_ms,
            "min_ms": min(float(row["kernel_ms"]) for row in samples),
            "trials": len(samples),
            "v06_ms": v06_ms,
            "speedup_vs_v06": v06_ms / kernel_ms,
        })

    best = {}
    for row in records:
        key = (row["word_bits"], row["batch"], row["physical_core"])
        if key not in best or row["kernel_ms"] < best[key]["kernel_ms"]:
            best[key] = row

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# APPT Physical-Core Design Space", "",
        "The APPT schedule is fixed at 7+7+6. Each physical core is searched with the same role-weight simplex; timings are medians across trials.", "",
        "| bits | batch | physical core | best weights | kernel ms | vs v0.6 | vs warp core |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for key, row in sorted(best.items()):
        bits, batch, core = key
        warp = best.get((bits, batch, "appt-online"))
        versus_warp = warp["kernel_ms"] / row["kernel_ms"] if warp else 1.0
        lines.append(
            f"| {bits} | {batch} | `{core}` | `{row['weights']}` | "
            f"{row['kernel_ms']:.6f} | {row['speedup_vs_v06']:.3f}x | "
            f"{versus_warp:.3f}x |")
    lines += [
        "", "The role weights are part of a core's physical mapping, not a transferable scheduler constant. A core comparison is valid only after each core has been independently balanced.",
    ]
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
