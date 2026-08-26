#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


VARIANTS = (
    "v06_hybrid2d",
    "v06_resident",
    "v07_full",
    "v07_resident_generic",
    "v07_resident_core",
)


def geometric_mean(values):
    values = list(values)
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summarize(rows):
    samples = defaultdict(lambda: defaultdict(list))
    for row in rows:
        variant = row["variant"]
        if variant not in VARIANTS:
            continue
        key = (int(row["word_bits"]), int(row["logN"]), int(row["batch"]))
        samples[key][variant].append(float(row["kernel_ms"]))

    records = []
    required = {"v06_hybrid2d", "v07_full", "v07_resident_generic",
                "v07_resident_core"}
    for (bits, logn, batch), point in sorted(samples.items()):
        missing = required - point.keys()
        if missing:
            raise ValueError(
                f"u{bits} logN={logn} batch={batch} is missing "
                f"{', '.join(sorted(missing))}")
        medians = {name: statistics.median(values)
                   for name, values in point.items()}
        v06_candidates = {
            name: value for name, value in medians.items()
            if name.startswith("v06_")
        }
        v06_selected = min(v06_candidates, key=v06_candidates.get)
        v06_ms = v06_candidates[v06_selected]
        v07_candidates = {
            name: medians[name] for name in (
                "v07_full", "v07_resident_generic", "v07_resident_core")
        }
        mg_selected = min(v07_candidates, key=v07_candidates.get)
        mg_ms = v07_candidates[mg_selected]
        records.append({
            "word_bits": bits,
            "logN": logn,
            "batch": batch,
            "v06_hybrid2d_ms": medians["v06_hybrid2d"],
            "v06_resident_ms": medians.get("v06_resident", ""),
            "v06_selected": v06_selected,
            "v06_best_ms": v06_ms,
            "v07_full_ms": medians["v07_full"],
            "v07_resident_generic_ms": medians["v07_resident_generic"],
            "v07_resident_core_ms": medians["v07_resident_core"],
            "mg_selected": mg_selected,
            "mg_best_ms": mg_ms,
            "lowering_speedup": (medians["v07_full"] /
                                 medians["v07_resident_generic"]),
            "core_speedup": (medians["v07_resident_generic"] /
                             medians["v07_resident_core"]),
            "mg_best_vs_v06": v06_ms / mg_ms,
            "v07_core_vs_v06": v06_ms / medians["v07_resident_core"],
            "v07_core_vs_v06_resident": (
                medians.get("v06_resident", 0.0) /
                medians["v07_resident_core"]
                if "v06_resident" in medians else ""),
            "trials": min(len(values) for values in point.values()),
        })
    if not records:
        raise ValueError("no resident-v0.6 comparison records")
    return records


def aggregate(records, key):
    groups = defaultdict(list)
    for row in records:
        groups[row[key]].append(row)
    result = []
    for value, rows in sorted(groups.items()):
        ratios = [row["mg_best_vs_v06"] for row in rows]
        result.append((value, len(rows), geometric_mean(ratios),
                       sum(ratio > 1.0 for ratio in ratios),
                       min(ratios), max(ratios)))
    return result


def render(records):
    ratios = [row["mg_best_vs_v06"] for row in records]
    lowering = [row["lowering_speedup"] for row in records]
    cores = [row["core_speedup"] for row in records]
    lines = [
        "# Resident M/G Candidate Ablation versus v0.6", "",
        "CUDA-event medians from order-rotated independent processes. v0.6-best "
        "is the faster of the mature Hybrid2D radix-4 path and, at logN=20, "
        "the established 10+10 resident dataflow path. M/G-best is selected "
        "only among these three new resident-lowering ablations; it is not the "
        "best implementation in the complete v0.7 repository.", "",
        "| bits | logN | batch | v0.6-best | ms | M/G full | lowered generic | lowered core | M/G-best | vs v0.6 |",
        "|---:|---:|---:|:--|---:|---:|---:|---:|:--|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['word_bits']} | {row['logN']} | {row['batch']} | "
            f"`{row['v06_selected']}` | {row['v06_best_ms']:.6f} | "
            f"{row['v07_full_ms']:.6f} | "
            f"{row['v07_resident_generic_ms']:.6f} | "
            f"{row['v07_resident_core_ms']:.6f} | "
            f"`{row['mg_selected']}` | {row['mg_best_vs_v06']:.3f}x |")

    lines.extend([
        "", "## Aggregate", "",
        f"Across {len(records)} workload points, M/G-best wins at "
        f"{sum(r > 1.0 for r in ratios)}/{len(ratios)} points and reaches "
        f"{geometric_mean(ratios):.3f}x v0.6-best geometric-mean throughput. "
        f"Resident boundary lowering contributes {geometric_mean(lowering):.3f}x "
        f"and the selected physical core contributes {geometric_mean(cores):.3f}x "
        "relative to their immediately preceding layers.", "",
        "| grouping | value | points | geomean vs v0.6 | wins | min | max |",
        "|:--|---:|---:|---:|---:|---:|---:|",
    ])
    for key, label in (("word_bits", "bits"), ("logN", "logN"),
                       ("batch", "batch")):
        for value, count, geomean, wins, minimum, maximum in aggregate(records, key):
            lines.append(
                f"| {label} | {value} | {count} | {geomean:.3f}x | "
                f"{wins}/{count} | {minimum:.3f}x | {maximum:.3f}x |")

    lines.extend([
        "", "## Interpretation", "",
        "This table is a controlled M/G-lowering ablation, not a release-level "
        "v0.6/v0.7 comparison. `v07_full` isolates the cost of materializing "
        "every logical boundary. "
        "`v07_resident_generic` changes only M-to-G lowering, and "
        "`v07_resident_core` additionally changes the physical execution-group "
        "core. The generated dataflow core is currently specialized only for "
        "10+10; shorter execution groups use the descriptor radix-4 fallback. "
        "At logN=20 the dataflow path is physically equivalent to the v0.6 "
        "10+10 dataflow schedule, so parity there is expected and is an explicit "
        "equivalence check rather than an independent speedup claim.", "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()
    with args.raw.open(newline="") as handle:
        records = summarize(csv.DictReader(handle))
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    args.markdown.write_text(render(records))


if __name__ == "__main__":
    main()
