#!/usr/bin/env python3
import argparse
import csv
import pathlib
import statistics


def summarize(rows, max_relative_range=0.03, timing_floor_ms=0.020):
    groups = {}
    for row in rows:
        groups.setdefault((row["group"], row["implementation"]), []).append(row)
    summaries = []
    for (_, _), samples in sorted(groups.items()):
        times = [float(row["kernel_ms"]) for row in samples]
        ordered = sorted(times)
        central = ordered[1:-1] if len(ordered) >= 5 else ordered
        median = statistics.median(times)
        relative_range = (max(times) - min(times)) / median
        central_relative_range = (max(central) - min(central)) / median
        if central_relative_range > max_relative_range:
            stability = "unstable"
        elif relative_range > max_relative_range:
            stability = "stable-with-outlier"
        else:
            stability = "stable"
        first = samples[0]
        summaries.append({
            **{field: first[field] for field in ("group", "implementation", "reference", "runner", "operator",
                                                  "precision", "direction", "normalization", "placement", "output_order",
                                                  "logN", "N", "batch", "modulus", "modulus_bits", "warmup", "repeat")},
            "trials": len(samples),
            "median_kernel_ms": median,
            "min_kernel_ms": min(times),
            "max_kernel_ms": max(times),
            "relative_range": relative_range,
            "central_relative_range": central_relative_range,
            "stability_class": stability,
            "timing_quality": "below-timing-floor" if median < timing_floor_ms else stability,
            "correct": int(all(row["correct"] == "1" for row in samples)),
        })
    by_group = {}
    for row in summaries:
        by_group.setdefault(row["group"], []).append(row)
    for rows_in_group in by_group.values():
        reference = next(row for row in rows_in_group if row["reference"] == "1")
        for row in rows_in_group:
            row["throughput_vs_reference"] = reference["median_kernel_ms"] / row["median_kernel_ms"]
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Summarize the matching-protocol external baseline suite.")
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--markdown", type=pathlib.Path)
    args = parser.parse_args()
    with args.input.open() as source:
        rows = list(csv.DictReader(source))
    summaries = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=summaries[0].keys(), lineterminator="\n")
        writer.writeheader()
        for row in summaries:
            formatted = dict(row)
            for field in ("median_kernel_ms", "min_kernel_ms", "max_kernel_ms", "relative_range",
                          "central_relative_range", "throughput_vs_reference"):
                formatted[field] = f"{row[field]:.6f}"
            writer.writerow(formatted)
    if args.markdown:
        lines = ["# V100 Matching-Protocol External Baselines", "",
                 "All rows use 1000 warmups, 100 repetitions, five independent process trials, and correctness checks.", "",
                 "Rows below 0.020 ms are retained but are not used for stable latency claims.", "",
                 "| Operator | logN | Batch | Output | Implementation | Median ms | vs cuButterfly/cuNTT | Timing quality |",
                 "|:--|--:|--:|:--|:--|--:|--:|:--|"]
        for row in summaries:
            lines.append(f"| {row['operator']} | {row['logN']} | {int(row['batch']):,} | {row['output_order']} | "
                         f"{row['implementation']} | {row['median_kernel_ms']:.6f} | {row['throughput_vs_reference']:.3f}x | {row['timing_quality']} |")
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
