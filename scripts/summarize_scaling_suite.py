#!/usr/bin/env python3
import argparse
import csv
import pathlib

from summarize_comprehensive_suite import read_rows, summarize


SERIES_FIELDS = (
    "operator", "precision", "direction", "normalization", "placement", "logN",
    "element_stride", "output_order", "implementation",
)


def analyze(rows, saturation_fraction=0.90, timing_floor_ms=0.020):
    summary = summarize(rows)
    series = {}
    for row in summary:
        key = tuple(row.get(field, "") for field in SERIES_FIELDS)
        series.setdefault(key, []).append(row)
    output = []
    for series_rows in series.values():
        ordered = sorted(series_rows, key=lambda row: int(row["performance_batch"]))
        eligible = [row for row in ordered
                    if row["median_kernel_ms"] >= timing_floor_ms and row["stability_class"] != "unstable"]
        saturation_basis = eligible if eligible else ordered
        peak = max(row["billion_points_s"] for row in saturation_basis)
        peak_row = max(saturation_basis, key=lambda row: row["billion_points_s"])
        saturated = [row for row in saturation_basis if row["billion_points_s"] >= saturation_fraction * peak]
        saturation_batch = min(int(row["performance_batch"]) for row in saturated)
        batch_one = next((row for row in ordered if int(row["performance_batch"]) == 1), None)
        for row in ordered:
            batch = int(row["performance_batch"])
            fraction = row["billion_points_s"] / peak if peak else 0.0
            if batch_one and batch_one["billion_points_s"]:
                throughput_speedup = row["billion_points_s"] / batch_one["billion_points_s"]
                batch_efficiency = throughput_speedup / batch
            else:
                throughput_speedup = 0.0
                batch_efficiency = 0.0
            region = "saturated" if fraction >= saturation_fraction else ("ramp" if fraction >= 0.5 else "launch-limited")
            timing_quality = "below-timing-floor" if row["median_kernel_ms"] < timing_floor_ms else row["stability_class"]
            eligible_for_saturation = row in saturation_basis
            output.append({
                **row,
                "peak_billion_points_s": peak,
                "peak_batch": int(peak_row["performance_batch"]),
                "fraction_of_peak": fraction,
                "saturation_fraction": saturation_fraction,
                "saturation_batch": saturation_batch,
                "throughput_speedup_vs_batch1": throughput_speedup,
                "batch_efficiency_vs_batch1": batch_efficiency,
                "scaling_region": region,
                "timing_floor_ms": timing_floor_ms,
                "timing_quality": timing_quality,
                "eligible_for_saturation": int(eligible_for_saturation),
            })
    return sorted(output, key=lambda row: (
        row["operator"], int(row["logN"]), row["implementation"], int(row["performance_batch"])))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for field in (
                "median_kernel_ms", "min_kernel_ms", "max_kernel_ms", "basis_kernel_ms",
                "million_transforms_s", "billion_points_s", "peak_billion_points_s",
            ):
                formatted[field] = f"{row[field]:.9f}"
            for field in (
                "relative_range", "central_relative_range", "throughput_vs_basis",
                "fraction_of_peak", "saturation_fraction", "throughput_speedup_vs_batch1",
                "batch_efficiency_vs_batch1",
                "timing_floor_ms",
            ):
                formatted[field] = f"{row[field]:.6f}"
            writer.writerow(formatted)


def write_markdown(path, rows):
    series = {}
    for row in rows:
        key = (row["operator"], row["precision"], row["logN"], row["implementation"])
        series.setdefault(key, []).append(row)
    lines = [
        "# V100 Orthogonal Length/Batch Scaling", "",
        "`Batch@90%` is the smallest measured batch reaching 90% of that",
        "implementation's peak measured `Gpoint/s` at fixed transform length.",
        "Rows below the configured 0.020 ms timing floor are retained but are not",
        "used as stable latency claims or peak/saturation references.", "",
        "## Saturation Summary", "",
        "| Operator | Numeric | logN | Implementation | Batch@90% | Peak batch | Peak Gpoint/s | Batch-1 ms |",
        "|:--|:--|--:|:--|--:|--:|--:|--:|",
    ]
    for key, series_rows in sorted(series.items(), key=lambda item: (item[0][0], int(item[0][2]), item[0][3])):
        first = series_rows[0]
        batch_one = next((row for row in series_rows if int(row["performance_batch"]) == 1), None)
        batch_one_text = f"{batch_one['median_kernel_ms']:.6f}" if batch_one else "-"
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {first['saturation_batch']} | "
            f"{first['peak_batch']} | {first['peak_billion_points_s']:.3f} | {batch_one_text} |")
    lines += ["", "## Detailed Scaling", "",
              "| Operator | logN | Batch | Implementation | Median ms | Gpoint/s | Peak fraction | vs basis | Region | Timing quality |",
              "|:--|--:|--:|:--|--:|--:|--:|--:|:--|:--|"]
    for row in rows:
        lines.append(
            f"| {row['operator']} | {row['logN']} | {int(row['performance_batch']):,} | "
            f"{row['implementation']} | {row['median_kernel_ms']:.6f} | "
            f"{row['billion_points_s']:.3f} | {row['fraction_of_peak']:.3f} | "
            f"{row['throughput_vs_basis']:.3f}x | {row['scaling_region']} | {row['timing_quality']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize orthogonal length/batch scaling.")
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", "-o", required=True, type=pathlib.Path)
    parser.add_argument("--markdown", type=pathlib.Path)
    parser.add_argument("--saturation-fraction", type=float, default=0.90)
    parser.add_argument("--timing-floor-ms", type=float, default=0.020)
    parser.add_argument("--require-stable", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.saturation_fraction <= 1.0:
        raise ValueError("saturation fraction must be in (0, 1]")
    raw = []
    for path in args.inputs:
        raw.extend(read_rows(path))
    if args.timing_floor_ms < 0.0:
        raise ValueError("timing floor must be non-negative")
    rows = analyze(raw, args.saturation_fraction, args.timing_floor_ms)
    write_csv(args.output, rows)
    if args.markdown:
        write_markdown(args.markdown, rows)
    unstable = sorted({row["case_id"] for row in rows if row["timing_quality"] == "unstable"})
    if args.require_stable and unstable:
        raise ValueError(f"unstable scaling cases: {unstable}")


if __name__ == "__main__":
    main()
