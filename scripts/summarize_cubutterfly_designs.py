#!/usr/bin/env python3
import argparse
import csv
import pathlib
import statistics

from residency_features import RESOURCE_FEATURE_FIELDS


CONFIG_FIELDS = ("backend", "compute_unit", "complex_multiply", "cross_twiddle", "local_exchange", "shared_layout", "fft_core", "stage_space", "tile_threads", "prefix_threads", "suffix_threads", "prefix_ept", "suffix_ept", "prefix_units_per_cta", "suffix_units_per_cta", "local_stages", "reorder_columns", "warp_stages", "pipeline_warps", "stage_handoff")
GROUP_FIELDS = ("operator", "precision", "direction", "normalization", "placement", "stage_matrices", "logN", "N", "batch", "element_stride", "batch_stride")


def main():
    parser = argparse.ArgumentParser(description="Rank cuButterfly kernel forms independently for each operator.")
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", "-o", required=True, type=pathlib.Path)
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        with path.open() as source:
            rows.extend(csv.DictReader(source))
    if not rows:
        raise ValueError("cuButterfly design sweep is empty")

    groups = {}
    for row in rows:
        key = tuple((row.get(field) or "linear") if field == "shared_layout" else row.get(field, "0")
                    for field in GROUP_FIELDS + CONFIG_FIELDS)
        groups.setdefault(key, []).append(row)

    summaries = []
    for key, samples in groups.items():
        kernel_samples = [float(row["kernel_ms"]) for row in samples]
        summaries.append({
            **dict(zip(GROUP_FIELDS + CONFIG_FIELDS, key)),
            "trials": len(samples),
            "median_kernel_ms": statistics.median(kernel_samples),
            "min_kernel_ms": min(kernel_samples),
            "max_kernel_ms": max(kernel_samples),
            "median_Gbutterfly_s": statistics.median(float(row["Gbutterfly_s"]) for row in samples),
            **{field: samples[0].get(field, "") for field in RESOURCE_FEATURE_FIELDS},
        })

    output = []
    group_keys = sorted({tuple(row[field] for field in GROUP_FIELDS) for row in summaries})
    for group_key in group_keys:
        group_rows = sorted((row for row in summaries if tuple(row[field] for field in GROUP_FIELDS) == group_key),
                            key=lambda row: row["median_kernel_ms"])
        best_ms = group_rows[0]["median_kernel_ms"]
        for rank, row in enumerate(group_rows, 1):
            output.append({
                "rank": rank,
                **row,
                "slowdown_vs_best": row["median_kernel_ms"] / best_ms,
            })

    fields = list(output[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in output:
            formatted = dict(row)
            for field in ("median_kernel_ms", "min_kernel_ms", "max_kernel_ms", "median_Gbutterfly_s", "slowdown_vs_best"):
                formatted[field] = f"{row[field]:.6f}"
            writer.writerow(formatted)


if __name__ == "__main__":
    main()
