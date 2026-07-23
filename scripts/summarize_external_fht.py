#!/usr/bin/env python3
import argparse
import csv
import pathlib
import statistics


GROUP_FIELDS = (
    "device", "compute_capability", "operator", "precision", "direction",
    "normalization", "placement", "backend", "compute_unit", "logN", "N", "batch",
)


def main():
    parser = argparse.ArgumentParser(description="Summarize external FWHT trials and optionally compare FP32 with cuButterfly.")
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--output", "-o", required=True, type=pathlib.Path)
    parser.add_argument("--cubutterfly-summary", type=pathlib.Path)
    parser.add_argument("--comparison-output", type=pathlib.Path)
    args = parser.parse_args()

    with args.input.open() as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("external FWHT input is empty")

    groups = {}
    for row in rows:
        key = tuple(row[field] for field in GROUP_FIELDS)
        groups.setdefault(key, []).append(row)

    summaries = []
    for key, samples in sorted(groups.items()):
        summaries.append({
            **dict(zip(GROUP_FIELDS, key)),
            "trials": len(samples),
            "median_kernel_ms": statistics.median(float(row["kernel_ms"]) for row in samples),
            "min_kernel_ms": min(float(row["kernel_ms"]) for row in samples),
            "max_kernel_ms": max(float(row["kernel_ms"]) for row in samples),
            "median_Gbutterfly_s": statistics.median(float(row["Gbutterfly_s"]) for row in samples),
            "max_roundtrip_error": max(float(row["max_roundtrip_error"]) for row in samples),
            "correct": int(all(row["correct"] == "1" for row in samples)),
        })

    fields = list(summaries[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            formatted = dict(row)
            for field in ("median_kernel_ms", "min_kernel_ms", "max_kernel_ms", "median_Gbutterfly_s", "max_roundtrip_error"):
                formatted[field] = f"{row[field]:.6f}"
            writer.writerow(formatted)

    if args.cubutterfly_summary:
        if not args.comparison_output:
            raise ValueError("--comparison-output is required with --cubutterfly-summary")
        with args.cubutterfly_summary.open() as source:
            internal = list(csv.DictReader(source))
        comparisons = []
        for external in summaries:
            if external["precision"] != "fp32":
                continue
            candidates = [row for row in internal if row["operator"] == "fwht" and row["precision"] == "fp32"
                          and row["logN"] == external["logN"] and row["rank"] == "1"]
            if not candidates:
                continue
            local = candidates[0]
            local_ms = float(local["median_kernel_ms"])
            comparisons.append({
                "logN": external["logN"],
                "N": external["N"],
                "batch": external["batch"],
                "external_backend": external["backend"],
                "external_ms": f"{external['median_kernel_ms']:.6f}",
                "external_Gbutterfly_s": f"{external['median_Gbutterfly_s']:.6f}",
                "cubutterfly_backend": local["backend"],
                "cubutterfly_compute_unit": local["compute_unit"],
                "cubutterfly_local_stages": local.get("local_stages", "0"),
                "cubutterfly_tile_threads": local["tile_threads"],
                "cubutterfly_ms": local["median_kernel_ms"],
                "cubutterfly_Gbutterfly_s": local["median_Gbutterfly_s"],
                "cubutterfly_vs_external_throughput": f"{external['median_kernel_ms'] / local_ms:.6f}",
                "external_speedup": f"{local_ms / external['median_kernel_ms']:.6f}",
            })
        comparisons.sort(key=lambda row: int(row["logN"]))
        args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
        with args.comparison_output.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=comparisons[0])
            writer.writeheader()
            writer.writerows(comparisons)


if __name__ == "__main__":
    main()
