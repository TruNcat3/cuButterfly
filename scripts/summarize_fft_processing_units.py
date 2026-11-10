#!/usr/bin/env python3
import argparse
import csv
import pathlib
import statistics


CONFIG_FIELDS = ("backend", "compute_unit", "fft_core", "tile_threads")


def main():
    parser = argparse.ArgumentParser(description="Select the best mapping of each FFT processing unit and compare it with cuFFT.")
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--output", "-o", required=True, type=pathlib.Path)
    args = parser.parse_args()

    with args.input.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("FFT processing-unit sweep is empty")

    configurations = {}
    for row in rows:
        key = (row["logN"], row["N"], row["batch"], *(row[field] for field in CONFIG_FIELDS))
        configurations.setdefault(key, []).append(float(row["kernel_ms"]))

    candidates = []
    for key, samples in configurations.items():
        log_n, n, batch, backend, compute_unit, fft_core, tile_threads = key
        candidates.append({
            "logN": int(log_n),
            "N": int(n),
            "batch": int(batch),
            "backend": backend,
            "compute_unit": compute_unit,
            "fft_core": "cufft" if backend == "cufft" else fft_core,
            "tile_threads": int(tile_threads),
            "trials": len(samples),
            "median_kernel_ms": statistics.median(samples),
            "min_kernel_ms": min(samples),
            "max_kernel_ms": max(samples),
        })

    selected = []
    for log_n in sorted({row["logN"] for row in candidates}):
        length_rows = [row for row in candidates if row["logN"] == log_n]
        cufft_rows = [row for row in length_rows if row["fft_core"] == "cufft"]
        if len(cufft_rows) != 1:
            raise ValueError(f"expected exactly one cuFFT configuration for logN={log_n}")
        cufft_ms = cufft_rows[0]["median_kernel_ms"]
        for core in sorted({row["fft_core"] for row in length_rows}):
            best = min((row for row in length_rows if row["fft_core"] == core), key=lambda row: row["median_kernel_ms"])
            selected.append({
                **best,
                "throughput_vs_cufft": cufft_ms / best["median_kernel_ms"],
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=selected[0].keys(), lineterminator="\n")
        writer.writeheader()
        for row in selected:
            formatted = dict(row)
            for field in ("median_kernel_ms", "min_kernel_ms", "max_kernel_ms"):
                formatted[field] = f"{row[field]:.6f}"
            formatted["throughput_vs_cufft"] = f"{row['throughput_vs_cufft']:.6f}"
            writer.writerow(formatted)


if __name__ == "__main__":
    main()
