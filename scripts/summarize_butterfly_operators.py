#!/usr/bin/env python3
import argparse
import csv
import pathlib
import statistics


def main():
    parser = argparse.ArgumentParser(description="Summarize FWHT and FFT cuButterfly mapping trials.")
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--output", "-o", required=True, type=pathlib.Path)
    args = parser.parse_args()

    with args.input.open() as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("butterfly operator input is empty")

    groups = {}
    for row in rows:
        key = (row["operator"], row["backend"], int(row["stage_space"]))
        groups.setdefault(key, []).append(row)

    fields = [
        "operator", "backend", "stage_space", "trials", "median_kernel_ms", "median_Gbutterfly_s",
        "speedup_vs_us1", "speedup_vs_temporal_tile", "speedup_vs_cufft",
    ]
    order = []
    for operator in ("fwht", "xor-zeta", "fft"):
        order.append((operator, "temporal-tile", 0))
        order.extend((operator, "stage-pipeline", stage_space) for stage_space in (1, 2, 4, 8))
    order.append(("fft", "cufft", 0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for operator, backend, stage_space in order:
            samples = groups[(operator, backend, stage_space)]
            median_ms = statistics.median(float(row["kernel_ms"]) for row in samples)
            median_gbutterfly = statistics.median(float(row["Gbutterfly_s"]) for row in samples)
            us1_ms = statistics.median(float(row["kernel_ms"]) for row in groups[(operator, "stage-pipeline", 1)])
            tile_ms = statistics.median(float(row["kernel_ms"]) for row in groups[(operator, "temporal-tile", 0)])
            cufft_ms = statistics.median(float(row["kernel_ms"]) for row in groups[("fft", "cufft", 0)]) if operator == "fft" else None
            writer.writerow({
                "operator": operator,
                "backend": backend,
                "stage_space": stage_space if backend == "stage-pipeline" else 0,
                "trials": len(samples),
                "median_kernel_ms": f"{median_ms:.6f}",
                "median_Gbutterfly_s": f"{median_gbutterfly:.6f}",
                "speedup_vs_us1": f"{us1_ms / median_ms:.6f}",
                "speedup_vs_temporal_tile": f"{tile_ms / median_ms:.6f}",
                "speedup_vs_cufft": "" if cufft_ms is None else f"{cufft_ms / median_ms:.6f}",
            })


if __name__ == "__main__":
    main()
