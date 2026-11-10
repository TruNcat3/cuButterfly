#!/usr/bin/env python3
import argparse
import csv
import pathlib
import statistics


def main():
    parser = argparse.ArgumentParser(description="Summarize cross-operator cuButterfly mapping trials.")
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--output", "-o", required=True, type=pathlib.Path)
    args = parser.parse_args()

    with args.input.open() as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("butterfly operator input is empty")

    groups = {}
    for row in rows:
        key = (row["operator"], row.get("stage_matrices", ""), row["backend"], int(row["stage_space"]))
        groups.setdefault(key, []).append(row)

    fields = [
        "operator", "stage_matrices", "backend", "stage_space", "trials", "median_kernel_ms", "median_Gbutterfly_s",
        "speedup_vs_us1", "speedup_vs_temporal_tile", "speedup_vs_cufft",
    ]
    order = []
    for operator in ("fwht", "structured-2x2", "subset-zeta", "superset-zeta", "xor-zeta", "fft"):
        matrix_contracts = sorted({key[1] for key in groups if key[0] == operator})
        for matrices in matrix_contracts:
            order.append((operator, matrices, "temporal-tile", 0))
            order.extend((operator, matrices, "stage-pipeline", stage_space) for stage_space in (1, 2, 4, 8))
            if operator == "fft":
                order.append((operator, matrices, "cufft", 0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for operator, stage_matrices, backend, stage_space in order:
            samples = groups[(operator, stage_matrices, backend, stage_space)]
            median_ms = statistics.median(float(row["kernel_ms"]) for row in samples)
            median_gbutterfly = statistics.median(float(row["Gbutterfly_s"]) for row in samples)
            us1_ms = statistics.median(float(row["kernel_ms"]) for row in groups[(operator, stage_matrices, "stage-pipeline", 1)])
            tile_ms = statistics.median(float(row["kernel_ms"]) for row in groups[(operator, stage_matrices, "temporal-tile", 0)])
            cufft_ms = statistics.median(float(row["kernel_ms"]) for row in groups[("fft", "", "cufft", 0)]) if operator == "fft" else None
            writer.writerow({
                "operator": operator,
                "stage_matrices": stage_matrices,
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
