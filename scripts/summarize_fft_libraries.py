#!/usr/bin/env python3
import argparse
import csv
import pathlib
import statistics


def read_rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def median(rows, field):
    return statistics.median(float(row[field]) for row in rows)


def main():
    parser = argparse.ArgumentParser(description="Combine the selected cuButterfly FFT and external-library trials.")
    parser.add_argument("--libraries", type=pathlib.Path, default=pathlib.Path("results/fft_libraries_v100_raw.csv"))
    parser.add_argument("--cubutterfly", type=pathlib.Path,
                        default=pathlib.Path("results/fft_cubutterfly_selected_v100_raw.csv"))
    parser.add_argument("--output", "-o", type=pathlib.Path,
                        default=pathlib.Path("results/fft_library_comparison_v100_summary.csv"))
    args = parser.parse_args()

    library_rows = read_rows(args.libraries)
    cubutterfly_rows = read_rows(args.cubutterfly)
    records = []
    for log_n in sorted({int(row["logN"]) for row in cubutterfly_rows}):
        selected = [row for row in cubutterfly_rows if int(row["logN"]) == log_n]
        cufft = [row for row in library_rows if int(row["logN"]) == log_n and row["library"] == "cufft"]
        vkfft = [row for row in library_rows if int(row["logN"]) == log_n and row["library"] == "vkfft"]
        if not cufft or not vkfft:
            raise ValueError(f"missing library trials for logN={log_n}")
        cub_ms = median(selected, "kernel_ms")
        cufft_ms = median(cufft, "kernel_ms")
        vkfft_ms = median(vkfft, "kernel_ms")
        records.append({
            "logN": log_n,
            "N": selected[0]["N"],
            "batch": selected[0]["batch"],
            "cubutterfly_backend": selected[0]["backend"],
            "cubutterfly_median_ms": f"{cub_ms:.9f}",
            "cufft_median_ms": f"{cufft_ms:.9f}",
            "vkfft_median_ms": f"{vkfft_ms:.9f}",
            "cubutterfly_vs_cufft_throughput": f"{cufft_ms / cub_ms:.6f}",
            "cubutterfly_vs_vkfft_throughput": f"{vkfft_ms / cub_ms:.6f}",
            "vkfft_vs_cufft_throughput": f"{cufft_ms / vkfft_ms:.6f}",
            "trials": min(len(selected), len(cufft), len(vkfft)),
            "correct": int(all(row["correct"] == "1" for row in selected + cufft + vkfft)),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=records[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
