#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS = Path("results/hybrid2d_matrix.csv")
OUTPUT = Path("figures")


def select(rows, log_n, word_bits, unit, modulus_bits=30):
    return next(
        row for row in rows
        if int(row["logN"]) == log_n
        and int(row["word_bits"]) == word_bits
        and int(row["modulus_bits"]) == modulus_bits
        and row["compute_unit"] == unit
        and row.get("mod_multiply", "shoup") == "shoup"
    )


def main():
    with RESULTS.open() as input_file:
        rows = list(csv.DictReader(input_file))
    logs = [12, 14, 16, 18, 20]
    OUTPUT.mkdir(exist_ok=True)

    styles = {
        (32, "radix2"): ("32-bit radix-2", "#0072B2", "o", "-"),
        (32, "radix4"): ("32-bit radix-4", "#009E73", "s", "-"),
        (64, "radix2"): ("64-bit radix-2", "#D55E00", "o", "--"),
        (64, "radix4"): ("64-bit radix-4", "#CC79A7", "s", "--"),
    }
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    for (word_bits, unit), (label, color, marker, linestyle) in styles.items():
        values = [float(select(rows, log_n, word_bits, unit)["kernel_points_s"]) / 1e9 for log_n in logs]
        axis.plot(logs, values, label=label, color=color, marker=marker, linestyle=linestyle, linewidth=2)
    axis.set_xlabel("NTT length (log2 N)")
    axis.set_ylabel("Throughput (Gpoint/s)")
    axis.set_xticks(logs)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "hybrid2d_length_scaling.svg")
    fig.savefig(OUTPUT / "hybrid2d_length_scaling.png", dpi=180)
    plt.close(fig)

    word_speedup = []
    radix4_32 = []
    radix4_64 = []
    for log_n in logs:
        r2_32 = float(select(rows, log_n, 32, "radix2")["kernel_ntt_s"])
        r4_32 = float(select(rows, log_n, 32, "radix4")["kernel_ntt_s"])
        r2_64 = float(select(rows, log_n, 64, "radix2")["kernel_ntt_s"])
        r4_64 = float(select(rows, log_n, 64, "radix4")["kernel_ntt_s"])
        word_speedup.append(r4_32 / r4_64)
        radix4_32.append(r4_32 / r2_32)
        radix4_64.append(r4_64 / r2_64)

    positions = list(range(len(logs)))
    width = 0.24
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar([x - width for x in positions], word_speedup, width, label="32-bit / 64-bit (radix-4)", color="#0072B2")
    axis.bar(positions, radix4_32, width, label="radix-4 / radix-2 (32-bit)", color="#009E73")
    axis.bar([x + width for x in positions], radix4_64, width, label="radix-4 / radix-2 (64-bit)", color="#D55E00")
    axis.axhline(1.0, color="#333333", linewidth=1)
    axis.set_xlabel("NTT length (log2 N)")
    axis.set_ylabel("Speedup")
    axis.set_xticks(positions, logs)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "hybrid2d_unit_speedup.svg")
    fig.savefig(OUTPUT / "hybrid2d_unit_speedup.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
