#!/usr/bin/env python3
import argparse
import csv
import pathlib


TABLES = {
    "Fft8": ("fft", 8, "cuButterfly-cuFFTDx"),
    "Fft14": ("fft", 14, "cuButterfly-cuFFTDx-direct"),
    "Fft18": ("fft", 18, "cuButterfly-cuFFTDx-online"),
    "Fft20": ("fft", 20, "cuButterfly-cuFFTDx-online"),
    "Fwht8Warp": ("fwht", 8, "cuButterfly-warp-register"),
    "Fwht15Warp": ("fwht", 15, "cuButterfly-warp-register"),
    "Fwht15Online": ("fwht", 15, "cuButterfly-online-radix4"),
    "Fwht20Online": ("fwht", 20, "cuButterfly-online-radix4"),
    "Ntt12Radix4": ("ntt", 12, "cuNTT-Hybrid2D-radix4"),
    "Ntt16Radix4": ("ntt", 16, "cuNTT-Hybrid2D-radix4"),
    "Ntt20Compact": ("ntt", 20, "cuNTT-compact-stage"),
    "Xor8Radix4": ("xor-zeta", 8, "cuButterfly-radix4"),
    "Xor20Online": ("xor-zeta", 20, "cuButterfly-online-radix4"),
}


def load_tables(path):
    with path.open() as source:
        rows = list(csv.DictReader(source))
    output = {}
    for name, (operator, log_n, implementation) in TABLES.items():
        anchors = sorted(
            {
                (int(row["performance_batch"]), float(row["median_kernel_ms"]))
                for row in rows
                if row.get("correct") == "1"
                and row["operator"] == operator
                and int(row["logN"]) == log_n
                and row["implementation"] == implementation
            }
        )
        if len(anchors) < 2:
            raise ValueError(f"runtime selector table {name} has fewer than two valid anchors")
        output[name] = anchors
    return output


def generate_header(path, tables):
    lines = [
        "// Generated from the archived V100 scaling summary. Do not edit.",
        "#pragma once",
        "#include <array>",
        "#include <cstddef>",
        "namespace cuntt::detail {",
        "struct GeneratedLatencyAnchor { std::size_t batch; double kernel_ms; };",
    ]
    for name, anchors in tables.items():
        values = ", ".join(f"GeneratedLatencyAnchor{{{batch}ULL, {latency:.9f}}}" for batch, latency in anchors)
        lines.append(f"inline constexpr std::array<GeneratedLatencyAnchor, {len(anchors)}> k{name}Anchors = {{{{{values}}}}};")
    lines.extend(["}  // namespace cuntt::detail", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Generate runtime selector latency anchors from measured scaling data.")
    parser.add_argument("--summary", type=pathlib.Path, required=True)
    parser.add_argument("--header", type=pathlib.Path, required=True)
    args = parser.parse_args()
    generate_header(args.header, load_tables(args.summary))


if __name__ == "__main__":
    main()
