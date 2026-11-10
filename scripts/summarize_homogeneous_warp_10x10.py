#!/usr/bin/env python3
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    samples = defaultdict(list)
    with args.input.open(newline="") as source:
        for row in csv.DictReader(source):
            key = (int(row["word_bits"]), int(row["batch"]), row["variant"],
                   int(row["scan_threads"]), int(row["scan_units"]),
                   int(row["scan_resident"]))
            samples[key].append(float(row["kernel_ms"]))

    rows = []
    for key, values in samples.items():
        bits, batch, variant, threads, units, resident = key
        rows.append({
            "word_bits": bits, "batch": batch, "variant": variant,
            "threads": threads, "units": units, "resident": resident,
            "kernel_ms": statistics.median(values), "trials": len(values),
        })
    baseline = {(r["word_bits"], r["batch"]): r["kernel_ms"]
                for r in rows if r["variant"] == "v06"}
    for row in rows:
        row["throughput_vs_v06"] = (
            baseline[(row["word_bits"], row["batch"])] / row["kernel_ms"])
    rows.sort(key=lambda r: (r["word_bits"], r["batch"], r["variant"],
                             r["threads"]))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["word_bits", "batch", "variant", "threads", "units",
              "resident", "kernel_ms", "trials", "throughput_vs_v06"]
    with args.csv.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Warp-Granular 10+10 Physical-Core Screen", "",
        "Median CUDA-event time across repeated trials; trial count, warmup, "
        "and repetitions are controlled by the benchmark environment.", "",
        "| bits | batch | v0.6 ms | best full packet | full ms | best half packet | half ms | selected mapping | selected ms | selected/v0.6 |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for bits in (32, 64):
        for batch in sorted({r["batch"] for r in rows if r["word_bits"] == bits}):
            selected = [r for r in rows if r["word_bits"] == bits and
                        r["batch"] == batch]
            v06 = next(r for r in selected if r["variant"] == "v06")
            warp1024 = next(r for r in selected if r["variant"] == "warp1024")
            warp256 = min((r for r in selected if r["variant"] == "warp256"),
                          key=lambda r: r["kernel_ms"])
            static = min((r for r in selected
                          if r["variant"] == "warp256_static"),
                         key=lambda r: r["kernel_ms"])
            static_io = min((r for r in selected
                             if r["variant"] == "warp256_static_io"),
                            key=lambda r: r["kernel_ms"])
            full = min((static, static_io), key=lambda r: r["kernel_ms"])
            half = min((r for r in selected if r["variant"] in
                        ("warp256_static_half", "warp256_static_io_half")),
                       key=lambda r: r["kernel_ms"])
            best = min((full, half), key=lambda r: r["kernel_ms"])
            packet = 4 if bits == 32 else 2
            full_name = f"{full['variant'].removeprefix('warp256_')}/t{full['threads']}"
            half_name = f"{half['variant'].removeprefix('warp256_')}/t{half['threads']}/r{packet}"
            if bits == 32 and half["variant"] == "warp256_static_io_half":
                weight = {1: "11:9", 4: "8:7", 16: "17:13"}.get(
                    batch, "11:9")
                half_name += f"/w{weight}"
            best_name = full_name if best is full else half_name
            lines.append(
                f"| {bits} | {batch} | {v06['kernel_ms']:.6f} | "
                f"{full_name} | {full['kernel_ms']:.6f} | "
                f"{half_name} | {half['kernel_ms']:.6f} | {best_name} | "
                f"{best['kernel_ms']:.6f} | {best['throughput_vs_v06']:.3f}x |")
    lines.extend([
        "", "## Interpretation", "",
        "Full packets fill 32-byte sectors; half packets trade transaction "
        "packing for one additional resident 128-thread CTA. The selected "
        "mapping jointly searches IO mode, shared layout, packet width, CTA "
        "width, and producer/consumer role balance.",
    ])
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
