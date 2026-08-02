#!/usr/bin/env python3
import argparse
import csv
import pathlib
import statistics

MIN_STABLE_MS = 0.020


def analyze(rows):
    groups = {}
    for row in rows:
        key = (int(row["logN"]), int(row["batch"]), row["backend"])
        groups.setdefault(key, []).append(row)

    output = []
    shapes = sorted({(log_n, batch) for log_n, batch, _ in groups})
    for log_n, batch in shapes:
        ours = groups.get((log_n, batch, "online-reorder"), [])
        cufft = groups.get((log_n, batch, "cufft"), [])
        if not ours or not cufft:
            raise ValueError(f"missing implementation for logN={log_n}, batch={batch}")
        ours_times = [float(row["kernel_ms"]) for row in ours]
        cufft_times = [float(row["kernel_ms"]) for row in cufft]
        ours_median = statistics.median(ours_times)
        cufft_median = statistics.median(cufft_times)
        mapping = ours[0]
        output.append({
            "logN": log_n,
            "batch": batch,
            "total_points": (1 << log_n) * batch,
            "local_stages": int(mapping["local_stages"]),
            "prefix_threads": int(mapping["prefix_threads"]),
            "prefix_ept": int(mapping["prefix_ept"]),
            "suffix_threads": int(mapping["suffix_threads"]),
            "suffix_ept": int(mapping["suffix_ept"]),
            "trials": len(ours_times),
            "cuntt_median_ms": ours_median,
            "cuntt_min_ms": min(ours_times),
            "cuntt_max_ms": max(ours_times),
            "cufft_median_ms": cufft_median,
            "cufft_min_ms": min(cufft_times),
            "cufft_max_ms": max(cufft_times),
            "throughput_vs_cufft": cufft_median / ours_median,
            "timing_quality": "stable" if min(ours_median, cufft_median) >= MIN_STABLE_MS else "short",
            "range_separation": "faster" if max(ours_times) < min(cufft_times)
                                else "slower" if min(ours_times) > max(cufft_times)
                                else "overlap",
        })
    return output


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    stable = [row for row in rows if row["timing_quality"] == "stable"]
    faster = sum(row["throughput_vs_cufft"] > 1.0 for row in stable)
    separated = sum(row["range_separation"] == "faster" for row in stable)
    lines = [
        "# FP64 FFT Length/Batch Robustness", "",
        "All rows use the per-length mapping selected by the V100 coarse scan, recurrence twiddles,",
        "and XOR-swizzled prefix staging. Trials alternate cuButterfly/cuFFT execution order.", "",
        f"Among {len(stable)} stable shapes with both medians at least {MIN_STABLE_MS:.3f} ms, cuButterfly has",
        f"higher median throughput in {faster}/{len(stable)} and non-overlapping faster trial ranges in",
        f"{separated}/{len(stable)}. Short rows remain measurements, not stable performance claims.", "",
        "| logN | Batch | Points | Split | Prefix | Suffix | cuButterfly ms | cuFFT ms | Throughput | Quality | Trial ranges |",
        "|--:|--:|--:|:--|:--|:--|--:|--:|--:|:--|:--|",
    ]
    for row in rows:
        split = f"{row['local_stages']}+{row['logN'] - row['local_stages']}"
        prefix = f"{row['prefix_threads']}/{row['prefix_ept']}"
        suffix = f"{row['suffix_threads']}/{row['suffix_ept']}"
        lines.append(
            f"| {row['logN']} | {row['batch']} | {row['total_points']} | `{split}` | `{prefix}` | `{suffix}` | "
            f"{row['cuntt_median_ms']:.6f} | {row['cufft_median_ms']:.6f} | "
            f"{row['throughput_vs_cufft']:.3f}x | {row['timing_quality']} | {row['range_separation']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize FP64 FFT robustness measurements.")
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--output", "-o", required=True, type=pathlib.Path)
    parser.add_argument("--markdown", type=pathlib.Path)
    args = parser.parse_args()
    with args.input.open(newline="") as source:
        rows = analyze(list(csv.DictReader(source)))
    write_csv(args.output, rows)
    if args.markdown:
        write_markdown(args.markdown, rows)


if __name__ == "__main__":
    main()
