#!/usr/bin/env python3
import argparse
import csv
import pathlib
import statistics


CONFIG_FIELDS = ("candidate_id", "group", "implementation", "reference", "static_score", "mapping_id",
                 "processing_unit", "prefix_log_n", "suffix_log_n", "prefix_threads", "suffix_threads",
                 "prefix_ept", "suffix_ept", "cross_twiddle", "direct_boundary", "operator", "precision", "direction",
                 "normalization", "placement", "backend", "fft_core", "logN", "N", "batch")


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["candidate_id"], []).append(row)
    candidates = []
    for samples in groups.values():
        times = [float(row["kernel_ms"]) for row in samples]
        first = samples[0]
        candidates.append({
            **{field: first.get(field, "") for field in CONFIG_FIELDS},
            "trials": len(samples), "median_kernel_ms": statistics.median(times),
            "min_kernel_ms": min(times), "max_kernel_ms": max(times),
            "correct": int(all(row["correct"] == "1" for row in samples)),
        })
    by_shape = {}
    for row in candidates:
        by_shape.setdefault(row["group"], []).append(row)
    output = []
    def shape_key(shape_rows):
        first = shape_rows[0]
        try:
            return (int(first["logN"]), int(first["batch"]), first["group"])
        except (KeyError, TypeError, ValueError):
            return (0, 0, first["group"])

    ordered_shapes = sorted(by_shape.values(), key=shape_key)
    for shape_rows in ordered_shapes:
        reference = next(row for row in shape_rows if row["reference"] == "1")
        internal = sorted((row for row in shape_rows if row["reference"] != "1"),
                          key=lambda row: row["median_kernel_ms"])
        for rank, row in enumerate(internal, 1):
            output.append({
                **row, "rank": rank, "cufft_median_ms": reference["median_kernel_ms"],
                "throughput_vs_cufft": reference["median_kernel_ms"] / row["median_kernel_ms"],
                "slowdown_vs_internal_best": row["median_kernel_ms"] / internal[0]["median_kernel_ms"],
            })
    return output


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for field in ("median_kernel_ms", "min_kernel_ms", "max_kernel_ms", "cufft_median_ms",
                          "throughput_vs_cufft", "slowdown_vs_internal_best"):
                formatted[field] = f"{row[field]:.6f}"
            writer.writerow(formatted)


def write_markdown(path, rows):
    winners = [row for row in rows if row["rank"] == 1]
    lines = ["# Generated V100 FFT Pipeline Search", "",
             "| logN | Batch | Split | Threads | EPT | Twiddle | Boundary | Median ms | cuFFT ms | Throughput ratio |",
             "|--:|--:|:--|:--|:--|:--|:--|--:|--:|--:|"]
    for row in winners:
        lines.append(f"| {row['logN']} | {row['batch']} | {row['prefix_log_n']}+{row['suffix_log_n']} | "
                     f"{row['prefix_threads']}+{row['suffix_threads']} | {row['prefix_ept']}+{row['suffix_ept']} | "
                     f"{row['cross_twiddle']} | {row['direct_boundary']} | {row['median_kernel_ms']:.6f} | {row['cufft_median_ms']:.6f} | "
                     f"{row['throughput_vs_cufft']:.3f}x |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Rank generated FFT mapping candidates.")
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--markdown", type=pathlib.Path)
    args = parser.parse_args()
    with args.input.open() as source:
        rows = list(csv.DictReader(source))
    summary = summarize(rows)
    write_csv(args.output, summary)
    if args.markdown:
        write_markdown(args.markdown, summary)


if __name__ == "__main__":
    main()
