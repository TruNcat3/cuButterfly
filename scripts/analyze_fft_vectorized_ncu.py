#!/usr/bin/env python3
import argparse
import csv
import pathlib


SUM_FIELDS = (
    "time_us", "dram_read_mib", "dram_write_mib", "shared_load_bank_conflicts",
    "shared_store_bank_conflicts", "warp_instructions", "fp32_thread_instructions",
)
WEIGHTED_FIELDS = (
    "dram_peak_pct", "l2_hit_pct", "l1_hit_pct", "active_warps_pct",
    "barrier_stall_pct", "long_scoreboard_stall_pct",
)


def number(row, field):
    value = row.get(field, "")
    return float(value) if value not in ("", None) else 0.0


def aggregate(rows):
    total_time = sum(number(row, "time_us") for row in rows)
    result = {field: sum(number(row, field) for row in rows) for field in SUM_FIELDS}
    for field in WEIGHTED_FIELDS:
        result[field] = (sum(number(row, field) * number(row, "time_us") for row in rows) / total_time
                         if total_time else 0.0)
    result["kernel_count"] = len(rows)
    result["shared_bank_conflicts"] = (
        result["shared_load_bank_conflicts"] + result["shared_store_bank_conflicts"])
    return result


def analyze(rows, batch):
    labels = {
        "pre-vector fixed": f"fft20_online_b{batch}",
        "post-vector fixed": f"post_vector_fixed_b{batch}",
        "post-vector selected": f"post_vector_selected_b{batch}",
        "cuFFT": f"post_vector_cufft_b{batch}",
    }
    output = []
    for name, label in labels.items():
        matches = [row for row in rows if row["label"] == label]
        if not matches:
            raise ValueError(f"missing NCU label {label}")
        output.append({"configuration": name, "label": label, **aggregate(matches)})
    baseline = output[0]
    for row in output:
        for field in ("time_us", "dram_read_mib", "dram_write_mib", "warp_instructions",
                      "shared_bank_conflicts"):
            row[f"{field}_vs_pre_vector"] = row[field] / baseline[field] if baseline[field] else 0.0
    return output


def attach_event_timings(output, pre_rows, post_rows, batch):
    group = f"fft20-fp32-batch{batch}"

    def fixed(rows):
        matches = [row for row in rows if row["group"] == group and
                   row["candidate_id"].startswith(f"fft20-fp32_b{batch}_p10s10_t512x512_e8x8_rec")]
        if len(matches) != 1:
            raise ValueError(f"expected one fixed CUDA-event row for {group}")
        return float(matches[0]["median_kernel_ms"])

    selected = [row for row in post_rows if row["group"] == group and row["rank"] == "1"]
    if len(selected) != 1:
        raise ValueError(f"expected one selected CUDA-event row for {group}")
    timings = {
        "pre-vector fixed": fixed(pre_rows),
        "post-vector fixed": fixed(post_rows),
        "post-vector selected": float(selected[0]["median_kernel_ms"]),
        "cuFFT": float(selected[0]["cufft_median_ms"]),
    }
    baseline = timings["pre-vector fixed"]
    for row in output:
        row["cuda_event_ms"] = timings[row["configuration"]]
        row["cuda_event_ms_vs_pre_vector"] = row["cuda_event_ms"] / baseline


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, batch):
    lines = [
        "# FFT Vectorized-Boundary NCU Attribution", "",
        f"All cuButterfly rows use FP32 forward `logN=20`, batch {batch}. The fixed",
        "rows use the same 512/512-thread, EPT 8/8 mapping, isolating the code change.",
        "NCU replay time is attribution-only; CUDA-event scans remain authoritative.", "",
        "| Configuration | CUDA-event ms | NCU time us | Kernels | DRAM read MiB | DRAM write MiB | Warp inst. | Bank conflicts | DRAM peak | Barrier stall | Scoreboard stall |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['configuration']} | {row.get('cuda_event_ms', 0.0):.6f} | "
            f"{row['time_us']:.3f} | {row['kernel_count']} | "
            f"{row['dram_read_mib']:.2f} | {row['dram_write_mib']:.2f} | "
            f"{row['warp_instructions']:.0f} | {row['shared_bank_conflicts']:.0f} | "
            f"{row['dram_peak_pct']:.1f}% | {row['barrier_stall_pct']:.1f}% | "
            f"{row['long_scoreboard_stall_pct']:.1f}% |")
    fixed = rows[1]
    selected = rows[2]
    lines += ["", "## Controlled Changes", "",
              f"- Vectorizing the same fixed mapping changes replay time by "
              f"{(fixed['time_us_vs_pre_vector'] - 1.0):+.1%}, warp instructions by "
              f"{(fixed['warp_instructions_vs_pre_vector'] - 1.0):+.1%}, and DRAM writes by "
              f"{(fixed['dram_write_mib_vs_pre_vector'] - 1.0):+.1%}. The matching CUDA-event "
              f"change is {(fixed.get('cuda_event_ms_vs_pre_vector', 1.0) - 1.0):+.1%}.",
              f"- CUDA-event timing selects the 256/128 mapping: it is "
              f"{(selected['cuda_event_ms'] / fixed['cuda_event_ms'] - 1.0):+.1%} versus the "
              f"post-vector fixed point. NCU replay reverses that ordering by "
              f"{(selected['time_us'] / fixed['time_us'] - 1.0):+.1%}; replay time must not be "
              f"used as the mapping-performance result.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Compare pre/post vectorized FFT NCU captures.")
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--pre-timing-summary", type=pathlib.Path)
    parser.add_argument("--post-timing-summary", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--markdown", type=pathlib.Path)
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        with path.open(newline="") as source:
            rows.extend(csv.DictReader(source))
    output = analyze(rows, args.batch)
    if bool(args.pre_timing_summary) != bool(args.post_timing_summary):
        raise ValueError("CUDA-event attribution requires both timing summaries")
    if args.pre_timing_summary:
        with args.pre_timing_summary.open(newline="") as source:
            pre_rows = list(csv.DictReader(source))
        with args.post_timing_summary.open(newline="") as source:
            post_rows = list(csv.DictReader(source))
        attach_event_timings(output, pre_rows, post_rows, args.batch)
    write_csv(args.output, output)
    if args.markdown:
        write_markdown(args.markdown, output, args.batch)


if __name__ == "__main__":
    main()
