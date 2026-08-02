#!/usr/bin/env python3
import argparse
import csv
import pathlib


SUM_FIELDS = (
    "time_us", "dram_read_mib", "dram_write_mib", "shared_load_bank_conflicts",
    "shared_store_bank_conflicts", "warp_instructions", "integer_thread_instructions",
    "fp64_thread_instructions",
)
WEIGHTED_FIELDS = (
    "dram_peak_pct", "l2_hit_pct", "l1_hit_pct", "active_warps_pct",
    "barrier_stall_pct", "long_scoreboard_stall_pct",
)
def labels(batch):
    return {name: f"fp64_{name}_b{batch}" for name in ("scalar", "cufftdx", "cufft")}


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
    result["dram_total_mib"] = result["dram_read_mib"] + result["dram_write_mib"]
    return result


def event_timings(path):
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    matches = {}
    for row in rows:
        if row["backend"] == "cufft":
            matches["cufft"] = float(row["median_kernel_ms"])
        elif row["fft_core"] == "cufftdx-block":
            if row.get("shared_layout") == "xor-swizzle":
                name = "xor_swizzle"
            else:
                name = "recurrence" if row.get("cross_twiddle") == "recurrence" else "cufftdx"
            matches[name] = float(row["median_kernel_ms"])
        elif row["fft_core"] == "scalar" and row["backend"] == "online-reorder":
            matches["scalar"] = float(row["median_kernel_ms"])
    missing = {"scalar", "cufftdx", "cufft"} - set(matches)
    if missing:
        raise ValueError(f"missing CUDA-event configurations: {sorted(missing)}")
    return matches


def analyze(rows, timings, batch):
    output = []
    for implementation, label in labels(batch).items():
        matches = [row for row in rows if row["label"] == label]
        if not matches:
            raise ValueError(f"missing NCU label {label}")
        output.append({
            "implementation": implementation,
            "label": label,
            "cuda_event_ms": timings[implementation],
            **aggregate(matches),
        })
    recurrence_label = f"fp64_recurrence_b{batch}"
    recurrence_rows = [row for row in rows if row["label"] == recurrence_label]
    if recurrence_rows:
        if "recurrence" not in timings:
            raise ValueError("recurrence NCU rows require a recurrence CUDA-event timing")
        output.append({
            "implementation": "recurrence", "label": recurrence_label,
            "cuda_event_ms": timings["recurrence"], **aggregate(recurrence_rows),
        })
    xor_label = f"fp64_xor_swizzle_b{batch}"
    xor_rows = [row for row in rows if row["label"] == xor_label]
    if xor_rows:
        if "xor_swizzle" not in timings:
            raise ValueError("XOR-swizzle NCU rows require an XOR-swizzle CUDA-event timing")
        output.append({
            "implementation": "xor_swizzle", "label": xor_label,
            "cuda_event_ms": timings["xor_swizzle"], **aggregate(xor_rows),
        })
    cufft = next(row for row in output if row["implementation"] == "cufft")
    for row in output:
        for field in ("cuda_event_ms", "time_us", "dram_total_mib", "warp_instructions",
                      "integer_thread_instructions", "fp64_thread_instructions",
                      "shared_bank_conflicts"):
            row[f"{field}_vs_cufft"] = row[field] / cufft[field] if cufft[field] else 0.0
    return output


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, kernels, batch):
    by_name = {row["implementation"]: row for row in rows}
    scalar = by_name["scalar"]
    cufftdx = by_name["cufftdx"]
    cufft = by_name["cufft"]
    recurrence = by_name.get("recurrence")
    xor_swizzle = by_name.get("xor_swizzle")
    cufftdx_kernels = [row for row in kernels if row["label"] == labels(batch)["cufftdx"]]
    first = next(row for row in cufftdx_kernels if "first_kernel" in row["kernel_name"])
    second = next(row for row in cufftdx_kernels if "second_kernel" in row["kernel_name"])
    recurrence_first = None
    recurrence_second = None
    if recurrence:
        recurrence_kernels = [row for row in kernels if row["label"] == f"fp64_recurrence_b{batch}"]
        recurrence_first = next(row for row in recurrence_kernels if "first_kernel" in row["kernel_name"])
        recurrence_second = next(row for row in recurrence_kernels if "second_kernel" in row["kernel_name"])
    xor_first = None
    xor_second = None
    if xor_swizzle:
        xor_kernels = [row for row in kernels if row["label"] == f"fp64_xor_swizzle_b{batch}"]
        xor_first = next(row for row in xor_kernels if "first_kernel" in row["kernel_name"])
        xor_second = next(row for row in xor_kernels if "second_kernel" in row["kernel_name"])

    def pass_row(name, row):
        conflicts = number(row, "shared_load_bank_conflicts") + number(row, "shared_store_bank_conflicts")
        return (f"| {name} | {number(row, 'time_us'):.3f} | {number(row, 'warp_instructions'):.0f} | "
                f"{number(row, 'integer_thread_instructions'):.0f} | "
                f"{number(row, 'fp64_thread_instructions'):.0f} | {conflicts:.0f} | "
                f"{number(row, 'dram_peak_pct'):.1f}% | {number(row, 'barrier_stall_pct'):.1f}% | "
                f"{number(row, 'long_scoreboard_stall_pct'):.1f}% |")

    lines = [
        "# FP64 FFT NCU Attribution", "",
        f"All rows are FP64 forward `logN=16`, batch {batch} on V100. CUDA-event medians use",
        "five process trials, 1000 warmups, and 100 repetitions. NCU replay time is",
        "attribution-only and is not used to rank the implementations.", "",
        "| Implementation | CUDA-event ms | NCU us | DRAM MiB | Warp inst. | Integer inst. | FP64 inst. | Shared conflicts | DRAM peak | Barrier stall | Scoreboard stall |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['implementation']} | {row['cuda_event_ms']:.6f} | {row['time_us']:.3f} | "
            f"{row['dram_total_mib']:.2f} | {row['warp_instructions']:.0f} | "
            f"{row['integer_thread_instructions']:.0f} | "
            f"{row['fp64_thread_instructions']:.0f} | {row['shared_bank_conflicts']:.0f} | "
            f"{row['dram_peak_pct']:.1f}% | {row['barrier_stall_pct']:.1f}% | "
            f"{row['long_scoreboard_stall_pct']:.1f}% |")
    lines += [
        "", "## Attribution", "",
        f"- Replacing the scalar unit with cuFFTDx reduces CUDA-event time by "
        f"{1.0 - cufftdx['cuda_event_ms'] / scalar['cuda_event_ms']:.1%} and NCU replay time by "
        f"{1.0 - cufftdx['time_us'] / scalar['time_us']:.1%}. It also reduces FP64 instructions by "
        f"{1.0 - cufftdx['fp64_thread_instructions'] / scalar['fp64_thread_instructions']:.1%}.",
        f"- Against cuFFT, cuFFTDx transfers {cufftdx['dram_total_mib_vs_cufft']:.3f}x the DRAM bytes and "
        f"executes {cufftdx['fp64_thread_instructions_vs_cufft']:.3f}x the FP64 instructions. The remaining "
        f"gap is therefore not explained by external traffic volume or double-precision arithmetic work.",
        f"- cuFFTDx executes {cufftdx['warp_instructions_vs_cufft']:.2f}x the warp instructions and incurs "
        f"{cufftdx['shared_bank_conflicts_vs_cufft']:.1f}x the shared-memory bank conflicts of cuFFT. "
        f"Its weighted barrier and long-scoreboard stalls are {cufftdx['barrier_stall_pct']:.1f}% and "
        f"{cufftdx['long_scoreboard_stall_pct']:.1f}%.",
    ]
    if recurrence:
        lines.append(
            f"- Twiddle recurrence changes CUDA-event time by "
            f"{recurrence['cuda_event_ms'] / cufftdx['cuda_event_ms'] - 1.0:+.1%}, replay time by "
            f"{recurrence['time_us'] / cufftdx['time_us'] - 1.0:+.1%}, and FP64 instructions by "
            f"{recurrence['fp64_thread_instructions'] / cufftdx['fp64_thread_instructions'] - 1.0:+.1%}.")
    if xor_swizzle:
        replay_delta = xor_swizzle['time_us_vs_cufft'] - 1.0
        replay_position = "below" if replay_delta < 0.0 else "above"
        lines.append(
            f"- XOR swizzle changes CUDA-event time by "
            f"{xor_swizzle['cuda_event_ms'] / recurrence['cuda_event_ms'] - 1.0:+.1%} and replay time by "
            f"{xor_swizzle['time_us'] / recurrence['time_us'] - 1.0:+.1%} versus linear recurrence. "
            f"It finishes {abs(replay_delta):.1%} {replay_position} cuFFT replay while still "
            f"executing {xor_swizzle['warp_instructions_vs_cufft']:.2f}x its warp instructions, "
            f"{xor_swizzle['integer_thread_instructions_vs_cufft']:.2f}x its integer instructions, and incurring "
            f"{xor_swizzle['shared_bank_conflicts_vs_cufft']:.1f}x its shared conflicts.")
    lines += [
        "", "## Prefix/Suffix Split", "",
        "| cuFFTDx pass | NCU us | Warp inst. | Integer inst. | FP64 inst. | Shared conflicts | DRAM peak | Barrier stall | Scoreboard stall |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
        pass_row("table prefix + twiddle/reorder", first),
        pass_row("table suffix + natural-order store", second),
    ]
    if recurrence:
        lines += [pass_row("recurrence prefix + twiddle/reorder", recurrence_first),
                  pass_row("recurrence suffix + natural-order store", recurrence_second)]
    if xor_swizzle:
        lines += [pass_row("XOR-swizzle prefix + twiddle/reorder", xor_first),
                  pass_row("XOR-swizzle suffix + natural-order store", xor_second)]
    if recurrence:
        lines += ["",
                  f"Recurrence reduces prefix replay time by "
                  f"{1.0 - number(recurrence_first, 'time_us') / number(first, 'time_us'):.1%}, warp instructions by "
                  f"{1.0 - number(recurrence_first, 'warp_instructions') / number(first, 'warp_instructions'):.1%}, "
                  f"and raises DRAM peak from {number(first, 'dram_peak_pct'):.1f}% to "
                  f"{number(recurrence_first, 'dram_peak_pct'):.1f}%. It spends "
                  f"{number(recurrence_first, 'fp64_thread_instructions') / number(first, 'fp64_thread_instructions') - 1.0:.1%} "
                  f"more FP64 instructions while registers, shared allocation, and waves/SM remain unchanged. "
                  f"The suffix changes by only "
                  f"{number(recurrence_second, 'time_us') / number(second, 'time_us') - 1.0:+.1%}.", "",
                  "The recurrence result confirms a useful architecture trade: idle FP64 capacity replaces "
                  "dependent twiddle-table service in the prefix. Shared conflicts remain essentially unchanged, "
                  "so the next target is the local exchange/address path rather than another twiddle change.", ""]
    if xor_swizzle:
        recurrence_integer = number(recurrence_first, 'integer_thread_instructions')
        xor_integer = number(xor_first, 'integer_thread_instructions')
        integer_delta = xor_integer / recurrence_integer - 1.0 if recurrence_integer else None
        integer_text = (f"Integer instructions change by {integer_delta:+.1%}. "
                        if integer_delta is not None else "")
        xor_summary = (
            f"XOR swizzle changes prefix replay time by "
            f"{number(xor_first, 'time_us') / number(recurrence_first, 'time_us') - 1.0:+.1%}, shared conflicts by "
            f"{(number(xor_first, 'shared_load_bank_conflicts') + number(xor_first, 'shared_store_bank_conflicts')) / (number(recurrence_first, 'shared_load_bank_conflicts') + number(recurrence_first, 'shared_store_bank_conflicts')) - 1.0:+.1%}, "
            f"and warp instructions by "
            f"{number(xor_first, 'warp_instructions') / number(recurrence_first, 'warp_instructions') - 1.0:+.1%}. "
            + integer_text
            + f"The suffix changes by {number(xor_second, 'time_us') / number(recurrence_second, 'time_us') - 1.0:+.1%}.")
        lines += [xor_summary, ""]
    else:
        lines += ["",
                  f"The prefix consumes {number(first, 'time_us') / cufftdx['time_us']:.1%} of replay time and is "
                  f"{number(first, 'time_us') / number(second, 'time_us'):.2f}x slower than the suffix. The suffix "
                  f"already runs in {number(second, 'time_us'):.3f} us, below either cuFFT pass (about 179 us).", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Attribute FP64 scalar/cuFFTDx/cuFFT NCU captures.")
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--timing-summary", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--markdown", required=True, type=pathlib.Path)
    args = parser.parse_args()
    with args.input.open(newline="") as source:
        kernels = list(csv.DictReader(source))
    if not kernels:
        raise ValueError("FP64 NCU summary is empty")
    rows = analyze(kernels, event_timings(args.timing_summary), args.batch)
    write_csv(args.output, rows)
    write_markdown(args.markdown, rows, kernels, args.batch)


if __name__ == "__main__":
    main()
