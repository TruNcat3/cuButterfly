#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


METRICS = (
    "time_us", "dram_read_mib", "dram_write_mib", "warp_instructions",
    "integer_thread_instructions", "active_warps_pct", "barrier_stall_pct",
    "wait_stall_pct", "long_scoreboard_stall_pct", "registers_per_thread",
    "shared_mem_bytes", "waves_per_sm",
)


def numeric(row, field):
    return float(row[field])


def ratio(value, reference):
    return value / reference if reference else 0.0


def analyze(rows):
    indexed = {row["label"]: row for row in rows}
    required = (
        "barrier_n20_b4", "streaming_generic_n20_b4",
        "streaming_resident_776_n20_b1", "streaming_resident_776_n20_b4",
        "streaming_specialized_n20_b4", "barrier_w32_n20_b4",
        "streaming_specialized_w32_n20_b4",
        "streaming_resident_776_w32_n20_b1",
    )
    missing = [label for label in required if label not in indexed]
    if missing:
        raise ValueError(f"missing NCU records: {', '.join(missing)}")

    comparisons = []
    groups = (
        (64, "barrier_n20_b4", (
            ("barrier", "barrier_n20_b4"),
            ("generic-7+7+6", "streaming_generic_n20_b4"),
            ("resident-7+7+6", "streaming_resident_776_n20_b4"),
            ("resident-10+10", "streaming_specialized_n20_b4"),
        )),
        (32, "barrier_w32_n20_b4", (
            ("barrier", "barrier_w32_n20_b4"),
            ("resident-10+10", "streaming_specialized_w32_n20_b4"),
        )),
    )
    for bits, reference_label, variants in groups:
        reference = indexed[reference_label]
        for variant, label in variants:
            source = indexed[label]
            record = {"word_bits": bits, "batch": 4, "variant": variant}
            for field in METRICS:
                value = numeric(source, field)
                base = numeric(reference, field)
                record[field] = value
                record[f"over_barrier_{field}"] = ratio(value, base)
            record["speedup_vs_barrier"] = ratio(
                numeric(reference, "time_us"), numeric(source, "time_us")
            )
            comparisons.append(record)

    b1 = indexed["streaming_resident_776_n20_b1"]
    b4 = indexed["streaming_resident_776_n20_b4"]
    scaling = {
        "time_ratio_b4_b1": ratio(numeric(b4, "time_us"), numeric(b1, "time_us")),
        "per_transform_time_ratio": ratio(numeric(b4, "time_us"), 4 * numeric(b1, "time_us")),
        "read_ratio_b4_b1": ratio(numeric(b4, "dram_read_mib"), numeric(b1, "dram_read_mib")),
        "write_ratio_b4_b1": ratio(numeric(b4, "dram_write_mib"), numeric(b1, "dram_write_mib")),
        "warp_ratio_b4_b1": ratio(numeric(b4, "warp_instructions"), numeric(b1, "warp_instructions")),
        "integer_ratio_b4_b1": ratio(
            numeric(b4, "integer_thread_instructions"),
            numeric(b1, "integer_thread_instructions"),
        ),
        "barrier_b1": numeric(b1, "barrier_stall_pct"),
        "barrier_b4": numeric(b4, "barrier_stall_pct"),
        "wait_b1": numeric(b1, "wait_stall_pct"),
        "wait_b4": numeric(b4, "wait_stall_pct"),
    }
    return comparisons, scaling


def fmt(value, digits=2):
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description="Attribute the resident 7+7+6 NCU checkpoint")
    parser.add_argument("summary")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()

    with Path(args.summary).open(newline="") as stream:
        comparisons, scaling = analyze(list(csv.DictReader(stream)))

    fields = list(comparisons[0])
    with Path(args.csv).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)

    lines = [
        "# Resident 7+7+6 NCU Attribution", "",
        "All batch-4 rows are matched single-kernel captures on Tesla V100. "
        "Speedup above one is better than the v0.6 barrier kernel.", "",
        "| bits | implementation | time us | speedup vs barrier | read ratio | write ratio | warp-inst ratio | integer-inst ratio | active warps | barrier stall | wait stall | long scoreboard | regs | shared B | waves/SM |",
        "|---:|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['word_bits']} | {row['variant']} | {fmt(row['time_us'])} | "
            f"{row['speedup_vs_barrier']:.3f}x | {row['over_barrier_dram_read_mib']:.3f}x | "
            f"{row['over_barrier_dram_write_mib']:.3f}x | {row['over_barrier_warp_instructions']:.3f}x | "
            f"{row['over_barrier_integer_thread_instructions']:.3f}x | {fmt(row['active_warps_pct'])}% | "
            f"{fmt(row['barrier_stall_pct'])}% | {fmt(row['wait_stall_pct'])}% | "
            f"{fmt(row['long_scoreboard_stall_pct'])}% | "
            f"{row['registers_per_thread']:.0f} | {row['shared_mem_bytes']:.0f} | {row['waves_per_sm']:.2f} |"
        )
    lines.extend([
        "", "## Resident 7+7+6 Batch Scaling", "",
        f"For uint64, batch 4 takes `{scaling['time_ratio_b4_b1']:.3f}x` the batch-1 time. "
        f"The per-transform time is therefore `{scaling['per_transform_time_ratio']:.3f}x` worse, "
        f"not better. DRAM read/write scale by `{scaling['read_ratio_b4_b1']:.3f}x/"
        f"{scaling['write_ratio_b4_b1']:.3f}x`, while warp/integer instructions scale only by "
        f"`{scaling['warp_ratio_b4_b1']:.3f}x/{scaling['integer_ratio_b4_b1']:.3f}x`. "
        f"Barrier stall remains dominant (`{scaling['barrier_b1']:.2f}%` at batch 1, "
        f"`{scaling['barrier_b4']:.2f}%` at batch 4), while wait stall is only "
        f"`{scaling['wait_b1']:.2f}%/{scaling['wait_b4']:.2f}%`.", "",
        "## Interpretation", "",
        "The resident physical core reduces instructions relative to the generic three-layer kernel, but fixed role ownership leaves runnable upstream work behind synchronization-stalled roles. Repeated readiness loads amplify measured DRAM traffic as batch grows. The next kernel must make scheduling work-conserving without introducing a single contended global queue; changing the butterfly arithmetic alone cannot close this gap.",
    ])
    Path(args.markdown).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
