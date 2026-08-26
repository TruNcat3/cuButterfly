#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path


SUM_FIELDS = (
    "time_us", "dram_read_mib", "dram_write_mib", "warp_instructions",
    "integer_thread_instructions",
)
WEIGHTED_FIELDS = (
    "active_warps_pct", "barrier_stall_pct", "long_scoreboard_stall_pct",
)
MAX_FIELDS = ("registers_per_thread", "shared_mem_bytes")
MIN_FIELDS = (
    "occupancy_register_block_limit", "occupancy_shared_block_limit",
    "occupancy_warp_block_limit",
)
OUTPUT_FIELDS = (
    "label", "word_bits", "backend", "kernel_count", *SUM_FIELDS,
    *WEIGHTED_FIELDS, *MAX_FIELDS, *MIN_FIELDS,
)


def value(row, field):
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return math.nan


def aggregate(rows):
    totals = []
    labels = {}
    for row in rows:
        labels.setdefault(row["label"], []).append(row)
    for label, group in labels.items():
        match = re.search(r"_w(32|64)_", label)
        word_bits = int(match.group(1)) if match else 0
        backend = "hierarchical-dataflow" if label.startswith("hierarchical_") else "hybrid2d"
        times = [value(row, "time_us") for row in group]
        valid_times = [time for time in times if math.isfinite(time)]
        total_time = sum(valid_times)
        result = {"label": label, "word_bits": word_bits, "backend": backend,
                  "kernel_count": len(group), "time_us": total_time}
        for field in SUM_FIELDS[1:]:
            result[field] = sum(v for v in (value(row, field) for row in group) if math.isfinite(v))
        for field in WEIGHTED_FIELDS:
            pairs = [(time, value(row, field)) for time, row in zip(times, group)
                     if math.isfinite(time) and math.isfinite(value(row, field))]
            result[field] = sum(time * metric for time, metric in pairs) / sum(time for time, _ in pairs) if pairs else math.nan
        for field in MAX_FIELDS:
            values = [value(row, field) for row in group if math.isfinite(value(row, field))]
            result[field] = max(values) if values else math.nan
        for field in MIN_FIELDS:
            values = [value(row, field) for row in group if math.isfinite(value(row, field))]
            result[field] = min(values) if values else math.nan
        totals.append(result)
    return sorted(totals, key=lambda row: (row["word_bits"], row["backend"]))


def fmt(value_, digits=2):
    return "n/a" if not math.isfinite(value_) else f"{value_:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description="Aggregate matched hierarchical/Hybrid2D NCU plans")
    parser.add_argument("summary")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    with Path(args.summary).open(newline="") as stream:
        totals = aggregate(list(csv.DictReader(stream)))
    if not totals:
        raise ValueError("no NCU summary records found")
    with Path(args.csv).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(totals)

    lines = [
        "# HierarchicalDataflow NCU attribution", "",
        "Hybrid2D totals include captured sequential pass kernels; a valid plan comparison requires two. Percentage metrics are time-weighted.", "",
        "| bits | backend | kernels | time us | read MiB | write MiB | integer inst | active warps | barrier stall | long scoreboard | regs/thread | shared B |",
        "|---:|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_bits = {}
    for row in totals:
        by_bits.setdefault(row["word_bits"], {})[row["backend"]] = row
        lines.append(
            f"| {row['word_bits']} | {row['backend']} | {row['kernel_count']} | {fmt(row['time_us'])} | "
            f"{fmt(row['dram_read_mib'])} | {fmt(row['dram_write_mib'])} | {fmt(row['integer_thread_instructions'], 0)} | "
            f"{fmt(row['active_warps_pct'])}% | {fmt(row['barrier_stall_pct'])}% | "
            f"{fmt(row['long_scoreboard_stall_pct'])}% | {fmt(row['registers_per_thread'], 0)} | "
            f"{fmt(row['shared_mem_bytes'], 0)} |"
        )
    lines.extend(["", "## Plan-level comparison", ""])
    for bits, backends in sorted(by_bits.items()):
        base = backends.get("hybrid2d")
        hierarchy = backends.get("hierarchical-dataflow")
        if not base or not hierarchy or base["kernel_count"] != 2 or hierarchy["kernel_count"] != 1:
            base_count = base["kernel_count"] if base else 0
            hierarchy_count = hierarchy["kernel_count"] if hierarchy else 0
            lines.append(
                f"- uint{bits}: incomplete capture (Hybrid2D kernels={base_count}, "
                f"hierarchical kernels={hierarchy_count}); no plan-level ratio is reported."
            )
            continue
        speedup = base["time_us"] / hierarchy["time_us"]
        read_ratio = hierarchy["dram_read_mib"] / base["dram_read_mib"]
        write_ratio = hierarchy["dram_write_mib"] / base["dram_write_mib"]
        warp_ratio = hierarchy["warp_instructions"] / base["warp_instructions"]
        integer_ratio = hierarchy["integer_thread_instructions"] / base["integer_thread_instructions"]
        lines.append(
            f"- uint{bits}: hierarchical/Hybrid2D speedup `{speedup:.3f}x`; "
            f"read ratio `{read_ratio:.3f}x`; write ratio `{write_ratio:.3f}x`; "
            f"warp-instruction ratio `{warp_ratio:.3f}x`; integer-instruction ratio "
            f"`{integer_ratio:.3f}x`; registers/thread `{hierarchy['registers_per_thread']:.0f}` "
            f"versus `{base['registers_per_thread']:.0f}`."
        )
    lines.extend([
        "", "## Interpretation", "",
        "Near-unit DRAM ratios mean the one-launch graph preserves, rather than removes, the single inter-layer global boundary. Performance differences are therefore attributed to CTA scheduling, instruction work, latency hiding, and synchronization instead of a different traffic-complexity class.",
    ])
    Path(args.markdown).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
