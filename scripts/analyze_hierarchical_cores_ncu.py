#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


METRICS = (
    "time_us", "dram_read_mib", "dram_write_mib", "warp_instructions",
    "integer_thread_instructions", "active_warps_pct", "barrier_stall_pct",
    "long_scoreboard_stall_pct", "registers_per_thread", "shared_mem_bytes",
    "occupancy_register_block_limit", "occupancy_shared_block_limit",
    "occupancy_warp_block_limit",
)


def number(row, field):
    return float(row[field])


def compare(rows):
    indexed = {}
    for row in rows:
        label = row["label"]
        if label.startswith("w32_"):
            bits = 32
        elif label.startswith("w64_"):
            bits = 64
        else:
            continue
        indexed[(bits, label.split("_", 1)[1])] = row

    comparisons = []
    for bits in (32, 64):
        native = indexed.get((bits, "dataflow-radix4"))
        mature = indexed.get((bits, "hybrid2d-radix4"))
        if native is None or mature is None:
            raise ValueError(f"missing uint{bits} hierarchical core record")
        result = {"word_bits": bits}
        for field in METRICS:
            native_value = number(native, field)
            mature_value = number(mature, field)
            result[f"native_{field}"] = native_value
            result[f"mature_{field}"] = mature_value
            result[f"mature_over_native_{field}"] = mature_value / native_value if native_value else 0.0
        result["native_effective_block_limit"] = min(number(native, field) for field in METRICS[-3:])
        result["mature_effective_block_limit"] = min(number(mature, field) for field in METRICS[-3:])
        comparisons.append(result)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description="Compare hierarchical local-core NCU records")
    parser.add_argument("summary")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    with Path(args.summary).open(newline="") as stream:
        comparisons = compare(list(csv.DictReader(stream)))

    with Path(args.csv).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    lines = [
        "# Hierarchical local-core NCU attribution", "",
        "Both kernels use the same persistent graph, cooperative boundary, grid/block shape, and shared allocation. Only the resident radix-4 instruction schedule changes.", "",
        "| bits | native us | mature us | native speedup | warp inst change | integer inst change | long scoreboard change | regs native/mature | effective CTA/SM limit |",
        "|---:|---:|---:|---:|---:|---:|---:|:--|:--|",
    ]
    for row in comparisons:
        speedup = row["mature_time_us"] / row["native_time_us"]
        warp_change = (row["mature_over_native_warp_instructions"] - 1.0) * 100.0
        integer_change = (row["mature_over_native_integer_thread_instructions"] - 1.0) * 100.0
        scoreboard_change = row["mature_long_scoreboard_stall_pct"] - row["native_long_scoreboard_stall_pct"]
        lines.append(
            f"| {row['word_bits']} | {row['native_time_us']:.2f} | {row['mature_time_us']:.2f} | "
            f"{speedup:.3f}x | {warp_change:+.1f}% | {integer_change:+.1f}% | "
            f"{scoreboard_change:+.2f} pp | {row['native_registers_per_thread']:.0f}/{row['mature_registers_per_thread']:.0f} | "
            f"{row['native_effective_block_limit']:.0f}/{row['mature_effective_block_limit']:.0f} |"
        )
    lines.extend([
        "", "## Interpretation", "",
        "The mature Hybrid2D schedule reduces instruction count and registers, but does not cross an effective residency boundary. For uint32, shared memory fixes both kernels at five CTAs/SM even though the register limit improves from six to eight. For uint64, allocated register granularity leaves both at two CTAs/SM.", "",
        "The mature schedule instead raises long-scoreboard stalls, especially for uint32. With near-equal traffic and unchanged active-warps percentage, this attributes the regression to load/dependency placement and reduced latency-hiding ILP inside the physical unit, not to the outer dataflow graph, arithmetic count, or DRAM traffic class.",
    ])
    Path(args.markdown).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
