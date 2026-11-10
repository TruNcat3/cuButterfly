#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def read_summary(path):
    return {row["label"]: row for row in csv.DictReader(path.open())}


def value(row, key):
    return float(row[key])


def change(before, after, key):
    return 100.0 * (value(after, key) / value(before, key) - 1.0)


def main():
    parser = argparse.ArgumentParser(
        description="Compare matched APPT register-tail NCU captures")
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    before_rows = read_summary(args.before)
    after_rows = read_summary(args.after)
    records = []
    for bits in (32, 64):
        for batch in (1, 4):
            label = f"register-tail_u{bits}_b{batch}"
            before = before_rows[label]
            after = after_rows[label]
            baseline = after_rows[f"v06_u{bits}_b{batch}"]
            records.append({
                "bits": bits,
                "batch": batch,
                "before_us": value(before, "time_us"),
                "after_us": value(after, "time_us"),
                "speedup": value(before, "time_us") / value(after, "time_us"),
                "loads": change(before, after, "global_load_sectors"),
                "stores": change(before, after, "global_store_sectors"),
                "warp": change(before, after, "warp_instructions"),
                "integer": change(before, after, "integer_thread_instructions"),
                "l1_before": value(before, "l1_hit_pct"),
                "l1_after": value(after, "l1_hit_pct"),
                "load_vs_v06": value(after, "global_load_sectors") /
                               value(baseline, "global_load_sectors"),
                "store_vs_v06": value(after, "global_store_sectors") /
                                value(baseline, "global_store_sectors"),
            })

    load_changes = [-row["loads"] for row in records]
    time_reductions = [100.0 * (1.0 - 1.0 / row["speedup"])
                       for row in records]
    l1_before = [row["l1_before"] for row in records]
    l1_after = [row["l1_after"] for row in records]
    lines = [
        "# APPT Coefficient-Tree Cache Attribution", "",
        "Matched V100 base-clock NCU replay. The baseline is the selected readiness-policy capture; the after capture adds CTA-local coefficient reuse for the producer and low tail and uses the corresponding role weights.", "",
        "| bits | batch | before us | cached us | speedup | load sectors change | store sectors change | warp instructions change | integer instructions change |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['bits']} | {row['batch']} | {row['before_us']:.1f} | "
            f"{row['after_us']:.1f} | {row['speedup']:.3f}x | "
            f"{row['loads']:+.2f}% | {row['stores']:+.2f}% | "
            f"{row['warp']:+.2f}% | {row['integer']:+.2f}% |")
    lines += [
        "", "The cache removes repeated coefficient requests rather than a materialized state: "
        f"global-load sectors fall by {min(load_changes):.1f}%--{max(load_changes):.1f}% and replay time falls by "
        f"{min(time_reductions):.1f}%--{max(time_reductions):.1f}%, while store sectors are unchanged.",
        "", f"The L1 hit-rate percentage falls from {min(l1_before):.1f}%--{max(l1_before):.1f}% to {min(l1_after):.1f}%--{max(l1_after):.1f}% because the removed requests were the repeatedly reused, L1-friendly part of the stream. The remaining state, readiness, and final coefficient accesses have poorer L1 locality and remain mostly L2-served.",
        "", f"The residual register-tail traffic is {min(row['load_vs_v06'] for row in records):.2f}x--{max(row['load_vs_v06'] for row in records):.2f}x the v0.6 load sectors and {min(row['store_vs_v06'] for row in records):.2f}x--{max(row['store_vs_v06'] for row in records):.2f}x the store sectors. For uint32 the store count is approximately N+N/8 sectors; for uint64 it is N+N/4. The state1 write is coalesced, but one fixed-c tail task writes natural-order values at 128-element strides. Removing that sector term requires a multi-c tail microgroup, not another readiness change.",
        "", "The current source additionally preserves the same coefficient tree across the low and high tail halves. That later change is validated by CUDA-event timing and correctness but is not included in this NCU capture.",
    ]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
