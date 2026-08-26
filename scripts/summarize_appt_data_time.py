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
                   int(row["scan_data_time"]), row["roles"],
                   int(row.get("role_mask", 7)))
            samples[key].append(float(row["kernel_ms"]))

    rows = []
    for key, values in samples.items():
        bits, batch, variant, data_time, roles, role_mask = key
        rows.append({
            "word_bits": bits, "batch": batch, "variant": variant,
            "data_time": data_time, "roles": roles,
            "role_mask": role_mask,
            "kernel_ms": statistics.median(values), "trials": len(values),
        })
    baselines = {
        (row["word_bits"], row["batch"], row["variant"]): row["kernel_ms"]
        for row in rows if row["variant"] != "data-time"
    }
    for row in rows:
        key = (row["word_bits"], row["batch"])
        row["throughput_vs_v06"] = baselines[key + ("v06",)] / row["kernel_ms"]
        row["throughput_vs_writer_final"] = (
            baselines[key + ("writer-final",)] / row["kernel_ms"])
    rows.sort(key=lambda row: (row["word_bits"], row["batch"],
                               row["variant"], row["data_time"],
                               row["role_mask"], row["roles"]))

    best = {}
    for row in rows:
        if row["variant"] != "data-time":
            continue
        key = (row["word_bits"], row["batch"], row["data_time"],
               row["role_mask"])
        if key not in best or row["kernel_ms"] < best[key]["kernel_ms"]:
            best[key] = row

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["word_bits", "batch", "variant", "data_time", "role_mask", "roles",
              "kernel_ms", "trials", "throughput_vs_v06",
              "throughput_vs_writer_final"]
    with args.csv.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# APPT Batch Data-Time Screen", "",
        "Median CUDA-event time. Each row selects the best P/T/W mapping at fixed `T_d`; `T_d=1` is the controlled schedule-equivalence point.",
        "",
        "Role mask bits are producer=1, tail=2, writer=4.", "",
        "| bits | batch | T_d | role mask | best weights | kernel ms | throughput/writer-final | throughput/v0.6 |",
        "|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for key in sorted(best):
        row = best[key]
        lines.append(
            f"| {row['word_bits']} | {row['batch']} | {row['data_time']} | "
            f"{row['role_mask']} | `{row['roles']}` | {row['kernel_ms']:.6f} | "
            f"{row['throughput_vs_writer_final']:.3f}x | "
            f"{row['throughput_vs_v06']:.3f}x |")
    useful = [row for row in best.values() if row["data_time"] > 1]
    overall = max(useful, key=lambda row: row["throughput_vs_writer_final"])
    all_role = [row for row in useful if row["role_mask"] == 7]
    best_all_role = max(all_role, key=lambda row: row["throughput_vs_writer_final"])
    lines.extend([
        "",
        "## Interpretation", "",
        f"The best temporal placement is mask `{overall['role_mask']}` at "
        f"{overall['word_bits']}-bit batch {overall['batch']} and `T_d={overall['data_time']}`: "
        f"{overall['throughput_vs_writer_final']:.3f}x writer-final and "
        f"{overall['throughput_vs_v06']:.3f}x v0.6 throughput.",
        "",
        f"The best all-role (`7`) temporal point reaches only "
        f"{best_all_role['throughput_vs_writer_final']:.3f}x writer-final. "
        "Keeping producer in spatial wavefront order therefore avoids a measurable downstream-readiness delay, but role placement alone does not close the physical-core gap.",
        "",
        "This data-time core reuses coefficient/layout state inside each physical role. Producer, tail, and writer still exchange transform state through global memory, so this experiment is a role-local temporal traversal rather than end-to-end on-chip subgraph residence.",
        "",
        "The `7+7+6` DAG, grouped state format, writer-final codelet, and global boundary count are fixed.",
    ])
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
