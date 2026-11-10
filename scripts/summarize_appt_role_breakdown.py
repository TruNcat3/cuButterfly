#!/usr/bin/env python3
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def median(rows, field):
    return statistics.median(float(row[field]) for row in rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize APPT online role timing")
    parser.add_argument("raw", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.raw.open(newline="")))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["kind"], int(row["bits"]), int(row["batch"]), int(row["role"]))].append(row)

    records = []
    for bits in (32, 64):
        for batch in (1, 4):
            online = [grouped[("online", bits, batch, role)] for role in range(3)]
            v06 = [grouped[("v06", bits, batch, role)] for role in range(2)]
            kernel_ms = median(online[0], "kernel_ms")
            v06_ms = median(v06[0], "kernel_ms")
            active_ns = 0.0
            wait_ns = 0.0
            for role, role_rows in enumerate(online):
                tasks = median(role_rows, "tasks")
                wait = median(role_rows, "wait_ns")
                compute = median(role_rows, "compute_ns")
                boundary = median(role_rows, "boundary_ns")
                active_ns += compute + boundary
                wait_ns += wait
                records.append({
                    "word_bits": bits,
                    "batch": batch,
                    "role": role,
                    "kernel_ms": kernel_ms,
                    "v06_ms": v06_ms,
                    "tasks": tasks,
                    "wait_us_per_task": wait / tasks / 1000.0,
                    "compute_us_per_task": compute / tasks / 1000.0,
                    "boundary_us_per_task": boundary / tasks / 1000.0,
                    "span_us": median(role_rows, "span_ns") / 1000.0,
                    "start_delay_us": median(role_rows, "start_delay_ns") / 1000.0,
                })
            grid = 80 * (3 if bits == 32 else 2)
            ideal_ms = active_ns / grid / 1.0e6
            for record in records[-3:]:
                record["work_conserving_lower_bound_ms"] = ideal_ms
                record["measured_pipeline_efficiency"] = ideal_ms / kernel_ms
                record["wait_share"] = wait_ns / (wait_ns + active_ns)
                record["online_over_v06"] = v06_ms / kernel_ms
                record["lower_bound_over_v06"] = v06_ms / ideal_ms

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# APPT Online Per-Role Breakdown", "",
        "Intrusive device-globaltimer diagnostics; normal performance kernels do not include these timers.", "",
        "| bits | batch | role | wait us/task | compute us/task | boundary us/task | role span us | start delay us |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['word_bits']} | {row['batch']} | {row['role']} | "
            f"{row['wait_us_per_task']:.2f} | {row['compute_us_per_task']:.2f} | "
            f"{row['boundary_us_per_task']:.2f} | {row['span_us']:.1f} | "
            f"{row['start_delay_us']:.1f} |"
        )
    lines += ["", "## Pipeline Lower Bound", "",
              "The lower bound divides summed measured active CTA time by resident grid size. "
              "It removes all readiness waiting and assumes perfect work-conserving role balance.", "",
              "| bits | batch | measured ms | v0.6 ms | current/v0.6 | active lower bound ms | efficiency | lower-bound/v0.6 | wait share |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for index in range(0, len(records), 3):
        row = records[index]
        lines.append(
            f"| {row['word_bits']} | {row['batch']} | {row['kernel_ms']:.4f} | "
            f"{row['v06_ms']:.4f} | {row['online_over_v06']:.3f}x | "
            f"{row['work_conserving_lower_bound_ms']:.4f} | "
            f"{100.0 * row['measured_pipeline_efficiency']:.1f}% | "
            f"{row['lower_bound_over_v06']:.3f}x | {100.0 * row['wait_share']:.1f}% |"
        )
    target = records[-3:]
    row = target[0]
    compute_ns_per_point = [
        target[0]["compute_us_per_task"] * 1000.0 / 16384,
        target[1]["compute_us_per_task"] * 1000.0 / 16384,
        target[2]["compute_us_per_task"] * 1000.0 / 8192,
    ]
    schedule_excess = row["kernel_ms"] - row["work_conserving_lower_bound_ms"]
    physical_excess = row["work_conserving_lower_bound_ms"] - row["v06_ms"]
    total_excess = row["kernel_ms"] - row["v06_ms"]
    lines += [
        "", "## Uint64 Batch-4 Attribution", "",
        f"Compute time normalized by task points is `{compute_ns_per_point[0]:.2f}`, "
        f"`{compute_ns_per_point[1]:.2f}`, and `{compute_ns_per_point[2]:.2f} ns/point` "
        "for folds 0/1/2. The butterfly work is balanced; there is no isolated slow fold formula.", "",
        f"Relative to v0.6, `{100.0 * schedule_excess / total_excess:.1f}%` of the measured "
        f"excess is above the work-conserving active-work lower bound, while "
        f"`{100.0 * physical_excess / total_excess:.1f}%` remains below that bound. "
        "The first part is the target for ready-task stealing and role migration. The residual "
        "requires reducing the extra global fold and using the resident radix-4/twiddle core.",
    ]
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
