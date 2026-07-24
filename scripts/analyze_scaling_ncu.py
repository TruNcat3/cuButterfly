#!/usr/bin/env python3
import argparse
import csv
import pathlib
import re


SUM_FIELDS = ("time_us", "dram_read_mib", "dram_write_mib", "warp_instructions",
              "fp32_thread_instructions", "shared_load_bank_conflicts", "shared_store_bank_conflicts")
WEIGHTED_FIELDS = ("dram_peak_pct", "l2_hit_pct", "l1_hit_pct", "active_warps_pct",
                   "barrier_stall_pct", "long_scoreboard_stall_pct")
MAX_FIELDS = ("registers_per_thread", "shared_mem_bytes")
LABEL = re.compile(r"^(fft|fwht)(\d+)_([^_]+)_b(\d+)$")


def number(row, field):
    value = row.get(field, "")
    return None if value in ("", None) else float(value)


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["label"], []).append(row)
    output = []
    for label, kernels in sorted(groups.items()):
        match = LABEL.match(label)
        if match is None:
            raise ValueError(f"unsupported crossover label {label}")
        operator, log_n, implementation, batch = match.groups()
        total_time = sum(number(row, "time_us") or 0.0 for row in kernels)
        record = {
            "label": label, "operator": operator, "logN": int(log_n),
            "implementation": implementation, "batch": int(batch), "kernels": len(kernels),
        }
        for field in SUM_FIELDS:
            record[field] = sum(number(row, field) or 0.0 for row in kernels)
        for field in WEIGHTED_FIELDS:
            values = [(number(row, field), number(row, "time_us")) for row in kernels]
            values = [(value, weight) for value, weight in values if value is not None and weight is not None]
            record[field] = sum(value * weight for value, weight in values) / sum(weight for _, weight in values) if values else 0.0
        for field in MAX_FIELDS:
            values = [number(row, field) for row in kernels if number(row, field) is not None]
            record[field] = max(values) if values else 0.0
        waves = [number(row, "waves_per_sm") for row in kernels if number(row, "waves_per_sm") is not None]
        record["total_waves_per_sm"] = sum(waves)
        points = (1 << record["logN"]) * record["batch"]
        record["dram_bytes_per_point"] = (record["dram_read_mib"] + record["dram_write_mib"]) * 1024 * 1024 / points
        record["profiled_gpoint_s"] = points / (total_time * 1.0e3) if total_time else 0.0
        output.append(record)
    return output


def conclusion(rows, operator, log_n, implementation, batches):
    selected = [row for row in rows if row["operator"] == operator and row["logN"] == log_n
                and row["implementation"] == implementation and row["batch"] in batches]
    selected.sort(key=lambda row: row["batch"])
    if len(selected) < 2:
        return f"{operator.upper()} logN={log_n} {implementation}: insufficient counter rows."
    low, high = selected[0], selected[-1]
    wave_ratio = high["total_waves_per_sm"] / low["total_waves_per_sm"] if low["total_waves_per_sm"] else 0.0
    traffic_change = high["dram_bytes_per_point"] / low["dram_bytes_per_point"] if low["dram_bytes_per_point"] else 0.0
    return (f"{operator.upper()} logN={log_n} {implementation}: waves/SM change {wave_ratio:.2f}x from "
            f"batch {low['batch']} to {high['batch']}; active warps {low['active_warps_pct']:.1f}% -> "
            f"{high['active_warps_pct']:.1f}%, DRAM peak {low['dram_peak_pct']:.1f}% -> "
            f"{high['dram_peak_pct']:.1f}%, bytes/point changes {traffic_change:.2f}x, barrier stalls "
            f"{low['barrier_stall_pct']:.1f}% -> {high['barrier_stall_pct']:.1f}%, and long-scoreboard "
            f"stalls {low['long_scoreboard_stall_pct']:.1f}% -> {high['long_scoreboard_stall_pct']:.1f}%.")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for field, value in row.items():
                if isinstance(value, float):
                    formatted[field] = f"{value:.6f}"
            writer.writerow(formatted)


def write_markdown(path, rows):
    lines = ["# V100 Scaling-Crossover Counter Attribution", "",
             "NCU replay time is used only for attribution; CUDA-event time remains the performance authority.", "",
             "| Workload | Batch | Impl | Kernels | Waves/SM | Active warps | DRAM peak | B/point | Barrier stall | Scoreboard stall | Reg/thread | Shared B |",
             "|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for row in rows:
        lines.append(f"| {row['operator']} logN={row['logN']} | {row['batch']} | {row['implementation']} | "
                     f"{row['kernels']} | {row['total_waves_per_sm']:.2f} | {row['active_warps_pct']:.1f}% | "
                     f"{row['dram_peak_pct']:.1f}% | {row['dram_bytes_per_point']:.1f} | "
                     f"{row['barrier_stall_pct']:.1f}% | {row['long_scoreboard_stall_pct']:.1f}% | "
                     f"{row['registers_per_thread']:.0f} | {row['shared_mem_bytes']:.0f} |")
    lines += ["", "## Measured Trends", "",
              "- " + conclusion(rows, "fft", 14, "direct", {16, 256, 1024}),
              "- " + conclusion(rows, "fft", 18, "online", {2, 16, 64}),
              "- " + conclusion(rows, "fft", 20, "online", {2, 8, 16}),
              "- " + conclusion(rows, "fwht", 15, "online", {4, 16}),
              "- " + conclusion(rows, "fwht", 15, "warp", {4, 16}), "",
              "Interpret the FWHT crossover by comparing the last two rows together: the selected unit changes only if the register-resident unit's added occupancy and lower exchange cost outweigh its low-batch launch cost.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Aggregate and interpret the selected V100 scaling-crossover counters.")
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--markdown", required=True, type=pathlib.Path)
    args = parser.parse_args()
    with args.input.open() as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("NCU summary is empty")
    aggregated = aggregate(rows)
    write_csv(args.output, aggregated)
    write_markdown(args.markdown, aggregated)


if __name__ == "__main__":
    main()
