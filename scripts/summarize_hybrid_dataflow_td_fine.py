#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def read_rows(paths):
    rows = []
    for path in paths:
        with path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        row["case"] = path.stem
        row["word_bits"] = int(row["word_bits"])
        row["logN"] = int(row["logN"])
        row["batch"] = int(row["batch"])
        row["role_stages"] = int(row["role_stages"])
        row["data_time"] = int(row["data_time"])
        row["kernel_ms"] = float(row["kernel_ms"])
        row["correct"] = int(row["correct"])
        rows.append(row)
    return sorted(rows, key=lambda row: (row["word_bits"], row["logN"], row["batch"], row["data_time"]))


def enrich(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["word_bits"], row["logN"], row["batch"])].append(row)
    for group in groups.values():
        best_ms = min(row["kernel_ms"] for row in group)
        previous_ms = None
        for row in group:
            replicas = row["role_stages"]
            row["full_replica_rounds"] = row["data_time"] // replicas
            row["tail_replicas"] = row["data_time"] % replicas
            row["tokens_per_replica"] = row["data_time"] / replicas
            token_count = 1 << (row["logN"] - row["role_stages"])
            row["token_count"] = token_count
            row["packet_count"] = math.ceil(token_count / row["data_time"])
            row["serial_slots_per_packet"] = math.ceil(row["data_time"] / replicas)
            row["serial_token_slots"] = row["packet_count"] * row["serial_slots_per_packet"]
            capacity = row["serial_token_slots"] * replicas
            row["token_capacity_utilization"] = token_count / capacity
            row["regret"] = row["kernel_ms"] / best_ms
            row["adjacent_speedup"] = "" if previous_ms is None else previous_ms / row["kernel_ms"]
            previous_ms = row["kernel_ms"]
    return groups


def write_csv(path, rows):
    fields = ["case", "word_bits", "logN", "batch", "role_stages", "data_time",
              "full_replica_rounds", "tail_replicas", "tokens_per_replica", "kernel_ms",
              "token_count", "packet_count", "serial_slots_per_packet", "serial_token_slots",
              "token_capacity_utilization",
              "adjacent_speedup", "regret", "correct"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def write_markdown(path, groups):
    lines = [
        "# HybridDataflow fine packet-depth scan",
        "",
        "All points use `Ti=1`; `Td` is scanned from 1 through 16.",
        "",
        "| bits | logN | batch | Ur | best Td | Td/Ur | packets | serial slots | utilization | best ms | Td1/best | <=2% plateau | negative adjacent steps |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for key in sorted(groups):
        group = groups[key]
        best = min(group, key=lambda row: row["kernel_ms"])
        td1 = next(row for row in group if row["data_time"] == 1)
        plateau = [str(row["data_time"]) for row in group if row["regret"] <= 1.02]
        negative = [str(row["data_time"]) for row in group
                    if row["adjacent_speedup"] != "" and row["adjacent_speedup"] < 1.0]
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {best['role_stages']} | {best['data_time']} | "
            f"{best['tokens_per_replica']:.2f} | {best['packet_count']} | {best['serial_token_slots']} | "
            f"{best['token_capacity_utilization']:.3f} | {best['kernel_ms']:.6f} | "
            f"{td1['kernel_ms'] / best['kernel_ms']:.3f}x | {','.join(plateau)} | {','.join(negative)} |")

    preferences = defaultdict(list)
    for (bits, log_n, batch), group in groups.items():
        best = min(group, key=lambda row: row["kernel_ms"])
        preferences[(bits, log_n)].append((batch, best["data_time"]))
    lines += ["", "## Winner map", "", "| bits | logN | batch -> best Td |", "|---:|---:|---|"]
    for key in sorted(preferences):
        mapping = ", ".join(f"{batch}->{td}" for batch, td in sorted(preferences[key]))
        lines.append(f"| {key[0]} | {key[1]} | {mapping} |")

    lines += ["", "## Full curves", ""]
    for key in sorted(groups):
        group = groups[key]
        curve = ", ".join(f"{row['data_time']}:{row['kernel_ms']:.6f}" for row in group)
        lines.append(f"- `{key[0]}-bit logN={key[1]} batch={key[2]}`: {curve}")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize fine HybridDataflow Td scans")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()
    rows = read_rows(args.inputs)
    if not rows or any(row["correct"] != 1 for row in rows):
        raise ValueError("fine Td scan is empty or contains an incorrect result")
    groups = enrich(rows)
    write_csv(args.csv, rows)
    write_markdown(args.markdown, groups)


if __name__ == "__main__":
    main()
