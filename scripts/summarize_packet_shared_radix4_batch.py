#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


PACKET = re.compile(r"packet128_b(\d+)_w(\d+)_(\d+)$")
CONTROL = re.compile(r"(v06|lane32)_b(\d+)$")


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows):
    batches = {}
    for row in rows:
        label = row["label"]
        packet = PACKET.fullmatch(label)
        control = CONTROL.fullmatch(label)
        if packet:
            batch, producer, consumer = map(int, packet.groups())
            entry = batches.setdefault(batch, {"packets": []})
            entry["packets"].append((float(row["kernel_ms"]), producer,
                                     consumer))
        elif control:
            name, batch = control.groups()
            batches.setdefault(int(batch), {"packets": []})[name] = float(
                row["kernel_ms"])
    result = []
    for batch in sorted(batches):
        entry = batches[batch]
        if "v06" not in entry or "lane32" not in entry or not entry["packets"]:
            raise ValueError(f"incomplete batch {batch}")
        packet_ms, producer, consumer = min(entry["packets"])
        best_control_name, best_control_ms = min(
            (("v0.6", entry["v06"]), ("lane32", entry["lane32"])),
            key=lambda item: item[1])
        result.append({
            "batch": batch,
            "v06_ms": entry["v06"],
            "lane32_ms": entry["lane32"],
            "packet128_ms": packet_ms,
            "packet_weights": f"{producer}:{consumer}",
            "packet_over_v06": entry["v06"] / packet_ms,
            "packet_over_lane32": entry["lane32"] / packet_ms,
            "best_control": best_control_name,
            "packet_over_best_control": best_control_ms / packet_ms,
            "weight_search_boundary": producer == min(
                candidate[1] for candidate in entry["packets"]),
        })
    return result


def write_csv(path, rows):
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_markdown(rows):
    lines = [
        "# Packet-Shared Radix-4 Batch Crossover", "",
        "CUDA-event medians after warmup. Packet role weights are independently "
        "selected at each batch.", "",
        "| batch | v0.6 (ms) | lane32 (ms) | packet128 (ms) | weights | packet/v0.6 | packet/lane32 | packet/best old |",
        "|--:|--:|--:|--:|:--|--:|--:|--:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch']} | {row['v06_ms']:.6f} | "
            f"{row['lane32_ms']:.6f} | {row['packet128_ms']:.6f} | "
            f"`{row['packet_weights']}` | {row['packet_over_v06']:.3f}x | "
            f"{row['packet_over_lane32']:.3f}x | "
            f"{row['packet_over_best_control']:.3f}x |")
    wins_v06 = sum(row["packet_over_v06"] > 1.0 for row in rows)
    wins_best = sum(row["packet_over_best_control"] > 1.0 for row in rows)
    ratios = [row["packet_over_best_control"] for row in rows]
    lines.extend(["", "## Interpretation", "",
                  f"Packet128 beats v0.6 at {wins_v06}/{len(rows)} points and "
                  f"the best legacy core at {wins_best}/{len(rows)} points. "
                  f"Its best-control-relative throughput ranges from "
                  f"{min(ratios):.3f}x to {max(ratios):.3f}x."])
    boundary_batches = [str(row["batch"]) for row in rows
                        if row["weight_search_boundary"]]
    if boundary_batches:
        lines.append(
            "The selected producer weight lies on the lower search boundary at "
            f"batch {', '.join(boundary_batches)}; those points do not establish "
            "a closed role-allocation optimum.")
    if wins_best == len(rows) and not boundary_batches:
        lines.append(
            "Packet128 wins the closed search against both controls at every point, "
            "supporting selector promotion for this numeric shape.")
    elif wins_v06 == len(rows):
        lines.append(
            "The core consistently replaces v0.6, but the lane32 crossover remains "
            "batch-dependent; retain per-batch selector entries.")
    else:
        lines.append(
            "The crossover is batch-dependent; retain v0.6 below the first stable packet win.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()
    rows = summarize(read_rows(args.raw))
    write_csv(args.csv, rows)
    args.markdown.write_text(make_markdown(rows))


if __name__ == "__main__":
    main()
