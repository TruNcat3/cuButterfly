#!/usr/bin/env python3
"""Compare dynamic SASS instruction mixes from NCU SourceCounters pages."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


CONTROL = {
    "BRA", "BRX", "JMP", "JMX", "CALL", "RET", "EXIT", "YIELD",
    "NANOSLEEP", "BSSY", "BSYNC", "BMOV", "BREAK", "CONT", "SSY",
    "SYNC", "PBK", "PCNT", "PRET",
}
SYNCHRONIZATION = {"BAR", "WARPSYNC", "MEMBAR", "ERRBAR", "CCTL"}
GLOBAL_MEMORY = {"LDG", "STG", "ATOM", "ATOMG", "RED", "REDG"}
SHARED_MEMORY = {"LDS", "STS", "ATOMS", "REDS"}
LOCAL_MEMORY = {"LDL", "STL"}


def normalized(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_number(value):
    value = value.strip().replace(",", "")
    if not value or value.lower() in {"-", "n/a", "nan", "no data"}:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def opcode(sass):
    text = sass.strip()
    text = re.sub(r"^@!?P(?:T|\d+)\s+", "", text)
    match = re.match(r"([A-Z][A-Z0-9]*)", text)
    return match.group(1) if match else "UNKNOWN"


def category(op):
    if op in GLOBAL_MEMORY:
        return "global memory"
    if op in SHARED_MEMORY:
        return "shared memory"
    if op in LOCAL_MEMORY:
        return "local memory"
    if op == "SHFL":
        return "shuffle"
    if op in SYNCHRONIZATION:
        return "synchronization/cache"
    if op in CONTROL:
        return "control"
    if op in {"NOP", "DEPBAR"}:
        return "scheduling"
    return "integer/arithmetic"


def read_source_instructions(path):
    """Return one record per SASS PC, selecting the richest inline view."""
    rows = list(csv.reader(path.read_text(errors="replace").splitlines()))
    header = None
    address_index = sass_index = instruction_index = None
    by_address = {}
    for row in rows:
        names = [normalized(cell) for cell in row]
        if ("address" in names and "instructions_executed" in names and
                names.count("source") >= 2):
            header = row
            address_index = names.index("address")
            instruction_index = names.index("instructions_executed")
            source_indices = [index for index, name in enumerate(names)
                              if name == "source"]
            sass_index = source_indices[1]
            continue
        if header is None:
            continue
        padded = row + [""] * (len(header) - len(row))
        address = padded[address_index].strip()
        if not address.lower().startswith("0x"):
            continue
        sass = padded[sass_index].strip()
        count = parse_number(padded[instruction_index])
        if not sass or sass in {"-", "..."} or count == 0:
            continue
        record = {"address": address, "sass": sass,
                  "opcode": opcode(sass), "instructions": count}
        previous = by_address.get(address)
        if previous is None or count > previous["instructions"]:
            by_address[address] = record
    if not by_address:
        raise ValueError(f"no SASS PC instruction records found in {path}")
    return list(by_address.values())


def aggregate(records):
    opcodes = defaultdict(float)
    categories = defaultdict(float)
    for record in records:
        op = record["opcode"]
        count = record["instructions"]
        opcodes[op] += count
        categories[category(op)] += count
    return {"total": sum(opcodes.values()), "opcodes": dict(opcodes),
            "categories": dict(categories), "pcs": len(records)}


def millions(value):
    return f"{value / 1.0e6:.3f}"


def make_report(baseline, candidate, baseline_name="v0.6",
                candidate_name="v0.7"):
    base_total = baseline["total"]
    candidate_total = candidate["total"]
    category_names = sorted(
        set(baseline["categories"]) | set(candidate["categories"]),
        key=lambda name: candidate["categories"].get(name, 0.0) -
        baseline["categories"].get(name, 0.0), reverse=True)
    lines = [
        "# v0.7 Dynamic Instruction Attribution", "",
        "Counts are dynamic warp instructions from NCU SourceCounters. SASS PCs "
        "are deduplicated across inline source-stack views.", "",
        f"| category | {baseline_name} (M) | {candidate_name} (M) | delta (M) | candidate/baseline |",
        "|:--|--:|--:|--:|--:|",
    ]
    for name in category_names:
        base = baseline["categories"].get(name, 0.0)
        value = candidate["categories"].get(name, 0.0)
        ratio = value / base if base else float("inf")
        ratio_text = f"{ratio:.3f}x" if ratio != float("inf") else "inf"
        lines.append(
            f"| {name} | {millions(base)} | {millions(value)} | "
            f"{millions(value - base)} | {ratio_text} |")
    lines.append(
        f"| **total** | **{millions(base_total)}** | "
        f"**{millions(candidate_total)}** | "
        f"**{millions(candidate_total - base_total)}** | "
        f"**{candidate_total / base_total:.3f}x** |")

    opcode_names = sorted(
        set(baseline["opcodes"]) | set(candidate["opcodes"]),
        key=lambda name: candidate["opcodes"].get(name, 0.0) -
        baseline["opcodes"].get(name, 0.0), reverse=True)
    lines.extend(["", "## Largest Opcode Deltas", "",
                  f"| opcode | {baseline_name} (M) | {candidate_name} (M) | delta (M) |",
                  "|:--|--:|--:|--:|"])
    for name in opcode_names[:12]:
        base = baseline["opcodes"].get(name, 0.0)
        value = candidate["opcodes"].get(name, 0.0)
        lines.append(f"| {name} | {millions(base)} | {millions(value)} | "
                     f"{millions(value - base)} |")

    delta = candidate_total - base_total
    positive = [(name, candidate["categories"].get(name, 0.0) -
                 baseline["categories"].get(name, 0.0))
                for name in category_names]
    positive = [(name, value) for name, value in positive if value > 0]
    lines.extend(["", "## Interpretation", ""])
    if delta <= 0:
        lines.append("The candidate does not have a dynamic instruction-count deficit.")
    elif positive:
        name, value = positive[0]
        lines.append(
            f"The largest positive category is **{name}**, contributing "
            f"{value / delta * 100.0:.1f}% of the net {millions(delta)}M-instruction "
            "gap. Optimize that physical-codelet path before further CTA scheduling scans.")
    lines.append(
        f"Source coverage: {baseline_name}={baseline['pcs']} PCs, "
        f"{candidate_name}={candidate['pcs']} PCs.")
    return "\n".join(lines) + "\n"


def write_opcode_csv(path, aggregates):
    names = sorted(set().union(*(value["opcodes"] for value in aggregates.values())))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["opcode", *aggregates])
        for name in names:
            writer.writerow([name, *(aggregates[label]["opcodes"].get(name, 0.0)
                                      for label in aggregates)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline-name", default="v0.6")
    parser.add_argument("--candidate-name", default="v0.7")
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    baseline = aggregate(read_source_instructions(args.baseline))
    candidate = aggregate(read_source_instructions(args.candidate))
    report = make_report(baseline, candidate, args.baseline_name,
                         args.candidate_name)
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end="")
    if args.csv:
        write_opcode_csv(args.csv, {args.baseline_name: baseline,
                                    args.candidate_name: candidate})


if __name__ == "__main__":
    main()
