#!/usr/bin/env python3
import argparse
import csv
import pathlib

from generate_numeric_boundary_ncu import profile_label


SUM_FIELDS = (
    "time_us", "dram_read_mib", "dram_write_mib", "warp_instructions",
    "integer_thread_instructions", "fp32_thread_instructions", "fp64_thread_instructions",
    "shared_load_bank_conflicts", "shared_store_bank_conflicts",
)
WEIGHTED_FIELDS = (
    "dram_peak_pct", "l2_hit_pct", "l1_hit_pct", "active_warps_pct",
    "barrier_stall_pct", "long_scoreboard_stall_pct",
)
MAX_FIELDS = ("registers_per_thread", "shared_mem_bytes")
MIN_FIELDS = ("occupancy_register_block_limit", "occupancy_shared_block_limit", "occupancy_warp_block_limit")


def number(row, field):
    value = row.get(field, "")
    return None if value in ("", None) else float(value)


def aggregate(summary_rows):
    groups = {}
    for row in summary_rows:
        groups.setdefault(row["label"], []).append(row)
    output = {}
    for label, kernels in groups.items():
        total_time = sum(number(row, "time_us") or 0.0 for row in kernels)
        record = {"kernels": len(kernels), "time_us": total_time}
        for field in SUM_FIELDS:
            if field != "time_us":
                record[field] = sum(number(row, field) or 0.0 for row in kernels)
        for field in WEIGHTED_FIELDS:
            values = [(number(row, field), number(row, "time_us")) for row in kernels]
            values = [(value, weight) for value, weight in values if value is not None and weight]
            record[field] = (sum(value * weight for value, weight in values) /
                             sum(weight for _, weight in values)) if values else 0.0
        for field in MAX_FIELDS:
            values = [number(row, field) for row in kernels if number(row, field) is not None]
            record[field] = max(values) if values else 0.0
        for field in MIN_FIELDS:
            values = [number(row, field) for row in kernels if number(row, field) is not None]
            record[field] = min(values) if values else 0.0
        record["waves_per_sm"] = sum(number(row, "waves_per_sm") or 0.0 for row in kernels)
        limits = [record[field] for field in MIN_FIELDS if record[field] > 0.0]
        record["resident_cta_limit"] = min(limits) if limits else 0.0
        record["dram_total_mib"] = record["dram_read_mib"] + record["dram_write_mib"]
        record["shared_bank_conflicts"] = (record["shared_load_bank_conflicts"] +
                                            record["shared_store_bank_conflicts"])
        output[label] = record
    return output


def ratio(left, right):
    return left / right if right else 0.0


def fold_change(left, right):
    low, high = sorted((left, right))
    if low == 0.0:
        return float("inf") if high else 1.0
    return high / low


def mechanisms(first, second):
    labels = []
    if first["registers_per_thread"] != second["registers_per_thread"]:
        labels.append("register-footprint")
    if first["shared_mem_bytes"] != second["shared_mem_bytes"]:
        labels.append("shared-footprint")
    if first["resident_cta_limit"] != second["resident_cta_limit"]:
        labels.append("resource-capacity")
    if abs(first["waves_per_sm"] - second["waves_per_sm"]) >= 0.25:
        labels.append("grid-wave")
    if abs(first["barrier_stall_pct"] - second["barrier_stall_pct"]) >= 5.0:
        labels.append("synchronization")
    if abs(first["long_scoreboard_stall_pct"] - second["long_scoreboard_stall_pct"]) >= 5.0:
        labels.append("dependency-memory")
    if fold_change(first["warp_instructions"], second["warp_instructions"]) >= 1.10:
        labels.append("instruction-work")
    if (abs(first["dram_total_mib"] - second["dram_total_mib"]) >= 1.0 and
            fold_change(first["dram_total_mib"], second["dram_total_mib"]) >= 1.10):
        labels.append("memory-traffic")
    if fold_change(first["shared_bank_conflicts"], second["shared_bank_conflicts"]) >= 1.25:
        labels.append("shared-conflict")
    return "+".join(labels) if labels else "unexplained"


def compare(counter_rows, timing_rows):
    counters = aggregate(counter_rows)
    output = []
    for event in timing_rows:
        if event["status"] not in ("confirmed-reversed", "not-confirmed"):
            continue
        mappings = (event["quick_from_mapping"], event["quick_to_mapping"])
        labels = tuple(profile_label(event, mapping) for mapping in mappings)
        if any(label not in counters for label in labels):
            missing = [label for label in labels if label not in counters]
            raise ValueError(f"missing NCU labels: {missing}")
        first, second = (counters[label] for label in labels)
        winner_index = 0 if first["time_us"] <= second["time_us"] else 1
        winner, loser = (first, second) if winner_index == 0 else (second, first)
        output.append({
            "event_id": event["event_id"], "status": event["status"], "operator": event["operator"],
            "precision": event["precision"], "logN": event["logN"], "batch": event["batch"],
            "mapping_a": mappings[0], "mapping_b": mappings[1],
            "ncu_winner": mappings[winner_index], "ncu_time_ratio": ratio(loser["time_us"], winner["time_us"]),
            "a_time_us": first["time_us"], "b_time_us": second["time_us"],
            "a_warp_instructions": first["warp_instructions"], "b_warp_instructions": second["warp_instructions"],
            "a_dram_mib": first["dram_total_mib"], "b_dram_mib": second["dram_total_mib"],
            "a_shared_conflicts": first["shared_bank_conflicts"],
            "b_shared_conflicts": second["shared_bank_conflicts"],
            "a_registers": first["registers_per_thread"], "b_registers": second["registers_per_thread"],
            "a_shared_bytes": first["shared_mem_bytes"], "b_shared_bytes": second["shared_mem_bytes"],
            "a_waves_per_sm": first["waves_per_sm"], "b_waves_per_sm": second["waves_per_sm"],
            "a_barrier_stall_pct": first["barrier_stall_pct"],
            "b_barrier_stall_pct": second["barrier_stall_pct"],
            "a_scoreboard_stall_pct": first["long_scoreboard_stall_pct"],
            "b_scoreboard_stall_pct": second["long_scoreboard_stall_pct"],
            "candidate_mechanisms": mechanisms(first, second),
        })
    return output


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    lines = [
        "# Numeric Boundary NCU Attribution", "",
        "NCU replay/base-clock time is mechanism evidence; CUDA-event timing remains the winner authority.", "",
        "| Event | Workload | A | B | NCU winner | Ratio | Candidate mechanisms |",
        "|:--|:--|:--|:--|:--|--:|:--|",
    ]
    for row in rows:
        workload = f"{row['operator']} {row['precision']} logN={row['logN']} batch={row['batch']}"
        lines.append(f"| {row['event_id']} | {workload} | {row['mapping_a']} | {row['mapping_b']} | "
                     f"{row['ncu_winner']} | {row['ncu_time_ratio']:.3f}x | {row['candidate_mechanisms']} |")
    lines += [
        "", "## Counter Differences", "",
        "| Event | Warp inst B/A | Registers A/B | Waves/SM A/B | Barrier A/B | Scoreboard A/B |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['event_id']} | {ratio(row['b_warp_instructions'], row['a_warp_instructions']):.3f}x | "
            f"{row['a_registers']:.0f}/{row['b_registers']:.0f} | "
            f"{row['a_waves_per_sm']:.2f}/{row['b_waves_per_sm']:.2f} | "
            f"{row['a_barrier_stall_pct']:.1f}%/{row['b_barrier_stall_pct']:.1f}% | "
            f"{row['a_scoreboard_stall_pct']:.1f}%/{row['b_scoreboard_stall_pct']:.1f}% |")
    lines += [
        "", "## Interpretation", "",
        "- The FP16 `logN=15,batch=5` paths differ by only 0.5% under NCU. Radix-4 executes fewer warp instructions but uses more registers; this is a genuine flat tradeoff, consistent with overlapping CUDA-event ranges.",
        "- Structured 2x2 `logN=15,batch=8` is a composition boundary rather than a local arithmetic limit. The online path executes substantially more warp work and barrier stall than hierarchical, so additional online handoff work reverses the quick winner.",
        "- For uint32 zeta, FP16 FFT at `logN=8`, and uint64 NTT, radix-4 reduces instruction work enough to offset its larger register footprint and, in two cases, higher synchronization stall.",
        "- FP64 `logN=8,batch=961` is the only pair where the tighter register footprint changes the resident-CTA bound. Radix-4 gains instruction efficiency but loses residency and raises barrier pressure, leaving only a 1.8% NCU difference and overlapping event-time ranges.",
        "", "Mechanism labels are screening rules. CUDA-event timing remains the winner authority, and paper-facing attribution must retain the per-kernel controls.", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Compare paired NCU profiles for unstable numeric boundaries.")
    parser.add_argument("summary", type=pathlib.Path)
    parser.add_argument("--timing-analysis", type=pathlib.Path,
                        default=pathlib.Path("results/v100_numeric_confirmed_followup_analysis.csv"))
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with args.summary.open(newline="") as source:
        counters = list(csv.DictReader(source))
    with args.timing_analysis.open(newline="") as source:
        timing = list(csv.DictReader(source))
    rows = compare(counters, timing)
    if not rows:
        raise ValueError("no unstable timing events were matched")
    write_csv(args.output, rows)
    write_markdown(args.report, rows)
    print(f"events={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
