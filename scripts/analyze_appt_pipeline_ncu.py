#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


IMPLEMENTATIONS = {
    "barrier": ("barrier_b1", "barrier_b4"),
    "resident-10+10": ("resident1010_b1", "resident1010_b4"),
    "APPT Us8 role1": ("appt_us8_td8_b1", "appt_us8_td8_b4"),
    "APPT Us8 role2": ("appt_us8_rs2_td8_b1", "appt_us8_rs2_td8_b4"),
}

OPTIONAL_IMPLEMENTATIONS = {
    "APPT Us7 role2 rep2 group2": (
        "appt_us7_rs2_rep2_ti2_b1", "appt_us7_rs2_rep2_ti2_b4"),
    "APPT online folds": ("appt_online_b1", "appt_online_b4"),
}


def value(row, name):
    return float(row[name])


def ratio(row, reference, name):
    return value(row, name) / value(reference, name)


def main():
    parser = argparse.ArgumentParser(description="Attribute APPT pipeline NCU cost and fixed overhead")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    args = parser.parse_args()

    with args.summary.open(newline="") as stream:
        indexed = {row["label"]: row for row in csv.DictReader(stream)}
    missing = [label for labels in IMPLEMENTATIONS.values() for label in labels if label not in indexed]
    if missing:
        raise ValueError("missing NCU rows: " + ", ".join(missing))

    implementations = dict(IMPLEMENTATIONS)
    for name, labels in OPTIONAL_IMPLEMENTATIONS.items():
        if all(label in indexed for label in labels):
            implementations[name] = labels

    barrier = indexed["barrier_b1"]
    records = []
    for implementation, (b1_label, b4_label) in implementations.items():
        b1 = indexed[b1_label]
        b4 = indexed[b4_label]
        steady_us = (value(b4, "time_us") - value(b1, "time_us")) / 3.0
        fixed_us = value(b1, "time_us") - steady_us
        records.append({
            "implementation": implementation,
            "batch1_us": value(b1, "time_us"),
            "batch4_us": value(b4, "time_us"),
            "fixed_us": fixed_us,
            "steady_us_per_transform": steady_us,
            "fixed_share_batch1_pct": 100.0 * fixed_us / value(b1, "time_us"),
            "fixed_share_batch4_pct": 100.0 * fixed_us / value(b4, "time_us"),
            "global_load_sector_ratio_b1": ratio(b1, barrier, "global_load_sectors"),
            "global_store_sector_ratio_b1": ratio(b1, barrier, "global_store_sectors"),
            "warp_instruction_ratio_b1": ratio(b1, barrier, "warp_instructions"),
            "integer_instruction_ratio_b1": ratio(b1, barrier, "integer_thread_instructions"),
            "dram_peak_pct_b1": value(b1, "dram_peak_pct"),
            "active_warps_pct_b1": value(b1, "active_warps_pct"),
            "barrier_stall_pct_b1": value(b1, "barrier_stall_pct"),
            "wait_stall_pct_b1": value(b1, "wait_stall_pct"),
            "long_scoreboard_stall_pct_b1": value(b1, "long_scoreboard_stall_pct"),
            "registers_per_thread": value(b1, "registers_per_thread"),
            "shared_mem_bytes": value(b1, "shared_mem_bytes"),
            "waves_per_sm": value(b1, "waves_per_sm"),
        })

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# APPT Pipeline NCU Attribution", "",
        "Matched Tesla V100 captures, uint64 `logN=20`. NCU reports one cooperative kernel "
        "for every implementation; the APPT backend does not use the CUDA Graph API.", "",
        "## Fixed-Cost Fit", "",
        "The two-point model is `T(batch)=T_fixed+batch*T_steady`. It includes cooperative "
        "admission and pipeline fill/drain in the fixed term; it does not misclassify repeated "
        "in-kernel handoffs as launch cost.", "",
        "| implementation | batch 1 us | batch 4 us | fixed us | steady us/transform | fixed share b1 | fixed share b4 |",
        "|:--|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['implementation']} | {row['batch1_us']:.1f} | {row['batch4_us']:.1f} | "
            f"{row['fixed_us']:.1f} | {row['steady_us_per_transform']:.1f} | "
            f"{row['fixed_share_batch1_pct']:.1f}% | {row['fixed_share_batch4_pct']:.1f}% |"
        )

    lines += [
        "", "## Batch-1 Hardware Attribution", "",
        "Ratios use the v0.6 barrier kernel as 1.0x.", "",
        "| implementation | load sectors | store sectors | warp inst | integer inst | DRAM peak | active warps | barrier stall | wait stall | long scoreboard | regs | shared B | waves/SM |",
        "|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['implementation']} | {row['global_load_sector_ratio_b1']:.2f}x | "
            f"{row['global_store_sector_ratio_b1']:.2f}x | {row['warp_instruction_ratio_b1']:.2f}x | "
            f"{row['integer_instruction_ratio_b1']:.2f}x | {row['dram_peak_pct_b1']:.1f}% | "
            f"{row['active_warps_pct_b1']:.1f}% | {row['barrier_stall_pct_b1']:.1f}% | "
            f"{row['wait_stall_pct_b1']:.1f}% | {row['long_scoreboard_stall_pct_b1']:.1f}% | "
            f"{row['registers_per_thread']:.0f} | {row['shared_mem_bytes']:.0f} | "
            f"{row['waves_per_sm']:.2f} |"
        )

    appt = next((row for row in records
                 if row["implementation"] == "APPT online folds"),
                next((row for row in records
                      if row["implementation"] == "APPT Us7 role2 rep2 group2"),
                     next(row for row in records
                          if row["implementation"] == "APPT Us8 role1")))
    baseline = next(row for row in records if row["implementation"] == "barrier")
    resident = next(row for row in records
                    if row["implementation"] == "resident-10+10")
    staged = next((row for row in records
                   if row["implementation"] == "APPT Us7 role2 rep2 group2"),
                  None)
    lines += [
        "", "## Conclusion", "",
        f"The online kernel has a `{appt['fixed_us']:.1f} us` fixed term and a "
        f"`{appt['steady_us_per_transform']:.1f} us/transform` fitted steady cost. "
        f"The mature v0.6 resident core needs `{resident['steady_us_per_transform']:.1f} us/transform`, "
        f"so the remaining steady-state gap is "
        f"`{appt['steady_us_per_transform'] / resident['steady_us_per_transform']:.2f}x`; "
        "launch or graph submission changes cannot remove it.", "",
    ]
    if staged is not None:
        lines += [
            f"Online cross-fold execution improves the staged APPT steady cost by "
            f"`{staged['steady_us_per_transform'] / appt['steady_us_per_transform']:.3f}x`, "
            f"but adds `{appt['fixed_us'] - staged['fixed_us']:.1f} us` of fill/drain cost. "
            f"This produces `{staged['batch1_us'] / appt['batch1_us']:.3f}x` at batch 1 and "
            f"`{staged['batch4_us'] / appt['batch4_us']:.3f}x` at batch 4, directly confirming "
            "that homogeneous load amortizes the online pipeline.", "",
            f"Relative to staged APPT, online execution reduces warp instructions by "
            f"`{100.0 * (1.0 - appt['warp_instruction_ratio_b1'] / staged['warp_instruction_ratio_b1']):.1f}%` "
            f"and integer instructions by "
            f"`{100.0 * (1.0 - appt['integer_instruction_ratio_b1'] / staged['integer_instruction_ratio_b1']):.1f}%`, "
            f"while registers fall from `{staged['registers_per_thread']:.0f}` to "
            f"`{appt['registers_per_thread']:.0f}` per thread. The remaining physical bottleneck is "
            f"the boundary/output layout: store sectors rise from "
            f"`{staged['global_store_sector_ratio_b1']:.2f}x` to "
            f"`{appt['global_store_sector_ratio_b1']:.2f}x`, DRAM utilization falls to "
            f"`{appt['dram_peak_pct_b1']:.1f}%`, and active warps fall to "
            f"`{appt['active_warps_pct_b1']:.1f}%`.",
        ]
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
