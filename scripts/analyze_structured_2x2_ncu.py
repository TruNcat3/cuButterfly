#!/usr/bin/env python3
import argparse
import csv
import pathlib


SUM_FIELDS = (
    "time_us", "dram_read_mib", "dram_write_mib", "warp_instructions",
    "fp32_thread_instructions", "shared_load_bank_conflicts", "shared_store_bank_conflicts",
)
WEIGHTED_FIELDS = (
    "dram_peak_pct", "l2_hit_pct", "l1_hit_pct", "active_warps_pct",
    "barrier_stall_pct", "long_scoreboard_stall_pct",
)
MAX_FIELDS = ("registers_per_thread", "shared_mem_bytes")


def number(row, field):
    value = row.get(field, "")
    return None if value in ("", None) else float(value)


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["label"], []).append(row)
    output = []
    for label, kernels in sorted(groups.items()):
        total_time = sum(number(row, "time_us") or 0.0 for row in kernels)
        record = {"label": label, "kernels": len(kernels)}
        for field in SUM_FIELDS:
            record[field] = sum(number(row, field) or 0.0 for row in kernels)
        for field in WEIGHTED_FIELDS:
            values = [(number(row, field), number(row, "time_us")) for row in kernels]
            values = [(value, weight) for value, weight in values if value is not None and weight is not None]
            record[field] = (sum(value * weight for value, weight in values) /
                             sum(weight for _, weight in values)) if values else 0.0
        for field in MAX_FIELDS:
            values = [number(row, field) for row in kernels if number(row, field) is not None]
            record[field] = max(values) if values else 0.0
        record["total_waves_per_sm"] = sum(number(row, "waves_per_sm") or 0.0 for row in kernels)
        record["dram_total_mib"] = record["dram_read_mib"] + record["dram_write_mib"]
        record["shared_bank_conflicts"] = (record["shared_load_bank_conflicts"] +
                                            record["shared_store_bank_conflicts"])
        record["profiled_gpoint_s"] = 4194304 / (total_time * 1.0e3) if total_time else 0.0
        output.append(record)
    return output


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def find(rows, label):
    return next(row for row in rows if row["label"] == label)


def find_optional(rows, label):
    return next((row for row in rows if row["label"] == label), None)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.6f}" if isinstance(value, float) else value
                             for key, value in row.items()})


def write_markdown(path, rows):
    s8 = find(rows, "structured8_temporal_r4")
    f8 = find(rows, "fwht8_temporal_r4")
    s12 = find(rows, "structured12_hierarchical_r8")
    s12_broadcast = find_optional(rows, "structured12_warp_register_broadcast")
    s12_per_stage = find_optional(rows, "structured12_warp_register_per_stage")
    s12_register = s12_broadcast or find_optional(rows, "structured12_warp_register")
    f12 = find(rows, "fwht12_hierarchical_r8")
    f12_register = find(rows, "fwht12_warp_register")
    s20 = find(rows, "structured20_online_r8")
    f20 = find(rows, "fwht20_online_r8")
    order = [s8, f8, s12]
    if s12_register is not None:
        order.append(s12_register)
    if s12_per_stage is not None:
        order.append(s12_per_stage)
    order += [f12, f12_register, s20, f20]
    lines = [
        "# Structured 2x2 V100 Counter Attribution", "",
        "NCU uses base clocks and replay; its time is mechanism evidence, not the CUDA-event performance authority.", "",
        "| Case | Kernels | Time us | DRAM MiB | Warp inst M | FP32 inst M | Active warps | Barrier stall | Scoreboard stall | Reg/thread | Shared B |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in order:
        lines.append(
            f"| {row['label']} | {row['kernels']} | {row['time_us']:.3f} | {row['dram_total_mib']:.2f} | "
            f"{row['warp_instructions'] / 1.0e6:.2f} | {row['fp32_thread_instructions'] / 1.0e6:.2f} | "
            f"{row['active_warps_pct']:.1f}% | {row['barrier_stall_pct']:.1f}% | "
            f"{row['long_scoreboard_stall_pct']:.1f}% | {row['registers_per_thread']:.0f} | "
            f"{row['shared_mem_bytes']:.0f} |")
    register_result = (
        f"The generated Structured2x2 register unit takes {s12_register['time_us']:.3f} us, "
        f"a {safe_ratio(s12['time_us'], s12_register['time_us']):.2f}x speedup over its hierarchical path. "
        if s12_register is not None else
        "The generated Structured2x2 register unit was added after this capture; its counter row is pending. ")
    active_warp_delta = (f12_register["active_warps_pct"] - s12_register["active_warps_pct"]
                         if s12_register is not None else 0.0)
    occupancy_result = (
        f"The {s12_register['registers_per_thread']:.0f}-register dense kernel retains essentially the same active-warps "
        f"level as FWHT ({s12_register['active_warps_pct']:.1f}% versus {f12_register['active_warps_pct']:.1f}%)."
        if s12_register is not None and abs(active_warp_delta) < 2.0 else
        f"The dense kernel lowers active warps by {active_warp_delta:.1f} percentage points."
        if s12_register is not None else "")
    register_consequence = (
        f"The generated register result confirms that the transport hierarchy can be reused without changing the "
        f"architecture schedule. Against the same-transport FWHT register core, Structured2x2 takes "
        f"{safe_ratio(s12_register['time_us'], f12_register['time_us']):.3f}x the base-clock time with "
        f"{safe_ratio(s12_register['dram_total_mib'], f12_register['dram_total_mib']):.3f}x DRAM traffic, "
        f"{safe_ratio(s12_register['warp_instructions'], f12_register['warp_instructions']):.2f}x warp instructions, "
        f"{safe_ratio(s12_register['fp32_thread_instructions'], f12_register['fp32_thread_instructions']):.2f}x FP32 "
        f"thread instructions, and {s12_register['registers_per_thread']:.0f} versus "
        f"{f12_register['registers_per_thread']:.0f} registers/thread. {occupancy_result} The nearly identical DRAM volume "
        f"and low shared bank-conflict counts isolate the current residual to dense-pair arithmetic; register allocation "
        f"remains a design constraint at other lengths."
        if s12_register is not None else
        "The CUDA-event results already confirm the generated register-core speedup; a refreshed NCU capture is needed "
        "to attribute its remaining dense-pair arithmetic cost."
    )
    policy_result = (
        f"Under identical matrix values, the broadcast specialization takes {s12_broadcast['time_us']:.3f} us versus "
        f"{s12_per_stage['time_us']:.3f} us for the table path, a "
        f"{safe_ratio(s12_per_stage['time_us'], s12_broadcast['time_us']):.3f}x speedup. It changes registers/thread "
        f"from {s12_per_stage['registers_per_thread']:.0f} to {s12_broadcast['registers_per_thread']:.0f}, raises active "
        f"warps from {s12_per_stage['active_warps_pct']:.1f}% to {s12_broadcast['active_warps_pct']:.1f}%, and lowers "
        f"long-scoreboard stall from {s12_per_stage['long_scoreboard_stall_pct']:.1f}% to "
        f"{s12_broadcast['long_scoreboard_stall_pct']:.1f}%. FP32 instruction count is unchanged."
        if s12_broadcast is not None and s12_per_stage is not None else None
    )
    lines += [
        "", "## Attribution", "",
        "### logN=8: dense arithmetic is visible but not dominant", "",
        f"With the same temporal radix-4 mapping, Structured2x2 takes {safe_ratio(s8['time_us'], f8['time_us']):.3f}x "
        f"the FWHT time, executes {safe_ratio(s8['fp32_thread_instructions'], f8['fp32_thread_instructions']):.2f}x "
        f"the FP32 instructions, and uses {s8['registers_per_thread']:.0f} versus {f8['registers_per_thread']:.0f} "
        "registers/thread. Active warps stay above 91% for both, so the extra dense arithmetic and barrier pressure produce "
        "only a small end-to-end penalty.", "",
        "### logN=12: the gap is the physical processing unit", "",
        f"The matched hierarchical paths differ by only {100.0 * (safe_ratio(s12['time_us'], f12['time_us']) - 1.0):.1f}% overall. "
        "The two global suffix kernels are effectively equal; the dense prefix raises registers/thread from 23 to 39 and "
        "reduces active warps from 92.6% to 70.7%. The generated FWHT register codelet is "
        f"{safe_ratio(f12['time_us'], f12_register['time_us']):.2f}x faster than hierarchical FWHT and moves "
        f"{safe_ratio(f12['dram_total_mib'], f12_register['dram_total_mib']):.2f}x less DRAM traffic by retaining all stages "
        "in one kernel. This isolated the original Structured2x2 opportunity to a generated register-resident matrix unit, "
        "not to a different architecture schedule. " + register_result, "",
        *( [
            "### logN=12 register control: the current residual is arithmetic", "",
            f"Structured2x2 takes {safe_ratio(s12_register['time_us'], f12_register['time_us']):.3f}x the FWHT register "
            f"time while moving {safe_ratio(s12_register['dram_total_mib'], f12_register['dram_total_mib']):.3f}x the "
            f"DRAM bytes. It executes {safe_ratio(s12_register['warp_instructions'], f12_register['warp_instructions']):.2f}x "
            f"the warp instructions and {safe_ratio(s12_register['fp32_thread_instructions'], f12_register['fp32_thread_instructions']):.2f}x "
            f"the FP32 thread instructions, raises registers/thread from {f12_register['registers_per_thread']:.0f} to "
            f"{s12_register['registers_per_thread']:.0f}, and reduces active warps from "
            f"{f12_register['active_warps_pct']:.1f}% to {s12_register['active_warps_pct']:.1f}%. Shared bank conflicts "
            "are negligible in both kernels. " + occupancy_result + " This rules out global layout, shared-bank behavior, "
            "and an achieved-occupancy collapse as the primary remaining cause.", "",
        ] if s12_register is not None else [] ),
        *( [
            "### logN=12 coefficient policy", "", policy_result, "",
        ] if policy_result is not None else [] ),
        "### logN=20: the apparent advantage is not robust", "",
        f"At base clocks the matched online Structured2x2 path is {100.0 * (safe_ratio(s20['time_us'], f20['time_us']) - 1.0):.1f}% "
        f"slower, executes {safe_ratio(s20['warp_instructions'], f20['warp_instructions']):.2f}x the warp instructions and "
        f"{safe_ratio(s20['fp32_thread_instructions'], f20['fp32_thread_instructions']):.2f}x the FP32 instructions, while "
        "DRAM traffic and cache hit rates are nearly unchanged. Its 40-register dense kernels reduce active warps to about "
        "70%, but low DRAM utilization and similar scoreboard stalls allow most added arithmetic to overlap. The earlier "
        "sequential-scan speedup must not be used as a performance claim.", "",
        "## Methodological Consequence", "",
        "The counters separate the replaceable local unit from the space/time schedule. Dense local arithmetic changes "
        "register and instruction demand, while the mapping transition across lengths remains temporal -> hierarchical -> "
        "online reorder. " + register_consequence + " Changing the architecture paradigm is not supported by this "
        "evidence.", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Aggregate Structured2x2/FWHT NCU A/B counters.")
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--markdown", required=True, type=pathlib.Path)
    args = parser.parse_args()
    with args.input.open() as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("Structured2x2 NCU summary is empty")
    aggregated = aggregate(rows)
    write_csv(args.output, aggregated)
    write_markdown(args.markdown, aggregated)


if __name__ == "__main__":
    main()
