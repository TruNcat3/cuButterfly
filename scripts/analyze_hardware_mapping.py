#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import sys


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def round_up(value, unit):
    return ceil_div(value, unit) * unit


def analyze_kernel(hardware, mapping, kernel, batch):
    warp_size = hardware["warp_size"]
    threads = kernel["block_x"] * kernel["block_y"]
    warps = ceil_div(threads, warp_size)

    registers_per_warp = round_up(
        kernel["registers_per_thread"] * warp_size,
        hardware.get("register_allocation_unit_per_warp", 1),
    )
    allocated_registers = registers_per_warp * warps
    allocated_shared = round_up(
        kernel["shared_bytes"],
        hardware.get("shared_allocation_unit_per_cta", 1),
    )

    limits = {
        "register": hardware["registers_per_sm"] // allocated_registers,
        "shared": hardware["shared_bytes_per_sm"] // allocated_shared,
        "thread": hardware["threads_per_sm"] // threads,
        "warp": hardware["warps_per_sm"] // warps,
        "hardware": hardware["max_ctas_per_sm"],
    }
    resident_ctas = min(limits.values())
    limiting_resources = "+".join(name for name, value in limits.items() if value == resident_ctas)

    ctas_per_transform = kernel["grid_x_per_transform"] * kernel["grid_y_per_transform"]
    total_ctas = ctas_per_transform * batch
    ctas_per_wave = hardware["sm_count"] * resident_ctas
    stage_end = kernel["stage_end"]
    unique_roots = 1 if stage_end == 0 else 1 << (stage_end - 1)
    n = 1 << mapping["log_n"]
    coefficient_bytes = 2 * n * batch * mapping["word_bytes"]

    return {
        "hardware": hardware["name"],
        "mapping": mapping["name"],
        "kernel": kernel["name"],
        "stage_begin": kernel["stage_begin"],
        "stage_end": stage_end,
        "stage_temporal_span": stage_end - kernel["stage_begin"],
        "threads_per_cta": threads,
        "warps_per_cta": warps,
        "data_lanes_per_cta": threads * mapping["coefficients_per_thread"],
        "data_temporal_reuse": mapping["coefficients_per_thread"],
        "allocated_registers_per_cta": allocated_registers,
        "allocated_shared_bytes_per_cta": allocated_shared,
        "resident_ctas_per_sm": resident_ctas,
        "resident_warps_per_sm": resident_ctas * warps,
        "occupancy_estimate": resident_ctas * warps / hardware["warps_per_sm"],
        "limiting_resource": limiting_resources,
        "ctas_per_transform": ctas_per_transform,
        "total_ctas": total_ctas,
        "waves": total_ctas / ctas_per_wave,
        "coefficient_bytes_min": coefficient_bytes,
        "unique_roots_max": unique_roots,
        "root_working_set_bytes": unique_roots * mapping["root_representation_bytes"],
        "root_working_set_over_l2": unique_roots * mapping["root_representation_bytes"] / hardware["l2_bytes"],
    }


def main():
    parser = argparse.ArgumentParser(description="Map an NTT schedule to GPU resource and working-set constraints.")
    parser.add_argument("--hardware", required=True, type=pathlib.Path)
    parser.add_argument("--mapping", required=True, type=pathlib.Path)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output", "-o", type=pathlib.Path)
    args = parser.parse_args()

    if args.batch <= 0:
        parser.error("--batch must be positive")

    hardware = json.loads(args.hardware.read_text())
    mapping = json.loads(args.mapping.read_text())
    records = [analyze_kernel(hardware, mapping, kernel, args.batch) for kernel in mapping["kernels"]]

    output = args.output.open("w", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    finally:
        if args.output:
            output.close()


if __name__ == "__main__":
    main()
