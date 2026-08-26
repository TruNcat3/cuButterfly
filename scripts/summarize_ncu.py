#!/usr/bin/env python3
import argparse
import csv
import pathlib
import re
import sys


METRICS = {
    "gpu__time_duration.sum": ("time_us", 1.0e-3),
    "dram__bytes_read.sum": ("dram_read_mib", 1.0 / (1024.0 * 1024.0)),
    "dram__bytes_write.sum": ("dram_write_mib", 1.0 / (1024.0 * 1024.0)),
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": ("global_load_sectors", 1.0),
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": ("global_store_sectors", 1.0),
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum": ("local_load_sectors", 1.0),
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum": ("local_store_sectors", 1.0),
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": ("dram_peak_pct", 1.0),
    "lts__t_sector_hit_rate.pct": ("l2_hit_pct", 1.0),
    "l1tex__t_sector_hit_rate.pct": ("l1_hit_pct", 1.0),
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": ("shared_load_bank_conflicts", 1.0),
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": ("shared_store_bank_conflicts", 1.0),
    "smsp__inst_executed.sum": ("warp_instructions", 1.0),
    "smsp__inst_executed.avg.per_cycle_active": ("instructions_per_active_cycle", 1.0),
    "smsp__sass_thread_inst_executed_op_integer_pred_on.sum": ("integer_thread_instructions", 1.0),
    "smsp__sass_thread_inst_executed_op_memory_pred_on.sum": ("memory_thread_instructions", 1.0),
    "smsp__sass_thread_inst_executed_op_control_pred_on.sum": ("control_thread_instructions", 1.0),
    "smsp__sass_thread_inst_executed_op_misc_pred_on.sum": ("misc_thread_instructions", 1.0),
    "smsp__sass_thread_inst_executed_op_conversion_pred_on.sum": ("conversion_thread_instructions", 1.0),
    "smsp__sass_inst_executed_op_global.sum": ("global_warp_instructions", 1.0),
    "smsp__sass_inst_executed_op_shared.sum": ("shared_warp_instructions", 1.0),
    "smsp__inst_executed_pipe_adu.sum": ("adu_warp_instructions", 1.0),
    "smsp__inst_executed_pipe_alu.sum": ("alu_warp_instructions", 1.0),
    "smsp__inst_executed_pipe_cbu.sum": ("cbu_warp_instructions", 1.0),
    "smsp__inst_executed_pipe_lsu.sum": ("lsu_warp_instructions", 1.0),
    "smsp__inst_executed_pipe_xu.sum": ("xu_warp_instructions", 1.0),
    "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum": ("fp32_thread_instructions", 1.0),
    "smsp__sass_thread_inst_executed_op_fp64_pred_on.sum": ("fp64_thread_instructions", 1.0),
    "smsp__inst_executed_pipe_tensor.sum": ("tensor_warp_instructions", 1.0),
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": ("tensor_active_pct", 1.0),
    "sm__warps_active.avg.pct_of_peak_sustained_active": ("active_warps_pct", 1.0),
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct": ("barrier_stall_pct", 1.0),
    "smsp__warp_issue_stalled_wait_per_warp_active.pct": ("wait_stall_pct", 1.0),
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": ("long_scoreboard_stall_pct", 1.0),
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct": ("short_scoreboard_stall_pct", 1.0),
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct": ("mio_throttle_stall_pct", 1.0),
    "launch__registers_per_thread": ("registers_per_thread", 1.0),
    "launch__shared_mem_per_block": ("shared_mem_bytes", 1.0),
    "launch__waves_per_multiprocessor": ("waves_per_sm", 1.0),
    "launch__occupancy_limit_registers": ("occupancy_register_block_limit", 1.0),
    "launch__occupancy_limit_shared_mem": ("occupancy_shared_block_limit", 1.0),
    "launch__occupancy_limit_warps": ("occupancy_warp_block_limit", 1.0),
}


def parse_number(value):
    value = value.strip().replace(",", "")
    if not value or value.lower() in {"n/a", "nan", "no data"}:
        return None
    return float(value)


def shorten_kernel_name(name):
    hybrid = re.search(r"hybrid2d_(first|second)_pass_kernel", name)
    if hybrid:
        return f"hybrid2d_{hybrid.group(1)}_pass"
    compact = re.search(r"compact_stage_kernel<([^>]+)>", name)
    if compact:
        return f"compact_stage<{compact.group(1)}>"
    return name.split("(", 1)[0].strip()


def normalize_dimension(value):
    return value.replace(",", "").strip()


def read_long_csv(path, lines, header_index):
    reader = csv.DictReader(lines[header_index:])
    kernels = {}
    for row in reader:
        metric = row.get("Metric Name", "")
        if metric not in METRICS:
            continue
        kernel_id = row.get("ID", "")
        key = (kernel_id, row.get("Kernel Name", ""), row.get("Grid Size", ""), row.get("Block Size", ""))
        record = kernels.setdefault(
            key,
            {
                "label": path.stem,
                "kernel_id": kernel_id,
                "kernel_name": shorten_kernel_name(row.get("Kernel Name", "")),
                "grid_size": normalize_dimension(row.get("Grid Size", "")),
                "block_size": normalize_dimension(row.get("Block Size", "")),
            },
        )
        output_name, scale = METRICS[metric]
        number = parse_number(row.get("Metric Value", ""))
        record[output_name] = "" if number is None else number * scale
    return list(kernels.values())


def read_wide_csv(path, lines, header_index):
    reader = csv.DictReader(lines[header_index:])
    kernels = []
    for row in reader:
        kernel_id = row.get("ID", "")
        kernel_name = row.get("Kernel Name", "")
        if not kernel_id or not kernel_name:
            continue
        record = {
            "label": path.stem,
            "kernel_id": kernel_id,
            "kernel_name": shorten_kernel_name(kernel_name),
            "grid_size": normalize_dimension(row.get("launch__grid_size", row.get("Grid Size", ""))),
            "block_size": normalize_dimension(row.get("launch__block_size", row.get("Block Size", ""))),
        }
        for metric, (output_name, scale) in METRICS.items():
            number = parse_number(row.get(metric, ""))
            record[output_name] = "" if number is None else number * scale
        kernels.append(record)
    return kernels


def read_ncu_csv(path):
    lines = path.read_text(errors="replace").splitlines()
    for index, line in enumerate(lines):
        parsed = next(csv.reader([line]), [])
        if "Metric Name" in parsed and "Kernel Name" in parsed:
            return read_long_csv(path, lines, index)
        if "Kernel Name" in parsed and any(metric in parsed for metric in METRICS):
            return read_wide_csv(path, lines, index)
    raise ValueError(f"no supported NCU CSV header found in {path}")


def is_derived_csv(path):
    """Identify this script's output and the APPT analysis, not malformed NCU input."""
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        header = set(next(csv.reader([line]), []))
        return ({"label", "kernel_id", "kernel_name"} <= header or
                {"implementation", "batch1_us", "batch4_us"} <= header)
    return False


def main():
    parser = argparse.ArgumentParser(description="Pivot Nsight Compute raw CSV files into one row per kernel.")
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", "-o", type=pathlib.Path)
    args = parser.parse_args()

    records = []
    for path in args.inputs:
        if ((args.output and path.resolve() == args.output.resolve()) or
                is_derived_csv(path)):
            print(f"skip derived CSV: {path}", file=sys.stderr)
            continue
        records.extend(read_ncu_csv(path))

    if not records:
        raise ValueError("no NCU kernel records found in the supplied inputs")

    fields = ["label", "kernel_id", "kernel_name", "grid_size", "block_size"] + [value[0] for value in METRICS.values()]
    output = args.output.open("w", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(
            output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    finally:
        if args.output:
            output.close()


if __name__ == "__main__":
    main()
