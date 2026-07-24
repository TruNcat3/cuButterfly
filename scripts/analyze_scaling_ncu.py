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
    if low["dram_bytes_per_point"] < 1.0:
        traffic = "the low-batch DRAM count is cache-resident/replay-sensitive"
    else:
        traffic_change = high["dram_bytes_per_point"] / low["dram_bytes_per_point"]
        traffic = f"bytes/point changes {traffic_change:.2f}x"
    return (f"{operator.upper()} logN={log_n} {implementation}: waves/SM change {wave_ratio:.2f}x from "
            f"batch {low['batch']} to {high['batch']}; active warps {low['active_warps_pct']:.1f}% -> "
            f"{high['active_warps_pct']:.1f}%, DRAM peak {low['dram_peak_pct']:.1f}% -> "
            f"{high['dram_peak_pct']:.1f}%, {traffic}, barrier stalls "
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


def find_row(rows, label):
    return next(row for row in rows if row["label"] == label)


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def write_markdown(path, rows, kernels):
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
    fft14_low = find_row(rows, "fft14_direct_b16")
    fft14_high = find_row(rows, "fft14_direct_b1024")
    cufft14_low = find_row(rows, "fft14_cufft_b16")
    fft18_online = find_row(rows, "fft18_online_b64")
    fft18_cufft = find_row(rows, "fft18_cufft_b64")
    fft20_online = find_row(rows, "fft20_online_b16")
    fft20_cufft = find_row(rows, "fft20_cufft_b16")
    fwht_online4 = find_row(rows, "fwht15_online_b4")
    fwht_online16 = find_row(rows, "fwht15_online_b16")
    fwht_warp4 = find_row(rows, "fwht15_warp_b4")
    fwht_warp16 = find_row(rows, "fwht15_warp_b16")
    fft20_passes = [row for row in kernels if row["label"] == "fft20_online_b16"]
    fft20_first = next(row for row in fft20_passes if "online_first" in row["kernel_name"])
    fft20_second = next(row for row in fft20_passes if "online_second" in row["kernel_name"])
    lines += ["", "## Measured Trends", "",
              "- " + conclusion(rows, "fft", 14, "direct", {16, 256, 1024}),
              "- " + conclusion(rows, "fft", 18, "online", {2, 16, 64}),
              "- " + conclusion(rows, "fft", 20, "online", {2, 8, 16}),
              "- " + conclusion(rows, "fwht", 15, "online", {4, 16}),
              "- " + conclusion(rows, "fwht", 15, "warp", {4, 16}), "",
              "Counts reported as zero for the smallest cases are not zero algorithmic traffic. `--cache-control none`, NCU replay, and a working set that fits in the 6 MiB L2 make those DRAM counters unsuitable for traffic ratios.", "",
              "## Architecture Attribution", "",
              "### FFT logN=14: grid concurrency threshold", "",
              f"The direct unit launches one 1024-thread CTA per transform and is limited to one resident CTA per SM by its register and {fft14_low['shared_mem_bytes']:.0f}-byte shared allocations. At batch 16 it exposes only {fft14_low['total_waves_per_sm']:.2f} waves/SM, whereas cuFFT exposes {cufft14_low['total_waves_per_sm']:.2f} waves/SM across two kernels. The direct grid therefore leaves most of the 80 SMs without work even though active CTAs individually report {fft14_low['active_warps_pct']:.1f}% active warps. At batch 1024 it reaches {fft14_high['total_waves_per_sm']:.1f} waves/SM and the grid-level deficit disappears. This attributes the low-batch gap to `Ub`/grid coverage, not to the local cuFFTDx arithmetic core.", "",
              "### FFT logN=18: useful-work ceiling", "",
              f"At batch 64, online composition has {fft18_online['active_warps_pct']:.1f}% active warps versus cuFFT's {fft18_cufft['active_warps_pct']:.1f}%, but executes {safe_ratio(fft18_online['warp_instructions'], fft18_cufft['warp_instructions']):.2f}x as many warp instructions and {safe_ratio(fft18_online['shared_load_bank_conflicts'] + fft18_online['shared_store_bank_conflicts'], fft18_cufft['shared_load_bank_conflicts'] + fft18_cufft['shared_store_bank_conflicts']):.2f}x as many shared-memory bank conflicts. Its barrier and long-scoreboard stalls are {fft18_online['barrier_stall_pct']:.1f}% and {fft18_online['long_scoreboard_stall_pct']:.1f}%, and it reaches only {fft18_online['dram_peak_pct']:.1f}% of peak DRAM versus cuFFT's {fft18_cufft['dram_peak_pct']:.1f}%. The early saturation is therefore caused by less useful work per active warp and exchange/synchronization overhead, not a lack of resident warps.", "",
              "### FFT logN=20: prefix/suffix imbalance", "",
              f"At batch 16, online and cuFFT both expose {fft20_online['total_waves_per_sm']:.1f} aggregate waves/SM. Online nevertheless executes {safe_ratio(fft20_online['warp_instructions'], fft20_cufft['warp_instructions']):.2f}x the warp instructions, incurs {safe_ratio(fft20_online['shared_load_bank_conflicts'] + fft20_online['shared_store_bank_conflicts'], fft20_cufft['shared_load_bank_conflicts'] + fft20_cufft['shared_store_bank_conflicts']):.2f}x the shared bank conflicts, and has {fft20_online['barrier_stall_pct']:.1f}% barrier plus {fft20_online['long_scoreboard_stall_pct']:.1f}% scoreboard stalls. The first pass consumes {float(fft20_first['time_us']) / fft20_online['time_us']:.1%} of profiled time and reaches only {float(fft20_first['dram_peak_pct']):.1f}% DRAM, while the second reaches {float(fft20_second['dram_peak_pct']):.1f}%. The remaining gap is a prefix-core/layout imbalance and synchronization problem; increasing occupancy alone cannot close it.", "",
              "### FWHT logN=15: mapping crossover", "",
              f"The warp-register unit uses {fwht_warp4['registers_per_thread']:.0f} registers/thread and {fwht_warp4['shared_mem_bytes']:.0f} shared bytes, limiting it to one CTA/SM. It launches only {fwht_warp4['total_waves_per_sm']:.2f} waves/SM at batch 4 and {fwht_warp16['total_waves_per_sm']:.2f} at batch 16, with nearly fixed NCU time. Online decomposition exposes {fwht_online4['total_waves_per_sm']:.2f} and {fwht_online16['total_waves_per_sm']:.2f} waves/SM, so it wins when four transforms cannot cover the GPU. By batch 16, the register unit's single-kernel path executes only {safe_ratio(fwht_warp16['warp_instructions'], fwht_online16['warp_instructions']):.2f}x the online warp instructions and avoids its shared exchanges and {fwht_online16['barrier_stall_pct']:.1f}% barrier stalls. The crossover is the expected tradeoff between `Ud` spatial decomposition at low batch and `Td` register residence once `Ub` supplies enough independent transforms.", "",
              "## Methodological Result", "",
              "The counters support separating three quantities in the selector: grid coverage (`Ub` and waves/SM), residency (register/shared limits), and useful work per resident warp (instruction, exchange, and stall costs). Occupancy is not a sufficient objective: both long online FFTs expose at least as many active warps as cuFFT while delivering a lower saturated ceiling.", ""]
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
    write_markdown(args.markdown, aggregated, rows)


if __name__ == "__main__":
    main()
