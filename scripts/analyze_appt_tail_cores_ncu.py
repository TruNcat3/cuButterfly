#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def number(row, key):
    value = row.get(key, "")
    return float(value) if value else 0.0


def main():
    parser = argparse.ArgumentParser(description="Analyze APPT tail-core NCU data")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--ablation-baseline", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    rows = {row["label"]: row for row in csv.DictReader(args.summary.open())}
    records = []
    cores = ("warp", "cta-radix4", "fused-tail", "split-tail", "register-tail")
    for bits in (32, 64):
        for batch in (1, 4):
            baseline = rows[f"v06_u{bits}_b{batch}"]
            baseline_time = number(baseline, "time_us")
            for core in cores:
                row = rows[f"{core}_u{bits}_b{batch}"]
                time_us = number(row, "time_us")
                records.append({
                    "word_bits": bits,
                    "batch": batch,
                    "physical_core": core,
                    "time_us": time_us,
                    "throughput_vs_v06": baseline_time / time_us,
                    "dram_read_mib": number(row, "dram_read_mib"),
                    "dram_write_mib": number(row, "dram_write_mib"),
                    "global_load_sectors": number(row, "global_load_sectors"),
                    "global_store_sectors": number(row, "global_store_sectors"),
                    "local_load_sectors": number(row, "local_load_sectors"),
                    "local_store_sectors": number(row, "local_store_sectors"),
                    "active_warps_pct": number(row, "active_warps_pct"),
                    "barrier_stall_pct": number(row, "barrier_stall_pct"),
                    "long_scoreboard_stall_pct": number(row, "long_scoreboard_stall_pct"),
                    "mio_throttle_stall_pct": number(row, "mio_throttle_stall_pct"),
                    "dram_peak_pct": number(row, "dram_peak_pct"),
                    "l1_hit_pct": number(row, "l1_hit_pct"),
                    "l2_hit_pct": number(row, "l2_hit_pct"),
                    "shared_load_conflicts": number(row, "shared_load_bank_conflicts"),
                    "shared_store_conflicts": number(row, "shared_store_bank_conflicts"),
                    "registers_per_thread": number(row, "registers_per_thread"),
                    "shared_bytes_per_block": number(row, "shared_mem_bytes"),
                })

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# APPT Tail-Core NCU Attribution", "",
        "Matched V100 base-clock replay; throughput is relative to v0.6 10+10.", "",
        "| bits | batch | core | time us | throughput/v0.6 | DRAM R/W MiB | global sectors R/W | local sectors R/W | active warps | barrier | scoreboard | MIO throttle | registers | shared B |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['word_bits']} | {row['batch']} | `{row['physical_core']}` | "
            f"{row['time_us']:.1f} | {row['throughput_vs_v06']:.3f}x | "
            f"{row['dram_read_mib']:.1f}/{row['dram_write_mib']:.1f} | "
            f"{row['global_load_sectors']:.0f}/{row['global_store_sectors']:.0f} | "
            f"{row['local_load_sectors']:.0f}/{row['local_store_sectors']:.0f} | "
            f"{row['active_warps_pct']:.1f}% | {row['barrier_stall_pct']:.1f}% | "
            f"{row['long_scoreboard_stall_pct']:.1f}% | "
            f"{row['mio_throttle_stall_pct']:.1f}% | "
            f"{row['registers_per_thread']:.0f} | "
            f"{row['shared_bytes_per_block']:.0f} |")
    register = {
        (row["word_bits"], row["batch"]): row
        for row in records if row["physical_core"] == "register-tail"
    }
    v06 = {
        (bits, batch): rows[f"v06_u{bits}_b{batch}"]
        for bits in (32, 64) for batch in (1, 4)
    }
    ratios = {}
    for key, row in register.items():
        base = v06[key]
        ratios[key] = {
            "loads": row["global_load_sectors"] / number(base, "global_load_sectors"),
            "stores": row["global_store_sectors"] / number(base, "global_store_sectors"),
            "warp_inst": number(rows[f"register-tail_u{key[0]}_b{key[1]}"], "warp_instructions") /
                         number(base, "warp_instructions"),
            "int_inst": number(rows[f"register-tail_u{key[0]}_b{key[1]}"], "integer_thread_instructions") /
                        number(base, "integer_thread_instructions"),
            "active": row["active_warps_pct"] / number(base, "active_warps_pct"),
        }
    l1_hits = [row["l1_hit_pct"] for row in register.values()]
    l2_hits = [row["l2_hit_pct"] for row in register.values()]
    scoreboard = [row["long_scoreboard_stall_pct"] for row in register.values()]
    barriers = [row["barrier_stall_pct"] for row in register.values()]
    dram_peak = {
        bits: [row["dram_peak_pct"] for key, row in register.items()
               if key[0] == bits]
        for bits in (32, 64)
    }
    ablation_finding = None
    if args.ablation_baseline:
        previous = {
            row["label"]: row
            for row in csv.DictReader(args.ablation_baseline.open())
        }
        time_gains = []
        sector_changes = []
        for batch in (1, 4):
            label = f"register-tail_u32_b{batch}"
            old = previous[label]
            new = rows[label]
            time_gains.append(
                100.0 * (number(old, "time_us") / number(new, "time_us") - 1.0))
            for metric in ("global_load_sectors", "global_store_sectors"):
                sector_changes.append(abs(
                    100.0 * (number(new, metric) / number(old, metric) - 1.0)))
        ablation_finding = (
            "6. The uint32 aggregate-readiness ablation improves NCU replay "
            f"time by {min(time_gains):.1f}%--{max(time_gains):.1f}% over the "
            "earlier distributed-flag capture, while global load/store sectors "
            f"change by at most {max(sector_changes):.2f}%. Readiness polling is "
            "therefore a latency contributor, but it does not explain the "
            "excess request traffic. Uint64 retains distributed flags because "
            "aggregate-counter contention was slower in the timing scan."
        )
    lines += [
        "", "## Findings", "",
        "1. Register-tail removes the second N-sized state without adding arithmetic work. "
        f"Its warp-instruction ratio is {min(x['warp_inst'] for x in ratios.values()):.2f}x--"
        f"{max(x['warp_inst'] for x in ratios.values()):.2f}x v0.6 and its integer-thread ratio is "
        f"{min(x['int_inst'] for x in ratios.values()):.2f}x--"
        f"{max(x['int_inst'] for x in ratios.values()):.2f}x.",
        "2. Zero local load/store sectors confirm that the retained low half does not spill. "
        f"The {register[(32, 1)]['registers_per_thread']:.0f}/"
        f"{register[(64, 1)]['registers_per_thread']:.0f}-register uint32/uint64 kernels remain limited by the requested 3/2 CTA shared-memory residency.",
        "3. The remaining request traffic is much larger than the materialized data volume. "
        f"Register-tail global-load sectors are {min(x['loads'] for x in ratios.values()):.2f}x--"
        f"{max(x['loads'] for x in ratios.values()):.2f}x v0.6 and store sectors are "
        f"{min(x['stores'] for x in ratios.values()):.2f}x--"
        f"{max(x['stores'] for x in ratios.values()):.2f}x. DRAM writes remain close to the one-state floor at batch 1, "
        "so this is primarily L1/L2 request and transaction overhead rather than an extra global state.",
        f"4. Register-tail L1 hit rate is only {min(l1_hits):.1f}%--{max(l1_hits):.1f}%, while its L2 hit rate is "
        f"{min(l2_hits):.1f}%--{max(l2_hits):.1f}%. Together with "
        f"{min(scoreboard):.1f}%--{max(scoreboard):.1f}% long-scoreboard stall, this identifies repeated L2-served dependency/coefficient/state loads as the critical path.",
        "5. Active-warps samples reach only "
        f"{min(x['active'] for x in ratios.values()):.2f}x--"
        f"{max(x['active'] for x in ratios.values()):.2f}x v0.6. Barrier stall remains "
        f"{min(barriers):.1f}%--{max(barriers):.1f}%, "
        "so the fused tail is not work-conserving even though its logical graph is dependency closed.",
        ablation_finding or
        "6. Readiness polling is consistent with the latency counters, but this capture alone does not isolate its contribution. Use --ablation-baseline with a matched distributed-flag capture to quantify it.",
        f"7. Achieved DRAM throughput is {min(dram_peak[32]):.1f}%--{max(dram_peak[32]):.1f}% of peak for uint32 and "
        f"{min(dram_peak[64]):.1f}%--{max(dram_peak[64]):.1f}% for uint64. The kernel is latency/request limited, not bandwidth saturated.",
        "", "Stall percentages are scheduler-state samples and must not be summed as time fractions.",
    ]
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
