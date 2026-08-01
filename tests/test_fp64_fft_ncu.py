#!/usr/bin/env python3
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_fp64_fft_ncu import aggregate, analyze  # noqa: E402


def kernel(label, time_us, warp_instructions, fp64_instructions, conflicts):
    return {
        "label": label,
        "time_us": str(time_us),
        "dram_read_mib": "64",
        "dram_write_mib": "64",
        "shared_load_bank_conflicts": str(conflicts),
        "shared_store_bank_conflicts": "0",
        "warp_instructions": str(warp_instructions),
        "fp64_thread_instructions": str(fp64_instructions),
        "dram_peak_pct": "50",
        "l2_hit_pct": "40",
        "l1_hit_pct": "30",
        "active_warps_pct": "20",
        "barrier_stall_pct": "10",
        "long_scoreboard_stall_pct": "5",
    }


def main():
    batch = 7
    rows = [
        kernel("fp64_scalar_b7", 30, 300, 600, 30),
        kernel("fp64_cufftdx_b7", 20, 200, 400, 20),
        kernel("fp64_cufft_b7", 10, 100, 500, 2),
    ]
    totals = aggregate(rows[:2])
    assert totals["time_us"] == 50
    assert totals["dram_total_mib"] == 256
    output = analyze(rows, {"scalar": 3.0, "cufftdx": 2.0, "cufft": 1.0}, batch)
    by_name = {row["implementation"]: row for row in output}
    assert by_name["cufftdx"]["warp_instructions_vs_cufft"] == 2.0
    assert by_name["cufftdx"]["fp64_thread_instructions_vs_cufft"] == 0.8
    assert by_name["scalar"]["cuda_event_ms_vs_cufft"] == 3.0


if __name__ == "__main__":
    main()
