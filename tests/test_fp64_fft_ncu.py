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
        "integer_thread_instructions": str(warp_instructions * 10),
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
        kernel("fp64_recurrence_b7", 15, 180, 450, 18),
        kernel("fp64_xor_swizzle_b7", 14, 170, 450, 9),
        kernel("fp64_cufft_b7", 10, 100, 500, 2),
    ]
    totals = aggregate(rows[:2])
    assert totals["time_us"] == 50
    assert totals["dram_total_mib"] == 256
    output = analyze(rows, {"scalar": 3.0, "cufftdx": 2.0, "recurrence": 1.5,
                            "xor_swizzle": 1.4, "cufft": 1.0}, batch)
    by_name = {row["implementation"]: row for row in output}
    assert by_name["cufftdx"]["warp_instructions_vs_cufft"] == 2.0
    assert by_name["xor_swizzle"]["integer_thread_instructions_vs_cufft"] == 1.7
    assert by_name["cufftdx"]["fp64_thread_instructions_vs_cufft"] == 0.8
    assert by_name["scalar"]["cuda_event_ms_vs_cufft"] == 3.0
    assert by_name["recurrence"]["time_us_vs_cufft"] == 1.5
    assert by_name["xor_swizzle"]["shared_bank_conflicts_vs_cufft"] == 4.5


if __name__ == "__main__":
    main()
