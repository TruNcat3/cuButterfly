#!/usr/bin/env python3
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_structured_2x2_ncu import aggregate


class Structured2x2NcuTest(unittest.TestCase):
    def test_aggregate_sums_kernels_and_time_weights_percentages(self):
        rows = [
            {"label": "case", "time_us": "1", "dram_read_mib": "2", "dram_write_mib": "3",
             "warp_instructions": "4", "fp32_thread_instructions": "5",
             "shared_load_bank_conflicts": "6", "shared_store_bank_conflicts": "7",
             "dram_peak_pct": "10", "active_warps_pct": "20", "registers_per_thread": "16",
             "shared_mem_bytes": "1024", "waves_per_sm": "2"},
            {"label": "case", "time_us": "3", "dram_read_mib": "5", "dram_write_mib": "7",
             "warp_instructions": "11", "fp32_thread_instructions": "13",
             "shared_load_bank_conflicts": "17", "shared_store_bank_conflicts": "19",
             "dram_peak_pct": "30", "active_warps_pct": "40", "registers_per_thread": "24",
             "shared_mem_bytes": "2048", "waves_per_sm": "4"},
        ]
        result = aggregate(rows)[0]
        self.assertEqual(result["kernels"], 2)
        self.assertEqual(result["time_us"], 4)
        self.assertEqual(result["dram_total_mib"], 17)
        self.assertEqual(result["warp_instructions"], 15)
        self.assertAlmostEqual(result["dram_peak_pct"], 25)
        self.assertAlmostEqual(result["active_warps_pct"], 35)
        self.assertEqual(result["registers_per_thread"], 24)
        self.assertEqual(result["total_waves_per_sm"], 6)


if __name__ == "__main__":
    unittest.main()
