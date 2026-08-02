#!/usr/bin/env python3
import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("analyze_scaling_ncu", ROOT / "scripts" / "analyze_scaling_ncu.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ScalingNcuAnalysisTest(unittest.TestCase):
    def test_aggregate_sums_kernels_and_weights_percentages(self):
        rows = [
            {"label": "fft18_online_b2", "time_us": "1", "dram_read_mib": "1", "dram_write_mib": "1",
             "dram_peak_pct": "20", "active_warps_pct": "30", "registers_per_thread": "40", "waves_per_sm": "1"},
            {"label": "fft18_online_b2", "time_us": "3", "dram_read_mib": "2", "dram_write_mib": "2",
             "dram_peak_pct": "40", "active_warps_pct": "50", "registers_per_thread": "48", "waves_per_sm": "2"},
        ]
        record = MODULE.aggregate(rows)[0]
        self.assertEqual(record["kernels"], 2)
        self.assertEqual(record["time_us"], 4)
        self.assertEqual(record["dram_peak_pct"], 35)
        self.assertEqual(record["registers_per_thread"], 48)
        self.assertEqual(record["total_waves_per_sm"], 3)


if __name__ == "__main__":
    unittest.main()
