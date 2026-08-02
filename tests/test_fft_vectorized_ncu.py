#!/usr/bin/env python3
import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_fft_vectorized_ncu", ROOT / "scripts" / "analyze_fft_vectorized_ncu.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FftVectorizedNcuTest(unittest.TestCase):
    def test_aggregate_sums_counters_and_weights_percentages(self):
        rows = [
            {"time_us": "1", "warp_instructions": "10", "dram_peak_pct": "20",
             "shared_load_bank_conflicts": "2", "shared_store_bank_conflicts": "3"},
            {"time_us": "3", "warp_instructions": "20", "dram_peak_pct": "40",
             "shared_load_bank_conflicts": "5", "shared_store_bank_conflicts": "7"},
        ]
        result = MODULE.aggregate(rows)
        self.assertEqual(result["time_us"], 4.0)
        self.assertEqual(result["warp_instructions"], 30.0)
        self.assertEqual(result["shared_bank_conflicts"], 17.0)
        self.assertEqual(result["dram_peak_pct"], 35.0)


if __name__ == "__main__":
    unittest.main()
