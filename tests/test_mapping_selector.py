#!/usr/bin/env python3
import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("select_mapping", ROOT / "scripts" / "select_mapping.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MappingSelectorTest(unittest.TestCase):
    def test_argument_parser_separates_mapping_and_processing_unit(self):
        parsed = MODULE.parse_args_list(["--backend", "online-reorder", "--compute-unit", "radix4"])
        self.assertEqual(parsed["backend"], "online-reorder")
        self.assertEqual(parsed["compute_unit"], "radix4")

    def test_interpolation_uses_log_batch_and_latency(self):
        samples = [{"batch": 1, "kernel_ms": 1.0}, {"batch": 4, "kernel_ms": 4.0}]
        predicted, confidence = MODULE.interpolate_log_latency(samples, 2)
        self.assertAlmostEqual(predicted, 2.0)
        self.assertEqual(confidence, "interpolated")

    def test_extrapolation_clamps_nonphysical_negative_slope(self):
        samples = [{"batch": 1, "kernel_ms": 2.0}, {"batch": 2, "kernel_ms": 1.0}]
        predicted, confidence = MODULE.interpolate_log_latency(samples, 4)
        self.assertAlmostEqual(predicted, 1.0)
        self.assertEqual(confidence, "extrapolated-high")

    def test_counter_evidence_reports_grid_underfill(self):
        row = {"operator": "fwht", "logN_int": 15, "batch": 4,
               "processing_unit": "warp-register", "parameters": {}}
        counters = [{"operator": "fwht", "logN": "15", "batch": "4", "implementation": "warp",
                     "total_waves_per_sm": "0.05", "registers_per_thread": "255",
                     "barrier_stall_pct": "1", "long_scoreboard_stall_pct": "6"}]
        evidence = MODULE.counter_evidence(row, counters)
        self.assertIn("grid underfills", evidence)
        self.assertIn("register-limited", evidence)

    def test_v100_evaluation_covers_multiple_operators(self):
        cases = MODULE.load_cases(ROOT / "config" / "v100_scaling_suite.json")
        rows = MODULE.load_rows(ROOT / "results" / "v100_scaling_full_summary.csv", cases)
        import json
        hardware = json.loads((ROOT / "configs" / "hardware" / "v100_sxm2_16gb.json").read_text())
        records, metrics = MODULE.evaluate(rows, hardware)
        self.assertGreater(len(records), 20)
        self.assertGreaterEqual(len({row["operator"] for row in records}), 3)
        self.assertGreaterEqual(metrics["top3_recall"], 0.9)


if __name__ == "__main__":
    unittest.main()
