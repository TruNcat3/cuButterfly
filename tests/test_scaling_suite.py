#!/usr/bin/env python3
import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_scaling_suite import expand
from run_comprehensive_suite import load_manifest, selected_cases
from summarize_scaling_suite import analyze


class ScalingSuiteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.space = json.loads((ROOT / "config" / "v100_scaling_space.json").read_text())
        cls.suite = expand(cls.space)

    def test_generated_space_has_orthogonal_coverage(self):
        self.assertEqual(len(self.space["workloads"]), 12)
        self.assertGreaterEqual(len(self.suite["cases"]), 180)
        self.assertEqual({case["operator"] for case in self.suite["cases"]}, {"fft", "ntt", "fwht", "xor-zeta"})
        for case in self.suite["cases"]:
            self.assertLessEqual(case["batch"] * (1 << case["logN"]), self.space["max_points"])

    def test_generated_suite_passes_comprehensive_validation(self):
        path = ROOT / "config" / "v100_scaling_suite.json"
        self.assertEqual(self.suite, json.loads(path.read_text()))
        document = load_manifest(path)
        self.assertGreater(len(selected_cases(document, "full", None)), len(selected_cases(document, "quick", None)))

    def test_scaling_metrics_find_saturation(self):
        raw = []
        for batch, kernel_ms in ((1, 1.0), (2, 1.0), (4, 1.1), (8, 2.0)):
            raw.append({
                "suite_case_id": f"case-{batch}", "suite_group": f"group-{batch}",
                "suite_tier": "quick", "suite_runner": "butterfly", "implementation": "unit",
                "reference": "0", "trial": "1", "performance_batch": str(batch),
                "preflight_correct": "1", "operator": "fwht", "precision": "fp32",
                "direction": "forward", "normalization": "none", "placement": "out-of-place",
                "logN": "8", "N": "256", "element_stride": "1", "batch_stride": "256",
                "kernel_ms": str(kernel_ms),
            })
        rows = analyze(raw, 0.90, timing_floor_ms=0.0)
        self.assertEqual(rows[0]["saturation_batch"], 4)
        self.assertEqual(rows[2]["scaling_region"], "saturated")
        self.assertAlmostEqual(rows[2]["batch_efficiency_vs_batch1"], 1.0 / 1.1)

    def test_sub_floor_point_cannot_define_peak(self):
        raw = []
        for batch, kernel_ms in ((1, 0.001), (64, 0.100), (128, 0.210)):
            raw.append({
                "suite_case_id": f"floor-{batch}", "suite_group": f"floor-group-{batch}",
                "suite_tier": "full", "suite_runner": "butterfly", "implementation": "unit",
                "reference": "0", "trial": "1", "performance_batch": str(batch),
                "preflight_correct": "1", "operator": "fwht", "precision": "fp32",
                "direction": "forward", "normalization": "none", "placement": "out-of-place",
                "logN": "8", "N": "256", "element_stride": "1", "batch_stride": "256",
                "kernel_ms": str(kernel_ms),
            })
        rows = analyze(raw, timing_floor_ms=0.020)
        self.assertEqual(rows[0]["eligible_for_saturation"], 0)
        self.assertEqual(rows[0]["peak_batch"], 64)


if __name__ == "__main__":
    unittest.main()
