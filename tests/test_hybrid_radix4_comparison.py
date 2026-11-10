#!/usr/bin/env python3
import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_hybrid_radix4_comparison import aggregate, build_rows


class HybridRadix4ComparisonTest(unittest.TestCase):
    def test_manifest_covers_three_numeric_length_regions(self):
        document = json.loads((ROOT / "config/v100_hybrid_radix4_comparison.json").read_text())
        self.assertEqual(document["schema_version"], 1)
        workloads = document["workloads"]
        self.assertEqual({(item["precision"], item["logN"]) for item in workloads},
                         {("uint32", 12), ("uint64", 10), ("uint64", 12)})
        for workload in workloads:
            names = {item["name"] for item in workload["implementations"]}
            self.assertIn("v0.7-pipeline-radix2", names)
            self.assertIn("v0.7-resident-radix4", names)
            self.assertEqual(workload["batches"], [1, 16, 80, 160, 256, 512])

    def test_best_mature_core_and_ratios(self):
        base = {"group": "g", "precision": "uint64", "logN": "10", "batch": "80",
                "correct": "1"}
        values = (
            ("v0.6-baseline", 4.0),
            ("v0.6-tile256", 2.0),
            ("v0.7-pipeline-radix2", 6.0),
            ("v0.7-resident-radix4", 1.5),
        )
        rows = build_rows([{**base, "implementation": name, "median_kernel_ms": str(time)}
                           for name, time in values])
        self.assertEqual(rows[0]["v06_best"], "v0.6-tile256")
        self.assertAlmostEqual(rows[0]["radix4_speedup_vs_pipeline"], 4.0)
        self.assertAlmostEqual(rows[0]["radix4_throughput_vs_v06"], 4.0 / 3.0)
        metrics = aggregate(rows)
        self.assertAlmostEqual(metrics[0]["radix4_throughput_vs_v06"], 4.0 / 3.0)
        self.assertEqual(metrics[0]["radix4_wins"], 1)


if __name__ == "__main__":
    unittest.main()
