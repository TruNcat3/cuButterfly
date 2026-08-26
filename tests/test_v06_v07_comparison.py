#!/usr/bin/env python3
import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_v06_v07_comparison import aggregate, build_rows


class VersionComparisonTest(unittest.TestCase):
    def test_manifest_has_four_roles_per_numeric_workload(self):
        document = json.loads((ROOT / "config/v100_v06_v07_comparison.json").read_text())
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual({workload["precision"] for workload in document["workloads"]}, {"uint32", "uint64"})
        for workload in document["workloads"]:
            self.assertEqual(len(workload["implementations"]), 4)
            self.assertEqual(sum(bool(item.get("reference")) for item in workload["implementations"]), 1)
            self.assertEqual(workload["batches"], [1, 16, 80, 160, 256, 512])
        self.assertEqual({(workload["precision"], workload["logN"]) for workload in document["workloads"]},
                         {("uint32", 12), ("uint64", 10), ("uint64", 12)})

    def test_ratios_and_aggregate(self):
        base = {"group": "g", "precision": "uint64", "logN": "12", "batch": "16",
                "stability_class": "stable", "correct": "1"}
        names_and_times = (
            ("v0.6-fixed-hybrid2d-radix2", 4.0),
            ("v0.6-search-hybrid2d-radix4", 2.0),
            ("v0.7-base-hybrid-dataflow-Ur1-Td4", 3.0),
            ("v0.7-search-hybrid-dataflow-static-table", 1.5),
        )
        rows = build_rows([{**base, "implementation": name, "median_kernel_ms": str(time)}
                           for name, time in names_and_times])
        self.assertEqual(rows[0]["fastest"], "v07_search")
        self.assertAlmostEqual(rows[0]["v06_search_speedup"], 2.0)
        self.assertAlmostEqual(rows[0]["v07_search_speedup"], 2.0)
        self.assertAlmostEqual(rows[0]["v07_search_vs_v06_search"], 4.0 / 3.0)
        metrics = aggregate(rows)
        self.assertAlmostEqual(metrics[0]["v07_search_vs_v06_search_geomean"], 4.0 / 3.0)
        self.assertEqual(metrics[0]["v07_search_wins"], 1)


if __name__ == "__main__":
    unittest.main()
