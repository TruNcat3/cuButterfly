#!/usr/bin/env python3
import csv
import importlib.util
import pathlib
import tempfile
import unittest
import unittest.mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_structured_coefficient_policies",
    ROOT / "scripts" / "summarize_structured_coefficient_policies.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StructuredCoefficientPolicySummaryTest(unittest.TestCase):
    def test_equivalent_policy_speedup(self):
        rows = [
            {"logN": "8", "N": "256", "batch": "16", "coefficient_policy": "broadcast",
             "kernel_ms": "2", "Gbutterfly_s": "4"},
            {"logN": "8", "N": "256", "batch": "16", "coefficient_policy": "broadcast",
             "kernel_ms": "4", "Gbutterfly_s": "2"},
            {"logN": "8", "N": "256", "batch": "16", "coefficient_policy": "per-stage",
             "kernel_ms": "6", "Gbutterfly_s": "1.5"},
            {"logN": "8", "N": "256", "batch": "16", "coefficient_policy": "per-stage",
             "kernel_ms": "8", "Gbutterfly_s": "1"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "raw.csv"
            output = pathlib.Path(directory) / "summary.csv"
            with source.open("w", newline="") as destination:
                writer = csv.DictWriter(destination, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with unittest.mock.patch("sys.argv", ["summary", str(source), "--output", str(output)]):
                MODULE.main()
            with output.open() as summary:
                record = next(csv.DictReader(summary))
            self.assertEqual(record["broadcast_ms"], "3.000000")
            self.assertEqual(record["per_stage_ms"], "7.000000")
            self.assertEqual(record["broadcast_speedup"], "2.333333")


if __name__ == "__main__":
    unittest.main()
