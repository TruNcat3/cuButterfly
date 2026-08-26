#!/usr/bin/env python3
import csv
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class HierarchicalDataflowSummaryTest(unittest.TestCase):
    def test_selects_fastest_hierarchical_point_and_reports_speedup(self):
        fields = [
            "word_bits", "logN", "batch", "backend", "n1_log",
            "rows_per_block", "threads_per_block", "data_time",
            "target_ctas_per_sm", "kernel_ms", "correct",
        ]
        rows = [
            dict(zip(fields, (32, 20, 4, "hybrid2d", 10, 2, 512, 0, 0, 0.30, 1))),
            dict(zip(fields, (32, 20, 4, "hierarchical-dataflow", 10, 2, 256, 1, 6, 0.25, 1))),
            dict(zip(fields, (32, 20, 4, "hierarchical-dataflow", 10, 4, 256, 1, 5, 0.20, 1))),
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            source = directory / "raw.csv"
            output = directory / "summary.csv"
            markdown = directory / "summary.md"
            with source.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/summarize_hierarchical_dataflow.py"),
                 str(source), "--csv", str(output), "--markdown", str(markdown)],
                check=True,
            )
            report = markdown.read_text()
            self.assertIn("1.500x", report)
            self.assertIn("| 4 | 1 | 256 | 5 | yes |", report)


if __name__ == "__main__":
    unittest.main()
