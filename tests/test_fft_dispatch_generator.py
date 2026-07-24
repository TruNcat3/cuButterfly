#!/usr/bin/env python3
import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("generate_fft_dispatch", ROOT / "scripts" / "generate_fft_dispatch.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FftDispatchGeneratorTest(unittest.TestCase):
    def rows(self):
        return [
            {"rank": "1", "logN": "18", "batch": "2", "candidate_id": "low", "prefix_log_n": "8",
             "suffix_log_n": "10", "prefix_threads": "256", "suffix_threads": "128", "prefix_ept": "8",
             "suffix_ept": "16", "cross_twiddle": "recurrence", "median_kernel_ms": "1",
             "cufft_median_ms": "1", "throughput_vs_cufft": "1"},
            {"rank": "1", "logN": "18", "batch": "16", "candidate_id": "mid", "prefix_log_n": "9",
             "suffix_log_n": "9", "prefix_threads": "128", "suffix_threads": "128", "prefix_ept": "16",
             "suffix_ept": "16", "cross_twiddle": "table", "median_kernel_ms": "1",
             "cufft_median_ms": "1", "throughput_vs_cufft": "1"},
            {"rank": "2", "logN": "18", "batch": "16", "candidate_id": "loser", "prefix_log_n": "9",
             "suffix_log_n": "9", "prefix_threads": "256", "suffix_threads": "256", "prefix_ept": "8",
             "suffix_ept": "8", "cross_twiddle": "table", "median_kernel_ms": "2",
             "cufft_median_ms": "1", "throughput_vs_cufft": "0.5"},
        ]

    def test_threshold_uses_log_batch_midpoint(self):
        entries = MODULE.select_dispatch(self.rows())
        self.assertEqual(entries[0]["max_batch"], 4)
        self.assertIsNone(entries[1]["max_batch"])

    def test_header_contains_selected_dimensions(self):
        document = {"schema_version": 1, "target": "v100-sm70", "entries": MODULE.select_dispatch(self.rows())}
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "dispatch.cuh"
            MODULE.generate_header(path, document)
            text = path.read_text()
            self.assertIn("log_n == 18U && batch <= 4ULL", text)
            self.assertIn("mapping = {8U, 256U, 128U, 8U, 16U, true}", text)


if __name__ == "__main__":
    unittest.main()
