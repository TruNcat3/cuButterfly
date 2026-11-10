#!/usr/bin/env python3
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_hierarchical_streaming_776_ncu import analyze
from summarize_hierarchical_streaming_comparison import summarize


class HierarchicalStreaming776AnalysisTest(unittest.TestCase):
    def test_ncu_ratios_and_batch_scaling(self):
        def row(label, time, read=10, write=5, wait=10):
            return {
                "label": label, "time_us": str(time), "dram_read_mib": str(read),
                "dram_write_mib": str(write), "warp_instructions": "100",
                "integer_thread_instructions": "200", "active_warps_pct": "25",
                "barrier_stall_pct": "5", "wait_stall_pct": str(wait),
                "long_scoreboard_stall_pct": "20", "registers_per_thread": "48",
                "shared_mem_bytes": "32768", "waves_per_sm": "1",
            }
        rows = [
            row("barrier_n20_b4", 100), row("streaming_generic_n20_b4", 200),
            row("streaming_resident_776_n20_b1", 50, read=10, write=5, wait=40),
            row("streaming_resident_776_n20_b4", 250, read=60, write=30, wait=35),
            row("streaming_specialized_n20_b4", 110),
            row("barrier_w32_n20_b4", 50), row("streaming_specialized_w32_n20_b4", 60),
            row("streaming_resident_776_w32_n20_b1", 40),
        ]
        comparisons, scaling = analyze(rows)
        resident = next(row for row in comparisons if row["variant"] == "resident-7+7+6")
        self.assertAlmostEqual(resident["speedup_vs_barrier"], 0.4)
        self.assertAlmostEqual(scaling["per_transform_time_ratio"], 1.25)
        self.assertAlmostEqual(scaling["read_ratio_b4_b1"], 6.0)

    def test_event_summary_joins_barrier(self):
        rows = [
            {"label": "barrier-w64-b1", "kernel_ms": "1", "correct": "-1"},
            {"label": "barrier-w64-b1", "kernel_ms": "1.2", "correct": "-1"},
            {"label": "resident776-w64-b1", "kernel_ms": "2", "correct": "-1"},
            {"label": "resident776-w64-b1", "kernel_ms": "2.4", "correct": "-1"},
        ]
        result = summarize(rows)
        resident = next(row for row in result if row["variant"] == "resident776")
        self.assertAlmostEqual(resident["speedup_vs_barrier"], 0.5)
        self.assertEqual(resident["trials"], 2)


if __name__ == "__main__":
    unittest.main()
