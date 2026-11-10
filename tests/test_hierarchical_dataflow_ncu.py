#!/usr/bin/env python3
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_hierarchical_dataflow_ncu import aggregate


class HierarchicalDataflowNcuTest(unittest.TestCase):
    def test_aggregates_two_pass_baseline(self):
        common = {
            "dram_read_mib": "10", "dram_write_mib": "8",
            "warp_instructions": "100", "integer_thread_instructions": "200",
            "active_warps_pct": "50", "barrier_stall_pct": "10",
            "long_scoreboard_stall_pct": "20", "registers_per_thread": "32",
            "shared_mem_bytes": "8192", "occupancy_register_block_limit": "4",
            "occupancy_shared_block_limit": "5", "occupancy_warp_block_limit": "4",
        }
        rows = [
            {**common, "label": "hybrid2d_w32_n20_b4", "time_us": "100"},
            {**common, "label": "hybrid2d_w32_n20_b4", "time_us": "150",
             "active_warps_pct": "70", "registers_per_thread": "40"},
            {**common, "label": "hierarchical_w32_n20_b4", "time_us": "200"},
        ]
        totals = {row["backend"]: row for row in aggregate(rows)}
        baseline = totals["hybrid2d"]
        self.assertEqual(baseline["kernel_count"], 2)
        self.assertEqual(baseline["time_us"], 250)
        self.assertEqual(baseline["dram_read_mib"], 20)
        self.assertEqual(baseline["registers_per_thread"], 40)
        self.assertAlmostEqual(baseline["active_warps_pct"], 62)


if __name__ == "__main__":
    unittest.main()
