#!/usr/bin/env python3
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_hierarchical_cores_ncu import compare


class HierarchicalCoresNcuTest(unittest.TestCase):
    def test_compares_local_cores_and_effective_limit(self):
        def row(label, time, registers, register_limit, scoreboard):
            return {
                "label": label, "time_us": str(time), "dram_read_mib": "10",
                "dram_write_mib": "5", "warp_instructions": "100",
                "integer_thread_instructions": "200", "active_warps_pct": "50",
                "barrier_stall_pct": "10", "long_scoreboard_stall_pct": str(scoreboard),
                "registers_per_thread": str(registers), "shared_mem_bytes": "16400",
                "occupancy_register_block_limit": str(register_limit),
                "occupancy_shared_block_limit": "5", "occupancy_warp_block_limit": "8",
            }
        rows = [
            row("w32_dataflow-radix4", 100, 38, 6, 30),
            row("w32_hybrid2d-radix4", 110, 32, 8, 40),
            row("w64_dataflow-radix4", 200, 48, 2, 20),
            row("w64_hybrid2d-radix4", 210, 44, 2, 25),
        ]
        result = compare(rows)
        self.assertEqual(result[0]["native_effective_block_limit"], 5)
        self.assertEqual(result[0]["mature_effective_block_limit"], 5)
        self.assertEqual(result[1]["native_effective_block_limit"], 2)
        self.assertAlmostEqual(result[0]["mature_over_native_time_us"], 1.1)


if __name__ == "__main__":
    unittest.main()
