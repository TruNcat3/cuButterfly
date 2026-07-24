#!/usr/bin/env python3
import pathlib
import json
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fft_design_space import classify_online_point, enumerate_online, load_codegen_points, load_space
from generate_fft_codegen import generate_dispatch, generate_header, load_selection


class FftDesignSpaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.space = load_space(ROOT / "config" / "fft_architecture_space.json")
        cls.hardware = cls.space["hardware_profiles"]["v100-sm70"]
        cls.compiled = load_codegen_points(ROOT / "config" / "v100_fft_codegen.json")

    def point(self, **overrides):
        point = {
            "prefix_log_n": 9,
            "suffix_log_n": 9,
            "prefix_threads": 256,
            "suffix_threads": 256,
            "prefix_ept": 8,
            "suffix_ept": 8,
        }
        point.update(overrides)
        return point

    def test_known_compiled_point_and_resources(self):
        point = self.point()
        status, reason = classify_online_point(point, self.hardware, self.compiled)
        self.assertEqual((status, reason), ("compiled", ""))
        self.assertEqual(point["prefix_units_per_cta"], 4)
        self.assertEqual(point["suffix_units_per_cta"], 4)
        self.assertEqual(point["prefix_tile_shared_bytes"], 16384)

    def test_selected_ept_is_compiled(self):
        point = self.point(prefix_ept=4, prefix_threads=128)
        status, reason = classify_online_point(point, self.hardware, self.compiled)
        self.assertEqual((status, reason), ("compiled", ""))

    def test_direct_codegen_points_are_explicit(self):
        self.assertIn((11, 256, 8), self.compiled)
        self.assertIn((12, 512, 8), self.compiled)
        self.assertNotIn((11, 256, 16), self.compiled)

    def test_direct_online_dimension_is_compiled(self):
        point = self.point(prefix_log_n=4, suffix_log_n=12, prefix_threads=128,
                           suffix_threads=512, suffix_ept=8)
        status, reason = classify_online_point(point, self.hardware, self.compiled)
        self.assertEqual((status, reason), ("compiled", ""))

    def test_dimension_beyond_direct_online_requires_layout_kernel(self):
        point = self.point(prefix_log_n=4, suffix_log_n=13, prefix_threads=128,
                           suffix_threads=512, suffix_ept=16)
        status, reason = classify_online_point(point, self.hardware, self.compiled)
        self.assertEqual(status, "requires-new-kernel")
        self.assertEqual(reason, "large-dimension-layout")

    def test_direct_online_rejects_multiple_units_per_cta(self):
        point = self.point(prefix_log_n=4, suffix_log_n=11, prefix_threads=128,
                           suffix_threads=256, suffix_ept=16)
        status, reason = classify_online_point(point, self.hardware, self.compiled)
        self.assertEqual((status, reason), ("requires-new-kernel", "suffix-direct-single-unit"))

    def test_invalid_coverage_is_hardware_infeasible(self):
        point = self.point(prefix_threads=32, prefix_ept=4)
        status, _ = classify_online_point(point, self.hardware, self.compiled)
        self.assertEqual(status, "hardware-infeasible")

    def test_enumerator_contains_compiled_and_future_points(self):
        rows = list(enumerate_online(self.space, "v100-sm70", [18], [256], [2, 4, 8, 32], ["recurrence"], self.compiled))
        statuses = {row["status"] for row in rows}
        self.assertIn("compiled", statuses)
        self.assertIn("awaiting-codegen", statuses)
        self.assertIn("requires-new-kernel", statuses)

    def test_codegen_selection_and_outputs(self):
        target, log_ns, pairs = load_selection(ROOT / "config" / "v100_fft_codegen.json", self.space)
        self.assertEqual(target, "v100-sm70")
        self.assertIn((256, 16), pairs)
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            generate_header(directory / "config.cuh", target, log_ns, pairs)
            generate_dispatch(directory / "dispatch.inc", target, pairs)
            self.assertIn("threads == 256 && ept == 16", (directory / "config.cuh").read_text())
            self.assertIn("CUNTT_CUFFTDX_ONLINE_LAUNCH(256, 16)", (directory / "dispatch.inc").read_text())

    def test_codegen_rejects_shared_memory_overflow(self):
        invalid = {
            "schema_version": 1,
            "target": "v100-sm70",
            "families": {"cufftdx-online": {
                "dimension_log_n": [10],
                "thread_ept_pairs": [[1024, 16]],
            }},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "invalid.json"
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "shared memory"):
                load_selection(path, self.space)


if __name__ == "__main__":
    unittest.main()
