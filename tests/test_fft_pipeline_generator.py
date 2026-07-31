#!/usr/bin/env python3
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fft_design_space import load_codegen_points, load_space
from generate_fft_pipeline import (enumerate_decomposition_topologies, enumerate_pipeline,
                                   load_pipeline, stage_partitions)


class FftPipelineGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = load_pipeline(ROOT / "config" / "v100_fft_pipeline.json")
        cls.architecture = load_space(ROOT / "config" / "fft_architecture_space.json")
        cls.codegen = load_codegen_points(ROOT / "config" / "v100_fft_codegen.json")
        cls.runnable, cls.backlog = enumerate_pipeline(cls.pipeline, cls.architecture, cls.codegen)
        cls.topologies = enumerate_decomposition_topologies(cls.pipeline)

    def test_runnable_points_are_bounded_per_shape(self):
        counts = {}
        for point in self.runnable:
            counts.setdefault((point["mapping_id"], point["batch"]), 0)
            counts[(point["mapping_id"], point["batch"])] += 1
        self.assertEqual(len(counts), 6)
        self.assertTrue(all(value <= (20 if mapping == "fft20-fp32" else 24)
                            for (mapping, _), value in counts.items()))
        self.assertTrue(all(point["selection_pool_size"] >= point["selection_limit"] for point in self.runnable))

    def test_fft18_contains_asymmetric_compiled_factorizations(self):
        splits = {(point["prefix_log_n"], point["suffix_log_n"]) for point in self.runnable
                  if point["mapping_id"] == "fft18-fp32"}
        self.assertEqual(splits, {(6, 12), (7, 11), (8, 10), (9, 9), (10, 8), (11, 7), (12, 6)})

    def test_stage_partition_is_an_ordered_composition(self):
        partitions = list(stage_partitions(18, 3, 3, 12))
        self.assertIn((6, 6, 6), partitions)
        self.assertIn((3, 7, 8), partitions)
        self.assertTrue(all(len(item) == 3 and sum(item) == 18 for item in partitions))

    def test_multidimensional_topologies_are_explicit_runtime_backlog(self):
        counts = {(item["logN"], item["decomposition_count"], item["status"])
                  for item in self.topologies}
        self.assertIn((18, 2, "compiled-runtime"), counts)
        self.assertIn((18, 3, "requires-multi-pass-runtime"), counts)
        self.assertIn((20, 4, "requires-multi-pass-runtime"), counts)
        self.assertTrue(all(sum(item["stages_per_decomposition"]) == item["logN"]
                            for item in self.topologies))

    def test_fft20_large_dimensions_are_compiled_mixed_unit_points(self):
        points = [point for point in self.runnable if point["mapping_id"] == "fft20-fp32"]
        splits = {(point["prefix_log_n"], point["suffix_log_n"]) for point in points}
        self.assertTrue({(8, 12), (9, 11), (11, 9), (12, 8)} <= splits)
        mixed = [point for point in points if point["prefix_processing_unit"] != point["suffix_processing_unit"]]
        self.assertTrue(mixed)
        self.assertFalse(self.backlog)

    def test_suffix_direct_points_include_both_boundary_realizations(self):
        points = [point for point in self.runnable
                  if point["mapping_id"] == "fft20-fp32" and point["suffix_log_n"] >= 11]
        self.assertEqual({point["direct_boundary"] for point in points},
                         {"direct-strided", "tiled-transpose"})
        self.assertTrue(all(point["id"].endswith("_tx")
                            for point in points if point["direct_boundary"] == "tiled-transpose"))

    def test_prefix_direct_points_include_both_boundary_realizations(self):
        points = [point for point in self.runnable
                  if point["mapping_id"] == "fft20-fp32" and point["prefix_log_n"] >= 11]
        self.assertEqual({point["direct_boundary"] for point in points},
                         {"direct-strided", "prefix-tiled-transpose"})
        self.assertTrue(all(point["id"].endswith("_ptx")
                            for point in points if point["direct_boundary"] == "prefix-tiled-transpose"))

    def test_generated_runtime_arguments_keep_dimensions_independent(self):
        point = next(point for point in self.runnable if point["prefix_log_n"] != point["suffix_log_n"])
        self.assertEqual(point["decomposition_count"], 2)
        self.assertEqual(sum(point["stages_per_decomposition"]), point["logN"])
        args = point["args"]
        self.assertIn("--prefix-threads", args)
        self.assertIn("--suffix-threads", args)
        self.assertIn("--prefix-ept", args)
        self.assertIn("--suffix-ept", args)
        self.assertIn("--direct-boundary", args)

    def test_each_shape_retains_measured_incumbent(self):
        expected_by_log_n = {18: [(9, 256, 256, 8, 8)],
                             20: [(10, 512, 512, 8, 8), (10, 256, 128, 16, 16)]}
        for log_n, expected_points in expected_by_log_n.items():
            batches = {point["batch"] for point in self.runnable if point["logN"] == log_n}
            for batch in batches:
                candidates = [point for point in self.runnable
                              if point["logN"] == log_n and point["batch"] == batch]
                actual = {(point["prefix_log_n"], point["prefix_threads"], point["suffix_threads"],
                           point["prefix_ept"], point["suffix_ept"]) for point in candidates}
                for expected in expected_points:
                    self.assertIn(expected, actual)


if __name__ == "__main__":
    unittest.main()
