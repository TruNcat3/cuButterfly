import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_ntt_streaming_space.py"
SPEC = importlib.util.spec_from_file_location("streaming_generator", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StreamingGeneratorTest(unittest.TestCase):
    def test_partitions_cover_supported_lengths(self):
        for log_n in range(12, 21):
            partitions = list(MODULE.stage_partitions(log_n))
            self.assertTrue(partitions)
            self.assertTrue(all(sum(point) == log_n for point in partitions))

    def test_dependency_graph_two_and_three_segments(self):
        self.assertTrue(MODULE.validate_dependency_graph(12, (6, 6)))
        self.assertTrue(MODULE.validate_dependency_graph(15, (5, 5, 5)))
        self.assertTrue(MODULE.validate_dependency_graph(17, (6, 6, 5)))

    def test_publications_are_per_dependency_wave_not_per_point(self):
        point = MODULE.make_point(15, 2, (5, 5, 5), 256, 2, 64)
        self.assertEqual(point["publication_tokens"], 128)
        self.assertLess(point["publication_tokens"], sum(point["task_counts"][:-1]))
        self.assertLess(point["publication_tokens"], 2 * (1 << 15))

    def test_single_transform_requires_outer_digit_for_wavefront(self):
        two = MODULE.make_point(20, 1, (10, 10), 256, 2, 64)
        three = MODULE.make_point(20, 1, (7, 7, 6), 256, 2, 64)
        self.assertFalse(two["single_transform_wavefront"])
        self.assertTrue(three["single_transform_wavefront"])

    def test_v100_resident_point_uses_measured_service_weights(self):
        point = MODULE.make_point(20, 16, (10, 10), 256, 2, 64)
        self.assertEqual(
            [mapping["cta_weight"] for mapping in point["subgraph_mappings"]],
            [9, 11],
        )

    def test_three_layer_point_selects_generated_wave_kernel(self):
        point64 = MODULE.make_point(20, 1, (7, 7, 6), 256, 2, 64)
        point32 = MODULE.make_point(20, 1, (7, 7, 6), 256, 3, 32)
        self.assertEqual(point64["generated_kernel"], "resident-7x7x6-wave")
        self.assertEqual(point64["resident_rows"], 32)
        self.assertEqual(point32["resident_rows"], 32)
        self.assertEqual(
            [mapping["cta_weight"] for mapping in point64["subgraph_mappings"]],
            [6, 6, 8],
        )
        self.assertTrue(point64["single_transform_wavefront"])

    def test_three_layer_permutations_select_generated_wave_kernel(self):
        partitions = ((6, 7, 7), (7, 6, 7), (7, 7, 6),
                      (6, 6, 8), (6, 8, 6), (8, 6, 6))
        for partition in partitions:
            point = MODULE.make_point(20, 1, partition, 256, 2, 32)
            self.assertTrue(point["compiled"])
            self.assertEqual(point["physical_units_per_cta"], 32)
            self.assertEqual(point["task_counts"],
                             MODULE.task_counts(20, partition, 1))
            self.assertTrue(MODULE.validate_dependency_graph(20, partition))

    def test_homogeneous_template_keeps_role_parameters_independent(self):
        point = MODULE.make_point(
            20, 16, (10, 10), 256, 2, 64,
            core="homogeneous-radix4", data_times=(2, 4),
            cta_weights=(8, 12),
        )
        self.assertEqual(point["generated_kernel"],
                         "homogeneous-two-level-10x10")
        self.assertTrue(point["compiled"])
        self.assertEqual(
            [mapping["data_time"] for mapping in point["subgraph_mappings"]],
            [2, 4],
        )
        self.assertEqual(
            [mapping["cta_weight"] for mapping in point["subgraph_mappings"]],
            [8, 12],
        )
        self.assertTrue(all(mapping["units_per_cta"] == 4
                            for mapping in point["subgraph_mappings"]))

    def test_warp_physical_cores_select_generated_instances(self):
        full_warp = MODULE.make_point(
            20, 4, (10, 10), 256, 1, 32,
            core="homogeneous-warp-radix2")
        warp256_128 = MODULE.make_point(
            20, 4, (10, 10), 128, 3, 32,
            core="homogeneous-warp256-radix2")
        warp256_256 = MODULE.make_point(
            20, 4, (10, 10), 256, 1, 64,
            core="homogeneous-warp256-radix2")
        warp256_static = MODULE.make_point(
            20, 4, (10, 10), 128, 2, 64,
            core="homogeneous-warp256-static-radix2")
        warp256_static_io = MODULE.make_point(
            20, 4, (10, 10), 256, 1, 32,
            core="homogeneous-warp256-static-io-radix2", data_space=4)
        warp256_coefficient_reuse = MODULE.make_point(
            20, 4, (10, 10), 128, 3, 32,
            core="homogeneous-warp256-coefficient-reuse-static-io-radix2",
            data_space=4)
        warp256_coefficient_reuse_depth6 = MODULE.make_point(
            20, 4, (10, 10), 128, 3, 32,
            core="homogeneous-warp256-coefficient-reuse-static-io-radix2",
            data_space=4, coefficient_reuse_stages=6)
        warp128_static_io = MODULE.make_point(
            20, 4, (10, 10), 128, 3, 32,
            core="homogeneous-warp128-static-io-radix2", data_space=4)
        warp128_static_io_dual = MODULE.make_point(
            20, 4, (10, 10), 256, 2, 32,
            core="homogeneous-warp128-static-io-radix2", data_space=4)
        warp128_coefficient_reuse_dual = MODULE.make_point(
            20, 4, (10, 10), 256, 2, 32,
            core="homogeneous-warp128-coefficient-reuse-static-io-radix2",
            data_space=4)
        warp128_vector_radix4_dual = MODULE.make_point(
            20, 4, (10, 10), 256, 2, 32,
            core="homogeneous-warp128-vector-radix4-static-io",
            data_space=4)
        warp64_static_io = MODULE.make_point(
            20, 4, (10, 10), 128, 3, 32,
            core="homogeneous-warp64-static-io-radix2", data_space=4)
        warp128_pipeline = MODULE.make_point(
            20, 4, (10, 10), 128, 3, 32,
            core="homogeneous-warp128-pipeline-static-io-radix2",
            data_space=4)
        warp128_cooperative = MODULE.make_point(
            20, 4, (10, 10), 128, 4, 32,
            core="homogeneous-warp128-cooperative-static-io-radix2",
            data_space=4)
        self.assertEqual(full_warp["generated_kernel"],
                         "homogeneous-warp-10x10")
        self.assertEqual(full_warp["physical_units_per_cta"], 8)
        self.assertEqual(warp256_128["generated_kernel"],
                         "homogeneous-warp256-10x10")
        self.assertEqual(warp256_128["physical_units_per_cta"], 4)
        self.assertEqual(warp256_256["physical_units_per_cta"], 8)
        self.assertEqual(warp256_static["generated_kernel"],
                         "homogeneous-warp256-static-10x10")
        self.assertEqual(warp256_static["physical_units_per_cta"], 4)
        self.assertEqual(warp256_static_io["generated_kernel"],
                         "homogeneous-warp256-static-io-10x10")
        self.assertEqual(warp256_static_io["physical_units_per_cta"], 8)
        self.assertEqual(
            warp256_coefficient_reuse["generated_kernel"],
            "homogeneous-warp256-coefficient-reuse-static-io-10x10")
        self.assertEqual(warp256_coefficient_reuse["physical_units_per_cta"], 4)
        self.assertTrue(all(
            mapping["coefficient_reuse_stages"] == 1
            for mapping in warp256_coefficient_reuse["subgraph_mappings"]))
        self.assertTrue(all(
            mapping["coefficient_reuse_stages"] == 6
            for mapping in warp256_coefficient_reuse_depth6["subgraph_mappings"]))
        self.assertEqual(warp128_static_io["generated_kernel"],
                         "homogeneous-warp128-static-io-10x10")
        self.assertEqual(warp128_static_io_dual["generated_kernel"],
                         "homogeneous-warp128-static-io-10x10")
        self.assertEqual(warp128_static_io_dual["physical_units_per_cta"], 8)
        self.assertEqual(
            warp128_coefficient_reuse_dual["generated_kernel"],
            "homogeneous-warp128-coefficient-reuse-static-io-10x10")
        self.assertEqual(
            warp128_coefficient_reuse_dual["physical_units_per_cta"], 8)
        self.assertTrue(all(
            mapping["coefficient_reuse_stages"] == 4
            for mapping in
            warp128_coefficient_reuse_dual["subgraph_mappings"]))
        self.assertEqual(
            warp128_vector_radix4_dual["generated_kernel"],
            "homogeneous-warp128-vector-radix4-static-io-10x10")
        self.assertEqual(
            warp128_vector_radix4_dual["physical_units_per_cta"], 8)
        warp128_vector_radix4_reuse = MODULE.make_point(
            20, 16, (10, 10), 256, 2, 32, "full-scratch", 1,
            core="homogeneous-warp128-vector-radix4-static-io",
            data_times=(1, 1), cta_weights=(8, 7), data_space=4,
            coefficient_reuse_stages=6)
        self.assertTrue(warp128_vector_radix4_reuse["compiled"])
        self.assertTrue(all(
            mapping["coefficient_reuse_stages"] == 6
            for mapping in
            warp128_vector_radix4_reuse["subgraph_mappings"]))
        warp128_vector_radix4_reuse7 = MODULE.make_point(
            20, 16, (10, 10), 256, 2, 32, "full-scratch", 1,
            core="homogeneous-warp128-vector-radix4-static-io",
            data_times=(1, 1), cta_weights=(8, 7), data_space=4,
            coefficient_reuse_stages=7)
        self.assertTrue(warp128_vector_radix4_reuse7["compiled"])
        self.assertTrue(all(
            mapping["coefficient_reuse_stages"] == 7
            for mapping in
            warp128_vector_radix4_reuse7["subgraph_mappings"]))
        warp128_vector_radix4_packed_stage6 = MODULE.make_point(
            20, 16, (10, 10), 256, 2, 32, "full-scratch", 1,
            core="homogeneous-warp128-vector-radix4-packed-stage6-static-io",
            data_times=(1, 1), cta_weights=(8, 7), data_space=4,
            coefficient_reuse_stages=6)
        self.assertEqual(
            warp128_vector_radix4_packed_stage6["generated_kernel"],
            "homogeneous-warp128-vector-radix4-packed-stage6-static-io-10x10")
        self.assertTrue(all(
            mapping["coefficient_reuse_stages"] == 6
            for mapping in
            warp128_vector_radix4_packed_stage6["subgraph_mappings"]))
        warp128_vector_radix4_packed_stage6_distributed = MODULE.make_point(
            20, 16, (10, 10), 256, 2, 32, "full-scratch", 1,
            core="homogeneous-warp128-vector-radix4-packed-stage6-distributed-static-io",
            data_times=(1, 1), cta_weights=(8, 7), data_space=4,
            coefficient_reuse_stages=6)
        self.assertEqual(
            warp128_vector_radix4_packed_stage6_distributed["generated_kernel"],
            "homogeneous-warp128-vector-radix4-packed-stage6-distributed-static-io-10x10")
        self.assertTrue(all(
            mapping["coefficient_reuse_stages"] == 6
            for mapping in
            warp128_vector_radix4_packed_stage6_distributed["subgraph_mappings"]))
        packet_shared_radix4 = MODULE.make_point(
            20, 16, (10, 10), 128, 4, 32, "full-scratch", 1,
            core="homogeneous-warp128-packet-shared-radix4-static-io",
            data_times=(1, 1), cta_weights=(9, 11), data_space=4)
        self.assertEqual(
            packet_shared_radix4["generated_kernel"],
            "homogeneous-warp128-packet-shared-radix4-static-io-10x10")
        self.assertEqual(packet_shared_radix4["physical_units_per_cta"], 4)
        self.assertEqual(warp64_static_io["generated_kernel"],
                         "homogeneous-warp64-static-io-10x10")
        self.assertEqual(warp128_static_io["physical_units_per_cta"], 4)
        self.assertEqual(warp64_static_io["physical_units_per_cta"], 4)
        self.assertEqual(warp128_pipeline["generated_kernel"],
                         "homogeneous-warp128-pipeline-static-io-10x10")
        self.assertEqual(warp128_pipeline["pipeline_buffers"], 1)
        self.assertEqual(warp128_cooperative["generated_kernel"],
                         "homogeneous-warp128-cooperative-static-io-10x10")
        self.assertEqual(warp128_cooperative["physical_units_per_cta"], 4)
        self.assertTrue(all(mapping["data_space"] == 4
                            for mapping in warp256_static_io["subgraph_mappings"]))
        self.assertTrue(full_warp["compiled"])
        self.assertTrue(warp256_128["compiled"])
        self.assertTrue(warp256_coefficient_reuse["compiled"])
        self.assertTrue(warp128_static_io["compiled"])
        self.assertTrue(warp128_static_io_dual["compiled"])
        self.assertTrue(warp128_coefficient_reuse_dual["compiled"])
        self.assertTrue(warp64_static_io["compiled"])
        self.assertTrue(warp128_pipeline["compiled"])
        self.assertTrue(warp128_cooperative["compiled"])


if __name__ == "__main__":
    unittest.main()
