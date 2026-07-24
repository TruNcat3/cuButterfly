#!/usr/bin/env python3
import copy
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from butterfly_design_space import (
    derive_architecture_point,
    derive_multidimensional_point,
    expand_design_choices,
    load_space,
    stage_partitions,
    validate_fft_projection,
    validate_hierarchy_factors,
)


class ButterflyDesignSpaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.space = load_space(ROOT / "config" / "butterfly_architecture_space.json")
        cls.fft = json.loads((ROOT / "config" / "fft_architecture_space.json").read_text())

    def test_all_operator_projections_and_fft_projection(self):
        self.assertEqual(len(self.space["operator_projections"]), 6)
        self.assertTrue(validate_fft_projection(self.space, self.fft))

    def test_ordered_factorizations_are_complete(self):
        self.assertEqual(stage_partitions(5, 1), [(5,)])
        self.assertEqual(stage_partitions(5, 2), [(1, 4), (2, 3), (3, 2), (4, 1)])
        self.assertEqual(len(stage_partitions(5, 3)), 6)

    def test_fixed_compute_budget_preserves_ideal_cycles(self):
        points = [
            derive_architecture_point(self.space, "ntt", 16, us, 128 // us, 8, "shared", "shared")
            for us in (1, 2, 4, 8, 16)
        ]
        self.assertEqual({point["spatial_butterfly_cells"] for point in points}, {128})
        self.assertEqual({point["ideal_body_cycles"] for point in points}, {4096})
        self.assertEqual([point["boundary_coefficient_words_per_cycle"] for point in points],
                         [256, 128, 64, 32, 16])
        self.assertEqual([point["interstage_coefficient_words_per_cycle"] for point in points],
                         [0, 128, 192, 224, 240])

    def test_operator_traits_change_coefficient_demand(self):
        fft = derive_architecture_point(self.space, "fft", 16, 8, 16, 8, "shared", "shared")
        fwht = derive_architecture_point(self.space, "fwht", 16, 8, 16, 8, "shared", "shared")
        self.assertEqual(fft["coefficient_streams_upper_bound"], 128)
        self.assertEqual(fwht["coefficient_streams_upper_bound"], 0)

    def test_batch_unfolding_is_independent_of_stage_and_data_unfolding(self):
        serial = derive_architecture_point(
            self.space, "fft", 12, 4, 16, 8, "shared", "shared", batch=10, ub=1)
        parallel = derive_architecture_point(
            self.space, "fft", 12, 4, 16, 8, "shared", "shared", batch=10, ub=4)
        for field in ("Us", "Ts", "Ud", "Td"):
            self.assertEqual(serial[field], parallel[field])
        self.assertEqual((serial["Ub"], serial["Tb"]), (1, 10))
        self.assertEqual((parallel["Ub"], parallel["Tb"]), (4, 3))
        self.assertEqual(serial["ideal_body_cycles"], 10 * parallel["ideal_body_cycles"] // 3)
        self.assertEqual(parallel["boundary_coefficient_words_per_cycle"],
                         4 * serial["boundary_coefficient_words_per_cycle"])
        self.assertEqual(parallel["coefficient_streams_upper_bound"],
                         4 * serial["coefficient_streams_upper_bound"])

    def test_hierarchy_factors_compose_to_logical_unfolding(self):
        factors = {
            "lane": {"Ud": 4},
            "warp": {"Us": 2, "Ud": 2},
            "cta": {"Us": 4, "Ub": 2},
        }
        self.assertTrue(validate_hierarchy_factors(8, 8, 2, factors))
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_hierarchy_factors(4, 8, 2, factors)

    def test_each_factorized_dimension_has_its_own_two_axis_mapping(self):
        point = derive_multidimensional_point(
            10, (4, 6), ({"Us": 2, "Ud": 4}, {"Us": 3, "Ud": 8}))
        self.assertEqual(point["dimension_boundaries"], 1)
        self.assertEqual(point["dimensions"][0]["Ts"], 2)
        self.assertEqual(point["dimensions"][1]["Ts"], 2)
        self.assertEqual(point["dimensions"][1]["Td"], 4)

    def test_runtime_backends_are_bound_to_architecture_families(self):
        bindings = self.space["runtime_bindings"]
        self.assertEqual(set(bindings["ntt_backends"]),
                         {"baseline", "tile256", "hybrid2d", "compact-stage", "stage-pipeline"})
        self.assertEqual(set(bindings["butterfly_backends"]),
                         {"temporal-tile", "hierarchical", "online-reorder", "warp-hybrid", "stage-pipeline", "cufft"})

    def test_filtered_processing_layout_realization_product_is_enumerable(self):
        choices = expand_design_choices(self.space, "fft", {
            "operator_core": ["scalar-radix4", "cufftdx-block"],
            "coefficient_form": ["native-table", "recurrence"],
            "kernel_form": ["online-reorder"],
            "threads_per_cta": [256],
        })
        self.assertEqual(len(choices), 4)
        self.assertEqual({point["kernel_form"] for point in choices}, {"online-reorder"})
        self.assertEqual({point["precision"] for point in choices}, {"fp32"})
        self.assertEqual({point["stage_group"] for point in choices}, {"radix4", "full-codelet"})
        self.assertTrue({
            "direction", "element_stride", "schedule", "coefficient_residence",
            "elements_per_thread", "global_order", "bank_mapping", "pipeline_depth",
            "selection_method", "candidate_status", "objective",
        } <= set(choices[0]))

    def test_operator_projection_rejects_invalid_coefficient_choice(self):
        with self.assertRaisesRegex(ValueError, "coefficient_form"):
            expand_design_choices(self.space, "fwht", {"coefficient_form": ["native-table"]})

    def test_core_precision_binding_rejects_invalid_cross_product(self):
        with self.assertRaisesRegex(ValueError, "no legal choices"):
            expand_design_choices(self.space, "fft", {
                "operator_core": ["wmma-dft8"], "precision": ["fp64"]})

    def test_every_declared_operator_core_has_a_legal_default_point(self):
        for operator, projection in self.space["operator_projections"].items():
            for core in projection["cores"]:
                with self.subTest(operator=operator, core=core):
                    points = expand_design_choices(
                        self.space, operator, {"operator_core": [core]})
                    self.assertGreaterEqual(len(points), 1)

    def test_residence_changes_feedback_traffic_not_logical_factors(self):
        shared = derive_architecture_point(self.space, "fft", 16, 8, 16, 8, "shared", "shared")
        global_point = derive_architecture_point(self.space, "fft", 16, 8, 16, 8, "global", "global")
        for field in ("Us", "Ts", "Ud", "Td", "ideal_body_cycles"):
            self.assertEqual(shared[field], global_point[field])
        self.assertEqual(shared["stage_feedback_bytes_above_residence"], 0)
        self.assertGreater(global_point["stage_feedback_bytes_above_residence"], 0)

    def test_invalid_fft_projection_is_rejected(self):
        invalid = copy.deepcopy(self.fft)
        invalid["axes"]["dimension"]["core"].append("unknown-core")
        with self.assertRaisesRegex(ValueError, "core"):
            validate_fft_projection(self.space, invalid)

    def test_incomplete_hierarchy_axis_is_rejected(self):
        invalid = copy.deepcopy(self.space)
        del invalid["objects"]["A_architecture"]["hierarchical_unfolding"]
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "invalid.json"
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "hierarchical_unfolding"):
                load_space(path)

    def test_unknown_runtime_family_is_rejected(self):
        invalid = copy.deepcopy(self.space)
        invalid["runtime_bindings"]["ntt_backends"]["invalid"] = "missing-family"
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "invalid.json"
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "unknown families"):
                load_space(path)


if __name__ == "__main__":
    unittest.main()
