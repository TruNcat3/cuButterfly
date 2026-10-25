"""CPU contracts for shared organization projection and calibration identity."""
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from butterfly_design_space import expand_design_choices, load_space
from calibration_space import command_for, mapping_point, predicted_sample, stratified_candidates
from stage_service_calibration import _bulk_service_point, _point_static_key, _service_points
from stage_service_model import curve_key


class GenericOrganizationProjectionTest(unittest.TestCase):
    def test_all_operator_projections_reject_invalid_or_truncated_partitions(self):
        operators = ("fft", "ntt", "fwht", "subset-zeta", "superset-zeta", "structured-2x2")
        invalid = ([4], [4, 5], [4, 0, 4], [4, -1, 5], [4.0, 4], [True, 7], ["4", 4], "4x4")
        for operator in operators:
            for backend in ("shared-iterative", "hierarchical", "online-reorder"):
                for partition in invalid:
                    with self.subTest(operator=operator, backend=backend, partition=partition):
                        point = mapping_point(
                            {"operator": operator, "logN": 8, "batch": 1,
                             "precision": "word32" if operator == "ntt" else "fp32"},
                            {"backend": backend, "stage_partition": partition,
                             "local_stages": 4, "flow_tile_log_n": 4})
                        with self.assertRaisesRegex(ValueError, "positive integer stages covering logN"):
                            predicted_sample(point)

    def test_all_operator_valid_projections_cover_every_stage(self):
        for operator in ("fft", "ntt", "fwht", "subset-zeta", "superset-zeta", "structured-2x2"):
            for backend in ("shared-iterative", "hierarchical", "online-reorder"):
                with self.subTest(operator=operator, backend=backend):
                    sample = predicted_sample(mapping_point(
                        {"operator": operator, "logN": 8, "batch": 1,
                         "precision": "word32" if operator == "ntt" else "fp32"},
                        {"backend": backend, "stage_partition": [4, 4],
                         "local_stages": 4, "flow_tile_log_n": 4}))
                    groups = sample["execution_groups_json"]
                    self.assertEqual(sum(group["stage_count"] for group in groups), 8)
                    self.assertTrue(all(type(group["stage_count"]) is int and group["stage_count"] > 0
                                        for group in groups))
                    self.assertTrue(all(group["compiler_local_resources_known"] is False for group in groups))

    def test_factor_macro_and_physical_coverage_are_distinct_and_validated(self):
        workload = {"operator": "fft", "logN": 8, "batch": 1, "precision": "fp64"}
        mapping = {"backend": "factor-streamed", "fft_core": "cufftdx-block",
                   "stage_partition": [4, 4], "factor_partition": [2, 2, 2, 2],
                   "factor_ept": 2, "factor_columns": 2}
        sample = predicted_sample(mapping_point(workload, mapping))
        self.assertEqual(sample["decomposition_count"], 2)
        self.assertEqual(sample["execution_group_count"], 4)
        self.assertEqual([group["stage_count"] for group in sample["execution_groups_json"]], [2, 2, 2, 2])
        for partition in ([4], [4, 5], [4, 0, 4], [4.0, 4], [True, 7]):
            with self.subTest(partition=partition):
                with self.assertRaisesRegex(ValueError, "factor_partition"):
                    predicted_sample(mapping_point(workload, {**mapping, "factor_partition": partition}))
        with self.assertRaisesRegex(ValueError, "logical.*endpoint"):
            predicted_sample(mapping_point(workload, {**mapping, "stage_partition": [3, 5],
                                                     "factor_partition": [4, 4]}))

    def test_invalid_implicit_local_extents_are_rejected_before_expansion(self):
        for backend in ("shared-iterative", "hierarchical", "online-reorder"):
            for local in (-1, 2.5, True):
                with self.subTest(backend=backend, local=local):
                    with self.assertRaisesRegex(ValueError, "stage extent"):
                        predicted_sample({"operator": "fwht", "logN": 8, "batch": 1,
                                          "precision": "fp32", "backend": backend, "local_stages": local})

    def test_explicit_register_partition_supplies_missing_geometry_axis_without_mutation(self):
        mapping = {"backend": "online-reorder", "fft_core": "register-tile",
                   "stage_partition": [8, 12], "prefix_threads": 128, "prefix_ept": 16,
                   "suffix_threads": 512, "suffix_ept": 16}
        workload = {"operator": "fft", "precision": "fp32", "logN": 20, "batch": 17}
        point = mapping_point(workload, mapping)
        encoded = point["mapping_json"]
        sample = predicted_sample(point)
        self.assertEqual([group["stage_count"] for group in sample["execution_groups_json"]], [8, 12])
        self.assertEqual(point["mapping_json"], encoded)
        self.assertNotIn("local_stages", mapping)
        self.assertNotIn("local_stages", sample)
        with self.assertRaisesRegex(ValueError, "partition.*prefix"):
            predicted_sample(mapping_point(workload, {**mapping, "local_stages": 6}))

    def test_default_enumeration_covers_every_bound_core(self):
        space = load_space(ROOT / "config" / "butterfly_architecture_space.json")
        for operator, projection in space["operator_projections"].items():
            with self.subTest(operator=operator):
                points = expand_design_choices(space, operator, {})
                self.assertEqual({p["operator_core"] for p in points}, set(projection["cores"]))
                self.assertEqual({p["candidate_status"] for p in points}, {"abstract"})

    def test_shared_organization_is_owned_by_mapping_for_all_operators(self):
        for operator in ("fft", "ntt", "fwht", "subset-zeta", "superset-zeta", "structured-2x2"):
            with self.subTest(operator=operator):
                mapping = {"backend": "shared-iterative", "local_stages": 4,
                           "flow_tile_log_n": 4, "stage_partition": [4, 4],
                           "local_stage_partitions": [[2, 2], [2, 2]],
                           "exchange_chunks": [0, 0], "compute_unit": "radix4",
                           "local_exchange": "shared", "tile_threads": 64,
                           "shared_layout": "writer-aligned", "stage_overlap": 0}
                point = mapping_point({"operator": operator, "logN": 8, "batch": 3,
                                       "precision": "word32" if operator == "ntt" else "fp32"}, mapping)
                point.update(tile_threads=256, shared_layout="linear", stage_overlap=1,
                             compute_unit="radix2", local_exchange="warp-register")
                sample = predicted_sample(point)
                self.assertEqual(sample["tile_threads"], 64)
                self.assertEqual(sample["compute_unit"], "radix4")
                self.assertEqual(sample["stage_overlap"], 0)
                self.assertEqual(sample["shared_layout"], "writer-aligned")
                self.assertEqual(sample["execution_group_count"], 2)
                self.assertEqual(sample["decomposition_count"], 2)
                self.assertEqual([g["local_stage_partition"] for g in sample["execution_groups_json"]],
                                 [[2, 2], [2, 2]])
                command = command_for("probe", point)
                for flag in ("--compute-unit", "--local-exchange", "--tile-threads",
                             "--local-stage-partitions", "--stage-overlap"):
                    self.assertNotIn(flag, command)

    def test_ntt_default_partition_uses_flow_tile_extent(self):
        point = mapping_point({"operator": "ntt", "logN": 10, "batch": 1, "precision": "word32"},
                              {"backend": "shared-iterative", "flow_tile_log_n": 4})
        sample = predicted_sample(point)
        self.assertEqual(sample["decomposition_count"], 3)
        self.assertEqual([g["stage_count"] for g in sample["execution_groups_json"]], [4, 4, 2])

    def test_ntt_fusion_keeps_logical_and_physical_counts_distinct(self):
        mapping = {"backend": "shared-iterative", "stage_partition": [2, 2, 4],
                   "boundary_mappings": [{"storage": "resident-fused", "buffers": 1},
                                         {"storage": "full-scratch", "buffers": 1}],
                   "threads_per_block": 96, "batch_tile_count": 3}
        sample = predicted_sample(mapping_point(
            {"operator": "ntt", "precision": "word64", "logN": 8, "batch": 7}, mapping))
        self.assertEqual(sample["decomposition_count"], 3)
        self.assertEqual(sample["execution_group_count"], 2)
        self.assertEqual([g["stage_count"] for g in sample["execution_groups_json"]], [4, 4])

    def test_prediction_does_not_mutate_input_partition(self):
        mapping = {"backend": "shared-iterative", "local_stages": 4, "stage_partition": []}
        point = {"operator": "fwht", "precision": "fp32", "logN": 8, "batch": 1,
                 "mapping_json": mapping}
        predicted_sample(point)
        self.assertEqual(mapping["stage_partition"], [])

    def test_mapping_changes_keep_distinct_calibration_identity(self):
        base = {"operator": "fwht", "precision": "fp32", "logN": 8, "batch": 3,
                "mapping_json": {"backend": "shared-iterative", "stage_partition": [4, 4]}}
        original = _point_static_key(base)
        for field, value in (("local_stage_partitions", [[2, 2], [2, 2]]),
                             ("tile_threads", 64), ("shared_layout", "writer-aligned"),
                             ("local_exchange", "shared"), ("stage_overlap", 1),
                             ("batch_tile_count", 2)):
            changed = {**base, "mapping_json": {**base["mapping_json"], field: value}}
            self.assertNotEqual(original, _point_static_key(changed))

    def test_screening_uses_mapping_backend_and_keeps_full_inventory(self):
        workload = {"operator": "fwht", "precision": "fp32", "logN": 8, "batch": 1}
        points = [mapping_point(workload, {"backend": backend, "local_stages": 4,
                  "tile_threads": 64}) for backend in ("shared-iterative", "hierarchical")]
        for point in points:
            point["backend"] = "stale-alias"
        ordered = stratified_candidates(points, 0)
        self.assertEqual(len(ordered), 2)
        self.assertEqual({json.loads(p["mapping_json"])["backend"] for p in ordered},
                         {"shared-iterative", "hierarchical"})

    def test_bulk_service_preserves_shared_cta_batching(self):
        point = {"stage_overlap": 1, "batch_tile_count": 1,
                 "mapping_json": {"backend": "shared-iterative", "stage_overlap": 0,
                                  "batch_tile_count": 4}}
        bulk = _bulk_service_point(point)
        self.assertEqual(bulk["batch_tile_count"], 4)
        self.assertEqual(json.loads(bulk["mapping_json"])["batch_tile_count"], 4)
        service, composition = _service_points([point])
        self.assertEqual(service, [point])
        self.assertEqual(composition, [])

    def test_overlap_chunks_reset_to_serial_service(self):
        point = {"mapping_json": {"backend": "online-reorder", "stage_overlap": 1,
                                  "batch_tile_count": 4}}
        bulk = _bulk_service_point(point)
        self.assertEqual(bulk["batch_tile_count"], 1)
        self.assertEqual(json.loads(bulk["mapping_json"])["stage_overlap"], 0)

    def test_shared_bulk_batch_shape_changes_grid_without_scratch_replication(self):
        workload = {"operator": "fwht", "precision": "fp32", "logN": 8, "batch": 5}
        mapping = {"backend": "shared-iterative", "stage_partition": [4, 4]}
        single = predicted_sample(mapping_point(workload, mapping))["execution_groups_json"]
        folded = predicted_sample(mapping_point(workload, {**mapping, "batch_tile_count": 4}))["execution_groups_json"]
        for before, after in zip(single, folded):
            self.assertEqual(before["grid_ctas"], 80)
            self.assertEqual(after["grid_ctas"], 32)
            self.assertEqual(after["batch_space"], 4)
            self.assertEqual(after["batch_time"], 2)
            self.assertEqual(before["live_shared_bytes"], after["live_shared_bytes"])

    def test_top_level_shared_batch_shape_is_not_load_only_identity(self):
        base = {"operator": "fwht", "batch_tile_count": 1,
                "mapping_json": {"backend": "shared-iterative"}}
        self.assertNotEqual(_point_static_key(base), _point_static_key({**base, "batch_tile_count": 4}))

    def test_service_shape_uses_bulk_batch_width_but_not_total_batch(self):
        sample = {"operator": "fwht", "precision": "fp32", "logN": 8, "batch": 5,
                  "mapping_json": {"backend": "shared-iterative", "batch_tile_count": 1}}
        group = {"index": 0, "stage_count": 4, "threads": 64, "core": "scalar"}
        original = curve_key(sample, group)
        folded = {**sample, "mapping_json": {"backend": "shared-iterative", "batch_tile_count": 4}}
        self.assertNotEqual(original, curve_key(folded, group))
        self.assertEqual(curve_key(folded, group), curve_key({**folded, "batch": 9}, group))
        self.assertEqual(curve_key(folded, {**group, "batch_time": 2}),
                         curve_key(folded, {**group, "batch_time": 3}))


if __name__ == "__main__":
    unittest.main()
