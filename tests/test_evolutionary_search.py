import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from calibration_space import candidate_id, mapping_point, predicted_sample, stratified_candidates
from evolutionary_search import evolutionary_candidates, neighbors


def _point(mapping, **semantics):
    base = {
        "operator": "fft",
        "precision": "fp64",
        "direction": "forward",
        "normalization": "none",
        "placement": "out-of-place",
        "logN": 12,
        "batch": 8,
        "element_stride": 2,
        "batch_stride": 8195,
    }
    base.update(semantics)
    return {
        **base,
        "backend": mapping["backend"],
        "mapping_json": json.dumps(mapping, sort_keys=True),
    }


def test_ntt_semantics_survive_every_neighbour():
    mapping = {
        "schema_version": 1,
        "kind": "ntt",
        "backend": "shared-iterative",
        "compute_unit": "radix4",
        "stage_partition": [3, 4, 5],
        "threads_per_block": 128,
        "stage_overlap": True,
        "batch_tile_count": 4,
    }
    point = _point(
        mapping,
        operator="ntt",
        precision="word32",
        modulus=2013265921,
        direction="inverse",
        input_order="natural",
        output_order="bit-reversed",
        input_strides=[2],
        output_strides=[3],
    )
    original = {key: point[key] for key in (
        "operator", "precision", "modulus", "direction", "input_order", "output_order",
        "input_strides", "output_strides", "element_stride", "batch_stride",
    )}
    proposals = neighbors(point)
    assert proposals
    assert all({key: proposal.get(key) for key in original} == original for proposal in proposals)


def test_stage_structural_neighbours_are_positive_unbounded_and_reset_metadata():
    mapping = {
        "schema_version": 1,
        "kind": "butterfly",
        "backend": "online-reorder",
        "fft_core": "cufftdx-block",
        "stage_partition": [3, 4, 5],
        "prefix_threads": 256,
        "suffix_threads": 128,
        "prefix_ept": 8,
        "suffix_ept": 4,
        "segment_mappings": [{"core": "cufftdx-block"}] * 3,
        "execution_group_mappings": [{"core": "cufftdx-block"}] * 3,
        "boundaries": [
            {"twiddle": "recurrence", "layout": "direct-strided", "residency": "global-scratch"},
            {"twiddle": "table", "layout": "tiled-transpose", "residency": "global-scratch"},
        ],
    }
    point = _point(mapping)
    proposals = neighbors(point)
    partitions = [json.loads(candidate["mapping_json"]).get("stage_partition") for candidate in proposals]
    assert any(len(partition) > 3 for partition in partitions)
    assert any(len(partition) < 3 for partition in partitions)
    assert any(partition != [3, 4, 5] and len(partition) == 3 for partition in partitions)
    for candidate, partition in zip(proposals, partitions):
        if partition != [3, 4, 5]:
            mutated = json.loads(candidate["mapping_json"])
            assert sum(partition) == 12 and all(value > 0 for value in partition)
            assert "segment_mappings" not in mutated
            assert "execution_group_mappings" not in mutated
            assert len(mutated["boundaries"]) == len(partition) - 1


def test_factor_neighbours_preserve_macro_endpoints_and_native_legality():
    imported = {
        "schema_version": 1,
        "kind": "butterfly",
        "backend": "factor-streamed",
        "fft_core": "cufftdx-block",
        "stage_partition": [8, 16],
        "factor_partition": [4, 4, 8, 8],
        "factor_ept": 16,
        "factor_columns": 8,
        "data_tiles_per_cta": 4,
        "prefetch_depth": 1,
        "factor_slices": 2,
    }
    imported_neighbours = neighbors(_point(imported, logN=24))
    factors = [json.loads(candidate["mapping_json"]).get("factor_partition") for candidate in imported_neighbours]
    assert any(candidate == [4, 4, 16] for candidate in factors)
    for candidate in imported_neighbours:
        mutated = json.loads(candidate["mapping_json"])
        if mutated.get("factor_partition"):
            cumulative = 0
            endpoints = set()
            for value in mutated["factor_partition"][:-1]:
                cumulative += value
                endpoints.add(cumulative)
            assert 8 in endpoints and sum(mutated["factor_partition"]) == 24

    native = dict(imported, fft_core="register-tile", factor_partition=[8, 8, 8], factor_ept=16)
    native_neighbours = neighbors(_point(native, logN=24))
    for candidate in native_neighbours:
        mutated = json.loads(candidate["mapping_json"])
        factors = mutated.get("factor_partition")
        if factors:
            assert len(set(factors)) == 1 and all(value % 2 == 0 for value in factors)
            assert mutated["factor_ept"] == 1 << (factors[0] // 2)


def test_neighbours_are_deterministic_unique_and_can_propose_outside_inventory():
    mapping = {
        "schema_version": 1,
        "kind": "butterfly",
        "backend": "shared-iterative",
        "fft_core": "scalar",
        "stage_partition": [6, 6],
        "tile_threads": 128,
        "stage_overlap": False,
        "batch_tile_count": 1,
    }
    point = _point(mapping)
    first = neighbors(point)
    second = neighbors(point)
    assert first == second
    assert len({candidate_id(candidate) for candidate in first}) == len(first)
    assert any(json.loads(candidate["mapping_json"])["stage_partition"] == [3, 3, 6] for candidate in first)


def test_evolutionary_candidates_use_multiple_winners_model_seeds_and_family_diversity():
    parent_a = _point({
        "schema_version": 1, "kind": "butterfly", "backend": "shared-iterative", "fft_core": "scalar",
        "stage_partition": [6, 6], "tile_threads": 128,
    })
    parent_b = _point({
        "schema_version": 1, "kind": "butterfly", "backend": "online-reorder", "fft_core": "cufftdx-block",
        "stage_partition": [4, 8], "prefix_threads": 256, "suffix_threads": 128,
    })
    seed = _point({
        "schema_version": 1, "kind": "butterfly", "backend": "factor-streamed", "fft_core": "cufftdx-block",
        "stage_partition": [12], "factor_partition": [4, 4, 4], "factor_ept": 4,
        "data_tiles_per_cta": 2, "prefetch_depth": 1,
    })
    measured = {candidate_id(parent_a)}
    output = evolutionary_candidates([parent_a, parent_b], [seed], measured, lambda candidate: 0.1 if candidate is seed else 1.0, limit=20)
    assert len(output) == 20
    assert candidate_id(parent_a) not in {candidate_id(candidate) for candidate in output}
    families = {(json.loads(candidate["mapping_json"]).get("backend"), json.loads(candidate["mapping_json"]).get("fft_core")) for candidate in output}
    assert {"shared-iterative", "online-reorder", "factor-streamed"} <= {family[0] for family in families}
    assert output == evolutionary_candidates([parent_a, parent_b], [seed], measured, lambda candidate: 0.1 if candidate is seed else 1.0, limit=20)


def test_invalid_mapping_is_explicitly_rejected():
    with pytest.raises(ValueError, match="schema_version=1"):
        neighbors({"operator": "fft", "mapping_json": "{}"})


def test_register_strategy_neighbours_are_explicit_and_fp64_safe():
    mapping = {
        "schema_version": 1, "kind": "butterfly", "backend": "online-reorder",
        "fft_core": "register-tile", "stage_partition": [6, 6],
        "prefix_threads": 128, "prefix_ept": 8,
        "suffix_threads": 128, "suffix_ept": 8,
    }
    proposals = [json.loads(candidate["mapping_json"]) for candidate in neighbors(_point(mapping, precision="fp32"))]
    assert any(item.get("prefix_codelet") == "cufftdx-thread" for item in proposals)
    assert any(item.get("prefix_shared_layout") == "xor" for item in proposals)
    fp64 = [json.loads(candidate["mapping_json"]) for candidate in neighbors(_point(mapping, precision="fp64"))]
    assert all(item.get("prefix_codelet") != "cufftdx-thread" for item in fp64)


def test_register_lane_neighbours_keep_columns_and_update_threads_and_ept():
    mapping = {
        "schema_version": 1, "kind": "butterfly", "backend": "online-reorder",
        "fft_core": "register-tile", "stage_partition": [6, 6],
        "prefix_threads": 128, "prefix_ept": 8,
        "suffix_threads": 128, "suffix_ept": 8,
    }
    workload = dict(operator="fft", precision="fp32", logN=12, batch=8)
    proposals = [json.loads(candidate["mapping_json"])
                 for candidate in neighbors(_point(mapping, **workload))]
    lane_two = [item for item in proposals if item.get("prefix_codelet_lanes") == 2]
    assert lane_two
    mutated = lane_two[0]
    assert mutated["prefix_threads"] == 256 and mutated["prefix_ept"] == 4
    projected = predicted_sample(_point(mutated, **workload))
    prefix = projected["execution_groups_json"][0]
    assert prefix["threads"] == 256 and prefix["elements_per_thread"] == 4


def test_register_partition_neighbours_revalidate_lane_geometry():
    mapping = {
        "schema_version": 1, "kind": "butterfly", "backend": "online-reorder",
        "fft_core": "register-tile", "stage_partition": [8, 8],
        "prefix_threads": 256, "prefix_ept": 2, "prefix_codelet_lanes": 8,
        "suffix_threads": 128, "suffix_ept": 8,
    }
    parent = _point(mapping, logN=16)
    original = json.loads(parent["mapping_json"])
    for candidate in neighbors(parent):
        mutated = json.loads(candidate["mapping_json"])
        partition = mutated.get("stage_partition")
        if partition == original["stage_partition"] or len(partition) != 2:
            continue
        radius = 1 << (partition[0] // 2)
        lanes = mutated.get("prefix_codelet_lanes", 1)
        assert lanes > 0 and lanes & (lanes - 1) == 0 and lanes <= radius
        assert mutated["prefix_threads"] % (radius * lanes) == 0
        columns = mutated["prefix_threads"] // (radius * lanes)
        assert mutated["prefix_ept"] == radius // lanes
        if lanes > 1:
            assert lanes * columns <= 32


def test_bounded_screening_reserves_distinct_register_lane_strategies():
    workload = dict(operator="fft", precision="fp32", logN=12, batch=8)
    base = dict(schema_version=1, kind="butterfly", backend="online-reorder",
                fft_core="register-tile", stage_partition=[6, 6],
                prefix_threads=128, prefix_ept=8, suffix_threads=128, suffix_ept=8)
    cooperative = dict(base, prefix_codelet_lanes=2, prefix_threads=256, prefix_ept=4)
    points = [mapping_point(workload, base), mapping_point(workload, cooperative)]
    screened = stratified_candidates(points, budget=2)
    lanes = {json.loads(item["mapping_json"]).get("prefix_codelet_lanes", 1)
             for item in screened}
    assert lanes == {1, 2}


def test_factor_policy_neighbours_preserve_each_physical_stage():
    mapping = {
        "schema_version": 1, "kind": "butterfly", "backend": "factor-streamed",
        "fft_core": "cufftdx-block", "stage_partition": [12],
        "factor_partition": [4, 4, 4], "factor_ept": 4, "factor_columns": 4,
        "data_tiles_per_cta": 1, "prefetch_depth": 0,
    }
    proposals = [json.loads(candidate["mapping_json"]) for candidate in neighbors(_point(mapping))]
    policies = [item.get("factor_io_policies") for item in proposals
                if item.get("factor_io_policies")]
    assert {tuple(value) for value in policies} >= {
        ("static-unrolled", "dynamic", "dynamic"),
        ("dynamic", "static-unrolled", "dynamic"),
        ("dynamic", "dynamic", "static-unrolled"),
    }


def test_parallel_frozen_scores_and_proposals_match_serial(monkeypatch):
    import multiprocessing
    import os
    from evolutionary_search import _inventory_score_keys
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork scoring is unavailable")
    points = [_point({"schema_version": 1, "kind": "butterfly", "backend": "shared-iterative",
                      "fft_core": "scalar", "stage_partition": [6, 6], "tile_threads": 128,
                      "batch_tile_count": i + 1}) for i in range(160)]
    # A closure exercises inherited model state and deterministic ties/errors.
    model = {"weight": 1.25}
    parent_pid = os.getpid()
    def score(point):
        if os.getpid() != parent_pid:
            assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        tile = json.loads(point["mapping_json"])["batch_tile_count"]
        if tile % 13 == 0:
            raise ValueError("unavailable projection")
        return (tile % 7) * model["weight"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-device")
    monkeypatch.setenv("CUBUTTERFLY_SEARCH_SCORE_WORKERS", "1")
    keys = _inventory_score_keys(points, score)
    serial = evolutionary_candidates([], points, set(), score, limit=16)
    monkeypatch.setenv("CUBUTTERFLY_SEARCH_SCORE_WORKERS", "2")
    assert _inventory_score_keys(points, score) == keys
    assert evolutionary_candidates([], points, set(), score, limit=16) == serial
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent-device"
    model["weight"] = 2.5
    changed = _inventory_score_keys(points, score)
    monkeypatch.setenv("CUBUTTERFLY_SEARCH_SCORE_WORKERS", "1")
    assert changed == _inventory_score_keys(points, score)
    assert changed != keys
