"""Resident switches remain reachable through the public bounded GA path."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evolutionary_search
import resident_mapping
from calibration_space import candidate_id, mapping_point


def shared_point(operator, parts, *, fused=False):
    mapping = dict(
        schema_version=1, kind="ntt" if operator == "ntt" else "butterfly",
        backend="shared-iterative", stage_partition=[4, 4, 8, 8] if fused else [8, 8, 8],
        tile_threads=128, shared_layout="writer-aligned", stage_overlap=False,
        batch_tile_count=3, exchange_chunks=[0, 0, 0],
    )
    if parts is not None:
        mapping["local_stage_partitions"] = copy.deepcopy(parts)
    if fused:
        if operator == "ntt":
            mapping["boundary_mappings"] = [dict(storage=value) for value in
                                            ("resident-fused", "full-scratch", "full-scratch")]
        else:
            mapping["boundaries"] = [dict(residency=value) for value in
                                     ("fused", "global-scratch", "global-scratch")]
    workload = dict(operator=operator, precision="word32" if operator == "ntt" else "fp32",
                    logN=24, batch=2, direction="inverse", normalization="none",
                    input_order="natural", output_order="bit-reversed",
                    element_stride=2, batch_stride=33554432)
    if operator == "ntt":
        workload["modulus"] = 2013265921
    return mapping_point(workload, mapping), mapping


@pytest.mark.parametrize("operator", ["fft", "fwht", "ntt"])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_each_physical_group_toggle_survives_tight_public_budget(monkeypatch, operator, fused, enabled):
    parts = [[2, 2, 2, 2] for _ in range(3)] if enabled else [[], [], []]
    point, mapping = shared_point(operator, parts, fused=fused)
    original = copy.deepcopy(point)
    monkeypatch.setattr(evolutionary_search, "MAX_NEIGHBORS", 3)
    proposals = evolutionary_search.neighbors(point)
    assert len(proposals) == 3
    assert len({candidate_id(proposal) for proposal in proposals}) == 3
    for index, proposal in enumerate(proposals):
        expected = copy.deepcopy(mapping)
        expected["local_stage_partitions"][index] = [] if enabled else [5, 3]
        assert json.loads(proposal["mapping_json"]) == expected
        for field in ("operator", "precision", "direction", "normalization", "input_order",
                      "output_order", "element_stride", "batch_stride", "modulus"):
            assert proposal.get(field) == point.get(field)
        resident_mapping.validate_resident_axes(expected, [8, 8, 8], backend="shared-iterative")
    assert point == original
    assert evolutionary_search.neighbors(point) == proposals


@pytest.mark.parametrize("parts", [None, [], [[], [], []]])
def test_implicit_and_explicit_disabled_axes_enable_only_one_group_first(parts):
    point, mapping = shared_point("fwht", parts)
    proposals = list(evolutionary_search._resident_axis_neighbors(point, mapping))
    assert [proposal["local_stage_partitions"] for proposal in proposals[:3]] == [
        [[5, 3], [], []], [[], [5, 3], []], [[], [], [5, 3]],
    ]
    assert any(proposal["local_stage_partitions"] == [[5, 3]] * 3 for proposal in proposals)


def test_mixed_groups_keep_switches_and_split_merge_move():
    parts = [[2, 2, 2, 2], [], [3, 5]]
    point, mapping = shared_point("fft", parts)
    proposals = list(evolutionary_search._resident_axis_neighbors(point, mapping))
    partitions = [proposal["local_stage_partitions"] for proposal in proposals]
    assert partitions[:3] == [[[], [], [3, 5]],
                              [[2, 2, 2, 2], [5, 3], [3, 5]],
                              [[2, 2, 2, 2], [], []]]
    for entry in ([1, 1, 2, 2, 2], [4, 2, 2], [1, 3, 2, 2]):
        assert [entry, [], [3, 5]] in partitions
    assert [[], [], []] in partitions
    assert len({json.dumps(proposal, sort_keys=True) for proposal in proposals}) == len(proposals)
    for proposal in proposals:
        resident_mapping.validate_resident_axes(proposal, [8, 8, 8], backend="shared-iterative")


def test_many_groups_use_linear_switches_and_skip_unchanged_parent():
    point, mapping = shared_point("fwht", None)
    mapping.update(stage_partition=[1] * 12, exchange_chunks=[0] * 12)
    point["logN"] = 12
    proposals = list(evolutionary_search._resident_axis_neighbors(point, mapping))
    assert len(proposals) == 13
    assert all(sum(bool(entry) for entry in proposal["local_stage_partitions"]) == 1
               for proposal in proposals[:12])
    assert proposals[-1]["local_stage_partitions"] == [[1]] * 12


def test_malformed_physical_axis_is_not_proposed():
    point, mapping = shared_point("fwht", [[], []])
    assert list(evolutionary_search._resident_axis_neighbors(point, mapping)) == []
