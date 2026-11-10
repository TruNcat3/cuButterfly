"""Static shared arithmetic work follows plain and resident kernel loops."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from calibration_space import predicted_sample
from shared_work_geometry import shared_work_geometry
import stage_cost_model as costs


PROFILE = dict(sm_count=1, capabilities=dict(global_feedback_bytes_per_second=1e30,
    equivalent_butterflies_per_second=1e6, interstage_shared_bytes_per_second=1e30,
    cta_barriers_per_second=1e30))


def project(operator="fwht", stages=(8,), threads=128, parts=None):
    mapping = dict(backend="shared-iterative", stage_partition=list(stages),
                   tile_threads=threads, threads_per_block=threads,
                   local_stage_partitions=parts or [])
    return predicted_sample(dict(operator=operator,
        precision="word32" if operator == "ntt" else "fp32",
        logN=sum(stages), batch=1, mapping_json=json.dumps(mapping)))


@pytest.mark.parametrize("threads,ratio", [(128, 1), (1024, 1/8)])
def test_plain_256_work_and_rate_prior(threads, ratio):
    geometry = shared_work_geometry(8, threads)
    assert geometry["allocated_lane_utilization"] == ratio
    assert geometry["allocated_lane_butterfly_slots"] == threads * 8
    assert geometry["source"] == "static-shared-allocated-lane-heuristic"
    assert geometry["resident_cta_limits_modeled"] is False
    assert "not issued arithmetic" in geometry["limitation"]
    assert "overestimate" in geometry["limitation"]
    assert len(geometry["leaves"]) == 8
    assert all(leaf["work_items"] == 128 and leaf["rounds"] == 1 and
               leaf["butterflies_per_work_item"] == 1 for leaf in geometry["leaves"])
    term = costs.stage_terms(project(threads=threads), PROFILE)[0]
    assert term["arithmetic_work"] == geometry
    assert term["compute_ms"] == pytest.approx(1000 * 256 * 8 / 2 / 1e6 / ratio)


def test_16_point_resident_two_stage_leaves():
    geometry = shared_work_geometry(4, 128, [2, 2])
    assert geometry["allocated_lane_utilization"] == 1/32
    assert geometry["data_space"] == 4
    assert geometry["data_time"] == 1
    assert all(leaf["work_items"] == 4 and leaf["rounds"] == 1 and
               leaf["butterflies_per_work_item"] == 4 for leaf in geometry["leaves"])


def test_variable_leaf_width_uses_weighted_work_not_envelope_for_cost():
    geometry = shared_work_geometry(8, 32, [1, 2, 5])
    assert [leaf["work_items"] for leaf in geometry["leaves"]] == [128, 64, 8]
    assert [leaf["rounds"] for leaf in geometry["leaves"]] == [4, 2, 1]
    assert [leaf["butterflies_per_work_item"] for leaf in geometry["leaves"]] == [1, 4, 80]
    assert geometry["data_space"] == 32 and geometry["data_time"] == 4
    assert geometry["allocated_lane_utilization"] == pytest.approx(1024 / (32 * (4 + 2 * 4 + 80)))
    value = project(stages=(8,), threads=32, parts=[[1, 2, 5]])
    group = value["execution_groups_json"][0]
    assert (group["data_space"], group["data_time"]) == (32, 4)
    assert costs.stage_terms(value, PROFILE)[0]["arithmetic_work"] == geometry


@pytest.mark.parametrize("operator", ["fft", "ntt", "fwht", "subset-zeta", "superset-zeta", "structured-2x2"])
@pytest.mark.parametrize("threads", [128, 1024])
def test_six_operator_mixed_groups_project_the_same_arithmetic(operator, threads):
    value = project(operator, (8, 8, 4), threads, [[4, 4], [], [2, 2]])
    groups = value["execution_groups_json"]
    assert [(g["data_space"], g["data_time"]) for g in groups] == [(16, 1), (128, 1), (4, 1)]
    assert [g["codelet"] for g in groups] == ["native-register-subgraph", "native", "native-register-subgraph"]
    terms = costs.stage_terms(value, PROFILE)
    assert [term["arithmetic_work"]["allocated_lane_utilization"] for term in terms] == [16/threads, 128/threads, 4/threads]
    assert len({term["key"] for term in terms}) == 3


def test_measured_curve_body_is_not_scaled_by_allocated_lane_heuristic(monkeypatch):
    value = project(stages=(4,), threads=128, parts=[[2, 2]])
    term = costs.stage_terms(value, PROFILE)[0]
    assert term["arithmetic_work"]["allocated_lane_utilization"] == 1/32
    assert costs._minimal_terms(value, {})[0]["arithmetic_work"] == term["arithmetic_work"]
    measured = dict(covered=True, kernel_ms=.007, source="measured-stage-curve", curve_key="resident")
    resources = dict(resource_feasibility="unknown", unknown_resources=[])
    monkeypatch.setattr(costs, "_measured_components", lambda sample, model: [(term, measured, resources, False)])
    fitted = dict(profile=PROFILE, parameters={})
    prediction = costs._predict_measured(value, fitted, explain=True)
    assert prediction["kernel_ms"] == .007
    assert prediction["stages"][0]["body_ms"] == .007
    assert prediction["stages"][0]["arithmetic_work"] == term["arithmetic_work"]
    assert prediction["whole_plan_fallback_used"] is False


def test_nonshared_rate_prior_is_unchanged():
    value = project(stages=(8,), threads=1024)
    value["backend"] = "online-reorder"
    value["mapping_json"] = {"backend": "online-reorder"}
    term = costs.stage_terms(value, PROFILE)[0]
    assert term["arithmetic_work"] is None
    assert term["compute_ms"] == pytest.approx(1000 * 256 * 8 / 2 / 1e6)


@pytest.mark.parametrize('tiles,expected_space,expected_time', [(1, 8, 1), (4, 32, 1), (32, 128, 2)])
def test_independent_tile_columns_share_cta_lanes(tiles, expected_space, expected_time):
    geometry = shared_work_geometry(4, 128, data_tiles_per_cta=tiles)
    assert geometry['useful_butterflies'] == tiles * 4 * 8
    assert geometry['data_space'] == expected_space
    assert geometry['data_time'] == expected_time
    assert geometry['data_tiles_per_cta'] == tiles
    assert all(leaf['work_items'] == tiles * 8 for leaf in geometry['leaves'])


def test_shared_multitile_projection_and_cost_consumption():
    mapping = dict(backend='shared-iterative', stage_partition=[4, 4], tile_threads=128,
                   data_tiles_per_cta=4, batch_tile_count=2)
    sample = predicted_sample(dict(operator='fft', precision='bf16', logN=8, batch=8,
                                   mapping_json=json.dumps(mapping)))
    groups = sample['execution_groups_json']
    for group in groups:
        assert group['data_tiles_per_cta'] == 4
        assert group['grid_ctas'] == 16  # (8 / 2 batches) * (16 / 4 columns)
        assert group['live_shared_bytes'] == 2 * 16 * 4 * 4
        assert group['data_space'] == 32
        assert group['data_time'] == 1
    expected = shared_work_geometry(4, 128, data_tiles_per_cta=4)
    for terms in [costs.stage_terms(sample, PROFILE), costs._minimal_terms(sample, {})]:
        assert all(term['arithmetic_work'] == expected for term in terms)
    # Explicit physical group count takes precedence over mapping metadata.
    groups[0]['data_tiles_per_cta'] = 2
    assert costs.stage_terms(sample, PROFILE)[0]['arithmetic_work']['data_tiles_per_cta'] == 2


@pytest.mark.parametrize('tiles', [0, -1, 3, 1.5])
def test_invalid_tile_column_geometry_rejected(tiles):
    with pytest.raises(ValueError):
        shared_work_geometry(4, 128, data_tiles_per_cta=tiles)


def test_multi_tile_resident_partition_is_not_silently_projected():
    with pytest.raises(ValueError, match='resident local partitions'):
        shared_work_geometry(4, 128, [2, 2], 2)
