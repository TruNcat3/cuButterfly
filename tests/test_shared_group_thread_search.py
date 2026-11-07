"""Shared launch widths follow physical groups through search and models."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evolutionary_search as search
import resident_mapping
import stage_cost_model as costs
import stage_service_model as service
from calibration_space import mapping_point, predicted_sample


def point(operator="fwht", widths=(128, 128, 64), fused=False):
    mapping = dict(schema_version=1, kind="ntt" if operator == "ntt" else "butterfly",
                   backend="shared-iterative", tile_threads=128, threads_per_block=128,
                   stage_partition=[4, 4, 8, 7] if fused else [8, 8, 7])
    if operator == "ntt":
        mapping["execution_group_mappings"] = [dict(core="radix2", threads_per_block=t) for t in widths]
        if fused:
            mapping["boundary_mappings"] = [dict(storage=s) for s in
                                            ("resident-fused", "full-scratch", "full-scratch")]
    else:
        mapping["execution_group_mappings"] = [dict(core="scalar", exchange="shared", ept=0,
            codelet="native", io_policy="dynamic", threads=t) for t in widths]
        if fused:
            mapping["boundaries"] = [dict(residency=s) for s in ("fused", "global-scratch", "global-scratch")]
    return mapping_point(dict(operator=operator, precision="word32" if operator == "ntt" else "fp32",
                              logN=23, batch=2), mapping), mapping


@pytest.mark.parametrize("operator", ["fft", "fwht", "ntt"])
@pytest.mark.parametrize("fused", [False, True])
def test_projection_and_public_neighbors_use_physical_group_widths(operator, fused):
    candidate, mapping = point(operator, fused=fused)
    sample = predicted_sample(candidate)
    assert [g["threads"] for g in sample["execution_groups_json"]] == [128, 128, 64]
    proposals = search.neighbors(candidate)
    thread_proposals = [json.loads(p["mapping_json"]) for p in proposals
                        if json.loads(p["mapping_json"]).get("stage_partition") == mapping["stage_partition"]
                        and json.loads(p["mapping_json"]).get("execution_group_mappings") != mapping["execution_group_mappings"]]
    widths = [resident_mapping.shared_group_threads(p, [8, 8, 7], operator) for p in thread_proposals]
    assert [128, 128, 32] in widths
    assert [128, 64, 64] in widths
    assert [64, 128, 64] in widths
    for p in search._structural_candidates(candidate, mapping, operator):
        assert "execution_group_mappings" not in p


@pytest.mark.parametrize("operator", ["fwht", "ntt"])
def test_thread_widths_separate_model_keys_and_arithmetic(operator):
    candidate, mapping = point(operator)
    sample = predicted_sample(candidate)
    group = sample["execution_groups_json"][2]
    changed = dict(group, threads=128)
    assert costs.primitive_key(sample, group) != costs.primitive_key(sample, changed)
    assert service.curve_key(sample, group) != service.curve_key(sample, changed)
    missing = dict(group)
    missing.pop("threads")
    assert service.curve_key(sample, missing) == service.curve_key(sample, group)
    profile = dict(sm_count=1, capabilities=dict(global_feedback_bytes_per_second=1e30,
        equivalent_butterflies_per_second=1e6, interstage_shared_bytes_per_second=1e30,
        cta_barriers_per_second=1e30))
    terms = costs.stage_terms(sample, profile)
    assert terms[2]["arithmetic_work"]["allocated_lane_utilization"] == 1
    widened = copy.deepcopy(sample)
    widened["execution_groups_json"][2]["threads"] = 128
    assert costs.stage_terms(widened, profile)[2]["compute_ms"] == pytest.approx(2 * terms[2]["compute_ms"])


def test_mismatched_group_count_is_rejected_before_projection():
    candidate, mapping = point(widths=(128, 64))
    with pytest.raises(ValueError, match="one entry per physical execution group"):
        predicted_sample(candidate)
    assert list(search._shared_thread_neighbors(candidate, mapping)) == []


@pytest.mark.parametrize("field", ["core", "exchange", "threads"])
def test_butterfly_partial_descriptor_does_not_inherit_global_defaults(field):
    _, mapping = point()
    mapping["execution_group_mappings"][0].pop(field)
    with pytest.raises(ValueError):
        resident_mapping.shared_group_threads(mapping, [8, 8, 7], "fwht")


def test_butterfly_omitted_ept_uses_parser_zero_default():
    _, mapping = point()
    mapping["execution_group_mappings"][0].pop("ept")
    assert resident_mapping.shared_group_threads(mapping, [8, 8, 7], "fwht") == [128, 128, 64]


def test_ntt_ir_core_alias_is_not_a_public_json_core():
    _, mapping = point("ntt")
    mapping["execution_group_mappings"][0]["core"] = "radix2-shoup"
    with pytest.raises(ValueError, match="core must be radix2"):
        resident_mapping.shared_group_threads(mapping, [8, 8, 7], "ntt")


def test_ntt_core_is_required_by_public_parser():
    _, mapping = point("ntt")
    mapping["execution_group_mappings"][0].pop("core")
    with pytest.raises(ValueError, match="explicit radix2 core"):
        resident_mapping.shared_group_threads(mapping, [8, 8, 7], "ntt")


def test_ntt_partial_descriptor_uses_struct_default_launch_width():
    _, mapping = point("ntt")
    mapping["execution_group_mappings"][2].pop("threads_per_block")
    assert resident_mapping.shared_group_threads(mapping, [8, 8, 7], "ntt") == [128, 128, 256]
    mapping["execution_group_mappings"] = []
    assert resident_mapping.shared_group_threads(mapping, [8, 8, 7], "ntt") == [128, 128, 128]
