"""New executable choices must survive search and retain stage-local costs."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibration_space
import evolutionary_search
import resident_mapping
import stage_cost_model
import stage_service_model
from research_compile_requests import compile_request
from compile_module import render_register_fft, render_shared


def shared_point(fused=False):
    mapping = dict(backend="shared-iterative", stage_partition=[3,3,4] if fused else [6,4],
        tile_threads=128, shared_layout="writer-aligned",
        local_stage_partitions=[[2,4],[1,3]], exchange_chunks=[0,0])
    if fused:
        mapping["boundaries"] = [dict(residency="fused"), dict(residency="global-scratch")]
    return calibration_space.mapping_point(dict(operator="fwht", precision="fp32", logN=10, batch=3), mapping)


def test_fused_groups_compile_and_mutate_at_physical_indices():
    point = shared_point(True)
    request = compile_request(point, "auto")
    assert request["stage_partition"] == [6,4]
    assert request["local_stage_partitions"] == [[2,4],[1,3]]
    assert "shared_resident_kernel<Operator,10,0,6,1,0,2,4>" in render_shared(request,80,"test")
    mapping = json.loads(point["mapping_json"])
    neighbors = list(evolutionary_search._resident_axis_neighbors(point,mapping))
    assert neighbors
    for neighbor in neighbors:
        resident_mapping.validate_resident_axes(neighbor,[6,4],backend="shared-iterative",operator="fwht")
    assert any(n["local_stage_partitions"] not in ([[2,4],[1,3]],[[],[]]) for n in neighbors)
    physical = calibration_space.predicted_sample(point)["execution_groups_json"]
    assert [group["stage_count"] for group in physical] == [6,4]
    assert [group["local_stage_partition"] for group in physical] == [[2,4],[1,3]]


def test_compile_policy_preserves_the_historical_auto_path():
    point = shared_point()
    assert compile_request(point,"auto") is not None
    with pytest.raises(ValueError):
        compile_request(point,"precompiled")
    mapping = json.loads(point["mapping_json"])
    mapping.update(local_stage_partitions=[],exchange_chunks=[])
    point["mapping_json"] = mapping
    assert compile_request(point,"auto") is None
    assert compile_request(point,"research") is not None


def register_point():
    return dict(operator="fft",precision="fp32",logN=22,batch=1,backend="online-reorder",fft_core="register-tile",
        local_stages=11,stage_partition=[11,11],prefix_threads=256,prefix_ept=32,prefix_codelet_lanes=2,
        prefix_codelet="cufftdx-thread",prefix_shared_layout="xor",suffix_threads=512,suffix_ept=16,
        local_stage_partitions=[[6,5],[]],exchange_chunks=[0,8])


def test_rectangular_prefix_and_chunk_render_separate_lowerings():
    point = register_point()
    source = render_register_fft(point,80,"test")
    assert "rectangular_prefix<6,5,32,4" in source
    assert "grouped_suffix<Suffix<Inverse>,4,Complex,8,11>" in source
    assert "launch_grouped_suffix<11,16,4,Inverse,Complex,8,11>" in source
    with pytest.raises(ValueError):
        render_register_fft(dict(point,local_stage_partitions=[[6,5],[11]]),80,"test")
    with pytest.raises(ValueError):
        render_register_fft(dict(point,exchange_chunks=[0,3]),80,"test")
    projected = calibration_space.predicted_sample(dict(point,mapping_json=json.dumps(point)))
    prefix, suffix = projected["execution_groups_json"]
    assert (prefix["elements_per_thread"],prefix["live_shared_bytes"],prefix["grid_ctas"]) == (32,65536,512)
    assert prefix["local_stage_partition"] == [6,5] and suffix["exchange_chunk"] == 8


@pytest.mark.parametrize("key_function",[stage_cost_model.primitive_key,stage_service_model.curve_key])
def test_service_identity_is_group_local(key_function):
    mapping = register_point()
    sample = dict(mapping,mapping_json=mapping)
    prefix = dict(index=0,first_stage=0,stage_count=11,core="register-tile",threads=256,elements_per_thread=32)
    suffix = dict(index=1,first_stage=11,stage_count=11,core="cufftdx-block",threads=512,elements_per_thread=16)
    opposite = dict(sample,mapping_json=dict(mapping,local_stage_partitions=[[5,6],[]],prefix_codelet_lanes=1))
    assert key_function(sample,prefix) != key_function(opposite,prefix)
    assert key_function(sample,suffix) == key_function(opposite,suffix)
    whole = dict(sample,mapping_json=dict(mapping,exchange_chunks=[]))
    assert key_function(sample,prefix) == key_function(whole,prefix)
    assert key_function(sample,suffix) != key_function(whole,suffix)


def test_shared_prior_reduces_exchange_without_deleting_arithmetic():
    point = shared_point()
    projected = calibration_space.predicted_sample(point)
    groups = projected["execution_groups_json"]
    groups = json.loads(groups) if isinstance(groups,str) else groups
    assert groups[0]["local_stage_partition"] == [2,4]
    assert groups[0]["codelet"] == "native-register-subgraph"
    profile = dict(sm_count=10,capabilities=dict(global_feedback_bytes_per_second=1e12,
        equivalent_butterflies_per_second=1e11,interstage_shared_bytes_per_second=1e13,cta_barriers_per_second=1e9))
    fused = stage_cost_model.stage_terms(projected,profile)
    old = copy.deepcopy(projected)
    old_groups = copy.deepcopy(groups)
    for group in old_groups:
        group.update(local_stage_partition=[],codelet="native")
    old["execution_groups_json"] = old_groups
    unfused = stage_cost_model.stage_terms(old,profile)
    assert fused[0]["points"] == unfused[0]["points"]
    assert fused[0]["stage_count"] == unfused[0]["stage_count"]
    assert fused[0]["compute_ms"] <= unfused[0]["compute_ms"]
    assert fused[0]["key"] != unfused[0]["key"]
