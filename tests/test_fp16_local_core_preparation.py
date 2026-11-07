"""CPU-only preparation checks for the paired native-half local core."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cufftdx_low_module import PACKING, render, validate_point
from research_compile_requests import compile_request


def public_point(**changes):
    value = dict(operator="fft", precision="fp16", accumulation="native", logN=12,
                 batch=5, direction="inverse", normalization="none",
                 mapping_json=dict(backend="temporal-tile", fft_core="cufftdx-block",
                                   tile_threads=256, prefix_ept=16, stage_partition=[12]))
    value.update(changes)
    return value


@pytest.mark.parametrize("policy", ["research", "auto", "on-demand"])
@pytest.mark.parametrize("encoded", [False, True])
def test_public_request_preserves_native_storage_arithmetic_and_packing(policy, encoded):
    value = public_point()
    if encoded:
        value["mapping_json"] = json.dumps(value["mapping_json"])
    request = compile_request(value, policy)
    # precision identifies the stored scalar type; accumulation is independent.
    assert request["precision"] == value["precision"] == "fp16"
    assert request["accumulation"] == "native"
    assert request["packing"] == PACKING
    shape = validate_point(request, 80)
    assert (shape["size"], shape["threads"], shape["ept"]) == (4096, 256, 16)
    assert shape["ffts_per_block"] == 2
    assert shape["pairs_per_block"] == 1


def test_fp32_compute_does_not_silently_use_native_half_core():
    with pytest.raises(ValueError, match="accumulation=native"):
        compile_request(public_point(accumulation="fp32"), "research")


def test_normalized_inverse_is_not_silently_replaced_by_final_store_scaling():
    with pytest.raises(ValueError, match="normalized inverse"):
        compile_request(public_point(normalization="inverse"), "research")


@pytest.mark.parametrize("changes,reason", [
    ({"tile_threads": 128}, "integral paired"),
    ({"stage_partition": [6, 6]}, "one local group"),
    ({"stage_overlap": True}, "range overlap"),
])
def test_invalid_geometry_or_split_mapping_is_rejected(changes, reason):
    value = public_point()
    value["mapping_json"].update(changes)
    with pytest.raises(ValueError, match=reason):
        compile_request(value, "research")


def test_precompiled_policy_does_not_claim_a_missing_linked_core():
    with pytest.raises(ValueError, match="standalone module compiler"):
        compile_request(public_point(), "precompiled")


def test_render_exposes_both_directions_and_group_launcher_without_compilation():
    request = compile_request(public_point(), "research")
    source = render(request, 80, "cpu-test-only")
    assert "Precision<__half>()" in source
    assert "FFTsPerBlock<2>()" in source
    assert "low_block_launch<FFT<true>,true>" in source
    assert "low_block_launch<FFT<false>,false>" in source
    assert "cubutterfly_module_launch_group_v1" in source


@pytest.mark.parametrize("changes", [
    {"compute_unit": "radix4"},
    {"shared_layout": "xor"},
    {"local_exchange": "warp-register"},
    {"complex_multiply": "three-mul"},
    {"prefix_codelet": "radix4"},
    {"prefix_twiddle": "table"},
    {"prefix_codelet_lanes": 2},
    {"boundaries": [{"layout": "direct-strided"}]},
    {"local_stage_partitions": [[6, 6]]},
    {"exchange_chunks": [2]},
    {"segment_mappings": [{"core": "scalar"}]},
    {"execution_group_mappings": [{"core": "scalar"}]},
    {"factor_partition": [12]},
    {"factor_overlap": True},
])
def test_public_projection_rejects_unimplemented_axes_instead_of_discarding_them(changes):
    value = public_point()
    value["mapping_json"].update(changes)
    with pytest.raises(ValueError):
        compile_request(value, "research")


def test_non_fft_operator_cannot_project_to_an_fft_module():
    with pytest.raises(ValueError, match="operator=fft"):
        compile_request(public_point(operator="fwht"), "research")


@pytest.mark.parametrize("threads,fpb", [(512, 2), (1024, 4)])
def test_public_wide_native_half_request_reaches_module_preparation(threads, fpb):
    value = public_point()
    value['mapping_json'].update(tile_threads=threads, prefix_ept=8)
    request = compile_request(value, 'research')
    assert request['tile_threads'] == threads
    assert request['prefix_ept'] == 8
    assert request['precision'] == 'fp16' and request['accumulation'] == 'native'
    assert request['packing'] == PACKING
    assert validate_point(request, 80)['ffts_per_block'] == fpb
