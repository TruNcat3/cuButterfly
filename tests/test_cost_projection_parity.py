"""Deterministic FFT axes must agree before and after plan normalization."""
import json
import os
import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from calibration_space import predicted_sample
from fit_local_cost_model import FEATURE_NAMES, feature_vector


@pytest.mark.parametrize("core,partition,ept,columns,expected_threads", [
    ("cufftdx-block", [9, 11], 8, 4, 256),
    ("cufftdx-block", [8, 10], 32, 4, 32),
    ("register-tile", [6, 6, 6], 8, 8, 64),
])
def test_factor_axes_follow_first_physical_factor(core, partition, ept, columns, expected_threads):
    log_n = sum(partition)
    mapping = dict(backend="factor-streamed", fft_core=core, stage_partition=[log_n],
                   factor_partition=partition, factor_ept=ept, factor_columns=columns,
                   tile_threads=128, local_stages=10, execution_group_mappings=[])
    point = dict(operator="fft", precision="fp32", logN=log_n, batch=4,
                 local_stages=10, mapping_json=json.dumps(mapping))
    projected = predicted_sample(point)
    # Expected public config from normalize_factor_mapping, including an
    # unequal partition whose second group's launch width differs.
    resolved = dict(projected, tile_threads=expected_threads, local_stages=partition[0],
                    group_cores=":".join([core] * len(partition)))
    assert feature_vector(projected) == feature_vector(resolved)
    assert projected["decomposition_count"] == 1
    assert projected["execution_group_count"] == len(partition)
    assert all(g["compiler_local_resources_known"] is False for g in projected["execution_groups_json"])


def test_register_adapter_resolves_mixed_cores_without_compiler_resource_claims():
    mapping = dict(backend="online-reorder", fft_core="register-tile", compute_unit="radix2",
                   local_stages=12, prefix_threads=256, prefix_ept=64, suffix_threads=32,
                   suffix_ept=8, execution_group_mappings=[])
    projected = predicted_sample(dict(operator="fft", precision="fp32", logN=18,
                                       batch=16, mapping_json=json.dumps(mapping)))
    features = dict(zip(FEATURE_NAMES, feature_vector(projected)))
    assert features["prefix_register_tile"] == features["suffix_cufftdx"] == 1
    assert features["compute_unit_radix2"] == 0
    assert features["max_group_live_kib"] == 128
    assert features["compiler_local_known_fraction"] == 0
    assert projected["group_cores"] == "register-tile:cufftdx-block"


@pytest.mark.parametrize("lanes,threads,ept,expected_data_time", [
    (2, 512, 32, 16), (4, 1024, 16, 8),
])
def test_register_cooperative_prefix_projection_matches_column_geometry(
        lanes, threads, ept, expected_data_time):
    mapping = dict(backend="online-reorder", fft_core="register-tile",
                   stage_partition=[12, 6], local_stages=12,
                   prefix_threads=threads, prefix_ept=ept,
                   prefix_codelet_lanes=lanes, suffix_threads=128, suffix_ept=4)
    projected = predicted_sample(dict(operator="fft", precision="fp32", logN=18,
                                       batch=4, mapping_json=json.dumps(mapping)))
    prefix, suffix = projected["execution_groups_json"]
    assert prefix["threads"] == threads
    assert prefix["elements_per_thread"] == ept
    assert prefix["data_time"] == expected_data_time
    assert suffix["threads"] == 128 and suffix["elements_per_thread"] == 4


def test_explicit_group_cores_are_not_replaced_by_default_adapter_cores():
    mapping = dict(backend="online-reorder", fft_core="cufftdx-block", local_stages=8,
                   execution_group_mappings=[dict(core="cufftdx-block"), dict(core="register-tile")])
    projected = predicted_sample(dict(operator="fft", precision="fp32", logN=16,
                                       batch=4, mapping_json=json.dumps(mapping)))
    assert projected["group_cores"] == "cufftdx-block:register-tile"


@pytest.mark.parametrize("operator,core,width", [
    ("fft", "scalar", 8), ("fwht", "cufftdx-block", 4),
    ("subset-zeta", "register-tile", 4), ("structured-2x2", "scalar", 4),
])
@pytest.mark.parametrize("log_n,local,batch", [(12, 8, 32), (7, 5, 3)])
def test_generic_hierarchical_projects_actual_ctas(operator, core, width, log_n, local, batch):
    mapping = dict(backend="hierarchical", fft_core=core, local_stages=local,
                   tile_threads=64, stage_partition=[log_n],
                   execution_group_mappings=[dict(core=core, threads=0, ept=8)])
    projected = predicted_sample(dict(operator=operator, precision="fp32", logN=log_n,
                                       batch=batch, mapping_json=json.dumps(mapping)))
    prefix, *tails = projected["execution_groups_json"]
    assert prefix["grid_ctas"] == batch*(1 << (log_n-local))
    assert prefix["live_shared_bytes"] == (1 << local)*width
    assert prefix["threads"] == 64
    assert prefix["data_time"] == max(1, (1 << (local-1))//64)
    total = batch*(1 << (log_n-1))
    for group in tails:
        assert group["grid_ctas"] == (total+255)//256
        assert group["threads"] == 256
        assert group["live_shared_bytes"] == 0
        assert group["data_time"] == 1
        assert group["data_space"] == min(total, 256)
        assert group["exchange"] == "global"
    assert all(g["core"] == "scalar" and g["elements_per_thread"] == 0
               for g in projected["execution_groups_json"])


@pytest.mark.parametrize("operator,core,width", [("fft", "scalar", 8), ("fwht", "register-tile", 4)])
def test_generic_online_uses_actual_columns_and_flat_candidate_axes(operator, core, width):
    projected = predicted_sample(dict(operator=operator, precision="fp32", logN=12, batch=3,
        backend="online-reorder", fft_core=core, local_stages=7, tile_threads=64, reorder_columns=3))
    prefix, suffix = projected["execution_groups_json"]
    assert prefix["grid_ctas"] == 3*32
    assert prefix["live_shared_bytes"] == 128*width
    assert suffix["grid_ctas"] == 3*43
    assert suffix["live_shared_bytes"] == 32*3*width
    assert suffix["threads"] == 64
    assert suffix["data_space"] == 48
    assert suffix["data_time"] == 1
    assert projected["group_cores"] == "scalar:scalar"


@pytest.mark.skipif(not os.environ.get("CUBUTTERFLY_TEST_PLAN_PROBE"), reason="requires an explicit GPU plan probe")
@pytest.mark.parametrize("operator,backend,core,log_n,local,batch,columns", [
    ("fft", "hierarchical", "scalar", 7, 5, 3, 1),
    ("fwht", "hierarchical", "scalar", 12, 8, 32, 1),
    ("subset-zeta", "online-reorder", "scalar", 10, 5, 3, 8),
    ("structured-2x2", "online-reorder", "scalar", 12, 7, 3, 4),
])
def test_generic_projection_matches_constructed_plan(operator, backend, core, log_n, local, batch, columns):
    mapping = dict(backend=backend, fft_core=core, local_stages=local, tile_threads=64,
                   reorder_columns=columns, compute_unit="radix2")
    point = dict(operator=operator, precision="fp32", logN=log_n, batch=batch,
                 mapping_json=json.dumps(mapping))
    if operator == "structured-2x2":
        point["stage_matrix"] = "1,0.25,-0.5,1"
    result = subprocess.run([os.environ["CUBUTTERFLY_TEST_PLAN_PROBE"], "--point-json", json.dumps(point)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    resolved = json.loads(result.stdout)
    assert resolved["status"] == "resolved"
    projected = predicted_sample({**point, "mapping_json": resolved["mapping_json"]})
    actual = resolved["execution_groups_json"]
    assert len(projected["execution_groups_json"]) == len(actual)
    for estimate, group in zip(projected["execution_groups_json"], actual):
        for field in ("stage_count", "core", "threads", "elements_per_thread", "grid_ctas",
                      "live_shared_bytes", "data_space", "data_time"):
            assert estimate[field] == group[field], (field, estimate, group)


@pytest.mark.skipif(not os.environ.get("CUBUTTERFLY_TEST_PLAN_PROBE"),
                    reason="requires an explicit GPU plan probe")
@pytest.mark.parametrize("layout", ["linear", "xor"])
@pytest.mark.parametrize("lanes", [1, 2, 4])
def test_register_cooperative_projection_matches_plan_probe(lanes, layout):
    radius = 1 << (6 // 2)
    mapping = dict(
        backend="online-reorder", fft_core="register-tile", local_stages=6,
        stage_partition=[6, 6], prefix_threads=radius * lanes * 4,
        prefix_ept=radius // lanes, prefix_codelet="native",
        prefix_shared_layout=layout, suffix_threads=128, suffix_ept=8,
        shared_layout="writer-aligned", cross_twiddle="recurrence",
        direct_boundary="direct-strided", reorder_columns=1,
    )
    mapping["prefix_codelet_lanes"] = lanes
    point = dict(operator="fft", precision="fp32", logN=12, batch=3,
                 mapping_json=json.dumps(mapping))
    result = subprocess.run([
        os.environ["CUBUTTERFLY_TEST_PLAN_PROBE"], "--point-json", json.dumps(point)
    ], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    resolved = json.loads(result.stdout)
    assert resolved["status"] == "resolved"
    projected = predicted_sample({**point, "mapping_json": resolved["mapping_json"]})
    actual = resolved["execution_groups_json"]
    assert len(projected["execution_groups_json"]) == len(actual) == 2
    for estimate, group in zip(projected["execution_groups_json"], actual):
        for field in ("threads", "elements_per_thread", "grid_ctas", "live_shared_bytes", "data_time"):
            assert estimate[field] == group[field], (field, estimate, group)
