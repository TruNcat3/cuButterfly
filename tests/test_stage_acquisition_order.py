"""CPU-only ordering tests for representative-first stage acquisition."""

import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from stage_acquisition_order import order_points, shared_stage_template_keys, shared_template_key


def shared(**overrides):
    mapping = {
        "schema_version": 1,
        "kind": "butterfly",
        "backend": "shared-iterative",
        "compute_unit": "radix2",
        "fft_core": "scalar",
        "local_exchange": "shared",
        "shared_layout": "writer-aligned",
        "tile_threads": 128,
        "stage_partition": [2, 2, 2],
        "boundaries": [
            {"twiddle": "table", "layout": "strided", "residency": "global-scratch"},
            {"twiddle": "table", "layout": "strided", "residency": "global-scratch"},
        ],
    }
    point = {
        "operator": "fft",
        "precision": "fp32",
        "accumulation": "native",
        "complex_multiply": "four-mul",
        "logN": 6,
        "batch": 1,
        "runtime_fingerprint": "build-a",
        "compute_capability": "8.0",
        "mapping_json": json.dumps(mapping),
    }
    point.update(overrides)
    return point


def ntt(**overrides):
    mapping = {
        "schema_version": 1,
        "kind": "ntt",
        "backend": "shared-iterative",
        "compute_unit": "radix2",
        "threads_per_block": 128,
        "dataflow_layout": "hermes-xor",
        "stage_partition": [3, 3],
    }
    point = {
        "operator": "ntt",
        "precision": "word32",
        "logN": 6,
        "batch": 1,
        "runtime_fingerprint": "build-a",
        "compute_capability": "8.0",
        "mapping_json": json.dumps(mapping),
    }
    point.update(overrides)
    return point


def shared_partition(partition, **overrides):
    point = shared(logN=sum(partition), **overrides)
    mapping = json.loads(point["mapping_json"])
    mapping["stage_partition"] = list(partition)
    mapping["boundaries"] = [
        {"twiddle": "table", "layout": "strided", "residency": "global-scratch"}
        for _ in range(max(0, len(partition) - 1))
    ]
    point["mapping_json"] = json.dumps(mapping)
    return point


def test_fused_segments_share_one_physical_template_but_max_work_is_representative():
    fused = shared(batch=2)
    fused_mapping = json.loads(fused["mapping_json"])
    fused_mapping["boundaries"][0]["residency"] = "fused"
    fused["mapping_json"] = json.dumps(fused_mapping)
    same_template_high_load = shared(batch=16)
    same_template_high_load["mapping_json"] = fused["mapping_json"]
    distinct_partition = shared(batch=8)

    assert shared_template_key(fused) == shared_template_key(same_template_high_load)
    ordered = order_points([fused, distinct_partition, same_template_high_load])
    assert ordered[0] is same_template_high_load
    assert {id(item) for item in ordered} == {id(fused), id(distinct_partition), id(same_template_high_load)}
    assert len({shared_template_key(item) for item in ordered[:2]}) == 2


def test_template_required_parameters_are_not_pooled():
    base = shared(batch=1)
    variants = [
        shared(threads=256),
        shared(shared_layout="linear"),
        shared(operator="fwht"),
        shared(precision="fp64"),
        shared(logN=8),
    ]
    for variant in variants:
        mapping = json.loads(variant["mapping_json"])
        if "threads" in variant:
            mapping["tile_threads"] = variant["threads"]
        if "shared_layout" in variant:
            mapping["shared_layout"] = variant["shared_layout"]
        variant["mapping_json"] = json.dumps(mapping)
        assert shared_template_key(base) != shared_template_key(variant)

    # Runtime invocation axes do not change the generated shared template.
    runtime_variant = shared(batch=32, inverse=True, normalization="inverse",
                             placement="in-place", modulus=998244353)
    assert shared_template_key(base) == shared_template_key(runtime_variant)


def test_stage_template_cover_precedes_fully_covered_composition():
    first_source = shared_partition((4, 2, 6), batch=1)
    second_source = shared_partition((6, 2, 4), batch=2)
    composed = shared_partition((4, 2, 2, 4), batch=32)

    first_keys = set(shared_stage_template_keys(first_source))
    second_keys = set(shared_stage_template_keys(second_source))
    composed_keys = set(shared_stage_template_keys(composed))
    assert shared_template_key(first_source) != shared_template_key(second_source)
    assert composed_keys <= first_keys | second_keys

    ordered = order_points([first_source, second_source, composed])
    assert ordered[:2] == [first_source, second_source]
    assert ordered[2] is composed


def test_non_shared_points_are_kept_and_not_inferred_from_shared_templates():
    points = [
        shared(batch=1),
        ntt(batch=64),
        {"backend": "temporal-tile", "logN": 8, "batch": 4, "id": "other-a"},
        {"backend": "temporal-tile", "logN": 8, "batch": 4, "id": "other-b"},
        shared(batch=8),
    ]
    ordered = order_points(points)
    assert len(ordered) == len(points)
    assert sorted(id(item) for item in ordered) == sorted(id(item) for item in points)
    assert all(shared_template_key(item) is not None for item in ordered[:2])
    assert any(item.get("backend") == "temporal-tile" for item in ordered[2:])


def test_shared_default_tiles_follow_backend_resolvers():
    butterfly = shared(logN=6)
    butterfly_mapping = json.loads(butterfly["mapping_json"])
    butterfly_mapping.pop("stage_partition")
    butterfly["mapping_json"] = json.dumps(butterfly_mapping)
    assert len(shared_stage_template_keys(butterfly)) == 1

    ntt_point = ntt(logN=12)
    ntt_mapping = json.loads(ntt_point["mapping_json"])
    ntt_mapping.pop("stage_partition")
    ntt_mapping["flow_tile_log_n"] = 0
    ntt_point["mapping_json"] = json.dumps(ntt_mapping)
    assert len(shared_stage_template_keys(ntt_point)) == 2
