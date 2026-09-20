"""CPU-only contract tests for batch-ring service descriptor resolution."""

import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from stage_composition import resolve_batch_services


def _group(index, first_stage, core, *, grid=32, resources=None):
    values = {
        "index": index,
        "first_stage": first_stage,
        "stage_count": 4,
        "core": core,
        "threads": 128,
        "data_space": 16,
        "data_time": 2,
        "live_shared_bytes": 2048,
        "dynamic_shared_bytes": 0,
        "compiler_registers_per_thread": 32,
        "compiler_local_bytes_per_thread": 0,
        "compiler_resources_known": True,
        "compiler_local_resources_known": True,
        "grid_ctas": grid,
        "batch_space": 8,
    }
    if resources:
        values.update(resources)
    return values


def _source(*, backend="shared-iterative", batch=31, tile=8, core="scalar"):
    mapping = {
        "backend": backend,
        "fft_core": core,
        "stage_partition": [4, 4],
        "stage_overlap": True,
        "batch_tile_count": tile,
        "shared_layout": "writer-aligned",
    }
    sample = {
        "operator": "fft",
        "precision": "fp32",
        "logN": 8,
        "batch": batch,
        "batch_stride": 256,
        "element_stride": 1,
        "backend": backend,
        "fft_core": core,
        "stage_overlap": True,
        "batch_tile_count": tile,
        "runtime_fingerprint": "build-a",
        "mapping_json": json.dumps(mapping, sort_keys=True, separators=(",", ":")),
    }
    cores = [core, core]
    if backend == "online-reorder":
        cores = ["register-tile", "cufftdx-block"]
    groups = [_group(index, index * 4, group_core, grid=tile)
              for index, group_core in enumerate(cores)]
    return {"status": "resolved", "runtime_fingerprint": "build-a",
            "sample": sample, "groups": groups}


def test_serial_point_is_not_applicable_without_invoking_describe():
    point = _source()["sample"]
    point["stage_overlap"] = False
    mapping = json.loads(point["mapping_json"])
    mapping["stage_overlap"] = False
    point["mapping_json"] = json.dumps(mapping)
    calls = []

    result = resolve_batch_services(point, _source(), lambda value: calls.append(value))

    assert result["status"] == "not-applicable"
    assert result["tail"] is None
    assert result["tile_count"] == 1
    assert calls == []


def test_resolves_tile_and_remainder_from_actual_serial_describes():
    source = _source(batch=31, tile=8)
    calls = []

    def describe(point):
        calls.append(point)
        batch = int(point["batch"])
        mapping = json.loads(point["mapping_json"])
        assert point["stage_overlap"] is False
        assert point["batch_tile_count"] == 1
        assert mapping["stage_overlap"] is False
        assert mapping["batch_tile_count"] == 1
        assert point["batch_stride"] == source["sample"]["batch_stride"]
        groups = [_group(0, 0, "scalar", grid=batch),
                  _group(1, 4, "scalar", grid=batch)]
        # The descriptor must come from the resolver, including physical
        # resource/data axes.  Only the load-dependent grid is changed.
        return {"status": "resolved", "runtime_fingerprint": "build-a",
                "sample": {**point, "runtime_fingerprint": "build-a"},
                "groups": groups}

    result = resolve_batch_services(source["sample"], source, describe)

    assert result["status"] == "resolved"
    assert result["tile_batch"] == 8
    assert result["tail_batch"] == 7
    assert result["tile_count"] == 4
    assert [call["batch"] for call in calls] == [8, 7]
    assert result["full"]["sample"]["batch"] == 8
    assert result["tail"]["sample"]["batch"] == 7
    assert result["full"]["groups"][0]["data_space"] == 16
    assert result["tail"]["groups"][0]["compiler_registers_per_thread"] == 32


def test_divisible_batch_has_no_tail_describe():
    source = _source(batch=16, tile=8)
    batches = []

    def describe(point):
        batches.append(point["batch"])
        return {"status": "resolved", "sample": {**point, "runtime_fingerprint": "build-a"},
                "runtime_fingerprint": "build-a",
                "groups": [_group(0, 0, "scalar", grid=8),
                            _group(1, 4, "scalar", grid=8)]}

    result = resolve_batch_services(source["sample"], source, describe)

    assert result["status"] == "resolved"
    assert result["tail_batch"] == 0
    assert result["tail"] is None
    assert batches == [8]


def test_factor_overlap_is_explicitly_unsupported_without_describe():
    source = _source(backend="factor-streamed")
    point = dict(source["sample"], factor_overlap=True)
    mapping = json.loads(point["mapping_json"])
    mapping["factor_overlap"] = True
    point["mapping_json"] = json.dumps(mapping)
    calls = []

    result = resolve_batch_services(point, source, lambda value: calls.append(value))

    assert result["status"] == "unsupported"
    assert "factor" in result["reason"]
    assert calls == []


def test_online_register_tile_batch_ring_is_supported():
    source = _source(backend="online-reorder", core="register-tile", batch=9, tile=4)

    def describe(point):
        batch = int(point["batch"])
        return {"status": "resolved", "runtime_fingerprint": "build-a",
                "sample": {**point, "runtime_fingerprint": "build-a"},
                "groups": [_group(0, 0, "register-tile", grid=batch),
                            _group(1, 4, "cufftdx-block", grid=batch)]}

    result = resolve_batch_services(source["sample"], source, describe)

    assert result["status"] == "resolved"
    assert result["tile_batch"] == 4 and result["tail_batch"] == 1


def test_unknown_source_resources_do_not_block_actual_serial_resources():
    source = _source(batch=9, tile=4)
    for group in source["groups"]:
        group.pop("dynamic_shared_bytes")
        group["compiler_local_resources_known"] = False
        group.pop("compiler_local_bytes_per_thread")

    def describe(point):
        batch = int(point["batch"])
        groups = [_group(0, 0, "scalar", grid=batch),
                  _group(1, 4, "scalar", grid=batch)]
        return {"status": "resolved", "sample": {**point, "runtime_fingerprint": "build-a"},
                "runtime_fingerprint": "build-a", "groups": groups}

    result = resolve_batch_services(source["sample"], source, describe)

    assert result["status"] == "resolved"
    assert result["full"]["groups"][0]["compiler_local_resources_known"] is True


def test_resource_or_runtime_identity_mismatch_is_unavailable():
    source = _source(batch=9, tile=4)

    def describe(point):
        batch = int(point["batch"])
        changed = {"compiler_registers_per_thread": 48} if batch == 1 else None
        return {"status": "resolved", "runtime_fingerprint": "build-b" if batch == 4 else "build-a",
                "sample": {**point, "runtime_fingerprint": "build-b" if batch == 4 else "build-a"},
                "groups": [_group(0, 0, "scalar", grid=batch, resources=changed),
                            _group(1, 4, "scalar", grid=batch, resources=changed)]}

    result = resolve_batch_services(source["sample"], source, describe)

    assert result["status"] == "unavailable"
    assert result["reason"] == "runtime-fingerprint-mismatch"


def test_integer_scalar_keeps_large_ntt_modulus_exact():
    import stage_composition

    modulus = 576460756061519873
    assert stage_composition._scalar(modulus) == modulus
    assert stage_composition._scalar(modulus) != stage_composition._scalar(modulus - 1)
    assert stage_composition._scalar(str(modulus)) == modulus
    assert stage_composition._scalar(modulus) != stage_composition._scalar(str(modulus - 1))
    assert stage_composition._scalar(str(modulus)) != stage_composition._scalar(modulus - 1)
