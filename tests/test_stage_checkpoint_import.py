"""CPU-only tests for importing compatible stage calibration journals."""

import copy
import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage_checkpoint_import as importer
import stage_service_calibration as calibration


def _fixture(tmp_path, *, batch=4):
    binary = tmp_path / "stage-probe"
    binary.write_bytes(b"stage-probe-binary")
    profile = {
        "device": "import-test-gpu",
        "compute_capability": "8.0",
        "global_memory_bytes": 1 << 40,
        "hardware": {
            "device": "import-test-gpu",
            "compute_capability": "8.0",
            "global_memory_bytes": 1 << 40,
            "free_memory_bytes": 1 << 40,
            "sm_count": 2,
            "max_blocks_per_sm": 4,
            "max_threads_per_sm": 1024,
            "registers_per_sm": 65536,
            "shared_bytes_per_sm": 49152,
        },
    }
    protocol = {
        "warmup": 1,
        "repeat": 2,
        "trials": 3,
        "max_points": 33,
        "mode": "full",
        "driver_version": calibration.VERSION,
    }
    identity = calibration._cache_identity(binary, profile, protocol, "ignored")
    point = {
        "operator": "fft",
        "precision": "fp32",
        "logN": 8,
        "batch": batch,
        "batch_stride": 256,
        "mapping_json": json.dumps({"backend": "fake", "fft_core": "scalar"}),
    }
    group = {
        "index": 0,
        "first_stage": 0,
        "stage_count": 1,
        "threads": 64,
        "units_per_cta": 1,
        "exchange": 0,
        "live_shared_bytes": 1024,
        "compiler_registers_per_thread": 16,
        "compiler_resources_known": True,
        "independent": True,
        "work_blocks": batch,
        "grid_ctas": batch,
        "trial_kernel_ms": [float(batch), float(batch) + 0.01, float(batch) - 0.01],
    }
    raw_description = {
        "schema": calibration.PROBE_SCHEMA,
        "status": "resolved",
        "point": point,
        "sample": point,
        "groups": [group],
        "execution_boundaries": [],
        "workspace_bytes": 4096,
        "runtime_fingerprint": "import-test-runtime",
        "hardware": profile["hardware"],
        "raw": {"status": "resolved", "groups": [group]},
    }
    static_key = "static-import-point"
    candidate = calibration._candidate_from_description(static_key, raw_description, profile)
    return binary, profile, protocol, identity, static_key, point, group, raw_description, candidate


def _record(point, group, candidate, *, role="train", trial_tag=None):
    row = {
        "schema": calibration.PROBE_SCHEMA,
        "status": "measured",
        "correct": True,
        "sample": copy.deepcopy(point),
        "groups": [copy.deepcopy(group)],
        "pairs": [],
        "plan_trial_kernel_ms": [],
        "candidate_key": candidate["key"],
        "batch": int(point["batch"]),
        "role": role,
        "protocol": {"warmup": 1, "repeat": 2, "trials": 3},
    }
    if trial_tag is not None:
        row["raw"] = {"trial_tag": trial_tag}
    return row


def _document(identity, protocol, static_key, description, candidate, records,
              *, composition_requests=None):
    return {
        "schema": calibration.SCHEMA,
        "version": calibration.VERSION,
        "status": "complete",
        "calibration_status": "complete",
        "identity": copy.deepcopy(identity),
        "protocol": copy.deepcopy(protocol),
        "records": copy.deepcopy(records),
        "raw_records": copy.deepcopy(records),
        "descriptions": {static_key: copy.deepcopy(description)},
        "candidates": [copy.deepcopy(candidate)],
        "composition_requests": copy.deepcopy(composition_requests or []),
        "coverage": {"coverage_complete": True, "validation_complete": True},
        "model": {"training_groups": 1},
    }


def _write_checkpoint(path, document):
    calibration._checkpoint_write(path, document)


def test_compatible_merge_preserves_identity_timing_and_is_not_complete(tmpdir):
    tmp_path = pathlib.Path(str(tmpdir))
    binary, profile, protocol, identity, static_key, point, group, description, candidate = _fixture(tmp_path)
    record = _record(point, group, candidate)
    source = tmp_path / "source.json"
    target = tmp_path / "target.json"
    _write_checkpoint(source, _document(identity, protocol, static_key, description, candidate, [record]))

    merged = importer.merge_stage_checkpoints(
        target, [source], binary=binary, profile=profile, protocol=protocol,
    )

    assert merged["identity"] == identity
    assert merged["records"][0]["groups"][0]["trial_kernel_ms"] == group["trial_kernel_ms"]
    assert static_key in merged["descriptions"]
    assert merged["candidates"]
    assert merged["status"] == "running"
    assert merged["calibration_status"] == "incomplete-import"
    assert merged["coverage"]["coverage_complete"] is False
    restored = calibration._load_checkpoint(target)
    assert restored["records"] == merged["records"]


@pytest.mark.parametrize("change", ("binary", "hardware", "timing"))
def test_identity_mismatch_is_rejected(tmpdir, change):
    tmp_path = pathlib.Path(str(tmpdir))
    binary, profile, protocol, identity, static_key, point, group, description, candidate = _fixture(tmp_path)
    source = tmp_path / "source.json"
    _write_checkpoint(source, _document(identity, protocol, static_key, description, candidate, []))
    target = tmp_path / "target.json"
    bad_binary = binary
    bad_profile = copy.deepcopy(profile)
    bad_protocol = copy.deepcopy(protocol)
    if change == "binary":
        bad_binary = tmp_path / "other-stage-probe"
        bad_binary.write_bytes(b"different-stage-probe")
    elif change == "hardware":
        bad_profile["hardware"]["gpu_uuid"] = "different-uuid"
    else:
        bad_protocol["repeat"] = 99

    with pytest.raises(ValueError, match="identity/protocol mismatch"):
        importer.merge_stage_checkpoints(
            target, [source], binary=bad_binary, profile=bad_profile,
            protocol=bad_protocol,
        )


def test_identical_rows_are_deduplicated_but_distinct_trials_are_retained(tmpdir):
    tmp_path = pathlib.Path(str(tmpdir))
    binary, profile, protocol, identity, static_key, point, group, description, candidate = _fixture(tmp_path)
    first = _record(point, group, candidate, trial_tag="trial-a")
    duplicate = copy.deepcopy(first)
    duplicate["candidate_key"] = "old-curve-key"
    different = _record(point, dict(group, trial_kernel_ms=[4.5, 4.6, 4.7]), candidate,
                        trial_tag="trial-b")
    target = _document(identity, protocol, static_key, description, candidate, [first])
    source = _document(identity, protocol, static_key, description, candidate, [duplicate, different])

    merged = importer.merge_checkpoint_documents(
        target, [source], expected_identity=identity, profile=profile, protocol=protocol,
    )

    assert len(merged["records"]) == 2
    timings = [row["groups"][0]["trial_kernel_ms"] for row in merged["records"]]
    assert group["trial_kernel_ms"] in timings
    assert [4.5, 4.6, 4.7] in timings
    assert len(merged["raw_records"]) == 2


def test_larger_extension_description_and_candidate_are_retained(tmpdir):
    tmp_path = pathlib.Path(str(tmpdir))
    binary, profile, protocol, identity, static_key, point4, group4, description4, candidate4 = _fixture(
        tmp_path, batch=4)
    point8 = dict(point4, batch=8)
    group8 = dict(group4, work_blocks=8, grid_ctas=8,
                  trial_kernel_ms=[8.0, 8.01, 7.99])
    description8 = dict(description4, point=point8, sample=point8, groups=[group8])
    candidate8 = calibration._candidate_from_description(static_key, description8, profile)
    record4 = _record(point4, group4, candidate4)
    record8 = _record(point8, group8, candidate8)

    merged = importer.merge_checkpoint_documents(
        _document(identity, protocol, static_key, description4, candidate4, [record4]),
        [_document(identity, protocol, static_key, description8, candidate8, [record8])],
        expected_identity=identity, profile=profile, protocol=protocol,
    )

    assert merged["descriptions"][static_key]["point"]["batch"] == 8
    assert max(int(item["requested_batch"]) for item in merged["candidates"]) == 8
    assert {int(item["batch"]) for item in merged["records"]} == {4, 8}
    assert merged["records"][0]["groups"][0]["trial_kernel_ms"] == group4["trial_kernel_ms"]
    assert merged["records"][1]["groups"][0]["trial_kernel_ms"] == group8["trial_kernel_ms"]


def test_composition_sidecar_is_loaded_and_rewritten(tmpdir):
    tmp_path = pathlib.Path(str(tmpdir))
    binary, profile, protocol, identity, static_key, point, group, description, candidate = _fixture(tmp_path)
    request = {"point": dict(point, stage_overlap=True), "reason": "pending", "tested": False}
    source = tmp_path / "source.json"
    _write_checkpoint(source, _document(identity, protocol, static_key, description, candidate, [],
                                        composition_requests=[request]))

    target = tmp_path / "target.json"
    importer.merge_stage_checkpoints(target, [source], binary=binary, profile=profile,
                                     protocol=protocol)
    loaded = calibration._load_checkpoint(target)

    assert loaded["composition_requests"] == [request]
    assert loaded["composition_request_count"] == 1
    assert "composition_requests" not in json.loads(target.read_text())


def test_acquisition_inventory_hashes_are_validated_merged_and_rewritten(tmpdir):
    tmp_path = pathlib.Path(str(tmpdir))
    binary, profile, protocol, identity, static_key, point, group, description, candidate = _fixture(tmp_path)
    source = tmp_path / "source.json"
    values = sorted({"a" * 64, "b" * 64})
    digest = calibration._points_hash(values)
    inventory = source.with_name(f"{source.stem}.inventory-{digest}.json")
    calibration._atomic_write(inventory, values)
    document = _document(identity, protocol, static_key, description, candidate, [])
    document["acquisition_inventory_ref"] = {
        "path": inventory.name, "sha256": digest, "count": len(values),
    }
    _write_checkpoint(source, document)

    target = tmp_path / "target.json"
    merged = importer.merge_stage_checkpoints(
        target, [source], binary=binary, profile=profile, protocol=protocol,
    )

    reference = merged["acquisition_inventory_ref"]
    rewritten = target.parent / reference["path"]
    assert json.loads(rewritten.read_text()) == values
    assert reference["sha256"] == digest
    loaded = calibration._load_checkpoint(target)
    assert loaded["acquisition_inventory_ref"] == reference
    assert loaded["coverage"]["pending_description_count"] == len(values)
