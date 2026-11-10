"""The extension must never steal its parent's GPU during CPU scoring."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

STUDY = Path(__file__).resolve().parents[1] / "results/batch_precision_20260920"
spec = importlib.util.spec_from_file_location("batch_precision_queue_test", STUDY / "queue.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)
study = queue.study


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_parent_pending_never_queries_gpu(monkeypatch):
    monkeypatch.setattr(study, "parent_resolved", lambda target: False)
    monkeypatch.setattr(queue.controller, "query_compute_clients", lambda uuid: pytest.fail("must not query/launch GPU"))
    assert "original 74-cell" in queue.prerequisite_clients("a100-40gb", "uuid")[0]


def test_parent_lock_blocks_even_when_journal_is_resolved(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ORIGINAL", tmp_path / "parent")
    monkeypatch.setattr(study, "parent_resolved", lambda target: True)
    lock = study.ORIGINAL / "a100-40gb/resume_when_idle.lock"
    with queue.controller.acquire_lock(lock):
        assert "still owns" in queue.prerequisite_clients("a100-40gb", "uuid")[0]


def test_compilation_must_finish_before_gpu_query(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ORIGINAL", tmp_path / "parent")
    monkeypatch.setattr(queue, "HERE", tmp_path / "new")
    monkeypatch.setattr(study, "parent_resolved", lambda target: True)
    monkeypatch.setattr(queue.controller, "query_compute_clients", lambda uuid: ["foreign-pid"])
    assert "precompilation" in queue.prerequisite_clients("a100-40gb", "uuid")[0]
    write(queue.HERE / "precompile/compilation.json", {"complete": True})
    assert queue.prerequisite_clients("a100-40gb", "uuid") == ["foreign-pid"]


def test_parent_requires_resolved_nonempty_cells(tmp_path):
    p = tmp_path / "a100-40gb/acceptance/acceptance.json"
    write(p, {"status": "complete-measured-matrix", "cells": [{"status": "pending"}]})
    assert not study.parent_resolved("a100-40gb", tmp_path)
    write(p, {"status": "complete-measured-matrix", "cells": []})
    assert not study.parent_resolved("a100-40gb", tmp_path)
    write(p, {"status": "complete-with-unavailable-cells", "cells": [{"status": "complete"}, {"status": "unavailable"}]})
    assert study.parent_resolved("a100-40gb", tmp_path)


@pytest.mark.parametrize("field,value", [("uuid", "wrong-gpu"), ("compile_mode", "auto"), ("repeat", 1), ("binary", "changed")])
def test_parent_identity_rejects_incompatible_reuse(field, value):
    build = {"compile_mode": "research", "binaries": {"bench": "hash"}}
    identity = {"device": {"uuid": study.TARGETS["a100-40gb"]}, "compile_mode": "research",
                "protocol": dict(study.PROTOCOL), "binaries": {"bench": "hash"}}
    if field == "uuid":
        identity["device"]["uuid"] = value
    elif field == "repeat":
        identity["protocol"][field] = value
    elif field == "binary":
        identity["binaries"]["bench"] = value
    else:
        identity[field] = value
    with pytest.raises(ValueError):
        study.validate_parent("a100-40gb", {"identity": identity}, build)


def test_freeze_rejects_mutation(tmp_path):
    p = tmp_path / "frozen.json"
    study.freeze(p, {"budget": 16})
    study.freeze(p, {"budget": 16})
    with pytest.raises(ValueError):
        study.freeze(p, {"budget": 32})
    assert study.read(p) == {"budget": 16}


def test_precision_is_preserved_by_fallback_projection():
    for precision in ("fp16", "bf16"):
        w = dict(operator="fft", precision=precision, accumulation="fp32", logN=12, batch=1)
        seed = study.baseline_seed(w)
        assert seed["semantics"] == w
        assert study.compile_request({**w, "mapping_json": seed["mapping"]}) is None


def test_parent_lock_handoff_is_retryable(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(study, "ORIGINAL", tmp_path)
    monkeypatch.setattr(study, "module", lambda *args: queue.controller)
    with queue.controller.acquire_lock(tmp_path / "a100-40gb/resume_when_idle.lock"):
        assert study.execute("a100-40gb") == 75
    message = capsys.readouterr().err
    assert queue.ExtensionController.recoverable_occupancy(message, 75)
    assert not queue.ExtensionController.recoverable_occupancy("numerical check failed", 75)


@pytest.mark.parametrize("changed", ["parent", "basis", "uuid", "snapshot", "source"])
def test_import_rechecks_parent_and_snapshot_binding(tmp_path, changed):
    source, imported = tmp_path / "parent", tmp_path / "import"
    parent = source / "acceptance/acceptance.json"
    basis = source / "calibration/stage_calibration.json"
    write(parent, {"status": "complete"})
    write(basis, {"status": "complete", "revision": 1})
    evidence = {"gpu_uuid": study.TARGETS["a100-40gb"], "parent_journal_sha256": study.sha(parent),
                "validated_basis": {"sha256": study.sha(basis)}, "files": {}}
    for name in ("base_profile.json", "stage_calibration.json", "schedule_calibration.json", "pipeline_schedule_calibration.json"):
        payload = {"gpu_uuid": study.TARGETS["a100-40gb"]}
        write(source / "files" / name, payload)
        write(imported / name, payload)
        evidence["files"][name] = {"source": str(source / "files" / name),
            "source_sha256": study.sha(source / "files" / name), "snapshot_sha256": study.sha(imported / name)}
    study.validate_import(evidence, "a100-40gb", source, imported)
    if changed == "parent":
        write(parent, {"status": "complete", "new_run": True})
    elif changed == "basis":
        write(basis, {"status": "complete", "revision": 2})
    elif changed == "uuid":
        evidence["gpu_uuid"] = study.TARGETS["a100-80gb"]
    elif changed == "snapshot":
        write(imported / "schedule_calibration.json", {"changed": True})
    else:
        write(source / "files/schedule_calibration.json", {"changed": True})
    with pytest.raises(ValueError):
        study.validate_import(evidence, "a100-40gb", source, imported)


def test_sidecar_references_are_preserved_and_cannot_escape():
    stage = {"acquisition_inventory_ref": {"path": "inventory-hash.json"},
             "coverage": {"composition_requests_ref": {"path": "composition-hash.json"}}}
    assert study.checkpoint_sidecars(stage) == {"inventory-hash.json", "composition-hash.json"}
    with pytest.raises(ValueError):
        study.checkpoint_sidecars({"acquisition_inventory_ref": {"path": "../another-gpu.json"}})
