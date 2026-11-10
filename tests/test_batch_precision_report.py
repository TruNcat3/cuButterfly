import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "results/batch_precision_20260920/report.py"
SPEC = importlib.util.spec_from_file_location("batch_precision_report", MODULE_PATH)
REPORT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REPORT)


def _sample(workload, kernel_ms, runtime="fp-runtime", selected="map-a", error=None):
    row = {
        **workload,
        "kernel_ms": str(kernel_ms),
        "correct": "1",
        "runtime_fingerprint": runtime,
        "mapping_json": json.dumps({"mapping": selected}),
    }
    if error is not None:
        row["max_error"] = str(error)
    return row


def _cell(identifier, workload, runtime="fp-runtime", status="complete", baseline=True, precision=None):
    workload = dict(workload)
    if precision is not None:
        workload["precision"] = precision
    selected_sample = _sample(workload, 1.0, runtime=runtime)
    measurements = [
        {"name": "cuButterfly", "trial": trial, "correct": True,
         "sample": _sample(workload, 1.0 + trial / 10, runtime=runtime, error=0.01)}
        for trial in range(3)
    ]
    if baseline:
        measurements.extend(
            {"name": "cuFFT", "trial": trial, "correct": True,
             "sample": _sample(workload, 2.0 + trial / 10, runtime=runtime)}
            for trial in range(3)
        )
    return {
        "id": identifier,
        "workload": workload,
        "status": status,
        "selected": {"name": "selected", "samples": [selected_sample]},
        "measurements": measurements,
        "baseline_plan": {"available": [{"name": "cuFFT"}], "unavailable": []},
    }


def _write_fixture(tmp_path, *, changed_uuid=False, changed_build=False, changed_protocol=False, partial=False, missing_baseline=False):
    tmp_path = Path(str(tmp_path))
    study = tmp_path / "study"
    original = tmp_path / "original"
    for root in (study, original):
        (root / "a100-40gb/acceptance").mkdir(parents=True)
        (root / "a100-80gb/acceptance").mkdir(parents=True)
    build = {"compile_mode": "research", "binaries": {"cubutterfly_bench": "digest"}, "runtime_fingerprint": "fp-runtime"}
    study_build = dict(build)
    if changed_build:
        study_build["binaries"] = {"cubutterfly_bench": "changed"}
    (original / "build.json").write_text(json.dumps(build))
    (study / "build.json").write_text(json.dumps(study_build))
    protocol = dict(REPORT.REQUIRED_PROTOCOL)
    if changed_protocol:
        protocol["repeat"] = 99
    (study / "manifest.json").write_text(json.dumps({"measurement_protocol": protocol,
        "targets": {"a100-40gb": "GPU-40", "a100-80gb": "GPU-80"}}))
    workload = {"operator": "fft", "precision": "fp32", "logN": 8, "batch": 1,
                "placement": "out-of-place", "normalization": "none"}
    population = {"schema": "batch-precision-study-v1", "cells": [
        {"id": "orig-id", "workload": workload, "suites": ["batch_scaling"], "source": "original", "input_bytes": 1024},
        {"id": "extension-id", "workload": {**workload, "precision": "fp16"}, "suites": ["precision_anchors"], "source": "extension", "input_bytes": 512},
    ], "unsupported": [], "limits": {}}
    (study / "population.json").write_text(json.dumps(population))
    (study / "workloads.json").write_text(json.dumps({"workloads": [workload]}))
    uuids = {"a100-40gb": "GPU-40", "a100-80gb": "GPU-80"}
    for target in REPORT.TARGETS:
        target_uuid = "GPU-changed" if changed_uuid and target == "a100-40gb" else uuids[target]
        identity = {"compile_mode": "research", "binaries": build["binaries"], "protocol": dict(REPORT.REQUIRED_PROTOCOL),
                    "device": {"uuid": uuids[target]}}
        new_identity = dict(identity)
        new_identity["device"] = {"uuid": target_uuid}
        original_cells = [_cell("orig-id", workload, status="partial" if partial else "complete")]
        new_cells = [_cell("extension-id", {**workload, "precision": "fp16"}, baseline=not missing_baseline)]
        (original / target / "acceptance/acceptance.json").write_text(json.dumps({"identity": identity, "cells": original_cells}))
        (study / target / "acceptance/acceptance.json").write_text(json.dumps({"identity": new_identity, "cells": new_cells}))
    return study, original


def test_identity_mismatches_are_rejected(tmpdir):
    for option in ("changed_uuid", "changed_build", "changed_protocol"):
        kwargs = {option: True}
        study, original = _write_fixture(Path(str(tmpdir)) / option, **kwargs)
        with pytest.raises(ValueError):
            REPORT.gather(study, original)


def test_equal_but_wrong_uuid_pair_is_not_a_valid_target(tmp_path):
    study, original = _write_fixture(tmp_path)
    for root in (study, original):
        path = root / "a100-40gb/acceptance/acceptance.json"
        doc = json.loads(path.read_text())
        doc["identity"]["device"]["uuid"] = "GPU-another-card"
        path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="frozen target"):
        REPORT.gather(study, original)
    (study / "a100-40gb/acceptance/acceptance.json").unlink()
    with pytest.raises(ValueError, match="frozen target"):
        REPORT.gather(study, original)


def test_frozen_population_hash_is_checked(tmp_path):
    study, original = _write_fixture(tmp_path)
    manifest = json.loads((study / "manifest.json").read_text())
    manifest["inputs"] = {"population.json": hashlib.sha256((study / "population.json").read_bytes()).hexdigest()}
    (study / "manifest.json").write_text(json.dumps(manifest))
    REPORT.gather(study, original)
    population = json.loads((study / "population.json").read_text())
    population["cells"].pop()
    (study / "population.json").write_text(json.dumps(population))
    with pytest.raises(ValueError, match="frozen study input"):
        REPORT.gather(study, original)


def test_partial_cell_is_not_counted_and_missing_baseline_is_not_a_win(tmpdir):
    study, original = _write_fixture(tmpdir, partial=True, missing_baseline=True)
    document = REPORT.gather(study, original)
    extension = [row for row in document["records"] if row["id"] == "extension-id"]
    assert extension and extension[0]["status"] == "complete"
    assert document["counts"]["partial"] == 2
    assert document["groups"] == []
    assert all(row["status"] == "missing-baseline" for row in document["curves"])
    assert all(float(row["throughput_B_per_kernel_ms_x1000"]) > 0 for row in document["curves"])


def test_incomplete_baseline_trials_are_missing_not_wins(tmpdir):
    study, original = _write_fixture(tmpdir)
    for target in REPORT.TARGETS:
        path = study / target / "acceptance/acceptance.json"
        journal = json.loads(path.read_text())
        journal["cells"][0]["measurements"] = [
            row for row in journal["cells"][0]["measurements"]
            if not (row["name"] == "cuFFT" and row["trial"] == 2)
        ]
        path.write_text(json.dumps(journal))
    document = REPORT.gather(study, original)
    rows = [row for row in document["curves"] if row["id"] == "extension-id"]
    assert rows and all(row["status"] == "missing-baseline" for row in rows)
    assert not any(group["precision"] == "fp16" for group in document["groups"])


def test_precision_groups_and_exact_id_reuse(tmpdir):
    study, original = _write_fixture(tmpdir)
    document = REPORT.gather(study, original)
    records = {(row["target"], row["id"]): row for row in document["records"]}
    assert records[("a100-40gb", "orig-id")]["source"] == "original"
    assert records[("a100-40gb", "extension-id")]["source"] == "extension"
    groups = {(row["precision"], row["baseline"]) for row in document["groups"]}
    assert ("fp32", "cuFFT") in groups
    assert ("fp16", "cuFFT") in groups
    # A different id with the same workload cannot acquire an original result.
    population = json.loads((study / "population.json").read_text())
    population["cells"][0]["id"] = "not-in-original"
    (study / "population.json").write_text(json.dumps(population))
    document = REPORT.gather(study, original)
    row = next(row for row in document["records"] if row["id"] == "not-in-original")
    assert row["status"] == "missing"
