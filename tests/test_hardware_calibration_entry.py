import json
import hashlib
import pathlib
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import calibrate_hardware as entry


def _workload_file(path):
    path.write_text(json.dumps({
        "schema": "cubutterfly-install-search-v1",
        "scope": "entry-test",
        "workloads": [
            {"operator": "fft", "precision": "fp32", "logN": 12, "batch": 4},
            {"operator": "fwht", "precision": "fp32", "logN": 12, "batch": 4},
        ],
    }))


def test_resolve_workloads_rejects_unknown_schema(tmpdir):
    tmp_path = pathlib.Path(str(tmpdir))
    path = tmp_path / "workloads.json"
    path.write_text(json.dumps({"schema": "wrong", "workloads": [{}]}))
    with pytest.raises(ValueError, match="unsupported workload configuration schema"):
        entry.load_workload_summary(path)


def test_dry_run_writes_a_complete_device_and_workload_plan(tmpdir, monkeypatch):
    tmp_path = pathlib.Path(str(tmpdir))
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    workloads = tmp_path / "workloads.json"
    _workload_file(workloads)
    identity = {
        "device": "Test GPU",
        "compute_capability": "8.0",
        "global_memory_bytes": 40 << 30,
        "source": "test",
    }
    monkeypatch.setattr(entry, "detect_identity", lambda path, required=False: identity)

    assert entry.main([
        "--build-dir", str(build_dir),
        "--profile-root", str(tmp_path / "profiles"),
        "--search-workloads", str(workloads),
        "--mode", "dry-run",
        "--search-budget", "0",
    ]) == 0

    manifest_path = tmp_path / "profiles" / "Test-GPU-sm80-42949672960B" / "migration_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "planned"
    assert manifest["device"] == identity
    assert manifest["workloads"]["workload_count"] == 2
    assert manifest["workloads"]["operators"] == ["fft", "fwht"]
    assert manifest["protocol"]["search_budget"] == 0
    assert manifest["protocol"]["seed_budget"] == 8
    assert manifest["calibration_command"][1].endswith("calibrate_local_hardware.py")


def test_execute_forwards_one_command_and_records_completion(tmpdir, monkeypatch):
    tmp_path = pathlib.Path(str(tmpdir))
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    workloads = tmp_path / "workloads.json"
    _workload_file(workloads)
    identity = {
        "device": "Test GPU",
        "compute_capability": "8.0",
        "global_memory_bytes": 40 << 30,
        "source": "test",
    }
    calls = []
    monkeypatch.setattr(entry, "detect_identity", lambda path, required=False: identity)

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(entry.subprocess, "run", fake_run)
    output = tmp_path / "output"
    assert entry.main([
        "--build-dir", str(build_dir),
        "--profile-dir", str(tmp_path / "profile"),
        "--output-dir", str(output),
        "--workloads", str(workloads),
        "--execute",
    ]) == 1
    assert len(calls) == 1
    assert calls[0][0][1].endswith("calibrate_local_hardware.py")
    manifest = json.loads((output / "migration_manifest.json").read_text())
    assert manifest["status"] != "complete"
    assert manifest["execution"]["status"] == "missing-calibration-manifest"


def test_execute_returns_one_when_selector_replay_fails(tmpdir, monkeypatch):
    tmp_path = pathlib.Path(str(tmpdir))
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    workloads = tmp_path / "workloads.json"
    _workload_file(workloads)
    identity = {
        "device": "Test GPU",
        "compute_capability": "8.0",
        "global_memory_bytes": 40 << 30,
        "source": "test",
    }
    calls = []
    monkeypatch.setattr(entry, "detect_identity", lambda path, required=False: identity)

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1].endswith("calibrate_local_hardware.py"):
            output = pathlib.Path(command[command.index("--output-dir") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "calibration_manifest.json").write_text(json.dumps({
                "schema": "cubutterfly-local-calibration-v1",
                "status": "ready-with-measured-mappings",
                "operator_families": ["fft"],
                "registry_path": str(output / "registry.json"),
                "registry_promoted_records": 1,
            }))
            (output / "operator_calibration.json").write_text("[]")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="selector replay failed")

    monkeypatch.setattr(entry.subprocess, "run", fake_run)
    output = tmp_path / "output"
    assert entry.main([
        "--build-dir", str(build_dir),
        "--profile-dir", str(tmp_path / "profile"),
        "--output-dir", str(output),
        "--workloads", str(workloads),
        "--execute",
    ]) == 1
    assert len(calls) == 2
    manifest = json.loads((output / "migration_manifest.json").read_text())
    assert manifest["status"] != "complete"
    assert manifest["execution"]["selector_replay"]["status"] == "failed"


def test_dry_run_with_existing_manifest_preserves_checkpoint_in_plan(tmpdir, monkeypatch):
    tmp_path = pathlib.Path(str(tmpdir))
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    workloads = tmp_path / "workloads.json"
    _workload_file(workloads)
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = {
        "schema": "cubutterfly-hardware-migration-v1",
        "status": "running",
        "checkpoint_token": "keep-this-checkpoint",
    }
    original = output / "migration_manifest.json"
    original.write_text(json.dumps(checkpoint))
    identity = {
        "device": "Test GPU",
        "compute_capability": "8.0",
        "global_memory_bytes": 40 << 30,
        "source": "test",
    }
    monkeypatch.setattr(entry, "detect_identity", lambda path, required=False: identity)

    assert entry.main([
        "--build-dir", str(build_dir),
        "--profile-dir", str(tmp_path / "profile"),
        "--output-dir", str(output),
        "--workloads", str(workloads),
        "--dry-run",
    ]) == 0
    assert (output / "migration_plan.json").is_file()
    assert json.loads(original.read_text()) == checkpoint
    plan = json.loads((output / "migration_plan.json").read_text())
    assert plan["status"] == "planned"


def test_dry_run_records_mapping_seed_paths_hashes_and_budget(tmpdir, monkeypatch):
    tmp_path = pathlib.Path(str(tmpdir))
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    workloads = tmp_path / "workloads.json"
    _workload_file(workloads)
    seeds = tmp_path / "mapping-seeds.json"
    seeds.write_text(json.dumps({"schema": "cubutterfly-mapping-seeds-v1", "seeds": []}))
    identity = {
        "device": "Test GPU",
        "compute_capability": "8.0",
        "global_memory_bytes": 40 << 30,
        "source": "test",
    }
    monkeypatch.setattr(entry, "detect_identity", lambda path, required=False: identity)

    assert entry.main([
        "--build-dir", str(build_dir),
        "--profile-dir", str(tmp_path / "profile"),
        "--search-workloads", str(workloads),
        "--mapping-seeds", str(seeds),
        "--seed-budget", "2",
        "--dry-run",
    ]) == 0

    manifest = json.loads((tmp_path / "profile" / "migration_manifest.json").read_text())
    record = manifest["mapping_seeds"]
    assert record["paths"] == [str(seeds.resolve())]
    assert record["files"] == [{
        "path": str(seeds.resolve()),
        "sha256": hashlib.sha256(seeds.read_bytes()).hexdigest(),
    }]
    assert manifest["protocol"]["seed_budget"] == 2
    command = manifest["calibration_command"]
    assert command[command.index("--seed-budget") + 1] == "2"
    assert command.count("--mapping-seeds") == 1


def test_resume_rejects_changed_explicit_mapping_seed_file(tmpdir, monkeypatch):
    tmp_path = pathlib.Path(str(tmpdir))
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    workloads = tmp_path / "workloads.json"
    _workload_file(workloads)
    seeds = tmp_path / "mapping-seeds.json"
    seeds.write_text("original\n")
    output = tmp_path / "output"
    resume = tmp_path / "resume.json"
    resume.write_text(json.dumps({
        "schema": "cubutterfly-hardware-migration-v1",
        "build_dir": str(build_dir),
        "profile_dir": str(output),
        "output_dir": str(output),
        "workloads": {"path": str(workloads),
                       "sha256": hashlib.sha256(workloads.read_bytes()).hexdigest()},
        "mapping_seeds": {
            "schema": "cubutterfly-mapping-seeds-v1",
            "paths": [str(seeds)],
            "files": [{"path": str(seeds),
                       "sha256": hashlib.sha256(seeds.read_bytes()).hexdigest()}],
        },
        "protocol": {"seed_budget": 2},
    }))
    identity = {
        "device": "Test GPU",
        "compute_capability": "8.0",
        "global_memory_bytes": 40 << 30,
        "source": "test",
    }
    monkeypatch.setattr(entry, "detect_identity", lambda path, required=False: identity)
    seeds.write_text("changed\n")

    with pytest.raises(ValueError, match="mapping seed file changed"):
        entry.main(["--resume-from", str(resume), "--dry-run"])


def test_coverage_exposes_seed_coverage_and_rejects_seed_omission(tmpdir):
    tmp_path = pathlib.Path(str(tmpdir))
    output = tmp_path / "output"
    output.mkdir()
    workload = {
        "operator": "fft", "precision": "fp32", "logN": 8, "batch": 1,
        "placement": "out-of-place", "normalization": "none",
    }
    def sample():
        return {**workload, "backend": "factor-streamed", "correct": True}
    measurements = [{
        "name": name, "status": "measured", "correct": True,
        "samples": [sample(), sample(), sample()], "median_kernel_ms": 1.0,
    } for name in ("seed-a", "seed-b", "search-a", "search-b")]
    (output / "search_measurements.json").write_text(json.dumps(measurements))
    (output / "search_coverage.json").write_text(json.dumps({
        "schema": "cubutterfly-search-coverage-v1",
        "search_protocol": {"search_budget": 2, "seed_budget": 2,
                             "search_finalists": 1, "operator_trials": 3},
        "workloads": [{
            "workload": workload, "complete": True, "enumerated": 4,
            "requested_screened_count": 4, "screened": 4, "budget_omitted": 0,
            "seed_coverage": {"available": 2, "requested": 2, "attempted": 1,
                              "confirmed": 1, "budget_omitted": 0},
            "candidates": [
                {"id": "seed-a", "status": "measured"},
                {"id": "seed-b", "status": "measured"},
                {"id": "search-a", "status": "measured"},
                {"id": "search-b", "status": "measured"},
            ],
        }],
    }))

    coverage = entry._workload_coverage(output, [workload])
    cell = coverage["cells"][0]
    assert cell["requested_screened_count"] == 4
    assert cell["screened_count"] == 4
    assert cell["seed_coverage"] == {
        "available": 2, "requested": 2, "attempted": 1,
        "confirmed": 1, "budget_omitted": 0,
    }
    assert coverage["search_protocol_complete"] is False


def test_coverage_accepts_rejected_seed_after_a_complete_attempt(tmpdir):
    tmp_path = pathlib.Path(str(tmpdir))
    output = tmp_path / "output"
    output.mkdir()
    workload = {
        "operator": "fft", "precision": "fp32", "logN": 8, "batch": 1,
        "placement": "out-of-place", "normalization": "none",
    }
    sample = {**workload, "backend": "factor-streamed", "correct": True}
    (output / "search_measurements.json").write_text(json.dumps([{
        "name": "search-winner", "status": "measured", "correct": True,
        "samples": [sample, sample, sample], "median_kernel_ms": 1.0,
    }]))
    (output / "search_coverage.json").write_text(json.dumps({
        "schema": "cubutterfly-search-coverage-v1",
        "search_protocol": {"search_budget": 1, "seed_budget": 1,
                             "search_finalists": 1, "operator_trials": 3},
        "workloads": [{
            "workload": workload, "complete": True, "enumerated": 2,
            "requested_screened_count": 2, "screened": 2, "budget_omitted": 0,
            "seed_coverage": {"available": 1, "requested": 1, "attempted": 1,
                              "confirmed": 0, "budget_omitted": 0},
            "candidates": [
                {"id": "seed-unavailable", "status": "unavailable",
                 "selection_reason": "historical-seed"},
                {"id": "search-winner", "status": "measured"},
            ],
        }],
    }))

    coverage = entry._workload_coverage(output, [workload])
    cell = coverage["cells"][0]
    assert cell["seed_coverage"]["attempted"] == cell["seed_coverage"]["requested"]
    assert cell["failed_candidate_count"] == 1
    assert cell["confirmed_candidate_count"] == 1
    assert coverage["search_protocol_complete"] is True
    assert coverage["complete"] is True
