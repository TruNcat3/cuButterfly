"""CPU-only contract tests for the unified stage-calibration entry point."""

import json
import pathlib
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import calibrate_hardware as entry
import stage_service_calibration as stage_calibration


def _workload():
    return {
        "operator": "fft",
        "precision": "fp32",
        "logN": 12,
        "batch": 4,
        "placement": "out-of-place",
        "direction": "forward",
        "normalization": "none",
        "accumulation": "native",
        "element_stride": 1,
        "batch_stride": 4096,
    }


def _sample(workload):
    return {**workload, "correct": True, "backend": "factor-streamed"}


def _protocol(mode, search_seconds):
    return {
        "search_budget": 1,
        "search_finalists": 1,
        "search_seconds": float(search_seconds),
        "compile_seconds": 4.0,
        "operator_trials": 3,
        "operator_warmup": 1,
        "operator_repeat": 2,
        "seed_budget": 0,
        "cost_model": "staged",
        "search_strategy": "model",
        "stage_calibration": mode,
    }


def _write_execution_artifacts(output, workload, mode, stage_status, stage_validated,
                               search_seconds):
    output.mkdir(parents=True, exist_ok=True)
    sample = _sample(workload)
    candidate = {
        "id": "winner",
        "name": "winner",
        "configuration": dict(workload),
        "status": "measured",
        "correct": True,
        "samples": [dict(sample) for _ in range(3)],
        "trials": 3,
        "median_kernel_ms": 1.0,
    }
    (output / "operator_calibration.json").write_text(json.dumps([candidate]))
    (output / "cost_model.json").write_text(json.dumps({
        "status": "calibrated-local",
        "training_rows": 3,
        "validation": {"holdout_top1_accuracy": 1.0,
                        "holdout_mean_latency_regret": 1.0},
    }))
    protocol = _protocol(mode, search_seconds)
    (output / "search_coverage.json").write_text(json.dumps({
        "schema": "cubutterfly-search-coverage-v1",
        "search_budget_per_workload": 1,
        "search_protocol": protocol,
        "measurement_identity": {"compile_mode": "auto"},
        "workloads": [{
            "workload": workload,
            "complete": True,
            "screened": 1,
            "enumerated": 1,
            "budget_omitted": 0,
            "candidates": [{"id": "winner", "status": "measured"}],
        }],
    }))
    (output / "search_measurements.json").write_text(json.dumps([{
        "name": "winner",
        "configuration": dict(workload),
        "status": "measured",
        "correct": True,
        "samples": [dict(sample) for _ in range(3)],
        "trials": 3,
        "median_kernel_ms": 1.0,
    }]))
    (output / "stage_service_profile.json").write_text(json.dumps({
        "validated": stage_validated,
        "calibration_status": stage_status,
        "coverage": {"coverage_complete": stage_status == "complete"},
    }))
    (output / "selector_replay.json").write_text(json.dumps([{
        "name": "winner",
        "correct": True,
        "mapping_matches": True,
        "sample": {**sample, "selected_implementation": "winner"},
    }]))
    (output / "selector_replay_status.json").write_text(json.dumps({
        "status": "passed", "result_count": 1,
    }))
    (output / "calibration_manifest.json").write_text(json.dumps({
        "schema": "cubutterfly-local-calibration-v1",
        "status": "ready-with-measured-mappings",
        "operator_families": ["fft"],
        "cost_model": str(output / "cost_model.json"),
        "registry_path": str(output / "registry.json"),
        "registry_promoted_records": 1,
        "stage_calibration_mode": mode,
    }))


def _fake_execute(monkeypatch, identity, workload, mode, stage_status,
                  stage_validated, search_seconds):
    monkeypatch.setattr(entry, "detect_identity", lambda path, required=False: identity)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if str(command[1]).endswith("calibrate_local_hardware.py"):
            output = pathlib.Path(command[command.index("--output-dir") + 1])
            _write_execution_artifacts(output, workload, mode, stage_status,
                                        stage_validated, search_seconds)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if str(command[1]).endswith("verify_local_selector.py"):
            output = pathlib.Path(command[command.index("--output") + 1])
            output.write_text(json.dumps([{
                "name": "winner",
                "correct": True,
                "mapping_matches": True,
                "sample": {**_sample(workload),
                           "selected_implementation": "winner"},
            }]))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected subprocess: {command}")

    monkeypatch.setattr(entry.subprocess, "run", fake_run)
    return calls


def test_full_default_is_unbounded_and_modes_forward_exactly(tmp_path):
    build = tmp_path / "build"
    args = entry.parse_args(["--build-dir", str(build)])
    entry._fill_protocol_defaults(args)
    assert args.stage_calibration == "full"
    assert args.search_seconds == 0
    assert args.compile_seconds == 0

    workloads = tmp_path / "workloads.json"
    workloads.write_text("{}")
    for mode, seconds in (("full", 17), ("bounded", 23), ("skip", 31)):
        parsed = entry.parse_args([
            "--build-dir", str(build), "--stage-calibration", mode,
            "--search-seconds", str(seconds),
        ])
        entry._fill_protocol_defaults(parsed)
        command = entry.calibration_command(parsed, tmp_path / "profile",
                                            tmp_path / "output", workloads)
        assert command[command.index("--stage-calibration") + 1] == mode
        assert command[command.index("--search-seconds") + 1] == str(float(seconds))


def test_bounded_and_skip_defaults_keep_legacy_300_second_search_budget(tmp_path):
    for mode in ("bounded", "skip"):
        args = entry.parse_args(["--build-dir", str(tmp_path),
                                 "--stage-calibration", mode])
        entry._fill_protocol_defaults(args)
        assert args.search_seconds == 300.0
        assert args.compile_seconds == 600.0


def test_stage_checkpoint_sources_forward_and_resume(tmp_path):
    source = tmp_path / "source.json"
    args = entry.parse_args(["--build-dir", str(tmp_path),
                             "--stage-import-checkpoint", str(source)])
    entry._fill_protocol_defaults(args)
    command = entry.calibration_command(args, tmp_path, tmp_path,
                                        tmp_path / "workloads.json")
    assert command[command.index("--stage-import-checkpoint") + 1] == str(source)
    manifest = tmp_path / "migration.json"
    manifest.write_text(json.dumps({"schema": "cubutterfly-hardware-migration-v1",
                                   "calibration_command": command}))
    resumed = entry.parse_args(["--resume-from", str(manifest)])
    entry._apply_resume_manifest(resumed)
    assert resumed.stage_import_checkpoint == [source]


def test_resume_restores_stage_protocol_and_legacy_manifests_disable_it(tmp_path,
                                                                         monkeypatch):
    build = tmp_path / "build"
    build.mkdir()
    workloads = tmp_path / "workloads.json"
    workloads.write_text(json.dumps({
        "schema": "cubutterfly-install-search-v1",
        "workloads": [{"logN": 8, "batch": 1}],
    }))
    identity = {"device": "Test GPU", "compute_capability": "8.0",
                "global_memory_bytes": 40 << 30, "source": "test"}
    monkeypatch.setattr(entry, "detect_identity", lambda path, required=False: identity)

    legacy_output = tmp_path / "legacy-output"
    legacy_manifest = tmp_path / "legacy.json"
    legacy_manifest.write_text(json.dumps({
        "schema": "cubutterfly-hardware-migration-v1",
        "build_dir": str(build), "profile_dir": str(legacy_output),
        "output_dir": str(legacy_output),
        "workloads": {"path": str(workloads)},
        "protocol": {"search_budget": 1},
    }))
    assert entry.main(["--resume-from", str(legacy_manifest), "--dry-run"]) == 0
    legacy_plan = json.loads((legacy_output / "migration_manifest.json").read_text())
    assert legacy_plan["protocol"]["stage_calibration"] == "skip"
    assert legacy_plan["protocol"]["seed_budget"] == 0
    assert legacy_plan["calibration_command"][legacy_plan["calibration_command"].index(
        "--stage-calibration") + 1] == "skip"

    staged_output = tmp_path / "staged-output"
    staged_manifest = tmp_path / "staged.json"
    staged_manifest.write_text(json.dumps({
        "schema": "cubutterfly-hardware-migration-v1",
        "build_dir": str(build), "profile_dir": str(staged_output),
        "output_dir": str(staged_output),
        "workloads": {"path": str(workloads)},
        "protocol": {"search_budget": 1, "search_finalists": 1,
                     "search_seconds": 12.0, "compile_seconds": 4.0,
                     "operator_trials": 3, "operator_warmup": 1,
                     "operator_repeat": 2, "seed_budget": 0,
                     "cost_model": "staged", "search_strategy": "model",
                     "stage_calibration": "bounded"},
    }))
    assert entry.main(["--resume-from", str(staged_manifest), "--dry-run"]) == 0
    staged_plan = json.loads((staged_output / "migration_manifest.json").read_text())
    assert staged_plan["protocol"]["stage_calibration"] == "bounded"
    assert staged_plan["protocol"]["search_seconds"] == 12.0
    assert staged_plan["protocol"]["resume_search"] is True


def test_stage_profile_gate_requires_validated_and_complete(tmp_path):
    workload = _workload()
    output = tmp_path / "output"
    _write_execution_artifacts(output, workload, "full", "complete", False, 0)
    protocol = _protocol("full", 0)
    summary = entry._execution_summary(output, [workload],
                                       expected_protocol=protocol,
                                       expected_compile_mode="auto")
    assert summary["stage_service_validated"] is False
    assert summary["complete"] is False

    (output / "stage_service_profile.json").write_text(json.dumps({
        "validated": True, "calibration_status": "incomplete",
    }))
    summary = entry._execution_summary(output, [workload],
                                       expected_protocol=protocol,
                                       expected_compile_mode="auto")
    assert summary["stage_service_validated"] is False

    (output / "stage_service_profile.json").write_text(json.dumps({
        "validated": True, "calibration_status": "complete",
    }))
    summary = entry._execution_summary(output, [workload],
                                       expected_protocol=protocol,
                                       expected_compile_mode="auto")
    assert summary["stage_service_validated"] is True
    assert summary["complete"] is True


@pytest.mark.parametrize(
    ("pending", "composition_status", "complete"),
    ((False, "unvalidated", True), (True, "unvalidated", False),
     (True, "warning-accuracy", False), (True, "passed", True)),
)
def test_service_success_does_not_hide_pending_composition(tmp_path, pending, composition_status, complete):
    workload = _workload()
    _write_execution_artifacts(tmp_path, workload, "full", "complete", True, 0)
    (tmp_path / "stage_service_profile.json").write_text(json.dumps({
        "validated": True, "calibration_status": "complete",
        "coverage": {"composition_request_count": int(pending)},
    }))
    (tmp_path / "cost_model.json").write_text(json.dumps({
        "status": "calibrated-local-serial",
        "validation": {"global_validation": {"passed": True}},
        "composition_validation": {"status": composition_status},
    }))
    summary = entry._execution_summary(tmp_path, [workload],
        expected_protocol=_protocol("full", 0), expected_compile_mode="auto")
    assert summary["stage_service_validated"]
    assert summary["complete"] is complete


@pytest.mark.parametrize(
    ("mode", "stage_status", "stage_validated", "expected_return"),
    (("full", "incomplete-budget", False, 1),
     ("bounded", "incomplete-budget", False, 0)),
)
def test_stage_mode_exit_and_manifest_status(tmp_path, monkeypatch, mode,
                                             stage_status, stage_validated,
                                             expected_return):
    workload = _workload()
    workloads = tmp_path / "workloads.json"
    workloads.write_text(json.dumps({
        "schema": "cubutterfly-install-search-v1",
        "workloads": [workload],
    }))
    identity = {"device": "Test GPU", "compute_capability": "8.0",
                "global_memory_bytes": 40 << 30, "source": "test"}
    calls = _fake_execute(monkeypatch, identity, workload, mode, stage_status,
                          stage_validated, 0)
    output = tmp_path / "output"
    result = entry.main([
        "--build-dir", str(tmp_path / "build"),
        "--profile-dir", str(tmp_path / "profile"),
        "--output-dir", str(output), "--workloads", str(workloads),
        "--stage-calibration", mode, "--search-seconds", "0",
        "--search-budget", "1", "--search-finalists", "1",
        "--seed-budget", "0", "--compile-seconds", "4",
        "--operator-trials", "3", "--operator-warmup", "1",
        "--operator-repeat", "2", "--execute",
    ])
    assert result == expected_return
    assert len(calls) == 2
    manifest = json.loads((output / "migration_manifest.json").read_text())
    expected_status = "complete" if stage_status == "complete" else "incomplete"
    assert manifest["status"] == expected_status
    assert manifest["protocol"]["stage_calibration"] == mode
    assert manifest["execution"]["stage_service_validated"] is stage_validated


def test_stage_probe_refuses_busy_gpu_before_running_command(tmp_path):
    calls = []

    def run(command):
        calls.append(command)
        return {"status": "measured"}

    checks = []

    def busy():
        checks.append(True)
        return False

    with pytest.raises(InterruptedError, match="gpu-not-exclusive"):
        stage_calibration._invoke(
            tmp_path / "cubutterfly_stage_microbench", _workload(), run, busy,
            warmup=1, repeat=2, trials=1, describe_only=False)
    assert calls == []
    assert len(checks) == 1
