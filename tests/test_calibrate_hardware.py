import json
import hashlib
import pathlib
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import calibrate_hardware


def _sample(workload, *, backend="factor-streamed", correct="1"):
    return {
        "operator": workload.get("operator", "fft"),
        "precision": workload["precision"],
        "logN": str(workload["logN"]),
        "batch": str(workload["batch"]),
        "placement": workload["placement"],
        "normalization": workload.get("normalization", "none"),
        "backend": backend,
        "correct": correct,
        "kernel_ms": "1.0",
    }


def _write_search_artifacts(output, workloads, candidate_specs, *, search_protocol=None,
                            measurement_identity=None, cell_options=None):
    search_protocol = search_protocol or {}
    cell_options = cell_options or {}
    measurements = []
    cells = []
    for workload, specs in zip(workloads, candidate_specs):
        candidates = []
        for spec in specs:
            name = spec["name"]
            status = spec.get("status", "measured")
            sample_workload = {**workload, **spec.get("sample_workload", {})}
            samples = [_sample(sample_workload, backend=spec.get("backend", "factor-streamed"))
                       for _ in range(spec.get("samples", 3))]
            candidate = {
                "id": name,
                "configuration": {**workload,
                                   "backend": spec.get("backend", "factor-streamed")},
                "status": status,
            }
            candidates.append(candidate)
            if spec.get("measurement", True):
                measurement_status = spec.get("measurement_status", status)
                measurements.append({
                    "name": name,
                    "configuration": candidate["configuration"],
                    "status": measurement_status,
                    "correct": spec.get("measurement_correct", measurement_status == "measured"),
                    "samples": samples,
                    "trials": len(samples),
                    "median_kernel_ms": 1.0,
                })
        options = cell_options.get(json.dumps(workload, sort_keys=True), {})
        cells.append({"workload": workload, "complete": options.get("complete", True),
                      "screened": options.get("screened", 1),
                      "enumerated": options.get("enumerated", len(candidates)),
                      "budget_omitted": options.get("budget_omitted", 7),
                      "stop_reason": options.get("stop_reason"),
                      "candidates": candidates})
    (output / "search_coverage.json").write_text(json.dumps({
        "schema": "cubutterfly-search-coverage-v1", "workloads": cells,
        "search_protocol": search_protocol,
        "measurement_identity": measurement_identity or {"compile_mode": "auto"},
    }))
    (output / "search_measurements.json").write_text(json.dumps(measurements))


def _write_calibration_manifest(output, *, cost_status="calibrated-local"):
    (output / "operator_calibration.json").write_text("[]")
    (output / "cost_model.json").write_text(json.dumps({
        "status": cost_status, "training_rows": 3,
        "validation": {"holdout_top1_accuracy": 1.0, "holdout_mean_latency_regret": 1.0},
    }))
    (output / "calibration_manifest.json").write_text(json.dumps({
        "schema": "cubutterfly-local-calibration-v1",
        "status": "ready-with-measured-mappings",
        "operator_families": ["fft", "fwht"],
        "cost_model": str(output / "cost_model.json"),
        "registry_path": str(output / "registry.json"),
        "registry_promoted_records": 1,
    }))


def _write_selector_status(output, status="passed", result_count=1):
    (output / "selector_replay.json").write_text(json.dumps([{"correct": True}] * result_count))
    (output / "selector_replay_status.json").write_text(json.dumps({
        "status": status, "result_count": result_count,
    }))


def _write_valid_selector_fixture(output, workload, name="winner-a"):
    sample = _sample(workload)
    samples = [{**sample} for _ in range(3)]
    calibration = [{"name": name, "status": "measured", "correct": True,
                    "samples": samples, "trials": 3, "median_kernel_ms": 1.0}]
    (output / "operator_calibration.json").write_text(json.dumps(calibration))
    replay_sample = {**sample, "selected_implementation": name}
    (output / "selector_replay.json").write_text(json.dumps([{
        "name": name, "correct": True, "mapping_matches": True,
        "sample": replay_sample,
    }]))
    (output / "selector_replay_status.json").write_text(json.dumps({
        "status": "passed", "result_count": 1,
    }))


def test_dry_run_writes_reproducible_migration_manifest(tmpdir):
    root = pathlib.Path(str(tmpdir))
    workloads = root / "workloads.json"
    workloads.write_text(json.dumps({
        "schema": "cubutterfly-install-search-v1",
        "scope": "test all operators",
        "workloads": [
            {"precision": "fp32", "logN": 8, "batch": 1},
            {"operator": "ntt", "precision": "word32", "logN": 8, "batch": 1},
        ],
    }))
    output = root / "profile"
    result = calibrate_hardware.main([
        "--build-dir", str(root / "build"),
        "--profile-dir", str(output),
        "--output-dir", str(output),
        "--search-workloads", str(workloads),
        "--dry-run",
    ])
    assert result == 0
    manifest = json.loads((output / "migration_manifest.json").read_text())
    assert manifest["status"] == "planned"
    assert manifest["workloads"]["workload_count"] == 2
    assert manifest["workloads"]["operators"] == ["fft", "ntt"]
    assert "calibrate_local_hardware.py" in " ".join(manifest["calibration_command"])


def test_execution_summary_requires_each_workload_and_repeated_internal_candidate(tmpdir):
    root = pathlib.Path(str(tmpdir))
    output = root / "output"
    output.mkdir()
    workloads = [
        {"operator": "fft", "precision": "fp32", "logN": 8, "batch": 1,
         "placement": "out-of-place", "normalization": "none"},
        {"operator": "fwht", "precision": "fp32", "logN": 8, "batch": 1,
         "placement": "out-of-place", "normalization": "none"},
    ]
    _write_search_artifacts(output, workloads, [
        [{"name": "fft-cufft", "backend": "cufft"},
         {"name": "fft-screened", "status": "screened", "samples": 1}],
        [{"name": "fwht-one-sample", "samples": 1}],
    ])
    _write_calibration_manifest(output)
    _write_selector_status(output, status="failed", result_count=0)

    summary = calibrate_hardware._execution_summary(output, workloads)

    assert summary["complete"] is False
    assert summary["workload_coverage_complete"] is False
    assert summary["workload_coverage"]["covered_count"] == 0
    assert summary["workload_coverage"]["missing_count"] == 2
    assert summary["workload_coverage"]["budget_omitted_count"] == 14
    assert summary["cost_model_validated"] is True
    assert summary["selector_replay"]["status"] == "failed"
    assert summary["operator_families"] == []


def test_execution_summary_counts_confirmed_cells_but_preserves_failures(tmpdir):
    root = pathlib.Path(str(tmpdir))
    output = root / "output"
    output.mkdir()
    workloads = [
        {"operator": "fft", "precision": "fp32", "logN": 8, "batch": 1,
         "placement": "out-of-place", "normalization": "none"},
        {"operator": "fwht", "precision": "fp32", "logN": 8, "batch": 1,
         "placement": "out-of-place", "normalization": "none"},
    ]
    _write_search_artifacts(output, workloads, [
        [{"name": "fft-confirmed"}],
        [{"name": "fwht-failed", "status": "unavailable", "measurement": False}],
    ])
    _write_calibration_manifest(output, cost_status="calibrated-local-warning")

    summary = calibrate_hardware._execution_summary(output, workloads)

    assert summary["workload_coverage"]["covered_count"] == 1
    assert summary["workload_coverage"]["missing_count"] == 1
    assert summary["workload_coverage"]["failed_candidate_count"] == 1
    assert summary["operator_families"] == ["fft"]
    assert summary["cost_model_status"] == "calibrated-local-warning"
    assert summary["cost_model_validated"] is False
    assert summary["complete"] is False


def test_execution_summary_rejects_semantically_mismatched_measured_samples(tmpdir):
    root = pathlib.Path(str(tmpdir))
    output = root / "output"
    output.mkdir()
    workload = {"operator": "fft", "precision": "fp32", "logN": 8, "batch": 1,
                "placement": "out-of-place", "normalization": "none"}
    _write_search_artifacts(output, [workload], [[{
        "name": "wrong-logn", "sample_workload": {"logN": 9}, "samples": 3,
    }],], search_protocol={"search_budget": 1, "search_finalists": 1,
                           "operator_trials": 3})
    _write_calibration_manifest(output)
    _write_selector_status(output)

    summary = calibrate_hardware._execution_summary(output, [workload])

    assert summary["workload_coverage"]["covered_count"] == 0
    assert summary["workload_coverage"]["cells"][0]["valid_candidate_count"] == 0
    assert summary["workload_coverage"]["cells"][0]["confirmed_candidate_count"] == 0


def test_execution_summary_requires_all_operator_trials_for_confirmation(tmpdir):
    root = pathlib.Path(str(tmpdir))
    output = root / "output"
    output.mkdir()
    workload = {"operator": "fwht", "precision": "fp32", "logN": 8, "batch": 1,
                "placement": "out-of-place", "normalization": "none"}
    _write_search_artifacts(output, [workload], [[{
        "name": "two-of-three", "samples": 2,
    }]], search_protocol={"search_budget": 1, "search_finalists": 1,
                          "operator_trials": 3},
                           cell_options={json.dumps(workload, sort_keys=True): {
                               "screened": 1, "enumerated": 1, "budget_omitted": 0,
                           }})
    _write_calibration_manifest(output)
    _write_selector_status(output)

    summary = calibrate_hardware._execution_summary(output, [workload])

    assert summary["workload_coverage"]["required_confirmation_trials"] == 3
    assert summary["workload_coverage"]["covered_count"] == 0
    assert summary["workload_coverage"]["cells"][0]["valid_candidate_count"] == 1
    assert summary["workload_coverage"]["cells"][0]["confirmed_candidate_count"] == 0


def test_execution_summary_accepts_complete_positive_search_protocol(tmpdir):
    root = pathlib.Path(str(tmpdir))
    output = root / "output"
    output.mkdir()
    workload = {"operator": "fft", "precision": "fp32", "logN": 8, "batch": 1,
                "placement": "out-of-place", "normalization": "none"}
    protocol = {"search_budget": 2, "search_finalists": 2, "search_seconds": 0.0,
                "compile_seconds": 600.0, "operator_trials": 3,
                "operator_warmup": 50, "operator_repeat": 50}
    _write_search_artifacts(output, [workload], [[
        {"name": "winner-a", "samples": 3},
        {"name": "winner-b", "samples": 3},
    ]], search_protocol=protocol, measurement_identity={"compile_mode": "auto"},
                           cell_options={json.dumps(workload, sort_keys=True): {
                               "complete": True, "screened": 2, "enumerated": 2,
                               "budget_omitted": 0,
                           }})
    _write_calibration_manifest(output)
    _write_valid_selector_fixture(output, workload)

    summary = calibrate_hardware._execution_summary(
        output, [workload], expected_protocol=protocol, expected_compile_mode="auto")

    assert summary["workload_coverage_complete"] is True
    assert summary["search_protocol_complete"] is True
    assert summary["protocol_identity"]["match"] is True
    assert summary["selector_replay"]["status"] == "passed"
    assert summary["complete"] is True


def test_execution_summary_rejects_budget_shortfall_even_with_mapping(tmpdir):
    root = pathlib.Path(str(tmpdir))
    output = root / "output"
    output.mkdir()
    workload = {"operator": "fft", "precision": "fp32", "logN": 8, "batch": 1,
                "placement": "out-of-place", "normalization": "none"}
    protocol = {"search_budget": 2, "search_finalists": 1, "operator_trials": 3}
    _write_search_artifacts(output, [workload], [[{"name": "one-winner", "samples": 3}]],
                            search_protocol=protocol,
                            cell_options={json.dumps(workload, sort_keys=True): {
                                "complete": True, "screened": 1, "enumerated": 2,
                                "budget_omitted": 1,
                            }})
    _write_calibration_manifest(output)
    _write_selector_status(output)

    summary = calibrate_hardware._execution_summary(output, [workload])

    assert summary["workload_coverage_complete"] is True
    assert summary["search_protocol_complete"] is False
    assert summary["workload_coverage"]["budget_omitted_count"] == 1
    assert summary["complete"] is False


def test_incumbent_covers_mapping_but_cannot_complete_search_protocol(tmpdir):
    root = pathlib.Path(str(tmpdir))
    output = root / "output"
    output.mkdir()
    workload = {"operator": "fft", "precision": "fp32", "logN": 8, "batch": 1,
                "placement": "out-of-place", "normalization": "none"}
    protocol = {"search_budget": 1, "search_finalists": 1, "operator_trials": 3}
    _write_search_artifacts(output, [workload], [[{
        "name": "screened-only", "status": "screened", "samples": 3,
        "measurement_status": "measured", "measurement_correct": True,
    }]], search_protocol=protocol,
                            cell_options={json.dumps(workload, sort_keys=True): {
                                "complete": True, "screened": 1, "enumerated": 1,
                                "budget_omitted": 0,
                            }})
    _write_calibration_manifest(output)
    _write_valid_selector_fixture(output, workload, name="incumbent-winner")

    summary = calibrate_hardware._execution_summary(output, [workload])
    cell = summary["workload_coverage"]["cells"][0]

    assert summary["workload_coverage_complete"] is True
    assert cell["incumbent_confirmed_candidate_ids"] == ["incumbent-winner"]
    assert cell["search_confirmed_candidate_ids"] == []
    assert summary["search_protocol_complete"] is False
    assert summary["complete"] is False


def test_missing_manifest_summary_is_stable_and_incomplete(tmpdir):
    output = pathlib.Path(str(tmpdir)) / "missing"
    output.mkdir()
    summary = calibrate_hardware._execution_summary(output, [{"operator": "fft", "logN": 8}])
    assert summary["status"] == "missing-calibration-manifest"
    assert summary["complete"] is False
    assert summary["workload_coverage"]["status"] == "missing-artifact"
    assert summary["missing_operator_families"] == ["fft"]


def test_resume_from_manifest_restores_protocol_and_legacy_compile_mode(tmpdir):
    root = pathlib.Path(str(tmpdir))
    build = root / "build"
    profile = root / "profile"
    output = root / "output"
    workloads = root / "workloads.json"
    workloads.write_text(json.dumps({
        "schema": "cubutterfly-install-search-v1", "workloads": [{"logN": 8, "batch": 1}],
    }))
    output.mkdir()
    (output / "operator_measurement_identity.json").write_text(json.dumps({"compile_mode": "research"}))
    resume = root / "migration_manifest.json"
    resume.write_text(json.dumps({
        "schema": "cubutterfly-hardware-migration-v1", "build_dir": str(build),
        "profile_dir": str(profile), "output_dir": str(output),
        "workloads": {"path": str(workloads)},
        "protocol": {"capability_trials": 2, "operator_trials": 4, "operator_warmup": 7,
                     "operator_repeat": 11, "verify_batches": 3, "search_budget": 9,
                     "search_finalists": 2, "search_seconds": 13.0, "compile_seconds": 17.0},
    }))
    assert calibrate_hardware.main(["--resume-from", str(resume), "--dry-run"]) == 0
    planned = json.loads((output / "migration_manifest.json").read_text())
    assert planned["compile_mode"] == "research"
    assert planned["protocol"]["operator_trials"] == 4
    assert planned["protocol"]["search_budget"] == 9
    assert planned["protocol"]["compile_seconds"] == 17.0
    assert planned["protocol"]["resume_search"] is True


def test_selector_replay_records_cpu_result(tmpdir, monkeypatch):
    root = pathlib.Path(str(tmpdir))
    output = root / "output"
    output.mkdir()
    calibration = output / "operator_calibration.json"
    calibration.write_text("[]")

    def fake_run(command, **kwargs):
        pathlib.Path(command[command.index("--output") + 1]).write_text("[]")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(calibrate_hardware.subprocess, "run", fake_run)
    result = calibrate_hardware._selector_replay(
        SimpleNamespace(build_dir=root / "build"), output,
        {"CUBUTTERFLY_COMPILE_MODE": "research"},
    )
    assert result["status"] != "passed"
    assert result["result_count"] == 0
    assert json.loads((output / "selector_replay_status.json").read_text())["status"] != "passed"


def test_selector_summary_rejects_empty_passed_replay(tmpdir):
    output = pathlib.Path(str(tmpdir)) / "output"
    output.mkdir()
    (output / "operator_calibration.json").write_text("[]")
    (output / "selector_replay.json").write_text("[]")
    (output / "selector_replay_status.json").write_text(json.dumps({
        "status": "passed", "result_count": 0,
    }))

    summary = calibrate_hardware._selector_replay_summary(output)

    assert summary["status"] == "failed"
    assert summary["result_count"] == 0


def test_selector_summary_marks_changed_calibration_stale(tmpdir):
    output = pathlib.Path(str(tmpdir)) / "output"
    output.mkdir()
    workload = {"operator": "fft", "precision": "fp32", "logN": 8, "batch": 1,
                "placement": "out-of-place", "normalization": "none"}
    _write_valid_selector_fixture(output, workload)
    calibration = output / "operator_calibration.json"
    original_hash = hashlib.sha256(calibration.read_bytes()).hexdigest()
    status = {"status": "passed", "result_count": 1, "calibration_sha256": original_hash}
    (output / "selector_replay_status.json").write_text(json.dumps(status))
    changed = json.loads(calibration.read_text())
    changed[0]["median_kernel_ms"] = 2.0
    calibration.write_text(json.dumps(changed))

    summary = calibrate_hardware._selector_replay_summary(output)

    assert summary["status"] == "stale"


def test_hidden_cuda_visibility_cannot_fallback_to_gpu0(tmpdir, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    identity = calibrate_hardware.detect_identity(pathlib.Path(str(tmpdir)))
    assert identity["source"] == "unavailable"
    with pytest.raises(RuntimeError, match="unable to detect GPU identity"):
        calibrate_hardware.detect_identity(pathlib.Path(str(tmpdir)), required=True)


def test_nvidia_smi_identity_is_marked_as_estimate(tmpdir, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(calibrate_hardware, "_benchmark_identity", lambda path: None)
    monkeypatch.setattr(calibrate_hardware, "subprocess", SimpleNamespace(run=lambda *args, **kwargs:
        SimpleNamespace(stdout="Test GPU, 8.0, 40960\n", returncode=0)))
    identity = calibrate_hardware.detect_identity(pathlib.Path(str(tmpdir)))
    assert identity["source"] == "nvidia-smi-estimate"
