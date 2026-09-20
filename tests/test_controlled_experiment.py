import json
import pathlib
import sys
import csv
import io

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_controlled_experiment as runner


def study(*, trials=3):
    return {
        "schema": runner.SCHEMA,
        "protocol": {"trials": trials, "warmup": 1, "repeat": 2, "verify_batches": 0},
        "cases": [{
            "id": "fft8-layout", "experiment": "layout",
            "workload": {"operator": "fft", "precision": "fp32", "logN": 8, "batch": 2,
                         "placement": "out-of-place", "normalization": "none"},
            "variants": [
                {"id": "linear", "mapping": {"backend": "shared-iterative", "stage_partition": [6, 6],
                                                "shared_layout": "linear"}},
                {"id": "writer", "mapping": {"backend": "shared-iterative", "stage_partition": [6, 6],
                                                "shared_layout": "writer-aligned"}},
            ],
            "comparisons": [{"baseline": "linear", "treatment": "writer", "changed_axes": ["shared_layout"]}],
        }],
    }


def test_prepare_is_cpu_only_and_generates_lossless_commands(tmpdir, monkeypatch):
    tmp_path = pathlib.Path(str(tmpdir))
    path = tmp_path / "study.json"
    path.write_text(json.dumps(study()))
    output = tmp_path / "prepared.json"

    def no_subprocess(*args, **kwargs):
        raise AssertionError("prepare must not query a GPU or execute a command")

    monkeypatch.setattr(runner.subprocess, "run", no_subprocess)
    document = runner.prepare_document(runner._load_json(path), path, tmp_path / "build")
    runner._atomic_write(output, document)

    variant = document["cases"][0]["variants"][0]
    assert document["status"] == "prepared"
    assert document["identity"]["gpu_queried"] is False
    assert document["identity"]["executable_claim"] is False
    assert "--mapping-json" in variant["preflight_command"]
    payload = json.loads(variant["preflight_command"][variant["preflight_command"].index("--mapping-json") + 1])
    assert payload["stage_partition"] == [6, 6]
    assert "--verify" in variant["preflight_command"]
    assert variant["preflight_command"][variant["preflight_command"].index("--verify-batches") + 1] == "0"


def test_prepare_resume_requires_compatible_prepared_journal(tmpdir):
    tmp_path = pathlib.Path(str(tmpdir))
    study_path = tmp_path / "study.json"
    study_path.write_text(json.dumps(study()))
    output = tmp_path / "journal.json"
    arguments = ["--study", str(study_path), "--build-dir", str(tmp_path / "build"), "--output", str(output)]

    assert runner.main(arguments + ["--mode", "prepare"]) == 0
    assert runner.main(arguments + ["--mode", "prepare", "--resume"]) == 0
    measured = json.loads(output.read_text())
    measured["status"] = "complete"
    runner._atomic_write(output, measured)
    with pytest.raises(runner.StudyError, match="non-prepared"):
        runner.main(arguments + ["--mode", "prepare", "--resume"])
    assert json.loads(output.read_text())["status"] == "complete"
    with pytest.raises(runner.StudyError, match="must not overwrite"):
        runner.main(["--study", str(study_path), "--build-dir", str(tmp_path / "build"),
                     "--output", str(study_path), "--mode", "prepare"])


def test_prepare_rejects_a_second_requested_axis():
    invalid = study()
    invalid["cases"][0]["variants"][1]["mapping"]["tile_threads"] = 256
    with pytest.raises(ValueError, match="undeclared mapping axes"):
        runner._validate_study(invalid)


def test_run_pairs_reverse_order_and_resume_without_reexecution(tmpdir, monkeypatch):
    tmp_path = pathlib.Path(str(tmpdir))
    study_path = tmp_path / "study.json"
    study_path.write_text(json.dumps(study()))
    build = tmp_path / "build"
    build.mkdir()
    (build / "cubutterfly_bench").write_bytes(b"mock-bench")
    output = tmp_path / "journal.json"
    calls = []

    mapping_by_layout = {
        "linear": {
            "schema_version": 1, "kind": "butterfly", "backend": "shared-iterative",
            "stage_partition": [6, 6], "shared_layout": "linear", "stage_overlap": False,
            "batch_tile_count": 1,
        },
        "writer-aligned": {
            "schema_version": 1, "kind": "butterfly", "backend": "shared-iterative",
            "stage_partition": [6, 6], "shared_layout": "writer-aligned", "stage_overlap": False,
            "batch_tile_count": 1,
        },
    }

    class Completed:
        def __init__(self, stdout="", returncode=0, stderr=""):
            self.stdout = stdout
            self.returncode = returncode
            self.stderr = stderr

    def fake_run(command, **kwargs):
        command = [str(value) for value in command]
        calls.append(command)
        if command[0] == "nvidia-smi":
            if "query-gpu=name" in " ".join(command):
                return Completed("A100, 8.0, 81920, GPU-test, 550.1\n")
            return Completed("")
        if "--device-identity" in command:
            return Completed("device,compute_capability,global_memory_bytes\nA100,8.0,85899345920\n")
        payload = json.loads(command[command.index("--mapping-json") + 1])
        layout = payload["shared_layout"]
        mapping = mapping_by_layout[layout]
        timed = "--verify" not in command
        latency = 1.0 if layout == "linear" else 0.5
        if timed:
            latency += 0.01 * sum(1 for value in calls if value == command)
        sample = {
            "device": "A100", "compute_capability": "8.0", "operator": "fft", "precision": "fp32",
            "placement": "out-of-place", "logN": "8", "batch": "2", "direction": "forward",
            "normalization": "none", "accumulation": "native", "element_stride": "1",
            "batch_stride": "256", "kernel_ms": str(latency), "correct": "-1" if timed else "1",
            "verified_batches": "0" if timed else "2", "mapping_json": json.dumps(mapping, sort_keys=True),
            "runtime_fingerprint": "mock-runtime-v1", "execution_groups_json": "[]",
            "workspace_bytes": "4096", "execution_group_count": "2",
        }
        fields = list(sample)
        stream = io.StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        writer.writerow([sample[field] for field in fields])
        return Completed(stream.getvalue())

    monkeypatch.setattr(runner, "require_exclusive_gpu", lambda: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    document = runner.run_document(
        {}, runner._load_json(study_path), study_path, build, output, "GPU-test", resume=False)

    case = document["cases"][0]
    assert document["status"] == "complete"
    assert case["trial_orders"][0]["order"] == ["linear", "writer"]
    assert case["trial_orders"][1]["order"] == ["writer", "linear"]
    comparison = case["comparisons"][0]
    assert comparison["status"] == "measured"
    assert comparison["performance_accepted"] is True
    assert comparison["actual_changed_axes"] == ["shared_layout"]
    assert comparison["median_treatment_over_baseline"] > 1.9
    measured_calls = len([call for call in calls if "--csv" in call and "--device-identity" not in call])

    resumed = runner.run_document(
        {}, runner._load_json(study_path), study_path, build, output, "GPU-test", resume=True)
    assert resumed["status"] == "complete"
    assert len([call for call in calls if "--csv" in call and "--device-identity" not in call]) == measured_calls


def test_runtime_mapping_alias_is_blocked(tmpdir, monkeypatch):
    tmp_path = pathlib.Path(str(tmpdir))
    study_path = tmp_path / "study.json"
    study_value = study()
    study_path.write_text(json.dumps(study_value))
    document = runner.prepare_document(study_value, study_path, tmp_path / "build")
    # This is an output-level check independent of CUDA execution: a runtime
    # that folds the requested treatment to the baseline must not be timed as
    # a distinct treatment.
    case = document["cases"][0]
    for variant in case["variants"]:
        mapping = {"schema_version": 1, "kind": "butterfly", "backend": "shared-iterative"}
        variant["preflight"] = {
            "status": "accepted", "resolved_mapping": mapping,
            "runtime_fingerprint": "same", "sample": {},
        }
        variant["trials"] = []
    result = runner._comparison_result(case, case["comparisons"][0])
    assert result["status"] == "blocked"
    assert result["resolved_mapping_alias"] is True


def test_partition_derived_boundaries_do_not_block_single_axis_pair():
    value = study()
    case = runner._validate_study(value)[0]["cases"][0]
    case["comparisons"][0]["changed_axes"] = ["stage_partition"]
    baseline = case["variants"][0]
    treatment = case["variants"][1]
    baseline_mapping = {
        "schema_version": 1,
        "kind": "butterfly",
        "backend": "shared-iterative",
        "stage_partition": [3, 3, 3, 3],
        "boundaries": [{"layout": "direct-strided"}] * 3,
    }
    treatment_mapping = {
        "schema_version": 1,
        "kind": "butterfly",
        "backend": "shared-iterative",
        "stage_partition": [6, 6],
        "boundaries": [{"layout": "direct-strided"}],
    }
    for variant, mapping, latency in (
        (baseline, baseline_mapping, 2.0), (treatment, treatment_mapping, 1.0)
    ):
        variant["preflight"] = {
            "status": "accepted", "resolved_mapping": mapping, "runtime_fingerprint": "same",
        }
        variant["trials"] = [
            {"trial": trial, "performance_accepted": True, "kernel_ms": latency}
            for trial in range(3)
        ]
    result = runner._comparison_result(case, case["comparisons"][0])
    assert result["status"] == "measured"
    assert result["actual_changed_axes"] == ["stage_partition"]


def test_partition_boundary_policy_change_is_not_derived():
    value = study()
    case = runner._validate_study(value)[0]["cases"][0]
    case["comparisons"][0]["changed_axes"] = ["stage_partition"]
    baseline = case["variants"][0]
    treatment = case["variants"][1]
    baseline_mapping = {
        "schema_version": 1,
        "kind": "butterfly",
        "backend": "shared-iterative",
        "stage_partition": [3, 3, 3, 3],
        "boundaries": [{"layout": "direct-strided", "residency": "global-scratch"}] * 3,
    }
    treatment_mapping = {
        "schema_version": 1,
        "kind": "butterfly",
        "backend": "shared-iterative",
        "stage_partition": [6, 6],
        "boundaries": [{"layout": "direct-strided", "residency": "fused"}],
    }
    for variant, mapping in ((baseline, baseline_mapping), (treatment, treatment_mapping)):
        variant["preflight"] = {
            "status": "accepted", "resolved_mapping": mapping, "runtime_fingerprint": "same",
        }
        variant["trials"] = [
            {"trial": trial, "performance_accepted": True, "kernel_ms": 1.0}
            for trial in range(3)
        ]
    result = runner._comparison_result(case, case["comparisons"][0])
    assert result["status"] == "blocked"
    assert result["actual_changed_axes"] == ["boundaries", "stage_partition"]
    assert "boundaries" in result["reason"]


def test_recomputed_success_discards_stale_blocked_fields():
    value = study()
    case = runner._validate_study(value)[0]["cases"][0]
    comparison = case["comparisons"][0]
    comparison["changed_axes"] = ["stage_partition"]
    comparison.update(
        status="blocked", performance_accepted=False,
        reason="runtime mapping changed undeclared axes: boundaries",
        resolved_mapping_alias=True, actual_changed_axes=["boundaries"],
    )
    mappings = (
        {"schema_version": 1, "kind": "butterfly", "backend": "shared-iterative",
         "stage_partition": [3, 3, 3, 3],
         "boundaries": [{"layout": "direct-strided"}] * 3},
        {"schema_version": 1, "kind": "butterfly", "backend": "shared-iterative",
         "stage_partition": [6, 6],
         "boundaries": [{"layout": "direct-strided"}]},
    )
    for variant, mapping, latency in zip(case["variants"], mappings, (2.0, 1.0)):
        variant["preflight"] = {
            "status": "accepted", "resolved_mapping": mapping, "runtime_fingerprint": "same",
        }
        variant["trials"] = [
            {"trial": trial, "performance_accepted": True, "kernel_ms": latency}
            for trial in range(3)
        ]
    result = runner._comparison_result(case, comparison)
    assert result["status"] == "measured"
    assert result["performance_accepted"] is True
    assert result["actual_changed_axes"] == ["stage_partition"]
    assert "reason" not in result
    assert "resolved_mapping_alias" not in result


def test_hardware_identity_keeps_cuda_and_nvml_memory_separate(tmpdir, monkeypatch):
    class Completed:
        def __init__(self, stdout):
            self.stdout = stdout
            self.returncode = 0
            self.stderr = ""

    def fake_run(command, **kwargs):
        if command[0] == "nvidia-smi":
            return Completed("A100, 8.0, 1, GPU-test, 550.1\n")
        return Completed("device,compute_capability,global_memory_bytes\nA100,8.0,85899345920\n")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    value = runner._query_hardware(pathlib.Path(str(tmpdir)) / "cubutterfly_bench", "GPU-test")
    assert value["global_memory_bytes"] == 85899345920
    assert value["cuda_global_memory_bytes"] == 85899345920
    assert value["nvml_memory_total_mib"] == 1


def test_sample_device_name_is_checked_against_hardware_identity():
    workload = study()["cases"][0]["workload"]
    sample = {
        "device": "Different GPU", "compute_capability": "8.0", "operator": "fft", "precision": "fp32",
        "placement": "out-of-place", "logN": "8", "batch": "2", "direction": "forward",
        "normalization": "none", "accumulation": "native", "element_stride": "1", "batch_stride": "256",
        "mapping_json": json.dumps({"backend": "shared-iterative"}), "runtime_fingerprint": "mock",
        "kernel_ms": "1", "correct": "-1", "verified_batches": "0",
    }
    with pytest.raises(RuntimeError, match="hardware mismatch for device"):
        runner._validate_sample(sample, workload, runner._expected_semantics(workload),
                                {"name": "A100", "compute_capability": "8.0"}, require_correctness=False)
