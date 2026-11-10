"""CPU contract checks for portable target orchestration, without CUDA probing."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_research_target as target


def test_prepare_uses_only_semantics_and_cpu_projection(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU preparation must not probe GPU or spawn benchmarks")
    monkeypatch.setattr(subprocess, "run", forbidden)
    args = target.parse_args(["--mode", "prepare", "--output-dir", str(tmp_path)])
    directory = target.prepare(args)
    document = target.read(directory / "modules.json")
    assert document["device_identity"] == {"status": "unmeasured-compilation-target", "sm": 70}
    assert document["module_count"] > 40
    assert not document["deferred_projection"]
    assert len({row["id"] for row in document["modules"]}) == document["module_count"]
    for row in document["modules"]:
        assert row["id"] == target.key(row["request"])
    from research_compile_requests import compile_request
    requests = {row["id"] for row in document["modules"]}
    for group in target.GROUPS:
        for workload in target.read(ROOT / "config/research_campaign" / f"{group}.json")["workloads"]:
            seed = target.fallback_seed(workload)
            request = compile_request(dict(workload, mapping_json=seed["mapping"]))
            assert request is None or target.key(request) in requests
    assert "kernel_ms" not in json.dumps(target.read(tmp_path / "mapping_seeds.json"))
    before = target.sha(directory / "modules.json")
    target.prepare(args)
    assert target.sha(directory / "modules.json") == before


def test_memory_screening_preserves_batch_and_semantic_identity():
    point = dict(operator="fft", precision="fp32", logN=20, batch=128)
    original = copy.deepcopy(point)
    assert target.input_bytes(point) == 1024 ** 3
    accepted16, deferred16 = target.partition_memory([point], 16 * 1024 ** 3)
    accepted32, deferred32 = target.partition_memory([point], 32 * 1024 ** 3)
    assert not accepted16 and deferred16[0]["workload"] == original
    assert accepted32 == [point] and not deferred32 and point == original
    assert "not proof" in deferred16[0]["reason"]


def test_memory_uses_strides_and_rns_channels():
    point = dict(operator="fwht", precision="fp32", logN=3, batch=3,
                 element_stride=2, batch_stride=21)
    assert target.input_bytes(point) == (42 + 14 + 1) * 4
    rns = dict(logN=8, batch=7, word_bits=64, moduli=[17, 97, 193, 257])
    assert target.input_bytes(rns) == 256 * 7 * 8 * 4
    assert target.input_bytes(dict(operator="fft", shape="64x256", batch=3, precision="fp64")) == 64 * 256 * 3 * 16


def test_plan_gate_accepts_only_fully_verified_contracts(tmp_path):
    args = target.parse_args(["--mode", "run", "--output-dir", str(tmp_path)])
    for workload in target.read(ROOT / "config/research_campaign/plan_workloads.json")["workloads"]:
        command = target.plan_command(args, workload)
        assert command[command.index("--shape") + 1] == workload["shape"]
        assert "--compare-cufft" in command and "--verify" in command
    with pytest.raises(ValueError, match="full-output verified"):
        target.plan_command(args, dict(operator="fwht", shape=[64], batch=1, precision="fp32"))


def test_plan_validation_rejects_partial_verification_and_bad_trials():
    workload = dict(shape="3x64", batch=7)
    sample = dict(logical_shape="3x64", physical_shape="8x128", batch="7", verified_batches="7",
        correct="1", trials="3", kernel_ms="1", cufft_ms="2", trial_kernel_ms="1:1:1", trial_cufft_ms="2:2:2")
    target.validate_plan(sample, workload, 3)
    for bad in ({"verified_batches": "1"}, {"physical_shape": "3x64"},
                {"trial_kernel_ms": "1:1"}, {"trial_cufft_ms": "nan:2:2"}, {"correct": "0"}):
        with pytest.raises(ValueError):
            target.validate_plan(dict(sample, **bad), workload, 3)


def test_arbitrary_fft_preserves_bluestein_physical_shape():
    for logical, physical in (("1000", "2048"), ("30x50", "64x128"), ("32x64", "32x64")):
        sample = dict(logical_shape=logical, physical_shape=physical, batch="1", verified_batches="1",
            correct="1", trials="1", kernel_ms="1", cufft_ms="2", trial_kernel_ms="1", trial_cufft_ms="2")
        target.validate_plan(sample, dict(shape=logical, batch=1), 1)
        with pytest.raises(ValueError, match="delegated"):
            target.validate_plan(dict(sample, algorithm="cufft-direct"), dict(shape=logical, batch=1), 1)


def test_plan_resume_rejects_foreign_result(tmp_path):
    args = target.parse_args(["--mode", "run", "--output-dir", str(tmp_path)])
    workload = dict(operator="fft", shape="32x64", precision="fp32", batch=1)
    target.write(tmp_path / "inputs/plan_workloads.json", {"workloads": [workload]})
    target.write(tmp_path / "plans/plans.json", [dict(workload=workload, gpu_uuid="GPU-other", correct=True,
        command=list(map(str, target.plan_command(args, workload))), sample={})])
    with pytest.raises(ValueError, match="different target"):
        target.run_plans(args, tmp_path, {"uuid": "GPU-target"})


def test_target_identity_distinguishes_model_memory_and_uuid():
    base = dict(uuid="GPU-123456789012-abcdef", name="Tesla V100-SXM2-16GB", memory_mib=16160)
    tags = {target.target_tag(base), target.target_tag(dict(base, memory_mib=32510)),
            target.target_tag(dict(base, name="Tesla V100-PCIE-16GB")),
            target.target_tag(dict(base, uuid="GPU-923456789012-abcdef"))}
    assert len(tags) == 4


def test_calibration_normalizes_real_probe_schema(tmp_path, monkeypatch):
    import schedule_cost_model
    import pipeline_schedule_model
    monkeypatch.setattr(target.os, "environ", dict(os.environ))
    args = target.parse_args(["--mode", "run", "--output-dir", str(tmp_path)])
    device = dict(uuid="GPU-test", name="Tesla V100", memory_mib=16384, driver="test")
    directory = tmp_path / "device"
    calls = []
    def guarded(command, uuid, log, quarantine):
        calls.append(list(map(str, command)))
        if "initialize_hardware_profile.py" in str(command[1]):
            target.write(directory / "calibration/caps/hardware_profile.json",
                dict(device="Tesla V100", compute_capability="7.0", capabilities={}))
        else:
            target.write(directory / "calibration/stage_calibration.json",
                dict(status="complete", coverage=dict(validation_complete=True)))
        return subprocess.CompletedProcess(command, 0, "", "")
    def capture(command):
        if command[-1] == "--hardware":
            return json.dumps(dict(device="Tesla V100", memory_bytes=16 * 1024**3, sm_count=80))
        return f"device,compute_capability,global_memory_bytes\nTesla V100,7.0,{16 * 1024**3}\n"
    def costs(binary, path, profile, run, exclusive):
        target.write(path, {"identity": "test"})
        return {"status": "calibrated"}
    monkeypatch.setattr(target, "guarded", guarded)
    monkeypatch.setattr(target, "capture", capture)
    monkeypatch.setattr(schedule_cost_model, "calibrate", costs)
    monkeypatch.setattr(pipeline_schedule_model, "calibrate", costs)
    paths = target.calibrate(args, directory, device)
    profile = target.read(directory / "calibration/base_profile.json")
    assert profile["global_memory_bytes"] == 16 * 1024**3
    assert profile["gpu_uuid"] == "GPU-test" and len(paths) == 5
    assert "--max-points" in calls[-1] and "1024" in calls[-1]
    target.calibrate(args, directory, device)
    assert len([row for row in calls if "initialize_hardware_profile.py" in row[1]]) == 1


def test_orchestration_calibrates_first_and_resumes_without_remeasurement(tmp_path, monkeypatch):
    monkeypatch.setattr(target.os, "environ", dict(os.environ))
    args = target.parse_args(["--mode", "run", "--output-dir", str(tmp_path)])
    device = dict(uuid="GPU-target-test", name="Tesla V100", memory_mib=16384, driver="test")
    directory = tmp_path / target.target_tag(device)
    calls = []
    identity = {"device": device, "protocol": {"repeat": 100}}
    monkeypatch.setattr(target, "target_device", lambda _: device)
    monkeypatch.setattr(target, "build_identity", lambda *a: identity)
    monkeypatch.setattr(target, "exclusive", lambda *a: True)
    def calibration(*a):
        calls.append("calibration")
        path = directory / "calibration/base_profile.json"
        target.write(path, dict(global_memory_bytes=16 * 1024**3))
        return [path]
    def guarded(command, uuid, log, quarantine):
        assert calls[0] == "calibration"
        dest = Path(command[command.index("--output-dir") + 1])
        calls.append(dest.name)
        for name in ("acceptance.json", "summary.json"):
            target.write(dest / name, {"status": "complete"})
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(target, "calibrate", calibration)
    monkeypatch.setattr(target, "guarded", guarded)
    monkeypatch.setattr(target, "run_plans", lambda *a: [])
    target.run_target(args)
    assert calls == ["calibration", *target.GROUPS, "composed"]
    assert target.read(directory / "campaign.json")["status"] == "resolved-declared-campaign"
    args.resume = True
    target.run_target(args)
    assert calls == ["calibration", *target.GROUPS, "composed"]
    assert target.read(directory / "report.json")["full_migration_qualified"] is False
    identity["protocol"]["repeat"] = 101
    with pytest.raises(ValueError, match="inputs changed"):
        target.run_target(args)


def test_acceptance_keeps_calibration_and_external_paths_explicit(tmp_path):
    args = target.parse_args(["--mode", "run", "--output-dir", str(tmp_path),
        "--gpuntt-binary", str(tmp_path / "gpuntt with spaces"), "--fht-python", sys.executable])
    command = target.acceptance_command(args, tmp_path / "target", "crypto")
    assert command[command.index("--gpuntt-binary") + 1] == str(tmp_path / "gpuntt with spaces")
    assert "--resume" in command and "--stage-checkpoint" in command
    assert not any("/tmp/gpuntt_merge_gap_bench_a100" in str(v) for v in command)


def test_monitor_rejects_post_command_contamination_and_preserves_evidence(tmp_path, monkeypatch):
    phase = tmp_path / "phase"
    target.write(phase / "trial.json", {"may_be_contaminated": True})
    class Process:
        pid = 12345
        def wait(self, timeout=None):
            return 0
        def poll(self):
            return 0
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(target, "exclusive", lambda *a: True)
    clients = iter([[], [777]])
    monkeypatch.setattr(target, "foreign_clients", lambda *a: next(clients))
    with pytest.raises(RuntimeError, match="after command"):
        target.guarded(["mock-benchmark"], "GPU-test", tmp_path / "log.txt", phase)
    assert not phase.exists()
    archives = list(tmp_path.glob("phase.excluded-*"))
    assert len(archives) == 1 and target.read(archives[0] / "trial.json")["may_be_contaminated"]
    assert target.read(archives[0] / "excluded.json")["reuse"] is False
