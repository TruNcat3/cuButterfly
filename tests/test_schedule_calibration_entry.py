import json
import pathlib
import re
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import calibrate_local_hardware as local_calibration


def _hardware_profile():
    return {
        "device": "Test GPU",
        "compute_capability": "8.0",
        "global_memory_bytes": 4096,
        "capabilities": {},
    }


def _schedule_probe_output():
    return {
        "schema": "cubutterfly-schedule-profile-v1",
        "hardware": {
            "device": "Test GPU",
            "compute_capability": "8.0",
            "global_memory_bytes": 4096,
            "sm_count": 20,
        },
        "protocol": {"warmup": 10, "repeat": 100, "trials": 3, "launch_mode": "direct"},
        "rows": [
            {
                "threads": 64,
                "grid_ctas": grid,
                "trial_kernel_ms": [0.004 + grid * 0.00001] * 3,
                "median_kernel_ms": 0.004 + grid * 0.00001,
            }
            for grid in (1, 20, 160)
        ],
    }


def test_cmake_installs_schedule_probe_and_cost_helper():
    cmake = (ROOT / "CMakeLists.txt").read_text()
    assert "add_executable(cubutterfly_schedule_microbench apps/cubutterfly_schedule_microbench.cu)" in cmake
    assert "target_link_libraries(cubutterfly_schedule_microbench PRIVATE CUDA::cudart nlohmann_json::nlohmann_json)" in cmake

    target_install = re.search(r"install\(TARGETS[^)]*cubutterfly_schedule_microbench[^)]*\)", cmake, re.DOTALL)
    assert target_install and "RUNTIME DESTINATION \"${CMAKE_INSTALL_BINDIR}\"" in target_install.group(0)

    helper_install = re.search(r"install\((?:PROGRAMS|FILES)[^)]*scripts/schedule_cost_model\.py[^)]*\)", cmake, re.DOTALL)
    assert helper_install and "${CMAKE_INSTALL_BINDIR}" in helper_install.group(0)


@pytest.mark.parametrize("stage_mode", ["skip", "full"])
def test_staged_entry_runs_schedule_probe_once_then_reuses_identity_cache(tmpdir, monkeypatch, stage_mode):
    tmp_path = pathlib.Path(str(tmpdir))
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    schedule_binary = build_dir / "cubutterfly_schedule_microbench"
    schedule_binary.write_bytes(b"schedule-probe-test-binary-v1")
    for name in ("cubutterfly_stage_microbench", "cubutterfly_pipeline_microbench"):
        (build_dir / name).write_bytes(name.encode())
    profile_dir = tmp_path / "profile"
    output_dir = tmp_path / "output"
    workloads = tmp_path / "workloads.json"
    workloads.write_text(json.dumps({
        "schema": "cubutterfly-install-search-v1",
        "scope": "schedule-entry-test",
        "workloads": [{"operator": "fft", "precision": "fp32", "logN": 8, "batch": 1}],
    }))

    schedule_runs = []
    exclusive_calls = []
    schedule_calls = []
    pipeline_calls = []
    profile = _hardware_profile()
    raw_probe = _schedule_probe_output()

    def fake_run(command, *, check=True):
        command = [str(value) for value in command]
        if any(value.endswith("initialize_hardware_profile.py") for value in command):
            profile_dir.mkdir(parents=True, exist_ok=True)
            (profile_dir / "hardware_profile.json").write_text(json.dumps(profile))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if any(value.endswith("check_hardware_profile.py") for value in command):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if pathlib.Path(command[0]).name == "cubutterfly_bench" and command[1:] == ["--device-identity"]:
            return SimpleNamespace(
                returncode=0,
                stdout="device,compute_capability,global_memory_bytes\nTest GPU,8.0,4096\n",
                stderr="",
            )
        if pathlib.Path(command[0]).name == schedule_binary.name:
            schedule_runs.append(command)
            return SimpleNamespace(returncode=0, stdout=json.dumps(raw_probe), stderr="")
        if pathlib.Path(command[0]).name == "cubutterfly_plan_probe" and command[1:] == ["--hardware"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps(raw_probe["hardware"]), stderr="")
        if pathlib.Path(command[0]).name == "cubutterfly_bench" and command[1:] == ["--list-processing-units"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if any(value.endswith("stage_cost_model.py") for value in command):
            output = pathlib.Path(command[command.index("--output") + 1])
            output.write_text(json.dumps({"schema": "test-cost-model"}))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected calibration command: {command}")

    def wrapped_schedule_calibration(binary, path, target_profile, run, exclusive):
        schedule_calls.append((pathlib.Path(binary), pathlib.Path(path)))
        return real_schedule_calibration(binary, path, target_profile, run, exclusive)

    real_schedule_calibration = local_calibration.calibrate_scheduling
    monkeypatch.setattr(local_calibration, "run", fake_run)
    monkeypatch.setattr(local_calibration, "require_exclusive_gpu", lambda: exclusive_calls.append("exclusive"))
    monkeypatch.setattr(local_calibration, "rank_model", lambda profile, output: 0)
    monkeypatch.setattr(local_calibration, "run_operator_calibration",
                        lambda args, output, cufftdx: {"status": "complete", "candidates": [], "operator_families": []})
    monkeypatch.setattr(local_calibration, "promote", lambda *args, **kwargs: 0)
    monkeypatch.setattr(local_calibration, "default_path", lambda: tmp_path / "registry.json")
    monkeypatch.setattr(local_calibration, "calibrate_scheduling", wrapped_schedule_calibration)
    def pipeline_calibration(binary, path, target_profile, run, exclusive):
        pipeline_calls.append(pathlib.Path(binary).name)
        return {"source": "minimal-batch-pipeline"}
    monkeypatch.setattr(local_calibration, "calibrate_pipeline_scheduling", pipeline_calibration)
    monkeypatch.setattr(local_calibration, "runtime_candidates", lambda *args: [])
    monkeypatch.setattr(local_calibration, "calibrate_stage_services", lambda *args, **kwargs: {"curves": {}})

    args = [
        "--build-dir", str(build_dir),
        "--profile-dir", str(profile_dir),
        "--output-dir", str(output_dir),
        "--search-workloads", str(workloads),
        "--cost-model", "staged", "--stage-calibration", stage_mode,
    ]
    monkeypatch.setattr(sys, "argv", ["calibrate_local_hardware.py", *args])
    assert local_calibration.main() == 0
    assert local_calibration.main() == 0

    schedule_path = output_dir / "schedule_calibration.json"
    assert schedule_path.is_file()
    assert len(schedule_calls) == 2
    assert schedule_calls[0] == (schedule_binary, schedule_path)
    assert len(schedule_runs) == 1
    assert json.loads(schedule_path.read_text())["summary"]["source"] == "measured direct launches of minimal CTA kernel"
    persisted_profile = json.loads((output_dir / "calibration_device.json").read_text())
    assert persisted_profile["scheduling"]["hardware"] == raw_probe["hardware"]
    assert pipeline_calls == (["cubutterfly_pipeline_microbench"] * 2 if stage_mode == "full" else [])
    if stage_mode == "full":
        assert persisted_profile["pipeline_scheduling"]["source"] == "minimal-batch-pipeline"
    assert exclusive_calls
