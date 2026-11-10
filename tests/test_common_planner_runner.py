"""CPU-only contracts for the bounded GPU command runner."""

import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("common_planner", ROOT / "scripts/check_common_planner.py")
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def row():
    return {"device": "fake", "compute_capability": "8.0", "correct": "1", "verified_batches": "1",
            "mapping_json": "{}", "runtime_fingerprint": "cpu-fake", "max_error": "0",
            "backend": "shared-iterative", "execution_groups_json": json.dumps([
                {"compiler_resources_known": True}])}


def test_matrix_is_bounded_and_cli_flags_exist():
    assert len(RUNNER.cases("correctness")) == 27
    assert len(RUNNER.cases("performance")) == 7
    assert len(RUNNER.cases("all")) == 34
    for case in RUNNER.cases("all") + RUNNER.cases("buffering"):
        command = RUNNER.command(case, Path("build"))
        source = (ROOT / "apps" / ("cuntt_bench.cpp" if case.operator == "ntt" else "cubutterfly_bench.cpp")).read_text()
        assert all(f'arg == "{argument}"' in source for argument in command if argument.startswith("--"))
        assert "--verify-batches" in command
        if case.precision == "word32":
            assert command[command.index("--modulus") + 1] == "998244353"


def test_every_batch_and_compiled_shared_resources_are_required():
    case = RUNNER.cases("correctness")[0]
    RUNNER.check_row(case, row())
    for override in ({"verified_batches": "0"}, {"correct": "0"},
                     {"execution_groups_json": '[{"compiler_resources_known": false}]'}):
        with pytest.raises(ValueError):
            RUNNER.check_row(case, {**row(), **override})


def test_dry_run_never_executes(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["runner", "--build", "missing-build", "--suite", "all", "--case-limit", "1"])
    monkeypatch.setattr(RUNNER.subprocess, "run", lambda *args, **kwargs: pytest.fail("GPU command in dry-run"))
    assert RUNNER.main() == 0
    assert "CUBUTTERFLY_COMPILE_MODE=research" in capsys.readouterr().out


def test_resume_uses_case_record_and_invalidates_binary_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBUTTERFLY_HARDWARE_MODEL", "/cpu-marker-hardware-model")
    build = tmp_path / "build"
    build.mkdir()
    binary = build / "cubutterfly_bench"
    binary.write_text("fake executable identity")
    output = tmp_path / "output"
    calls = []
    profile = tmp_path / "profile"
    profile.mkdir()
    metrics = {**row(), "selection_reason": f'cpu-fake; profile="{profile}"'}
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(metrics))
    writer.writeheader()
    writer.writerow(metrics)

    def fake_run(command, env, directory, *args):
        assert env["CUBUTTERFLY_HARDWARE_MODEL"] == "/cpu-marker-hardware-model"
        assert args[-1] == ("/usr/lib/xorg/Xorg",)
        calls.append(command)
        (directory / "stdout.txt").write_text(stream.getvalue())
        (directory / "stderr.txt").write_text("")
        return subprocess.CompletedProcess(command, 0, stream.getvalue(), "")

    monkeypatch.setattr(RUNNER, "guarded_run", fake_run)
    monkeypatch.setattr(RUNNER.signal, "signal", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["runner", "--build", str(build), "--run", "--resume",
                                      "--gpu-uuid", "GPU-00000000-0000-0000-0000-000000000006",
                                      "--idle-graphics-name", "/usr/lib/xorg/Xorg",
                                      "--output", str(output), "--case-limit", "1"])
    assert RUNNER.main() == 0
    assert RUNNER.main() == 0
    assert len(calls) == 1
    binary.write_text("updated fake executable")
    assert RUNNER.main() == 0
    assert len(calls) == 2
    assert (output / RUNNER.cases("correctness")[0].name / "stdout.txt").is_file()
    result = json.loads((output / RUNNER.cases("correctness")[0].name / "result.json").read_text())
    assert result["selection_provenance"]["source"] == "unknown"
    assert result["design_point"]["stage_partition"] is None
    assert result["idle_graphics_names"] == ["/usr/lib/xorg/Xorg"]
    assert result["planner_input_identities"]["idle_graphics_names"] == ["/usr/lib/xorg/Xorg"]


def snapshot(pids=()):
    return dict(pids=list(pids), gpu_utilization=0, memory_utilization=0, memory_used_mib=0)


def test_guard_does_not_spawn_on_busy_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(RUNNER, "gpu_snapshot", lambda uuid: snapshot([987]))
    monkeypatch.setattr(RUNNER.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("spawned on busy GPU"))
    with pytest.raises(RUNNER.GPUInterruption) as error:
        RUNNER.guarded_run(["fake"], {}, tmp_path, "fake-uuid", 1, 0.01, 64)
    assert error.value.status == "waiting-for-gpu"


@pytest.mark.parametrize("process_type", ["G", "C", "C+G"])
def test_explicit_graphics_allowlist_never_ignores_compute_work(tmp_path, monkeypatch, process_type):
    name = "/usr/lib/xorg/Xorg"
    initial = dict(snapshot([2016]), processes=[dict(pid=2016, type="G", name=name)])
    observed = dict(snapshot([123, 2016]), processes=[dict(pid=2016, type=process_type, name=name)])
    observations = iter([initial, observed])
    monkeypatch.setattr(RUNNER, "gpu_snapshot", lambda uuid: next(observations))
    monkeypatch.setattr(RUNNER, "descendants", lambda pid: set())
    killed = []
    monkeypatch.setattr(RUNNER.os, "killpg", lambda pid, sig: killed.append(pid))

    class Child:
        pid, returncode = 123, 0
        def poll(self):
            return 0
        def wait(self, timeout):
            return 0

    monkeypatch.setattr(RUNNER.subprocess, "Popen", lambda *args, **kwargs: Child())
    if process_type == "G":
        assert RUNNER.guarded_run(["fake"], {}, tmp_path, "fake-uuid", 1, 0.01, 64, (name,)).returncode == 0
        assert not killed
    else:
        with pytest.raises(RUNNER.GPUInterruption) as error:
            RUNNER.guarded_run(["fake"], {}, tmp_path, "fake-uuid", 1, 0.01, 64, (name,))
        assert error.value.snapshot["foreign_pids"] == [2016]
        assert killed == [123, 123]


def test_graphics_process_is_not_allowed_by_default(tmp_path, monkeypatch):
    observed = dict(snapshot([2016]), processes=[dict(pid=2016, type="G", name="/usr/lib/xorg/Xorg")])
    monkeypatch.setattr(RUNNER, "gpu_snapshot", lambda uuid: observed)
    monkeypatch.setattr(RUNNER.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("spawned without explicit display permission"))
    with pytest.raises(RUNNER.GPUInterruption, match="waiting-for-gpu"):
        RUNNER.guarded_run(["fake"], {}, tmp_path, "fake-uuid", 1, 0.01, 64)


def test_guard_foreign_work_stops_only_owned_group(tmp_path, monkeypatch):
    observations = iter([snapshot(), snapshot([123, 124, 987])])
    monkeypatch.setattr(RUNNER, "gpu_snapshot", lambda uuid: next(observations))
    monkeypatch.setattr(RUNNER, "descendants", lambda pid: {124})
    killed = []
    monkeypatch.setattr(RUNNER.os, "killpg", lambda pid, sig: killed.append((pid, sig)))

    class Child:
        pid = 123
        def wait(self, timeout):
            return 0

    def popen(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        return Child()

    monkeypatch.setattr(RUNNER.subprocess, "Popen", popen)
    with pytest.raises(RUNNER.GPUInterruption) as error:
        RUNNER.guarded_run(["fake"], {}, tmp_path, "fake-uuid", 1, 0.01, 64)
    assert error.value.status == "interrupted-by-foreign-gpu-work"
    assert error.value.snapshot["foreign_pids"] == [987]
    assert killed == [(123, RUNNER.signal.SIGTERM), (123, RUNNER.signal.SIGKILL)]


@pytest.mark.parametrize("failure", [RUNNER.DriverStopped("SIGTERM"), RuntimeError("GPU observation failed")])
def test_guard_cleans_owned_process_on_stop_or_observation_failure(tmp_path, monkeypatch, failure):
    observations = iter([snapshot(), failure])
    def observe(uuid):
        value = next(observations)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(RUNNER, "gpu_snapshot", observe)
    killed = []
    monkeypatch.setattr(RUNNER.os, "killpg", lambda pid, sig: killed.append(pid))
    class Child:
        pid = 123
        def wait(self, timeout):
            return 0
    monkeypatch.setattr(RUNNER.subprocess, "Popen", lambda *args, **kwargs: Child())
    with pytest.raises(type(failure)):
        RUNNER.guarded_run(["fake"], {}, tmp_path, "fake-uuid", 1, 0.01, 64)
    assert killed == [123, 123]


def test_uuid_required_and_interruption_checkpoint_is_retryable(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(sys, "argv", ["runner", "--build", str(tmp_path), "--run"])
    with pytest.raises(SystemExit):
        RUNNER.main()
    monkeypatch.setattr(RUNNER.signal, "signal", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["runner", "--build", str(tmp_path), "--run", "--case-limit", "1",
                                      "--gpu-uuid", "GPU-00000000-0000-0000-0000-000000000006",
                                      "--output", str(tmp_path)])
    def interrupted(*args):
        raise RUNNER.GPUInterruption("interrupted-by-foreign-gpu-work", snapshot([987]))
    monkeypatch.setattr(RUNNER, "guarded_run", interrupted)
    assert RUNNER.main() == 1
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["retryable"] and status["remaining"] == 1 and status["updated_unix"] > 0
    assert status["status"] == "interrupted-by-foreign-gpu-work"
    assert RUNNER.cases("correctness")[0].name in (tmp_path / "summary.json").read_text()


@pytest.mark.parametrize("confidence,mode,expected", [
    ("validated-static-model", "validated-static-model", "static-model-estimate"),
    ("unmeasured-static-model", "validated-static-model", "static-model-estimate"),
    ("hardware-rate-prior", "mixed-static-model-and-hardware-rate-prior", "rate-prior"),
    ("unmeasured-feasible", "unavailable", "unmeasured-fallback"),
    (None, "mixed-static-model-and-hardware-rate-prior", "unknown"),
])
def test_model_sources_are_explicit_and_estimates_are_not_measurements(confidence, mode, expected):
    metrics = dict(auto_select="1", selection_confidence=confidence,
                   selection_reason=f'static ranking; mode={mode}; profile="/profiles/a100"; finalists=4')
    result = RUNNER.selection_provenance(metrics, "cuButterfly hardware resources: live-device-fallback\n")
    assert result["source"] == expected
    assert result["exact_workload_measurement_used"] is (None if expected == "unknown" else False)
    assert result["profile_directory"] == "/profiles/a100"
    assert result["hardware_resource_sources"] == ["live-device-fallback"]


def test_design_point_retains_actual_group_geometry_without_inventing_unknowns():
    metrics = dict(mapping_json=json.dumps(dict(backend="shared-iterative", stage_partition=[4, 4],
                                               local_stages=4, stage_overlap=False)),
                   stage_overlap="1", batch_tile_count="4", execution_groups_json=json.dumps([
                       dict(first_stage=0, stage_count=8, grid_ctas=16, threads=128, core="radix2",
                            prefetch_depth=0, local_stage_partition=[4, 4])]))
    result = RUNNER.design_point(metrics)
    assert result["logical_segment_count"] == 2 and result["physical_group_count"] == 1
    assert result["physical_stage_partition"] == [8]
    assert result["groups"][0]["grid_ctas"] == 16 and result["groups"][0]["threads"] == 128
    assert result["groups"][0]["prefetch_depth"] == 0
    assert result["overlap"]["stage_overlap"] is False
    assert result["overlap"]["batch_tile_count"] == 4
    assert result["tile"]["flow_tile_log_n"] is None and result["measured_residency"] is None


@pytest.mark.parametrize("filename", ["hardware_profile.json", "calibration_device.json", "cost_model.json"])
def test_explicit_profile_performance_inputs_invalidate_but_unrelated_files_do_not(tmp_path, filename):
    hardware = tmp_path / "hardware_model.json"
    hardware.write_text("hardware identity")
    performance = tmp_path / filename
    performance.write_text("first performance inputs")
    env = {"CUBUTTERFLY_HARDWARE_MODEL": str(hardware)}
    before = RUNNER.planner_input_identities(env)
    (tmp_path / "unrelated-notes.txt").write_text("unrelated")
    assert RUNNER.planner_input_identities(env) == before
    performance.write_text("changed performance inputs")
    assert RUNNER.planner_input_identities(env) != before


def test_missing_profile_input_and_helper_creation_invalidate(tmp_path):
    env = {"CUBUTTERFLY_HARDWARE_MODEL": str(tmp_path / "hardware_model.json")}
    before = RUNNER.planner_input_identities(env, tmp_path)
    assert before["CUBUTTERFLY_HARDWARE_MODEL"]["profile_siblings"]["cost_model.json"] == {"state": "missing"}
    (tmp_path / "cost_model.json").write_text("new cost model")
    assert RUNNER.planner_input_identities(env, tmp_path) != before
    before = RUNNER.planner_input_identities(env, tmp_path)
    (tmp_path / "rank_static_candidates.py").write_text("new ranker")
    assert RUNNER.planner_input_identities(env, tmp_path) != before


def test_actual_reported_profile_files_are_checked_before_resume(tmp_path):
    provenance = RUNNER.selection_provenance(
        {"selection_reason": f'static ranking; profile="{tmp_path}"'}, "")
    previous = dict(selection_provenance=provenance,
                    consumed_profile_identity=RUNNER.consumed_profile_identity(provenance))
    assert RUNNER.profile_resume_matches(previous)
    (tmp_path / "unrelated.txt").write_text("unrelated")
    assert RUNNER.profile_resume_matches(previous)
    (tmp_path / "hardware_profile.json").write_text("new profile rates")
    assert not RUNNER.profile_resume_matches(previous)
    assert not RUNNER.profile_resume_matches({"selection_provenance": {"profile_directory": None}})


def test_buffering_suite_uses_three_explicit_existing_mapping_requests():
    selected = RUNNER.cases("buffering")
    assert len(selected) == 3 and len({case.name for case in selected}) == 3
    expected = [(20, [10, 10], 2, 2), (20, [10, 10], 4, 0), (18, [9, 9], 4, 2)]
    for case, (log_n, partition, columns, depth) in zip(selected, expected):
        command = RUNNER.command(case, Path("build"))
        mapping = json.loads(command[command.index("--mapping-json") + 1])
        assert command[command.index("--warmup") + 1] == "5"
        assert command[command.index("--repeat") + 1] == "20"
        assert case.log_n == log_n and case.batch == 4 and case.precision == "fp64"
        assert mapping == dict(backend="factor-streamed", fft_core="cufftdx-block",
                               factor_partition=partition, factor_columns=columns,
                               prefetch_depth=depth, factor_ept=8, data_tiles_per_cta=4,
                               cross_twiddle="recurrence", shared_layout="writer-aligned")
        assert "--auto-select" not in command and "--backend" not in command
        metrics = {**row(), "verified_batches": "4", "mapping_json": json.dumps(mapping)}
        RUNNER.check_row(case, metrics)
        with pytest.raises(ValueError, match="compiled group resources"):
            RUNNER.check_row(case, {**metrics, "execution_groups_json": '[{}]'})
        changed = {**mapping, "prefetch_depth": 1}
        with pytest.raises(ValueError, match="parameters were changed"):
            RUNNER.check_row(case, {**metrics, "mapping_json": json.dumps(changed)})
