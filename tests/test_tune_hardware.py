import json
import pathlib
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import tune_hardware


def _manifest(root, *, status="incomplete", search_budget=64, workload_hash="workload-a"):
    path = root / "migration_manifest.json"
    path.write_text(json.dumps({
        "schema": "cubutterfly-hardware-migration-v1",
        "device": {"device": "Test GPU", "compute_capability": "8.0",
                    "global_memory_bytes": 80 * 1024**3, "source": "test"},
        "build_dir": str(root / "build"),
        "profile_dir": str(root / "profile"),
        "output_dir": str(root),
        "compile_mode": "research",
        "workloads": {"path": str(root / "workloads.json"), "sha256": workload_hash,
                       "workload_count": 1},
        "protocol": {"search_budget": search_budget, "compile_seconds": 600.0},
        "status": status,
    }))
    return path


class _Process:
    def __init__(self, callback=None, *, returncode=0, stdout="", stderr=""):
        self.pid = 12345
        self.returncode = returncode
        self._callback = callback
        self._stdout = stdout
        self._stderr = stderr
        self._polled = False

    def poll(self):
        if not self._polled:
            self._polled = True
            if self._callback:
                self._callback()
        return self.returncode

    def communicate(self, **kwargs):
        return self._stdout, self._stderr

    def wait(self, timeout=None):
        return self.returncode


def _fake_popen(monkeypatch, calls, callback=None, *, returncode=0, stdout="", stderr=""):
    def fake(command, **kwargs):
        calls.append((command, kwargs))
        return _Process(callback, returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(tune_hardware.subprocess, "Popen", fake)


def test_defaults_and_dry_run_persist_next_round_command(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, search_budget=64)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    args = tune_hardware.parse_args(["--resume-from", str(manifest)])
    assert args.time_budget == tune_hardware.DEFAULT_TIME_BUDGET
    assert args.round_budget == tune_hardware.DEFAULT_ROUND_BUDGET
    assert args.max_rounds == tune_hardware.DEFAULT_MAX_ROUNDS
    assert args.poll_seconds == tune_hardware.DEFAULT_POLL_SECONDS

    def no_process(*args, **kwargs):
        raise AssertionError("dry-run must not launch the calibrator")

    monkeypatch.setattr(tune_hardware.subprocess, "Popen", no_process)
    assert tune_hardware.main(["--resume-from", str(manifest), "--dry-run"]) == 0
    state = json.loads((tmp_path / "tuner_state.json").read_text())
    assert state["status"] == "planned"
    assert state["planned_search_budget"] == 80
    assert state["planned_command"][-1] == "80"
    assert "--search-strategy" in state["planned_command"]
    assert state["planned_command"][state["planned_command"].index("--search-strategy") + 1] == "evolutionary"
    assert state["planned_command"][state["planned_command"].index("--cost-model") + 1] == "staged"


def test_warning_rounds_are_valid_and_budget_increases(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, status="incomplete", search_budget=10)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(tune_hardware, "gpu_is_busy", lambda target: False)
    calls = []

    def mark_incomplete():
        document = json.loads(manifest.read_text())
        document["status"] = "incomplete"
        manifest.write_text(json.dumps(document))

    _fake_popen(monkeypatch, calls, mark_incomplete)
    result = tune_hardware.main([
        "--resume-from", str(manifest), "--time-budget", "30", "--round-budget", "4", "--max-rounds", "2",
    ])
    assert result == 0
    assert [command[command.index("--search-budget") + 1] for command, _ in calls] == ["14", "18"]
    state = json.loads((tmp_path / "tuner_state.json").read_text())
    assert state["completed_rounds"] == 2
    assert state["status"] == "complete-with-warning"
    assert [round_["status"] for round_ in state["rounds"]] == ["warning", "warning"]


def test_busy_gpu_pauses_without_launching_or_killing(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(tune_hardware, "gpu_is_busy", lambda target: True)
    calls = []
    _fake_popen(monkeypatch, calls)
    assert tune_hardware.main(["--resume-from", str(manifest), "--time-budget", "20"]) == 0
    assert calls == []
    state = json.loads((tmp_path / "tuner_state.json").read_text())
    assert state["status"] == "paused"
    assert state["pause_reason"] == "gpu-busy"
    assert state["completed_rounds"] == 0
    assert state["next_round"] == 1


def test_watch_polls_busy_gpu_until_available_and_passes_uuid_to_child(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, status="complete")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    busy = iter([True, False])
    monkeypatch.setattr(tune_hardware, "gpu_is_busy", lambda target: next(busy))
    sleeps = []
    monkeypatch.setattr(tune_hardware.time, "sleep", lambda seconds: sleeps.append(seconds))
    calls = []
    _fake_popen(monkeypatch, calls)
    assert tune_hardware.main([
        "--resume-from", str(manifest), "--gpu", "GPU-test", "--watch", "--poll-seconds", "15",
        "--time-budget", "30", "--max-rounds", "1",
    ]) == 0
    assert sleeps == [15]
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-test"
    state = json.loads((tmp_path / "tuner_state.json").read_text())
    assert state["status"] == "complete"
    assert state["completed_rounds"] == 1


def test_child_busy_result_is_paused_and_nonbusy_failure_is_distinct(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(tune_hardware, "gpu_is_busy", lambda target: False)
    monkeypatch.setattr(tune_hardware, "run_calibration", lambda *args: {
        "returncode": 1, "stdout": "", "stderr": "target GPU is not exclusive; active compute processes: pid",
        "failed": False, "timed_out": False,
    })
    assert tune_hardware.main(["--resume-from", str(manifest)]) == 0
    state = json.loads((tmp_path / "tuner_state.json").read_text())
    assert state["status"] == "paused"
    assert state["pause_reason"] == "gpu-busy"

    state["status"] = "paused"
    (tmp_path / "tuner_state.json").write_text(json.dumps(state))
    monkeypatch.setattr(tune_hardware, "run_calibration", lambda *args: {
        "returncode": 17, "stdout": "", "stderr": "compile failed",
        "failed": False, "timed_out": False,
    })
    assert tune_hardware.main(["--resume-from", str(manifest)]) == 1
    state = json.loads((tmp_path / "tuner_state.json").read_text())
    assert state["status"] == "failed"
    assert state["rounds"][-1]["status"] == "failed"


def test_deadline_interrupts_only_owned_process_group(tmp_path, monkeypatch):
    calls = []

    class HungProcess(_Process):
        def poll(self):
            return None

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            return None

    def fake_popen(command, **kwargs):
        return HungProcess(stdout="checkpoint", stderr="")

    monkeypatch.setattr(tune_hardware.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tune_hardware, "_remaining", lambda deadline: -1.0)
    monkeypatch.setattr(tune_hardware.os, "killpg", lambda pid, sig: calls.append(("killpg", pid, sig)))
    result = tune_hardware.run_calibration(["calibrator"], {}, 0.0)
    assert result["timed_out"] is True
    assert result["stdout"] == "checkpoint"
    assert ("killpg", 12345, tune_hardware.signal.SIGINT) in calls


def test_verbose_child_does_not_block_on_full_output_pipes():
    import os
    import time
    result = tune_hardware.run_calibration([
        sys.executable, "-c", "import sys; sys.stdout.write('x'*200000); sys.stderr.write('y'*200000)"
    ], os.environ.copy(), time.monotonic() + 10)
    assert not result["timed_out"] and result["returncode"] == 0
    assert len(result["stdout"]) == len(result["stderr"]) == 200000


def test_resume_uses_saved_round_and_rejects_identity_changes(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, status="incomplete", search_budget=10)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(tune_hardware, "gpu_is_busy", lambda target: False)
    assert tune_hardware.main([
        "--resume-from", str(manifest), "--dry-run", "--round-budget", "4", "--max-rounds", "3", "--time-budget", "30",
    ]) == 0
    state_path = tmp_path / "tuner_state.json"
    state = json.loads(state_path.read_text())
    state.update(status="paused", completed_rounds=1, next_round=2)
    state_path.write_text(json.dumps(state))

    calls = []

    def mark_complete():
        document = json.loads(manifest.read_text())
        document["status"] = "complete"
        manifest.write_text(json.dumps(document))

    _fake_popen(monkeypatch, calls, mark_complete)
    assert tune_hardware.main([
        "--resume-from", str(manifest), "--round-budget", "4", "--max-rounds", "3", "--time-budget", "30",
    ]) == 0
    assert calls[0][0][calls[0][0].index("--search-budget") + 1] == "18"
    assert json.loads(state_path.read_text())["completed_rounds"] == 2

    document = json.loads(manifest.read_text())
    document["workloads"]["sha256"] = "changed"
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="workload configuration changed"):
        tune_hardware.main([
            "--resume-from", str(manifest), "--dry-run", "--round-budget", "4", "--max-rounds", "3", "--time-budget", "30",
        ])


def test_gpu_conflict_is_rejected_when_target_is_detectable(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        tune_hardware.main(["--resume-from", str(manifest), "--gpu", "3", "--dry-run"])
