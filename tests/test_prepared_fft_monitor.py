"""CPU-only lifecycle tests: no GPU query, compilation, or measurement."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("prepared_fft_monitor", ROOT / "scripts/watch_prepared_fft_experiment.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
import run_fft_intra_stage_experiment as runner


def prepared():
    return dict(status="failed", preparation_complete=True, error_type="RuntimeError",
        error="GPU is busy; no experiment started: test", modules=[{}] * 20,
        identity=dict(gpu=M.GPU_UUID, python=sys.executable, library="unused", nvcc="unused",
                      mathdx="unused", jit_cache="unused", compile_policy="research",
                      chain=8, repeats=20, trials=5, warmup=5, sm=80),
        prepared_cases=[dict(name=f"case{i}") for i in range(4)], cases=[])


def publish(path, report):
    path.write_text(json.dumps(report))


def snapshot(pids=()):
    return dict(pids=list(pids), processes=[], gpu_utilization=0, memory_utilization=0,
                memory_used_mib=18, gpu_uuid=M.GPU_UUID)


@pytest.fixture
def paths(tmp_path):
    summary, control = tmp_path / "measurement" / "summary.json", tmp_path / "control"
    summary.parent.mkdir()
    control.mkdir()
    publish(summary, prepared())
    return summary, control


def test_initial_busy_is_retryable_but_unknown_failure_stops():
    report = prepared()
    state = M.campaign_state(report)
    assert state["status"] == "waiting-for-gpu" and state["retryable"]
    for error, error_type in (("nonfinite output", "RuntimeError"),
                              ("specialization compilation failed", "RuntimeError"),
                              ("resume requires identical build", "RuntimeError"),
                              ("unexpected exception", "ValueError")):
        state = M.campaign_state(dict(report, error=error, error_type=error_type))
        assert state["status"] == "failed" and not state["retryable"]
    with pytest.raises(ValueError):
        M.campaign_state(dict(report, preparation_complete=False))
    with pytest.raises(ValueError):
        M.campaign_state(dict(report, identity=dict(report["identity"], gpu="wrong")))
    combined = dict(report, error_type="ForeignGPUInterruption", cases=[dict(failures=["nonfinite output"])])
    assert M.campaign_state(combined)["retryable"] is False
    rejected = M.campaign_state(dict(report, status="capabilities-rejected", error="shared memory exceeds limit"))
    assert rejected["status"] == "failed" and not rejected["retryable"]
    assert rejected["source_status"] == "capabilities-rejected"


def test_busy_idle_run_reuses_watcher_and_preserves_original(paths, monkeypatch):
    summary, control = paths
    original = summary.read_bytes()
    monkeypatch.setattr(M, "validate_cpu", lambda *args: None)
    observations = iter([snapshot([987654]), snapshot()])
    monkeypatch.setattr(M.idle, "gpu_snapshot", lambda uuid: next(observations, snapshot()))
    real_watch = M.idle.watch
    observed = []
    def fast_watch(root, uuid, command, **kwargs):
        observed.append((uuid, command, kwargs))
        child = [sys.executable, "-c",
                 'from pathlib import Path; Path("status.json").write_text(\'{"status":"complete","remaining":0}\')']
        return real_watch(root, uuid, child, poll_seconds=.005, idle_seconds=.01)
    monkeypatch.setattr(M.idle, "watch", fast_watch)
    assert M.monitor(summary, control) == 0
    assert summary.read_bytes() == original
    assert (control / "initial_summary.json").read_bytes() == original
    assert observed[0][0] == M.GPU_UUID
    assert observed[0][2] == dict(poll_seconds=10, idle_seconds=30, max_idle_memory_mib=64, idle_graphics_names=())
    events = (control / "monitor_events.jsonl").read_text()
    assert "waiting-for-idle" in events and "confirming-idle" in events and "finished" in events


def test_monitor_and_driver_singletons_never_overwrite(paths):
    summary, control = paths
    (control / "status.json").write_text("preserve")
    with (control / "adapter.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            M.monitor(summary, control)
    with (control / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert M.drive(summary, control) == 73
    assert (control / "status.json").read_text() == "preserve"


def test_driver_foreign_interruption_preserves_trials_then_resumes(paths, monkeypatch):
    summary, control = paths
    report = prepared()
    report["cases"] = [dict(name="case0", status="running", providers={"factor": dict(trial_us=[12.5])})]
    publish(summary, report)
    monkeypatch.setattr(M.idle, "gpu_snapshot", lambda uuid: snapshot([987654321]))
    def interrupted(argv):
        try:
            time.sleep(.2)
        except M.ForeignGPUInterruption as error:
            report.update(status="failed", error_type=type(error).__name__, error=str(error))
            publish(summary, report)
            return 1
        pytest.fail("foreign-work signal was not delivered")
    monkeypatch.setattr(runner, "main", interrupted)
    assert M.drive(summary, control, poll_seconds=.01) == 1
    state = json.loads((control / "status.json").read_text())
    assert state["status"] == "interrupted-by-foreign-gpu-work" and state["retryable"]
    assert M.read_summary(summary)["cases"][0]["providers"]["factor"]["trial_us"] == [12.5]
    monkeypatch.setattr(M.idle, "gpu_snapshot", lambda uuid: snapshot())
    def completed(argv):
        assert M.read_summary(summary)["cases"][0]["providers"]["factor"]["trial_us"] == [12.5]
        report.update(status=M.COMPLETE, cases=[dict(name=f"case{i}", status=M.COMPLETE) for i in range(4)])
        publish(summary, report)
        return 0
    monkeypatch.setattr(runner, "main", completed)
    assert M.drive(summary, control, poll_seconds=.01) == 0
    assert M.read_summary(control / "status.json")["remaining"] == 0


def test_real_failure_and_resume_rejection_stop(paths, monkeypatch):
    summary, control = paths
    def failed(argv):
        report = M.read_summary(summary)
        report.update(status="failed", error="nonfinite output", error_type="RuntimeError")
        publish(summary, report)
        return 1
    monkeypatch.setattr(runner, "main", failed)
    assert M.drive(summary, control) == 1
    assert M.read_summary(control / "status.json")["retryable"] is False
    publish(summary, prepared())
    original = summary.read_bytes()
    def rejected(argv):
        publish(summary.parent / "resume_failure.json", dict(error="identity mismatch"))
        return 1
    monkeypatch.setattr(runner, "main", rejected)
    assert M.drive(summary, control) == 2
    assert M.read_summary(control / "status.json")["retryable"] is False
    assert summary.read_bytes() == original


def test_stale_busy_is_not_retried_after_unconfirmed_exit(paths, monkeypatch):
    summary, control = paths
    monkeypatch.setattr(runner, "main", lambda argv: 9)
    assert M.drive(summary, control) == 9
    state = M.read_summary(control / "status.json")
    assert state["status"] == "failed" and state["retryable"] is False


def test_sigterm_saves_checkpoint_and_cleans_only_own_child(paths, monkeypatch):
    summary, control = paths
    children = []
    def wait_for_signal(argv):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        children.append(child)
        try:
            time.sleep(1)
        except M.MonitorStopped as error:
            report = M.read_summary(summary)
            report.update(error=str(error), error_type=type(error).__name__)
            publish(summary, report)
            return 1
    monkeypatch.setattr(runner, "main", wait_for_signal)
    timer = threading.Timer(.05, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    try:
        assert M.drive(summary, control) == 143
        assert children[0].wait(timeout=2) == -signal.SIGTERM
    finally:
        timer.cancel()
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=2)
    state = M.read_summary(control / "status.json")
    assert state["status"] == "stopped-manually" and not state["retryable"]


def test_resume_command_contains_original_protocol_and_no_prepare(paths):
    summary, _ = paths
    command = M.resume_command(prepared(), summary)
    assert command[0] == sys.executable and command[-1] == "--resume"
    assert "--prepare-only" not in command
    assert command[command.index("--gpu") + 1] == M.GPU_UUID
    assert command[command.index("--output") + 1] == str(summary.parent)
    assert command[command.index("--trials") + 1] == "5"
