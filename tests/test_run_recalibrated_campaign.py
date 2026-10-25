import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_recalibrated_campaign", ROOT / "scripts" / "run_recalibrated_campaign.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _digest(document):
    body = dict(document)
    body.pop("manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _artifact(path, content):
    path.write_bytes(content)
    return {
        "path": str(path),
        "required": True,
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _write_plan(tmp_path, phases=None):
    phases = phases or {
        "prerequisites": [],
        "correctness": [],
        "profile_gate": [],
        "comparison": [],
    }
    current_library = _artifact(tmp_path / "libcuntt.so", b"library-v1")
    current_butterfly = _artifact(tmp_path / "butterfly", b"butterfly-v1")
    baseline = _artifact(tmp_path / "vkfft_bench", b"baseline-v1")
    document = {
        "schema": "cubutterfly-recalibrated-campaign-v1",
        "campaign_id": "test-campaign",
        "builds": {
            "current": {"artifacts": {
                "current_library": current_library,
                "current_butterfly": current_butterfly,
            }},
            "baseline": {"artifacts": {"baseline_fft": baseline}},
        },
        "files": {},
        "phases": phases,
    }
    document["manifest_sha256"] = _digest(document)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document, indent=2) + "\n")
    return path, current_library["path"]


def _command(code):
    return "python3 -c {}".format(shlex.quote(code))


def _complete_calibration(tmp_path, status="complete"):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"source": "test"}) + "\n")
    path = tmp_path / "calibration_manifest.json"
    path.write_text(json.dumps({"status": status, "registry_path": str(registry)}) + "\n")
    return path


@pytest.mark.parametrize("resume", [False, True])
def test_cancelled_campaign_blocks_start_and_resume_without_state_writes(tmp_path, monkeypatch, resume):
    manifest, _ = _write_plan(tmp_path)
    original = manifest.read_bytes()
    (tmp_path / "cancellation.json").write_text(json.dumps({"reason": "repair semantics first"}))
    monkeypatch.setattr(runner, "wait_for_calibration", lambda *_: pytest.fail("must not wait"))
    monkeypatch.setattr(runner, "fingerprint_builds", lambda *_args, **_kwargs: pytest.fail("must not execute"))
    with pytest.raises(runner.CampaignError, match="campaign cancelled.*repair semantics first"):
        runner.run_campaign(manifest, "GPU-test", tmp_path / "calibration.json", resume=resume)
    assert not (tmp_path / "campaign_queue_state.json").exists()
    assert manifest.read_bytes() == original


def test_cancelled_campaign_remains_available_for_dry_run(tmp_path, monkeypatch):
    manifest, _ = _write_plan(tmp_path)
    (tmp_path / "cancellation.json").write_text("{}")
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must not launch"))
    assert runner.main(["--manifest", str(manifest), "--dry-run"]) == 0


def test_malformed_cancellation_marker_fails_closed(tmp_path):
    manifest, _ = _write_plan(tmp_path)
    (tmp_path / "cancellation.json").write_text("invalid JSON")
    with pytest.raises(runner.CampaignError, match="could not read JSON.*cancellation"):
        runner.run_campaign(manifest, "GPU-test", tmp_path / "calibration.json")


def test_dry_run_consumes_all_current_manifest_commands_without_starting_process(capsys):
    manifest = ROOT / "results" / "recalibrated_campaign_20261006" / "manifest.json"
    assert runner.main(["--manifest", str(manifest), "--dry-run", "--gpu-uuid", "GPU-test"]) == 0
    output = capsys.readouterr().out
    document = json.loads(manifest.read_text())
    expected = sum(len(document.get("phases", {}).get(phase, [])) for phase in runner.PHASES)
    assert expected >= 40
    assert "TOTAL_COMMANDS={}".format(expected) in output
    assert "comparison[{}]".format(len(document.get("phases", {}).get("comparison", [])) - 1) in output
    for _, _, step in runner._command_steps(document):
        argv, _, _ = runner.parse_command(step["run"], "GPU-test", runner.ROOT)
        assert argv[0] not in ("python", "python3")


def test_incomplete_calibration_blocks_every_command(tmpdir, monkeypatch):
    tmp_path = Path(str(tmpdir))
    marker = tmp_path / "started"
    phases = {
        "prerequisites": [{
            "id": "build",
            "cpu_only": True,
            "run": _command("from pathlib import Path; Path({!r}).write_text('started')".format(str(marker))),
        }],
        "correctness": [], "profile_gate": [], "comparison": [],
    }
    manifest, _ = _write_plan(tmp_path, phases)
    calibration = _complete_calibration(tmp_path, "incomplete")
    monkeypatch.setattr(runner, "require_exclusive_gpu", lambda: pytest.fail("GPU guard must not run"))
    with pytest.raises(runner.CampaignError, match="not successful"):
        runner.run_campaign(
            manifest, "GPU-test", calibration,
            state_path=tmp_path / "state.json", poll_seconds=0,
        )
    assert not marker.exists()


def test_resume_reuses_only_matching_hashes_and_python3_becomes_sys_executable(tmpdir, monkeypatch):
    tmp_path = Path(str(tmpdir))
    marker = tmp_path / "runs"
    code = "from pathlib import Path; p=Path({!r}); p.write_text(str(int(p.read_text() or '0')+1)) if p.exists() else p.write_text('1')".format(str(marker))
    command = _command(code)
    phases = {
        "prerequisites": [{"id": "build", "run": command}],
        "correctness": [{"id": "correct", "cpu_only": True, "run": command}],
        "profile_gate": [{"id": "summarize_profile", "run": command}],
        "comparison": [{"id": "summarize_comparison", "run": command}],
    }
    manifest, library_path = _write_plan(tmp_path, phases)
    calibration = _complete_calibration(tmp_path)
    monkeypatch.setattr(runner, "require_exclusive_gpu", lambda: pytest.fail("CPU-only commands need no GPU guard"))
    state = tmp_path / "state.json"
    assert runner.run_campaign(manifest, "GPU-test", calibration, state_path=state, poll_seconds=0) == 0
    assert marker.read_text() == "4"
    first_state = json.loads(state.read_text())
    assert first_state["status"] == "completed"
    assert first_state["phases"]["prerequisites"]["steps"][0]["argv"][0] == sys.executable

    assert runner.run_campaign(
        manifest, "GPU-test", calibration,
        state_path=state, resume=True, poll_seconds=0,
    ) == 0
    assert marker.read_text() == "4"

    Path(library_path).write_bytes(b"changed-library")
    with pytest.raises(runner.CampaignError, match="hash mismatch|hash changed"):
        runner.run_campaign(
            manifest, "GPU-test", calibration,
            state_path=state, resume=True, poll_seconds=0,
        )


def test_profile_failure_is_not_reported_as_completed(tmpdir, monkeypatch):
    tmp_path = Path(str(tmpdir))
    marker = tmp_path / "comparison-ran"
    passing = _command("pass")
    failing = _command("raise SystemExit(9)")
    comparison = _command("from pathlib import Path; Path({!r}).write_text('ran')".format(str(marker)))
    phases = {
        "prerequisites": [{"id": "build", "run": passing}],
        "correctness": [{"id": "correct", "cpu_only": True, "run": passing}],
        "profile_gate": [{"id": "profile-fails", "cpu_only": True, "run": failing}],
        "comparison": [{"id": "comparison-still-runs", "cpu_only": True, "run": comparison}],
    }
    manifest, _ = _write_plan(tmp_path, phases)
    calibration = _complete_calibration(tmp_path)
    monkeypatch.setattr(runner, "require_exclusive_gpu", lambda: pytest.fail("CPU-only commands need no GPU guard"))
    state = tmp_path / "state.json"
    assert runner.run_campaign(manifest, "GPU-test", calibration, state_path=state, poll_seconds=0) == 1
    document = json.loads(state.read_text())
    assert document["status"] == "completed-with-failures"
    assert document["phases"]["profile_gate"]["status"] == "failed"
    assert marker.read_text() == "ran"


def test_registry_path_is_injected_and_resume_rejects_registry_change(tmpdir, monkeypatch):
    tmp_path = Path(str(tmpdir))
    marker = tmp_path / "registry-seen"
    code = "from pathlib import Path; Path({!r}).write_text(__import__('os').environ['CUBUTTERFLY_REGISTRY'])".format(str(marker))
    phases = {
        "prerequisites": [{"id": "build", "run": _command("pass")}],
        "correctness": [{"id": "correct", "cpu_only": True, "run": _command(code)}],
        "profile_gate": [], "comparison": [],
    }
    manifest, _ = _write_plan(tmp_path, phases)
    calibration = _complete_calibration(tmp_path)
    registry = tmp_path / "registry.json"
    monkeypatch.setattr(runner, "require_exclusive_gpu", lambda: pytest.fail("CPU-only commands need no GPU guard"))
    state = tmp_path / "state.json"
    assert runner.run_campaign(manifest, "GPU-test", calibration, state_path=state, poll_seconds=0) == 0
    assert marker.read_text() == str(registry.resolve())
    registry.write_text("changed\n")
    with pytest.raises(runner.CampaignError, match="registry"):
        runner.run_campaign(manifest, "GPU-test", calibration, state_path=state, resume=True, poll_seconds=0)


def test_gpu_conflict_stops_only_owned_session(tmpdir, monkeypatch):
    class FakeChild:
        pid = 424242
        returncode = -15

        def __init__(self):
            self.stopped = False

        def poll(self):
            return self.returncode if self.stopped else None

    child = FakeChild()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(runner, "gpu_pids", lambda uuid: [os.getpid()])
    monkeypatch.setattr(runner, "descendant", lambda pid, owner: False)
    monkeypatch.setattr(runner, "stop_owned", lambda process: setattr(process, "stopped", True))
    log = Path(str(tmpdir)) / "gpu.log"
    result = runner._run_gpu_process(["/bin/true"], {}, log, "GPU-test", 0, None)
    assert result["status"] == "gpu-conflict"
    assert os.getpid() in result["contaminated_by_foreign_pids"]
    assert child.stopped


def test_large_memory_phase_runs_and_preexisting_gpu_conflict_stops_queue(tmp_path, monkeypatch):
    marker = tmp_path / "large-ran"
    later = tmp_path / "later-ran"
    phases = {
        "prerequisites": [], "correctness": [],
        "large_memory": [{"id": "large", "cpu_only": True,
            "run": _command("from pathlib import Path; Path({!r}).touch()".format(str(marker)))}],
        "profile_gate": [{"id": "needs-gpu", "run": _command("pass")}],
        "comparison": [{"id": "later", "cpu_only": True,
            "run": _command("from pathlib import Path; Path({!r}).touch()".format(str(later)))}],
    }
    manifest, _ = _write_plan(tmp_path, phases)
    calibration = _complete_calibration(tmp_path)
    def busy(_):
        raise runner.CampaignError("foreign process")
    monkeypatch.setattr(runner, "_ensure_gpu_exclusive", busy)
    state = tmp_path / "state.json"
    assert runner.run_campaign(manifest, "GPU-test", calibration, state_path=state) == 1
    assert marker.exists() and not later.exists()
    assert json.loads(state.read_text())["status"] == "gpu-not-exclusive"


def _reviewed_plan(tmp_path, phases=None):
    path, _ = _write_plan(tmp_path, phases)
    manifest = json.loads(path.read_text())
    artifacts = manifest["builds"]["current"]["artifacts"]
    artifacts["current_ntt"] = _artifact(tmp_path / "ntt", b"ntt-v1")
    manifest["measurement_contract"] = dict(
        gpu_uuid="GPU-test", compile_mode="research",
        execution_environment=runner.execution_environment(),
        hardware=dict(device="test GPU", compute_capability="8.0", global_memory_bytes=1024))
    manifest["manifest_sha256"] = _digest(manifest)
    path.write_text(json.dumps(manifest))
    calibration_path = _complete_calibration(tmp_path)
    calibration = json.loads(calibration_path.read_text())
    calibration.update(manifest["measurement_contract"]["hardware"])
    calibration["selector_replay"] = {"status": "passed"}
    calibration["measurement_identity"] = dict(
        compile_mode="research", library_sha256=artifacts["current_library"]["sha256"],
        binary_sha256={Path(artifacts[name]["path"]).name: artifacts[name]["sha256"]
                       for name in ("current_butterfly", "current_ntt")})
    calibration_path.write_text(json.dumps(calibration))
    return path, manifest, calibration_path, calibration


@pytest.mark.parametrize("field,value,match", [
    ("device", "other GPU", "hardware"),
    ("global_memory_bytes", 2048, "hardware"),
    ("compile_mode", "default", "compile mode"),
    ("library_sha256", "old-library", "shared library"),
])
def test_reviewed_calibration_rejects_cross_identity(tmp_path, monkeypatch, field, value, match):
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.delenv("CUBUTTERFLY_PLATFORM_TUNING", raising=False)
    _, manifest, _, calibration = _reviewed_plan(tmp_path)
    runner.validate_calibration_binding(manifest, calibration, "GPU-test")
    target = calibration["measurement_identity"] if field in ("compile_mode", "library_sha256") else calibration
    target[field] = value
    with pytest.raises(runner.CampaignError, match=match):
        runner.validate_calibration_binding(manifest, calibration, "GPU-test")


@pytest.mark.parametrize("phase", ["correctness", "profile_gate"])
def test_reviewed_gate_failure_stops_before_comparison(tmp_path, monkeypatch, phase):
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.delenv("CUBUTTERFLY_PLATFORM_TUNING", raising=False)
    marker = tmp_path / "must-not-run"
    phases = {key: [] for key in runner.PHASES}
    phases[phase] = [{"id": "bad-check", "cpu_only": True,
                      "run": _command("raise SystemExit(9)")}]
    phases["comparison"] = [{"id": "timing", "cpu_only": True,
                             "run": _command("from pathlib import Path; Path({!r}).touch()".format(str(marker)))}]
    path, _, calibration, _ = _reviewed_plan(tmp_path, phases)
    monkeypatch.setattr(runner, "validate_runtime_hardware", lambda *args: {})
    assert runner.run_campaign(path, "GPU-test", calibration) == 1
    assert not marker.exists()
    state = json.loads((tmp_path / "campaign_queue_state.json").read_text())
    assert state["status"] == "blocked-" + phase


def test_reviewed_detects_library_change_within_step(tmp_path, monkeypatch):
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.delenv("CUBUTTERFLY_PLATFORM_TUNING", raising=False)
    library = tmp_path / "libcuntt.so"
    phases = {key: [] for key in runner.PHASES}
    phases["correctness"] = [{"id": "mutate", "cpu_only": True,
        "run": _command("from pathlib import Path; Path({!r}).write_bytes(b'changed')".format(str(library)))}]
    path, _, calibration, _ = _reviewed_plan(tmp_path, phases)
    monkeypatch.setattr(runner, "validate_runtime_hardware", lambda *args: {})
    assert runner.run_campaign(path, "GPU-test", calibration) == 1
    state = json.loads((tmp_path / "campaign_queue_state.json").read_text())
    assert state["status"] == "identity-changed"


def test_retry_preserves_but_does_not_reuse_contaminated_output(tmp_path):
    raw = tmp_path / "comparison" / "raw.csv"
    raw.parent.mkdir()
    raw.write_text("contaminated trial\n")
    state = {}
    runner.quarantine_retry_outputs({"id": "timing", "outputs": [str(raw)]}, state,
                                    tmp_path / "state.json", "comparison", {"status": "gpu-conflict"})
    assert not raw.exists()
    preserved = state["attempt_history"][0]["quarantined_outputs"][0]["preserved"]
    assert Path(preserved).read_text() == "contaminated trial\n"
    with pytest.raises(runner.CampaignError, match="strictly inside"):
        runner.quarantine_retry_outputs({"id": "bad", "outputs": [str(tmp_path)]}, state,
                                        tmp_path / "state.json", "comparison", {})
