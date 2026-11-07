"""CPU-only orchestration tests; no subprocess here launches GPU work."""
import copy
import json
from pathlib import Path
import signal
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper/tools"))
import run_controlled_case_queue as queue


def prepare_queue(tmp_path, count=1):
    study = dict(schema=queue.controlled.SCHEMA,
                 protocol=dict(trials=2, warmup=1, repeat=2, verify_batches=0), cases=[])
    for index in range(count):
        study["cases"].append(dict(id=f"case-{index}", experiment="layout", workload=dict(
            operator="fwht", precision="fp64", logN=8, batch=1), variants=[
                dict(id="linear", mapping=dict(backend="shared-iterative", stage_partition=[4, 4], shared_layout="linear")),
                dict(id="writer", mapping=dict(backend="shared-iterative", stage_partition=[4, 4], shared_layout="writer-aligned"))],
            comparisons=[dict(baseline="linear", treatment="writer", changed_axes=["shared_layout"])]))
    source = tmp_path / "study.json"
    source.write_text(json.dumps(study))
    build = tmp_path / "build"
    build.mkdir()
    (build / "cubutterfly_bench").write_bytes(b"mock-benchmark")
    output = tmp_path / "queue"
    manifest = queue.prepare(source, build, output, "GPU-test")
    return source, build, output, manifest


def release(row, manifest):
    directory = Path(row["directory"])
    study_path = directory / "study.json"
    study = queue.read(study_path)
    journal = queue.read(directory / "prepared.json")
    journal.update(status="complete", completed=True)
    frozen = manifest["identity"]
    journal["identity"].update(
        binaries={name: frozen["binaries"][name] for name in journal["identity"]["binaries"]},
        source_sha256=frozen["source_sha256"], compile_mode=frozen["compile_mode"],
        gpu_uuid=frozen["gpu_uuid"], device=dict(uuid="GPU-test", name="Fake GPU",
                                                compute_capability="8.0", global_memory_bytes=40 * 1024**3))
    case = journal["cases"][0]
    case["status"] = "complete"
    for variant in case["variants"]:
        variant["preflight"] = dict(status="accepted", resolved_mapping=copy.deepcopy(variant["mapping"]))
        variant["trials"] = [dict(trial=index, status="measured", performance_accepted=True, kernel_ms=1.0)
                             for index in range(study["protocol"]["trials"])]
    case["comparisons"] = [queue.controlled._comparison_result(case, comparison) for comparison in case["comparisons"]]
    path = directory / "guards/0000/release/timing.json"
    path.parent.mkdir(parents=True)
    queue.controlled._atomic_write(path, journal)
    return path


def test_prepare_and_skip_compatible_completed_release(tmp_path, monkeypatch):
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **k: pytest.fail("completed release must skip launch"))
    source, build, output, manifest = prepare_queue(tmp_path)
    row = manifest["cases"][0]
    path = release(row, manifest)
    assert queue.prepare(source, build, output, "GPU-test") == manifest
    collection = queue.run(manifest, output, "2099-01-01T00:00:00+00:00")
    assert collection["status"] == "complete"
    assert collection["schema"] == queue.SCHEMA
    assert collection["records"][row["id"]]["release_path"] == str(path)
    assert collection["records"][row["id"]]["release_sha256"] == queue.sha(path)
    assert collection["records"][row["id"]]["identity"]["study_path"] == str(Path(row["directory"]) / "study.json")


@pytest.mark.parametrize("change", ["source", "binary", "subset"])
def test_prepare_rejects_changed_frozen_inputs(tmp_path, change):
    source, build, output, manifest = prepare_queue(tmp_path)
    if change == "source":
        value = queue.read(source)
        value["scope"] = "changed"
        source.write_text(json.dumps(value))
    elif change == "binary":
        (build / "cubutterfly_bench").write_bytes(b"changed")
    else:
        path = Path(manifest["cases"][0]["directory"]) / "study.json"
        path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="identity changed"):
        queue.prepare(source, build, output, "GPU-test")


def test_deadline_prevents_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **k: pytest.fail("expired deadline launched process"))
    _, _, output, manifest = prepare_queue(tmp_path)
    result = queue.run(manifest, output, "2020-01-01T00:00:00+00:00")
    assert result["status"] == "deadline"
    assert result["records"]["case-0"]["status"] == "pending"
    with pytest.raises(ValueError, match="timezone"):
        queue.deadline("2026-10-27T12:00:00")


@pytest.mark.parametrize("change", ["noncomplete", "missing-trial", "identity", "hardware"])
def test_invalid_release_is_never_accepted(tmp_path, change):
    _, _, _, manifest = prepare_queue(tmp_path)
    row = manifest["cases"][0]
    path = release(row, manifest)
    value = queue.read(path)
    if change == "noncomplete":
        value["status"] = "complete-with-gaps"
    elif change == "missing-trial":
        value["cases"][0]["variants"][0]["trials"].pop()
    elif change == "identity":
        value["identity"]["source_sha256"] = {}
    else:
        value["identity"]["device"]["uuid"] = "GPU-other"
    queue.controlled._atomic_write(path, value)
    if change in {"identity", "hardware"}:
        with pytest.raises(ValueError, match="identity"):
            queue.validate_release(path, row, manifest)
    else:
        assert queue.validate_release(path, row, manifest) is None


def test_failed_case_continues_to_next_case(tmp_path, monkeypatch):
    calls = []
    class Failed:
        def __init__(self, command, **kwargs):
            calls.append(command)
        def wait(self):
            return 1
    monkeypatch.setattr(queue.subprocess, "Popen", Failed)
    _, _, output, manifest = prepare_queue(tmp_path, count=2)
    result = queue.run(manifest, output, "2099-01-01T00:00:00+00:00")
    assert len(calls) == 2
    assert all(cmd[cmd.index("--gpu-uuid") + 1] == "GPU-test" for cmd in calls)
    assert result["status"] == "complete-with-gaps"
    assert all(row["status"] == "gap" for row in result["records"].values())


def test_interrupt_forwards_sigint_and_waits_bounded():
    events = []
    class Owned:
        def poll(self):
            return None
        def send_signal(self, sig):
            events.append(sig)
        def wait(self, timeout):
            events.append(timeout)
    queue.interrupt_guard(Owned())
    assert events == [signal.SIGINT, 12]


def test_interrupt_escalates_only_owned_descendant_groups(monkeypatch):
    events = []
    class Owned:
        pid = 100
        def poll(self):
            return None
        def send_signal(self, sig):
            events.append(("signal", sig))
        def wait(self, timeout):
            if timeout == 12:
                raise queue.subprocess.TimeoutExpired("guard", timeout)
            events.append(("wait", timeout))
    monkeypatch.setattr(queue.subprocess, "check_output", lambda *a, **k:
                        "100 1 100\n101 100 101\n102 101 101\n200 1 200\n")
    monkeypatch.setattr(queue.os, "killpg", lambda pgid, sig: events.append(("kill", pgid, sig)))
    queue.interrupt_guard(Owned())
    assert {event[1] for event in events if event[0] == "kill"} == {100, 101}
    assert events[-1] == ("wait", 2)


def test_guard_child_uses_its_own_attempt_journal(tmp_path, monkeypatch):
    source, build, output, manifest = prepare_queue(tmp_path)
    row = manifest["cases"][0]
    attempt = Path(row["directory"]) / "guards/0000/attempts/test"
    attempt.mkdir(parents=True)
    monkeypatch.setenv("XD_REPORT_DIR", str(attempt))
    calls = []
    monkeypatch.setattr(queue.controlled, "main", lambda args: calls.append(args) or 0)
    assert queue.main(["--study", str(source), "--build-dir", str(build), "--output-dir", str(output),
                       "--gpu-uuid", "GPU-test", "--until", "2099-01-01T00:00:00+00:00",
                       "--guard-child", row["id"]]) == 0
    assert calls[0][calls[0].index("--output") + 1] == str(attempt / "timing.json")
