import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SPEC = importlib.util.spec_from_file_location(
    "prepared_queue", Path(__file__).resolve().parents[1] / "paper/tools/run_prepared_queue.py")
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


def configure(tmp_path, monkeypatch, commands, extra=()):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"commands": commands}))
    output = tmp_path / "queue.json"
    args = ["queue", "--plan", str(plan), "--output", str(output), "--gpu-uuid", "GPU-test"]
    for command in commands:
        args += ["--command-id", command["id"]]
    monkeypatch.setattr(sys, "argv", args + list(extra))
    monkeypatch.setattr(queue, "require_exclusive_gpu", lambda: pytest.fail("unexpected GPU query"))
    return output


def test_interrupted_writer_never_starts_followers(tmp_path, monkeypatch):
    prerequisite = tmp_path / "acceptance.json"
    prerequisite.write_text(json.dumps({"status": "interrupted"}))
    commands = [dict(id="follow", argv=[sys.executable, "-c", "print('must not run')"])]
    output = configure(tmp_path, monkeypatch, commands,
                       ["--after-session", "prior", "--after-journal", str(prerequisite)])
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        assert argv[:2] == ["tmux", "has-session"]
        return subprocess.CompletedProcess(argv, 1)
    monkeypatch.setattr(queue.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="prerequisite ended"):
        queue.main()
    document = json.loads(output.read_text())
    assert document["status"] == "stopped" and document["records"] == []
    assert len(calls) == 1


def test_failed_command_stops_the_queue(tmp_path, monkeypatch):
    commands = [dict(id="first", argv=[sys.executable, "-c", "raise SystemExit(7)"]),
                dict(id="second", argv=[sys.executable, "-c", "print('must not run')"])]
    output = configure(tmp_path, monkeypatch, commands)
    with pytest.raises(RuntimeError, match="command failed: first"):
        queue.main()
    document = json.loads(output.read_text())
    assert document["status"] == "stopped"
    assert len(document["records"]) == 1
    assert document["records"][0]["returncode"] == 7


def test_success_preserves_order_and_rejects_duplicate_journal(tmp_path, monkeypatch):
    commands = [dict(id=name, argv=[sys.executable, "-c", f"print({name!r})"])
                for name in ["first", "second"]]
    output = configure(tmp_path, monkeypatch, commands)
    queue.main()
    document = json.loads(output.read_text())
    assert document["status"] == "completed-commands"
    assert [Path(record["log"]).read_text().strip() for record in document["records"]] == ["first", "second"]
    with pytest.raises(ValueError, match="queue journal exists"):
        queue.main()


def test_wrong_gpu_is_rejected_before_launch(tmp_path, monkeypatch):
    commands = [dict(id="wrong", requires_gpu=True, env={"CUDA_VISIBLE_DEVICES": "GPU-other"},
                     argv=[sys.executable, "-c", "print('must not run')"])]
    output = configure(tmp_path, monkeypatch, commands)
    with pytest.raises(SystemExit):
        queue.main()
    assert not output.exists()
