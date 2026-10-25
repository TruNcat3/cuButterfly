"""CPU-only parity checks for persistent broker transport and native client."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import semantic_compile_broker as broker
import semantic_compile_service as service


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    executable = tmp_path_factory.mktemp("compile-client") / "client"
    subprocess.run(["c++", "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    str(ROOT / "scripts/semantic_compile_client.cpp"), "-o", str(executable)], check=True)
    return executable


@pytest.fixture
def setup(tmp_path, client):
    root = tmp_path / "source"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/compile_module.py").write_text("# fixture compiler\n")
    (root / "scripts/resident_mapping.py").write_text("# fixture resident\n")
    nvcc = tmp_path / "nvcc"
    nvcc.write_text("#!/bin/sh\nprintf 'fixture compiler\\n'\n")
    nvcc.chmod(0o700)
    config = dict(root=str(root), mathdx=str(tmp_path / "mathdx"), nvcc=str(nvcc),
                  sm=80, cache=str(tmp_path / "cache"))
    directory = tmp_path / "broker"
    document = broker.initialize(directory, config)
    endpoint = tmp_path / "service.sock"
    events = tmp_path / "attempt.jsonl"
    environment = dict(os.environ, CUBUTTERFLY_PREPARE_BROKER=str(directory),
                       CUBUTTERFLY_PREPARE_EVENTS=str(events), CUBUTTERFLY_COMPILE_SOCKET=str(endpoint))
    process = subprocess.Popen([sys.executable, str(ROOT / "scripts/semantic_compile_service.py"),
                                "--broker", str(directory), "--socket", str(endpoint)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 5
    while not endpoint.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    if not endpoint.exists():
        process.terminate()
        out, err = process.communicate(timeout=5)
        pytest.fail(f"service failed to start: {out!r} {err!r}")
    fixture = SimpleNamespace(directory=directory, document=document, config=config,
                              events=events, endpoint=endpoint, env=environment,
                              process=process, client=client, root=root)
    try:
        yield fixture
    finally:
        if process.poll() is None:
            process.terminate()
        process.communicate(timeout=5)
        assert not endpoint.exists()


def mapping():
    return dict(backend="shared-iterative", operator="fft", precision="fp32",
                logN=8, stage_partition=[8], threads=128)


def arguments(setup, point=None, **changes):
    args = [str(setup.root / "scripts/compile_module.py"), "--mapping", json.dumps(point or mapping())]
    for key, value in {**setup.config, **changes}.items():
        args.extend(["--" + key, str(value)])
    return args


def invoke(setup, args=None, **environment):
    return subprocess.run([str(setup.client), *(args or arguments(setup))],
                          env={**setup.env, **environment}, capture_output=True, text=True, timeout=10)


def record(setup, state="ready"):
    identifier = broker._id({"broker_id": setup.document["id"], "mapping": mapping()})
    path = Path(setup.config["cache"]) / "entry/module.so"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture module")
    value = dict(id=identifier, broker_id=setup.document["id"], status=state)
    if state == "ready":
        value.update(path=str(path), module_stat=broker._stat(path))
    else:
        value["error"] = "fixture compilation failed"
    broker._atomic(setup.directory / "records" / (identifier + ".json"), value)
    return path


def event_values(path):
    return [{key: value for key, value in json.loads(line).items() if key != "time_ns"}
            for line in path.read_text().splitlines()]


def original(setup, monkeypatch, capsys, args):
    original_events = setup.events.with_name("original.jsonl")
    monkeypatch.setenv("CUBUTTERFLY_PREPARE_BROKER", str(setup.directory))
    monkeypatch.setenv("CUBUTTERFLY_PREPARE_EVENTS", str(original_events))
    code = broker.main(args)
    output = capsys.readouterr()
    return code, output, original_events


def test_ready_exact_mapping_and_stdout_parity(setup, monkeypatch, capsys):
    path = record(setup)
    code, output, _ = original(setup, monkeypatch, capsys, arguments(setup))
    result = invoke(setup)
    assert result.returncode == code == 0
    assert result.stdout == output.out == str(path) + "\n"
    assert result.stderr == output.err == ""
    assert not setup.events.exists()
    # A different mapping must queue, even though a ready record exists.
    result = invoke(setup, arguments(setup, {**mapping(), "threads": 256}))
    assert result.returncode == 1 and result.stdout == ""
    assert event_values(setup.events)[0]["kind"] == "compile_pending"


@pytest.mark.parametrize("state", ["pending", "failed", "running"])
def test_unresolved_event_and_exit_parity(setup, monkeypatch, capsys, state):
    if state != "pending":
        record(setup, state)
    code, output, events = original(setup, monkeypatch, capsys, arguments(setup))
    result = invoke(setup)
    assert result.returncode == code == 1
    assert result.stdout == output.out == "" and result.stderr == output.err
    assert event_values(setup.events) == event_values(events)
    assert event_values(events)[0]["kind"] == "compile_" + state


@pytest.mark.parametrize("fault", ["module", "compiler", "config", "script"])
def test_runtime_guards_match_original(setup, monkeypatch, capsys, fault):
    path = record(setup)
    args = arguments(setup)
    if fault == "module":
        path.write_bytes(b"changed fixture module with different size")
    elif fault == "compiler":
        Path(setup.config["nvcc"]).write_text("changed compiler")
    elif fault == "config":
        args = arguments(setup, sm=90)
    else:
        args[0] = "wrong-script.py"
    code, output, events = original(setup, monkeypatch, capsys, args)
    result = invoke(setup, args)
    assert result.returncode == code == 1
    assert result.stderr == output.err
    assert event_values(setup.events) == event_values(events)


@pytest.mark.parametrize("setting,reason", [("CUBUTTERFLY_WORKER_MAX_VMAS", "vma-count"),
                                            ("CUBUTTERFLY_WORKER_MAX_RSS_BYTES", "rss-bytes")])
def test_resource_guard_precedes_ready_lookup(setup, setting, reason):
    record(setup)
    result = invoke(setup, **{setting: "1"})
    assert result.returncode == 75 and result.stdout == ""
    event = event_values(setup.events)[0]
    assert event["kind"] == "worker_resource_limit"
    assert event["reason"].startswith(reason + " reached")
    assert event["resources"]["pid"] == os.getpid()
    assert len(event_values(setup.events)) == 1


def test_unavailable_service_records_fail_closed_event(setup):
    record(setup)
    result = invoke(setup, CUBUTTERFLY_COMPILE_SOCKET=str(setup.endpoint.with_name("absent.sock")))
    assert result.returncode == 1 and result.stdout == ""
    assert "unavailable" in result.stderr
    assert event_values(setup.events)[0]["kind"] == "broker_error"


def test_concurrent_clients_keep_attempt_events_separate(setup):
    def request(index):
        events = setup.events.with_name(f"attempt-{index}.jsonl")
        result = invoke(setup, CUBUTTERFLY_PREPARE_EVENTS=str(events))
        return result, event_values(events)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(request, range(8)))
    assert all(result.returncode == 1 and len(events) == 1 and
               events[0]["kind"] == "compile_pending" for result, events in results)
    assert not setup.events.exists()


@pytest.mark.parametrize("payload", [b"", b"\0\0\0\x05x", service._strings(["wrong-protocol"])])
def test_malformed_protocol_rejected_without_stopping_service(setup, payload):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(2)
        connection.connect(str(setup.endpoint))
        connection.sendall(struct.pack("!I", len(payload)) + payload)
        response = service._frame(connection)
        assert struct.unpack_from("!I", response)[0] == 1
    path = record(setup)
    assert invoke(setup).stdout == str(path) + "\n"


def test_wrong_parent_and_bound_broker_rejected(setup, monkeypatch):
    values = ["cubutterfly-compile-service-v1", str(os.getppid()), os.getcwd(),
              str(setup.directory), str(setup.events), "", "", os.environ["PATH"], *arguments(setup)]
    code, out, err = service.dispatch(setup.directory, values, os.getpid(), os.getuid() + 1)
    assert code == 1 and "credentials" in err and out == ""
    values[3] = str(setup.directory.with_name("other-broker"))
    code, out, err = service.dispatch(setup.directory, values, os.getpid(), os.getuid())
    assert code == 1 and "service broker" in err
    assert event_values(setup.events)[0]["kind"] == "broker_error"


def test_existing_socket_refused_and_not_removed(setup):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/semantic_compile_service.py"),
                             "--broker", str(setup.directory), "--socket", str(setup.endpoint)],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode != 0 and setup.endpoint.exists()
    assert invoke(setup).returncode == 1


def test_startup_rejects_changed_sources(setup):
    (setup.root / "scripts/compile_module.py").write_text("# changed source\n")
    endpoint = setup.endpoint.with_name("other.sock")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/semantic_compile_service.py"),
                             "--broker", str(setup.directory), "--socket", str(endpoint)],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode != 0 and "source or compiler identity changed" in result.stderr
    assert not endpoint.exists()


@pytest.mark.parametrize("failure", ["authentication", "protocol", "disconnect"])
@pytest.mark.parametrize("existing", [False, True])
def test_native_records_unrecorded_service_failures(setup, failure, existing):
    if existing:
        setup.events.write_text('{"kind":"compile_pending","request_id":"previous"}\n')
    endpoint = setup.endpoint.with_name("failure.sock")
    failures = []
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(endpoint))
        listener.listen(1)
        listener.settimeout(5)
        def reply():
            try:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(5)
                    values = service._decode(service._frame(connection))
                    if failure == "disconnect":
                        return
                    if failure == "authentication":
                        values[1] = "0"
                        pid, uid, _ = struct.unpack("3i", connection.getsockopt(
                            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
                        code, out, err = service.dispatch(setup.directory, values, pid, uid)
                        assert code == 1
                    else:
                        code, out, err = 1, "", "semantic compile service: malformed protocol\n"
                    payload = struct.pack("!I", code) + service._strings([out, err])
                    connection.sendall(struct.pack("!I", len(payload)) + payload)
            except Exception as error:
                failures.append(error)
        thread = threading.Thread(target=reply)
        thread.start()
        try:
            result = invoke(setup, CUBUTTERFLY_COMPILE_SOCKET=str(endpoint))
        finally:
            thread.join(timeout=6)
        assert not thread.is_alive() and not failures
    assert result.returncode == 1 and result.stdout == ""
    events = event_values(setup.events)
    assert len(events) == 1 + int(existing)
    assert events[-1]["kind"] == "broker_error"


def test_native_event_write_failure_has_distinct_stderr_marker(setup):
    result = invoke(setup, CUBUTTERFLY_PREPARE_EVENTS=str(setup.directory),
                    CUBUTTERFLY_COMPILE_SOCKET=str(setup.endpoint.with_name("missing.sock")))
    assert result.returncode == 1
    assert "semantic compile client: preparation event recording failed:" in result.stderr


def test_disconnected_peer_does_not_stop_real_service(setup):
    values = ["cubutterfly-compile-service-v1", str(os.getppid()), os.getcwd(),
              str(setup.directory), str(setup.events), "", "", os.environ["PATH"], *arguments(setup)]
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(setup.endpoint))
        payload = service._strings(values)
        connection.sendall(struct.pack("!I", len(payload)) + payload)
    path = record(setup)
    result = invoke(setup)
    assert result.returncode == 0 and result.stdout == str(path) + "\n"
