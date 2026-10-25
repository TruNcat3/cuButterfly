"""CPU-only checks for the MEASURE compilation handoff."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import semantic_compile_broker as broker


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "source"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "compile_module.py").write_text("# frozen compiler\n")
    (root / "scripts" / "resident_mapping.py").write_text("# frozen resident\n")
    (root / "scripts" / "cufftdx_capabilities.py").write_text("# frozen provider parser\n")
    nvcc = tmp_path / "nvcc"
    nvcc.write_text("compiler fixture\n")
    monkeypatch.setattr(broker.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="nvcc test version\n"))
    config = dict(root=str(root), mathdx=str(tmp_path / "mathdx"), nvcc=str(nvcc),
                  sm=80, cache=str(tmp_path / "cache"))
    directory = tmp_path / "broker"
    document = broker.initialize(directory, config)
    return SimpleNamespace(directory=directory, config=config, document=document,
                           events=tmp_path / "events.jsonl", root=root)


def _mapping(log_n=8):
    return dict(backend="shared-iterative", operator="fft", precision="fp32",
                logN=log_n, stage_partition=[log_n], threads=128)


def _miss(setup, mapping=None):
    with pytest.raises(RuntimeError, match="pending"):
        broker.runtime_request(setup.directory, setup.config, mapping or _mapping(), setup.events)
    return next((setup.directory / "requests").glob("*.json"))


def _fake_compiler(setup, monkeypatch, fail=False):
    calls = []
    def compile_mapping(mapping, **kwargs):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        calls.append((mapping, kwargs))
        if fail:
            raise RuntimeError("nvcc fixture rejected specialization")
        path = Path(kwargs["cache"]) / str(mapping["logN"]) / "module.so"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture module")
        return path
    monkeypatch.setattr(broker, "_compiler", lambda root: SimpleNamespace(compile_mapping=compile_mapping))
    return calls


def test_missing_is_queued_and_runtime_never_compiles(setup, monkeypatch):
    monkeypatch.setattr(broker, "_compiler", lambda *args: pytest.fail("runtime compiled"))
    path = _miss(setup)
    assert broker._read(path)["mapping"] == _mapping()
    assert broker.status(setup.directory) == dict(pending=1, running=0, ready=0, failed=0)
    assert broker._read(setup.events)["kind"] == "compile_pending"


def test_ready_is_verified_reused_without_worker_or_new_event(setup, monkeypatch):
    _miss(setup)
    calls = _fake_compiler(setup, monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "existing-selector")
    assert broker.work(setup.directory, jobs=2)["ready"] == 1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "existing-selector"
    before = setup.events.read_text()
    path = broker.runtime_request(setup.directory, setup.config, _mapping(), setup.events)
    assert path.read_bytes() == b"fixture module"
    assert setup.events.read_text() == before
    broker.work(setup.directory)
    assert len(calls) == 1
    assert calls[0][1]["sm"] == 80
    assert calls[0][1]["root"] == setup.root


def test_failure_is_preserved_and_recorded_until_explicit_retry(setup, monkeypatch):
    _miss(setup)
    calls = _fake_compiler(setup, monkeypatch, fail=True)
    assert broker.work(setup.directory)["failed"] == 1
    with pytest.raises(RuntimeError, match="failed"):
        broker.runtime_request(setup.directory, setup.config, _mapping(), setup.events)
    event = json.loads(setup.events.read_text().splitlines()[-1])
    assert event["kind"] == "compile_failed" and "nvcc fixture" in event["error"]
    broker.work(setup.directory)
    assert len(calls) == 1
    _fake_compiler(setup, monkeypatch)
    assert broker.work(setup.directory, retry_failed=True)["ready"] == 1


@pytest.mark.parametrize("field,value", [("sm", 90), ("cache", "/different/cache"),
                                         ("root", "/different/root"), ("mathdx", "/different/mathdx")])
def test_runtime_config_mismatch_records_attempt_error(setup, field, value):
    with pytest.raises(ValueError, match="configuration"):
        broker.runtime_request(setup.directory, {**setup.config, field: value}, _mapping(), setup.events)
    assert broker._read(setup.events)["kind"] == "broker_error"
    assert not list((setup.directory / "requests").glob("*.json"))


def test_source_change_blocks_worker_and_reinitialization(setup):
    (setup.root / "scripts" / "compile_module.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="source or compiler"):
        broker.work(setup.directory)
    with pytest.raises(ValueError, match="identity changed"):
        broker.initialize(setup.directory, setup.config)


def test_compiler_change_and_ready_module_change_are_fatal(setup, monkeypatch):
    _miss(setup)
    _fake_compiler(setup, monkeypatch)
    broker.work(setup.directory)
    path = broker.runtime_request(setup.directory, setup.config, _mapping(), setup.events)
    path.write_bytes(b"modified fixture")
    with pytest.raises(ValueError, match="module missing or changed"):
        broker.runtime_request(setup.directory, setup.config, _mapping(), setup.events)
    Path(setup.config["nvcc"]).write_text("changed compiler\n")
    with pytest.raises(ValueError, match="compiler changed"):
        broker.runtime_request(setup.directory, setup.config, _mapping(), setup.events)


def test_atomic_concurrent_missing_requests_and_events(setup):
    def request(index):
        with pytest.raises(RuntimeError):
            broker.runtime_request(setup.directory, setup.config, _mapping(8 + index % 2), setup.events)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(request, range(32)))
    requests = list((setup.directory / "requests").glob("*.json"))
    assert len(requests) == 2
    assert all(broker._read(path)["id"] == path.stem for path in requests)
    events = [json.loads(line) for line in setup.events.read_text().splitlines()]
    assert len(events) == 32 and all(event["kind"] == "compile_pending" for event in events)
    assert not list((setup.directory / "requests").glob(".publish-*"))


def test_running_request_never_waits_for_worker_lock(setup):
    import fcntl
    path = _miss(setup)
    request = broker._read(path)
    broker._atomic(setup.directory / "records" / path.name,
                   {"id": request["id"], "broker_id": setup.document["id"], "status": "running"})
    with (setup.directory / "locks" / (path.stem + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="running"):
            broker.runtime_request(setup.directory, setup.config, _mapping(), setup.events)


def test_abandoned_running_request_is_resumed(setup, monkeypatch):
    path = _miss(setup)
    request = broker._read(path)
    broker._atomic(setup.directory / "records" / path.name,
                   {"id": request["id"], "broker_id": setup.document["id"], "status": "running"})
    calls = _fake_compiler(setup, monkeypatch)
    assert broker.work(setup.directory)["ready"] == 1
    assert len(calls) == 1


def test_wrapper_cli_mismatch_is_recorded(setup, monkeypatch):
    monkeypatch.setenv("CUBUTTERFLY_PREPARE_BROKER", str(setup.directory))
    monkeypatch.setenv("CUBUTTERFLY_PREPARE_EVENTS", str(setup.events))
    args = ["wrong-compiler.py", "--mapping", json.dumps(_mapping())]
    for key, value in setup.config.items():
        args.extend(["--" + key, str(value)])
    assert broker.main(args) == 1
    assert broker._read(setup.events)["kind"] == "broker_runtime_error"


def test_wrapper_cli_ready_stdout_is_only_module_path(setup, monkeypatch, capsys):
    _miss(setup)
    _fake_compiler(setup, monkeypatch)
    broker.work(setup.directory)
    monkeypatch.setenv("CUBUTTERFLY_PREPARE_BROKER", str(setup.directory))
    monkeypatch.setenv("CUBUTTERFLY_PREPARE_EVENTS", str(setup.events))
    args = [str(setup.root / "scripts" / "compile_module.py"), "--mapping", json.dumps(_mapping())]
    for key, value in setup.config.items():
        args.extend(["--" + key, str(value)])
    assert broker.main(args) == 0
    output = capsys.readouterr()
    assert Path(output.out.strip()).is_file()
    assert output.out.count("\n") == 1 and output.err == ""


def test_wrapper_cli_missing_emits_one_event(setup, monkeypatch):
    monkeypatch.setenv("CUBUTTERFLY_PREPARE_BROKER", str(setup.directory))
    monkeypatch.setenv("CUBUTTERFLY_PREPARE_EVENTS", str(setup.events))
    args = [str(setup.root / "scripts" / "compile_module.py"), "--mapping", json.dumps(_mapping())]
    for key, value in setup.config.items():
        args.extend(["--" + key, str(value)])
    assert broker.main(args) == 1
    assert len(setup.events.read_text().splitlines()) == 1


def test_worker_bounded_parallelism_and_all_requests_complete(setup, monkeypatch):
    import threading
    import time
    for log_n in range(8, 14):
        _miss(setup, _mapping(log_n))
    lock = threading.Lock()
    active = maximum = completed = 0
    def compile_mapping(mapping, **kwargs):
        nonlocal active, maximum, completed
        with lock:
            active += 1
            maximum = max(active, maximum)
        time.sleep(0.02)
        path = Path(kwargs["cache"]) / str(mapping["logN"]) / "module.so"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        with lock:
            active -= 1
            completed += 1
        return path
    monkeypatch.setattr(broker, "_compiler", lambda root: SimpleNamespace(compile_mapping=compile_mapping))
    assert broker.work(setup.directory, jobs=2)["ready"] == 6
    assert maximum == 2 and completed == 6


def test_worker_signal_finishes_active_and_keeps_remaining_pending(setup, monkeypatch):
    import signal
    import threading
    import time
    for log_n in range(8, 11):
        _miss(setup, _mapping(log_n))
    original = signal.getsignal(signal.SIGTERM)
    calls = []
    def compile_mapping(mapping, **kwargs):
        calls.append(mapping)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.02)
        path = Path(kwargs["cache"]) / "module.so"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        return path
    monkeypatch.setattr(broker, "_compiler", lambda root: SimpleNamespace(compile_mapping=compile_mapping))
    result = broker.work(setup.directory, jobs=1)
    assert result["interrupted"] and result["ready"] == 1 and result["pending"] == 2
    assert len(calls) == 1 and signal.getsignal(signal.SIGTERM) == original


def test_seed_cache_filters_library_time_sm_and_deduplicates(setup, monkeypatch):
    library = setup.root / "frozen-library.so"
    library.write_bytes(b"frozen library")
    timestamp = 10_000_000_000
    os.utime(library, ns=(timestamp, timestamp))
    cache = Path(setup.config["cache"])
    def manifest(name, mapping, mtime, sm=80):
        path = cache / name / "manifest.json"
        broker._atomic(path, {"schema": "cubutterfly-module-v1", "sm": sm,
                              "mapping": mapping, "module_path": "/untrusted/old/module.so"})
        os.utime(path, ns=(mtime, mtime))
    manifest("old", _mapping(7), timestamp - 1)
    manifest("recent", _mapping(8), timestamp)
    manifest("duplicate", _mapping(8), timestamp + 1)
    manifest("wrong-sm", _mapping(9), timestamp + 1, sm=90)
    result = broker.seed_cache_since(setup.directory, library)
    assert result["recent_manifests"] == 3 and result["unique_mappings"] == 1
    assert result["queued"] == 1 and result["ready"] == result["failed"] == 0
    queued = list((setup.directory / "requests").glob("*.json"))
    assert len(queued) == 1
    request = broker._read(queued[0])
    assert request["mapping"] == _mapping(8) and request["broker_id"] == setup.document["id"]
    calls = _fake_compiler(setup, monkeypatch)
    assert broker.work(setup.directory)["ready"] == 1
    assert len(calls) == 1 and calls[0][0] == _mapping(8)
    result = broker.seed_cache_since(setup.directory, library)
    assert result["ready"] == 1 and result["queued"] == 0
    assert broker._read(setup.directory / "cache_seed_status.json")["source"]["library"] == str(library)


def test_seed_reports_recent_invalid_manifest_without_old_history_read(setup):
    library = setup.root / "frozen-library.so"
    library.write_bytes(b"frozen")
    timestamp = library.stat().st_mtime_ns
    for name, mtime in (("old-broken", timestamp - 1), ("recent-broken", timestamp + 1)):
        path = Path(setup.config["cache"]) / name / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text("not json")
        os.utime(path, ns=(mtime, mtime))
    result = broker.seed_cache_since(setup.directory, library)
    assert result["recent_manifests"] == 1
    assert len(result["invalid_manifests"]) == 1
    assert "recent-broken" in result["invalid_manifests"][0]["path"]
    assert result["queued"] == 0


def test_watch_stays_alive_for_new_work_updates_heartbeat_and_stops(setup, monkeypatch):
    import signal
    calls = _fake_compiler(setup, monkeypatch)
    original = signal.getsignal(signal.SIGTERM)
    sleeps = []
    def pause(seconds):
        sleeps.append(seconds)
        assert 0 < seconds <= 1
        heartbeat = broker._read(setup.directory / "producer_status.json")
        assert heartbeat["status"] == "running" and heartbeat["pid"] == os.getpid()
        if len(sleeps) == 1:
            _miss(setup)
        else:
            os.kill(os.getpid(), signal.SIGTERM)
    monkeypatch.setattr(broker.time, "sleep", pause)
    sources = broker._sources
    source_calls = []
    def read_sources(config):
        source_calls.append(config)
        return sources(config)
    monkeypatch.setattr(broker, "_sources", read_sources)
    result = broker.watch(setup.directory, jobs=2)
    assert result["interrupted"] and result["ready"] == 1
    assert len(calls) == 1 and len(source_calls) == 1
    assert len(sleeps) == 2 and signal.getsignal(signal.SIGTERM) == original
    final = broker._read(setup.directory / "producer_status.json")
    assert final["status"] == "stopped" and final["ready"] == 1
    assert final["broker_id"] == setup.document["id"]


def test_watch_signal_during_compile_finishes_only_active_work(setup, monkeypatch):
    import signal
    for log_n in range(8, 11):
        _miss(setup, _mapping(log_n))
    calls = []
    def compile_mapping(mapping, **kwargs):
        calls.append(mapping)
        os.kill(os.getpid(), signal.SIGTERM)
        path = Path(kwargs["cache"]) / "module.so"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        return path
    monkeypatch.setattr(broker, "_compiler", lambda root: SimpleNamespace(compile_mapping=compile_mapping))
    result = broker.watch(setup.directory, jobs=1)
    assert result["interrupted"] and result["ready"] == 1 and result["pending"] == 2
    assert len(calls) == 1
    assert broker._read(setup.directory / "producer_status.json")["status"] == "stopped"


def test_watch_delayed_seed_does_not_block_heartbeat_or_runtime_compile(setup, monkeypatch):
    import signal
    import threading
    seed_started = threading.Event()
    seed_finished = threading.Event()
    pause_event = threading.Event()
    def delayed_seed(directory, library, timeout, *, _stop_requested):
        seed_started.set()
        while not _stop_requested():
            pause_event.wait(0.005)
        seed_finished.set()
        return {"interrupted": True}
    monkeypatch.setattr(broker, "seed_cache_since", delayed_seed)
    calls = _fake_compiler(setup, monkeypatch)
    pauses = []
    def pause(seconds):
        pauses.append(seconds)
        assert seed_started.wait(1) and not seed_finished.is_set()
        heartbeat = broker._read(setup.directory / "producer_status.json")
        assert heartbeat["status"] == "running"
        if len(pauses) == 1:
            _miss(setup)
        else:
            assert len(calls) == 1 and broker.status(setup.directory)["ready"] == 1
            os.kill(os.getpid(), signal.SIGTERM)
    monkeypatch.setattr(broker.time, "sleep", pause)
    result = broker.watch(setup.directory, seed_library=setup.root / "library.so")
    assert result["ready"] == 1 and seed_finished.is_set()
    final = broker._read(setup.directory / "producer_status.json")
    assert final["status"] == "stopped" and final["seed_status"] == "stopped"
    assert final["seed_error"] is None


def test_watch_seed_exception_is_retained_in_status(setup, monkeypatch):
    import signal
    import threading
    seed_finished = threading.Event()
    def failed_seed(*args, **kwargs):
        seed_finished.set()
        raise OSError("fixture cache scan failed")
    monkeypatch.setattr(broker, "seed_cache_since", failed_seed)
    _fake_compiler(setup, monkeypatch)
    def pause(seconds):
        assert seed_finished.wait(1)
        os.kill(os.getpid(), signal.SIGTERM)
    monkeypatch.setattr(broker.time, "sleep", pause)
    broker.watch(setup.directory, seed_library=setup.root / "library.so")
    final = broker._read(setup.directory / "producer_status.json")
    assert final["seed_status"] == "failed"
    assert final["seed_error"] == "fixture cache scan failed"


def test_seed_shutdown_stops_before_reading_next_manifest(setup):
    library = setup.root / "library.so"
    library.write_bytes(b"fixture")
    cache = Path(setup.config["cache"])
    path = cache / "unused" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text("invalid fixture that must not be read")
    result = broker.seed_cache_since(setup.directory, library, _stop_requested=lambda: True)
    assert result["interrupted"] and result["recent_manifests"] == 0
    assert result["invalid_manifests"] == []
