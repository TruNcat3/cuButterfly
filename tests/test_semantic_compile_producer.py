"""Standalone producer scheduling checks; compilation is CPU-only and faked."""
import fcntl
from collections import deque
import os
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import semantic_compile_producer as producer

broker = producer.broker


@pytest.fixture
def setup(tmp_path, monkeypatch):
    directory = tmp_path / "broker"
    for name in ("requests", "records", "locks"):
        (directory / name).mkdir(parents=True)
    config = dict(root=str(tmp_path), mathdx=str(tmp_path), nvcc="fake", sm=80,
                  cache=str(tmp_path / "cache"))
    document = dict(id="test-broker", config=config)
    calls = []
    def compile_mapping(mapping, **kwargs):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        calls.append(mapping["number"])
        path = Path(kwargs["cache"]) / str(mapping["number"]) / "module.so"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"module")
        return path
    compiler = SimpleNamespace(compile_mapping=compile_mapping)
    monkeypatch.setattr(broker, "_validated_worker", lambda path: (document, compiler))
    def request(number, state=None):
        mapping = dict(number=number)
        identifier = broker._id(dict(broker_id=document["id"], mapping=mapping))
        broker._atomic(directory / "requests" / (identifier + ".json"),
                       dict(id=identifier, broker_id=document["id"], mapping=mapping, timeout=1))
        if state:
            broker._atomic(directory / "records" / (identifier + ".json"),
                           dict(id=identifier, broker_id=document["id"], status=state))
        return identifier
    return SimpleNamespace(directory=directory, document=document, calls=calls,
                           compiler=compiler, request=request)


def test_refill_is_parallel_and_does_not_globally_rescan(setup, monkeypatch):
    for number in range(20):
        setup.request(number)
    original = setup.compiler.compile_mapping
    barrier = threading.Barrier(4)
    # Directory order intentionally does not determine which first four are picked.
    first = set()
    first_lock = threading.Lock()
    def concurrent(mapping, **kwargs):
        with first_lock:
            first.add(mapping["number"])
            synchronize = len(first) <= 4
        if synchronize:
            barrier.wait(timeout=3)
        return original(mapping, **kwargs)
    setup.compiler.compile_mapping = concurrent
    monkeypatch.setattr(broker, "status", lambda *a: pytest.fail("full status scan"))
    result = producer.run(setup.directory, jobs=4, discovery_seconds=60)
    assert result["ready"] == 20 and result["completed"] == 20
    assert result["discovery_scans"] == 1
    assert len(setup.calls) == 20


def test_watch_discovers_new_arrivals_without_rereading_known_records(setup, monkeypatch):
    setup.request(0)
    original = setup.compiler.compile_mapping
    def compile_mapping(mapping, **kwargs):
        result = original(mapping, **kwargs)
        if mapping["number"] == 0:
            setup.request(1)
        return result
    setup.compiler.compile_mapping = compile_mapping
    reads = []
    original_read = broker._read
    def read(path):
        if Path(path).parent.name == "records":
            reads.append(Path(path).stem)
        return original_read(path)
    monkeypatch.setattr(broker, "_read", read)
    result = producer.run(setup.directory, jobs=2, watch=True, discovery_seconds=0.02,
                          _stop_requested=lambda: len(setup.calls) == 2)
    assert result["ready"] == 2 and result["interrupted"]
    assert max(reads.count(identifier) for identifier in set(reads)) == 1


def test_priority_closes_small_cases_and_preserves_background(setup, tmp_path):
    identifiers = [setup.request(number) for number in range(16)]
    queue = producer.Queue(setup.directory)
    queue.broker_id = setup.document["id"]
    queue.discover()
    queue.background = deque(identifiers)
    report = tmp_path / "results.json"
    broker._atomic(report, {"waiting_compilation": {"large": identifiers[5:], "small": identifiers[3:5]}})
    assert queue.priorities([str(tmp_path / "*.json")]) == []
    order = [queue.pop() for _ in identifiers]
    assert set(order[:2]) == set(identifiers[3:5])
    assert order[3] not in identifiers[3:]
    assert set(order) == set(identifiers)
    assert len(list((setup.directory / "requests").glob("*.json"))) == 16


def test_priority_reports_cached_and_latest_case_wins(setup, tmp_path, monkeypatch):
    first, second = setup.request(0), setup.request(1)
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    broker._atomic(old, {"waiting_compilation": {"case": [first]}})
    broker._atomic(new, {"waiting_compilation": {"case": [second]}})
    os.utime(old, ns=(1, 1))
    queue = producer.Queue(setup.directory)
    queue.broker_id = setup.document["id"]
    queue.discover()
    assert queue.priorities([str(tmp_path / "*.json")]) == []
    monkeypatch.setattr(broker, "_read", lambda *a: pytest.fail("unchanged report read"))
    assert queue.priorities([str(tmp_path / "*.json")]) == []
    assert queue.pop() == second


def test_failed_records_are_never_retried_and_stale_running_recovers(setup):
    setup.request(0, "failed")
    setup.request(1, "running")
    result = producer.run(setup.directory, jobs=2)
    assert result["failed"] == 1 and result["ready"] == 1
    assert setup.calls == [1]
    producer.run(setup.directory, jobs=2)
    assert setup.calls == [1]


def test_contended_request_lock_is_deferred_then_recovered(setup):
    identifier = setup.request(0, "running")
    lock = (setup.directory / "locks" / (identifier + ".lock")).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    def release():
        time.sleep(0.05)
        fcntl.flock(lock, fcntl.LOCK_UN)
    thread = threading.Thread(target=release)
    thread.start()
    try:
        result = producer.run(setup.directory, discovery_seconds=0.02)
    finally:
        thread.join()
        lock.close()
    assert result["ready"] == 1 and setup.calls == [0]


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_signal_finishes_active_without_scheduling_more(setup, monkeypatch, signum):
    for number in range(5):
        setup.request(number)
    original = setup.compiler.compile_mapping
    def compile_mapping(mapping, **kwargs):
        os.kill(os.getpid(), signum)
        time.sleep(0.05)
        return original(mapping, **kwargs)
    setup.compiler.compile_mapping = compile_mapping
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "original")
    result = producer.run(setup.directory, jobs=1)
    assert result["status"] == "stopped" and result["interrupted"]
    assert result["ready"] == 1 and result["pending"] == 4
    assert result["active"] == 0 and len(setup.calls) == 1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "original"


def test_conflicting_standalone_producer_is_rejected(setup):
    with (setup.directory / "standalone_producer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="another standalone"):
            producer.run(setup.directory)


def test_compile_failure_is_counted_and_preserved(setup):
    setup.request(0)
    def fail(mapping, **kwargs):
        setup.calls.append(mapping["number"])
        raise RuntimeError("fake nvcc failure")
    setup.compiler.compile_mapping = fail
    result = producer.run(setup.directory)
    assert result["failed"] == 1 and result["completed"] == 1
    assert producer.run(setup.directory)["failed"] == 1
    assert setup.calls == [0]
