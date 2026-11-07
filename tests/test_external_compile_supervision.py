"""CPU producer lifetime must be independent of interrupted GPU drivers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_full_semantic_refresh as refresh


@pytest.fixture
def service(tmp_path, monkeypatch):
    script = tmp_path / 'scripts/semantic_compile_producer.py'
    script.parent.mkdir()
    directory = tmp_path / 'broker'
    directory.mkdir()
    script.write_text('''import fcntl,json,os,sys,time
from pathlib import Path
p=Path(sys.argv[sys.argv.index('--broker')+1])
with (p/'standalone_producer.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 while True:
  r=dict(schema='cubutterfly-semantic-compile-producer-v1',broker_id='fixture',
         pid=os.getpid(),status='running',heartbeat_ns=time.time_ns())
  t=p/'status.tmp'; t.write_text(json.dumps(r)); t.replace(p/'producer_status.json')
  time.sleep(.02)
''')
    child = subprocess.Popen([sys.executable, str(script), '--broker', str(directory)],
                             start_new_session=True)
    monkeypatch.setattr(refresh, 'ROOT', tmp_path)
    deadline = time.monotonic() + 5
    while not (directory / 'producer_status.json').exists():
        assert child.poll() is None and time.monotonic() < deadline
        time.sleep(.01)
    try:
        yield SimpleNamespace(child=child, directory=directory)
    finally:
        refresh.stop_owned(child)


def test_external_service_remains_alive_after_consumer_cleanup(service):
    observer = refresh.ExternalCompileProducer(service.directory, 'fixture')
    assert observer.pid == service.child.pid and observer.poll() is None
    refresh.stop_producer(observer)
    assert service.child.poll() is None and observer.poll() is None


def test_external_service_wrong_broker_rejected(service):
    with pytest.raises(RuntimeError, match='identity or heartbeat'):
        refresh.ExternalCompileProducer(service.directory, 'different')
    assert service.child.poll() is None


@pytest.mark.parametrize('field,value', [('status', 'stopped'), ('heartbeat_ns', 1),
                                      ('pid', 1), ('broker_id', 'different')])
def test_changed_receipt_rejected(service, monkeypatch, field, value):
    observer = refresh.ExternalCompileProducer(service.directory, 'fixture')
    read = refresh.calibration.read
    def altered(path):
        result = read(path)
        result[field] = value
        return result
    monkeypatch.setattr(refresh.calibration, 'read', altered)
    with pytest.raises(RuntimeError, match='identity or heartbeat'):
        observer.poll()


def test_command_and_pid_start_identity_checked(service, monkeypatch):
    observer = refresh.ExternalCompileProducer(service.directory, 'fixture')
    monkeypatch.setattr(observer, '_start_ticks', lambda: 'reused-pid')
    with pytest.raises(RuntimeError, match='identity or heartbeat'):
        observer.poll()


def test_missing_lock_rejected(service, monkeypatch):
    observer = refresh.ExternalCompileProducer(service.directory, 'fixture')
    monkeypatch.setattr(refresh.fcntl, 'flock', lambda *args: None)
    with pytest.raises(RuntimeError, match='does not hold'):
        observer.poll()


def test_wrong_command_rejected(service, monkeypatch):
    observer = refresh.ExternalCompileProducer(service.directory, 'fixture')
    monkeypatch.setattr(refresh, 'ROOT', Path('/wrong-source-root'))
    with pytest.raises(RuntimeError, match='command does not match'):
        observer.poll()
