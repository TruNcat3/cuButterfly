"""CPU process-lifecycle checks; no GPU or campaign source changes."""
import fcntl
import importlib.util
import json
from pathlib import Path
import signal
import sys

import pytest

SPEC = importlib.util.spec_from_file_location('queue_supervisor', Path(__file__).resolve().parents[1] / 'scripts/supervise_semantic_queue.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_failed_child_records_exit_without_retry(tmp_path):
    code = MODULE.supervise(tmp_path, tmp_path / 'run', [sys.executable, '-c', 'print("once", flush=True); raise SystemExit(7)'], .01)
    state = json.loads((tmp_path / 'run/status.json').read_text())
    assert code == 7
    assert state['status'] == 'stopped'
    assert state['returncode'] == 7
    assert (tmp_path / 'run/queue.log').read_text().strip() == 'once'


def test_child_signal_and_success_are_distinct(tmp_path):
    command = [sys.executable, '-c', 'import os,signal; os.kill(os.getpid(), signal.SIGTERM)']
    assert MODULE.supervise(tmp_path, tmp_path / 'terminated', command, .01) == -signal.SIGTERM
    state = json.loads((tmp_path / 'terminated/status.json').read_text())
    assert state['child_signal'] == 'SIGTERM'
    assert MODULE.supervise(tmp_path, tmp_path / 'finished', [sys.executable, '-c', 'pass'], .01) == 0
    assert json.loads((tmp_path / 'finished/status.json').read_text())['status'] == 'finished'


def test_second_supervisor_cannot_launch_or_overwrite_status(tmp_path):
    state_dir = tmp_path / 'rejected'
    with (tmp_path / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            MODULE.supervise(tmp_path, state_dir, [sys.executable, '-c', 'raise SystemExit(0)'], .01)
    assert not (state_dir / 'status.json').exists()


def test_continuation_heartbeat_reads_active_shard_separately(tmp_path):
    measurement = tmp_path / 'conformance/shard-0000/measurement'
    measurement.mkdir(parents=True)
    (measurement / 'results.json').write_text(json.dumps({'status': 'running',
        'rows': [{'id': 'new'}], 'remaining_case_ids': ['next']}))
    (tmp_path / 'conformance/controller.json').write_text(json.dumps({'state': {
        'status': 'running', 'processed_case_ids': ['old'], 'quarantine': [{'id': 'failed'}],
        'shards': [{'output': str(measurement)}]}}))
    observed = MODULE.progress(tmp_path)['conformance_continuation']
    assert observed['processed_case_count'] == 1
    assert observed['quarantine_count'] == 1
    assert observed['latest_shard']['rows'] == 1
    assert observed['latest_shard']['remaining'] == 1
    assert observed['recorded_case_count'] == 3
    assert observed['unacknowledged_shard_rows'] == 1
    assert observed['latest_shard']['checkpoint_age_seconds'] >= 0
    assert 'rank_conformance' not in MODULE.progress(tmp_path)
    assert MODULE.progress(tmp_path)['comparison']['state'] == 'pending-conformance'


def test_acknowledged_shard_rows_are_not_counted_twice(tmp_path):
    measurement = tmp_path / 'conformance/shard-0000/measurement'
    measurement.mkdir(parents=True)
    (measurement / 'results.json').write_text(json.dumps({'status': 'stopped-correctness-failure',
        'rows': [{'id': 'new'}, {'id': 'failed'}], 'remaining_case_ids': ['next']}))
    (tmp_path / 'conformance/controller.json').write_text(json.dumps({'state': {
        'status': 'running', 'processed_case_ids': ['old', 'new'], 'quarantine': [{'id': 'failed'}],
        'shards': [{'output': str(measurement)}]}}))
    observed = MODULE.progress(tmp_path)['conformance_continuation']
    assert observed['recorded_case_count'] == 3
    assert observed['unacknowledged_shard_rows'] == 0
