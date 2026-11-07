"""CPU-only driver contract tests: no GPU queries, workers or compiler launches."""
import fcntl
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import run_fp16_local_core_study as study
from watch_idle_campaign import exit_action


@pytest.fixture
def tmp_path(tmpdir):
    return Path(str(tmpdir))


def test_trace_selection_uses_measured_best_and_deduplicates_default():
    def task(name, variant, threads=256, ept=16):
        return dict(id=name, n=4096, batch=5, variant=variant,
                    mapping=dict(tile_threads=threads, prefix_ept=ept))
    tasks = [task('wide', 'local', 512, 8), task('default', 'local'),
             task('limited', 'local', 1024, 4), task('scalar', 'scalar'), task('vendor', 'cufft')]
    manifest = dict(tasks=tasks, config=dict(trace_cases=[dict(n=4096, batch=5)]))
    completed = {t['id']: dict(status='complete', graph_timing=dict(median_ms=i+1))
                 for i, t in enumerate(tasks)}
    completed['limited'] = dict(status='resource-limit')
    assert study.trace_selection(manifest, completed) == ['wide', 'default', 'vendor']
    completed['default']['graph_timing']['median_ms'] = .1
    assert study.trace_selection(manifest, completed) == ['default', 'vendor']


def fake_child(polls, returncode=0):
    child = SimpleNamespace(pid=12345, returncode=returncode)
    child.poll = Mock(side_effect=polls)
    return child


def test_attempt_rejects_foreign_work_before_launch(tmp_path, monkeypatch):
    popen, stop = Mock(), Mock()
    monkeypatch.setattr(study, 'foreign', lambda *args: [777])
    monkeypatch.setattr(study.subprocess, 'Popen', popen)
    monkeypatch.setattr(study, 'stop_owned', stop)
    with pytest.raises(study.Interrupted, match='before launch'):
        study.run_attempt(['worker'], tmp_path / 'attempt', 'GPU-test')
    popen.assert_not_called()
    stop.assert_not_called()


@pytest.mark.parametrize('polls,observations', [([None], [[], [777]]), ([0], [[], [777]])])
def test_foreign_interruption_cleans_only_owned_child(tmp_path, monkeypatch, polls, observations):
    child = fake_child(polls)
    foreign, stop, popen = Mock(side_effect=observations), Mock(), Mock(return_value=child)
    monkeypatch.setattr(study, 'foreign', foreign)
    monkeypatch.setattr(study, 'stop_owned', stop)
    monkeypatch.setattr(study.subprocess, 'Popen', popen)
    with pytest.raises(study.Interrupted, match='foreign GPU work'):
        study.run_attempt(['worker'], tmp_path / 'attempt', 'GPU-test')
    stop.assert_called_once_with(child)
    assert popen.call_args[1]['start_new_session'] is True
    assert popen.call_args[1]['env']['CUDA_VISIBLE_DEVICES'] == 'GPU-test'
    # Foreign PID is observation only and is never passed to cleanup.
    assert foreign.call_args_list[-1][0] == ('GPU-test', child)


def test_attempt_timeout_is_a_failure_and_still_cleans_owned_child(tmp_path, monkeypatch):
    child, stop = fake_child([None]), Mock()
    monkeypatch.setattr(study, 'foreign', lambda *args: [])
    monkeypatch.setattr(study, 'stop_owned', stop)
    monkeypatch.setattr(study.subprocess, 'Popen', Mock(return_value=child))
    monkeypatch.setattr(study.time, 'monotonic', Mock(side_effect=[0, 2]))
    with pytest.raises(TimeoutError):
        study.run_attempt(['worker'], tmp_path / 'attempt', 'GPU-test', timeout=1)
    stop.assert_called_once_with(child)


def test_existing_attempt_is_not_reused(tmp_path, monkeypatch):
    attempt = tmp_path / 'attempt'
    attempt.mkdir()
    popen = Mock()
    monkeypatch.setattr(study.subprocess, 'Popen', popen)
    with pytest.raises(FileExistsError):
        study.run_attempt(['worker'], attempt, 'GPU-test')
    popen.assert_not_called()


def test_foreign_classifier_excludes_only_owned_descendants(monkeypatch):
    # Use real existing PIDs solely for /proc existence; mock every GPU query.
    parent, own = os.getppid(), os.getpid()
    monkeypatch.setattr(study, 'gpu_pids', lambda uuid: [parent, own])
    monkeypatch.setattr(study, 'descendant', lambda pid, root: pid == own)
    assert study.foreign('GPU-test', SimpleNamespace(pid=own)) == [parent]
    assert study.foreign('GPU-test') == [parent, own]


def test_driver_lock_prevents_second_run_before_any_gpu_work(tmp_path, monkeypatch):
    runner = Mock()
    monkeypatch.setattr(study, 'run', runner)
    monkeypatch.setattr(sys, 'argv', ['study', '--run', '--output', str(tmp_path)])
    with (tmp_path / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert study.main() == 73
    runner.assert_not_called()


def test_watcher_retries_only_fresh_foreign_interruptions():
    interrupted = dict(status='interrupted-by-foreign-gpu-work', retryable=True, updated_unix=11)
    assert exit_action(75, interrupted, 10) == 'retry'
    assert exit_action(75, interrupted, 12) == 'failed'
    assert exit_action(1, dict(status='failed', retryable=False), 10) == 'failed'
    assert exit_action(73, {}, 10) == 'wait-for-driver'
    assert exit_action(0, dict(status='complete-gpu-pending-cpu-review', remaining=0), 10) == 'finished'


@pytest.mark.parametrize('failure,retryable', [(study.Interrupted('foreign work'), True),
                                             (RuntimeError('numerical failure'), False)])
def test_driver_checkpoints_interruption_without_promoting_attempt(tmp_path, monkeypatch, failure, retryable):
    manifest = dict(gpu_uuid='GPU-test', worker_sha256='fixed', compiler_sha256='fixed',
                    tasks=[dict(id='only-task')])
    study.save(tmp_path / 'manifest.json', manifest)
    study.save(tmp_path / 'cpu_preparation.json', dict(scoring_complete=True, compile_ready=True))
    monkeypatch.setattr(study, 'sha256', lambda path: 'fixed')
    monkeypatch.setattr(study, 'run_attempt', Mock(side_effect=failure))
    monkeypatch.setattr(study.signal, 'signal', lambda *args: None)
    if retryable:
        assert study.run(tmp_path) == 75
    else:
        with pytest.raises(RuntimeError, match='numerical failure'):
            study.run(tmp_path)
    state = study.read(tmp_path / 'status.json')
    assert state['retryable'] is retryable
    assert state['measured_rows'] == 0
    assert not (tmp_path / 'measure' / 'only-task' / 'complete.json').exists()
    assert state['status'] == ('interrupted-by-foreign-gpu-work' if retryable else 'failed')


@pytest.mark.parametrize('field,value', [('task_id', 'wrong'), ('mode', 'capture'),
                                       ('manifest_sha256', 'wrong'), ('status', 'failed')])
def test_resume_rejects_mismatched_result_identity(tmp_path, field, value):
    record = dict(task_id='task', mode='measure', manifest_sha256='fixed', status='resource-limit')
    record[field] = value
    study.save(tmp_path / 'result.json', record)
    with pytest.raises(ValueError, match='identity/status'):
        study.checked_result(tmp_path, 'task', 'measure', 'fixed')


def test_resume_checks_record_hash(tmp_path):
    study.save(tmp_path / 'result.json', dict(task_id='task', mode='measure',
                                            manifest_sha256='fixed', status='resource-limit'))
    with pytest.raises(ValueError, match='result hash'):
        study.checked_result(tmp_path, 'task', 'measure', 'fixed', {'result_sha256': 'wrong'})


def resource_dump(variants=((0, 0), (0, 1), (1, 0), (1, 1))):
    return '\n'.join(' Function _ZNlow_block_pairE' + f'Lb{inverse}ELb{aligned}EEEv:' +
                     f'\n  REG:{96 + 16*inverse + 8*aligned} STACK:0 SHARED:0 LOCAL:{4*inverse} CONSTANT[0]:393'
                     for inverse, aligned in variants)


def test_resource_parser_retains_all_four_variants_and_worst_entry():
    entries = study.local_core_compiled_resources(resource_dump())
    assert len(entries) == 4
    assert {(e['inverse'], e['aligned_io']) for e in entries} == {(False, False), (False, True), (True, False), (True, True)}
    assert [e['reg'] for e in entries] == [96, 104, 112, 120]
    # The largest entry is last: neither an old two-entry slice nor forward-only
    # resource accounting may declare this point feasible at a 112-reg limit.
    assert max(e['reg'] for e in entries)*512 > 112*512
    assert max(e['local'] for e in entries) == 4


@pytest.mark.parametrize('raw', [
    '', resource_dump(((0, 0), (1, 0))), resource_dump(((0, 0), (0, 1), (1, 0))),
    resource_dump(((0, 0), (0, 1), (1, 0), (1, 0))),
    resource_dump().replace('REG:96', 'REG:invalid'),
    resource_dump().replace('REG:96', 'REG:0'),
    resource_dump().replace('REG:96', 'REG:256'),
    resource_dump().replace('LOCAL:0', 'LOCAL:-1', 1),
    resource_dump().replace(' STACK:0', '', 1),
    resource_dump() + '\n Function unexpected:\n  REG:20 STACK:0 SHARED:0 LOCAL:0',
    resource_dump() + '\n REG:25',
])
def test_resource_parser_rejects_missing_duplicate_or_abnormal_entries(raw):
    with pytest.raises(ValueError):
        study.local_core_compiled_resources(raw)
