"""Idle/resume lifecycle checks with fake GPU observations and tiny CPU children."""
import fcntl
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SPEC = importlib.util.spec_from_file_location('idle_monitor',
    Path(__file__).resolve().parents[1] / 'scripts/watch_idle_campaign.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)

XML = '''<nvidia_smi_log><gpu><uuid>GPU-test</uuid><product_name>A100</product_name>
<fb_memory_usage><total>81920 MiB</total><used>18 MiB</used></fb_memory_usage>
<utilization><gpu_util>0 %</gpu_util><memory_util>0 %</memory_util></utilization>
<processes>{processes}</processes></gpu></nvidia_smi_log>'''


def snapshot():
    return M.parse_gpu_snapshot(XML.format(processes=''), 'GPU-test')


def test_idle_requires_all_metrics_and_no_graphics_or_compute_process():
    idle = snapshot()
    assert M.is_idle(idle, 64)
    for key, value in [('pids', [42]), ('memory_used_mib', 65),
                       ('gpu_utilization', 1), ('memory_utilization', 1)]:
        assert not M.is_idle(dict(idle, **{key: value}), 64)
    raw = XML.format(processes='<process_info><pid>42</pid><type>G</type></process_info>')
    assert not M.is_idle(M.parse_gpu_snapshot(raw, 'GPU-test'), 64)


def test_explicit_idle_display_exception_never_permits_compute_or_load():
    raw = XML.format(processes='<process_info><pid>42</pid><type>G</type>'
        '<process_name>/usr/lib/xorg/Xorg</process_name></process_info>')
    s = M.parse_gpu_snapshot(raw, 'GPU-test')
    allowed = ['/usr/lib/xorg/Xorg']
    assert not M.is_idle(s, 64)
    assert M.is_idle(s, 64, allowed)
    assert not M.is_idle(dict(s, gpu_utilization=1), 64, allowed)
    assert not M.is_idle(dict(s, memory_used_mib=65), 64, allowed)
    for kind in ('C', 'C+G'):
        assert not M.is_idle(M.parse_gpu_snapshot(raw.replace('<type>G', '<type>'+kind),
                                                 'GPU-test'), 64, allowed)


@pytest.mark.parametrize('raw', [XML.format(processes='N/A'),
    XML.format(processes='').replace('0 %', 'N/A'),
    XML.format(processes='').replace('18 MiB', 'nan MiB'),
    XML.format(processes='').replace('GPU-test', 'GPU-other')])
def test_unavailable_observation_never_becomes_idle(raw):
    with pytest.raises(ValueError):
        M.parse_gpu_snapshot(raw, 'GPU-test')


def test_retry_requires_fresh_explicit_interruption_not_stale_status():
    interruption = dict(status='interrupted-by-foreign-gpu-work', retryable=True, updated_unix=12)
    assert M.exit_action(1, interruption, 11) == 'retry'
    assert M.exit_action(1, interruption, 13) == 'failed'
    assert M.exit_action(1, dict(interruption, retryable=False), 11) == 'failed'
    assert M.exit_action(1, dict(interruption, status='failed'), 11) == 'failed'
    assert M.exit_action(0, {'status': 'complete', 'remaining': 1}, 11) == 'failed'
    assert M.exit_action(0, {'status': 'complete', 'remaining': 0}, 11) == 'finished'
    assert M.exit_action(73, {}, 11) == 'wait-for-driver'


def test_singleton_cannot_overwrite_or_launch(tmp_path):
    path = tmp_path / 'monitor_status.json'
    path.write_text('preserve')
    with (tmp_path / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            M.watch(tmp_path, 'GPU-test', [sys.executable, '-c', 'raise AssertionError'])
    assert path.read_text() == 'preserve'


def test_query_failure_and_busy_reset_idle_window_before_launch(tmp_path, monkeypatch):
    # Use an observation error the real nvidia-smi path can raise, then a busy GPU.
    observations = iter([ValueError('NVML unavailable'), dict(snapshot(), pids=[42]), snapshot()])
    queried = []

    def query(uuid):
        value = next(observations, snapshot())
        queried.append(value)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(M, 'gpu_snapshot', query)
    command = [sys.executable, '-c',
        'from pathlib import Path; Path("status.json").write_text(\'{"status":"complete","remaining":0}\')']
    assert M.watch(tmp_path, 'GPU-test', command, .01, .025) == 0
    state = json.loads((tmp_path / 'monitor_status.json').read_text())
    events = [json.loads(line)['status'] for line in (tmp_path / 'monitor_events.jsonl').read_text().splitlines()]
    assert events[:4] == ['starting', 'waiting-for-gpu-query', 'waiting-for-idle', 'confirming-idle']
    assert len(queried) >= 6
    assert state['attempts'] == 1 and state['status'] == 'finished'


def test_interrupted_child_retries_once_then_completes(tmp_path, monkeypatch):
    monkeypatch.setattr(M, 'gpu_snapshot', lambda uuid: snapshot())
    command = [sys.executable, '-c', '''
import json, time
from pathlib import Path
p = Path('count')
n = int(p.read_text()) + 1 if p.exists() else 1
p.write_text(str(n))
state = dict(status='interrupted-calibration', retryable=True, updated_unix=time.time()) if n == 1 else dict(status='complete', remaining=0)
Path('status.json').write_text(json.dumps(state))
raise SystemExit(1 if n == 1 else 0)
''']
    assert M.watch(tmp_path, 'GPU-test', command, .01, .01) == 0
    state = json.loads((tmp_path / 'monitor_status.json').read_text())
    assert state['attempts'] == 2 and state['status'] == 'finished'
    assert (tmp_path / 'count').read_text() == '2'


def test_actual_failure_stops_instead_of_reusing_stale_retry_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(M, 'gpu_snapshot', lambda uuid: snapshot())
    (tmp_path / 'status.json').write_text(json.dumps(
        dict(status='interrupted-calibration', retryable=True, updated_unix=1)))
    assert M.watch(tmp_path, 'GPU-test', [sys.executable, '-c', 'raise SystemExit(7)'], .01, 0) == 7
    state = json.loads((tmp_path / 'monitor_status.json').read_text())
    assert state['attempts'] == 1 and state['status'] == 'failed'


def test_completed_campaign_does_not_query_gpu_or_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(M, 'gpu_snapshot', lambda uuid: pytest.fail('no query expected'))
    (tmp_path / 'status.json').write_text('{"status":"complete","remaining":0}')
    assert M.watch(tmp_path, 'GPU-test', [sys.executable, '-c', 'raise AssertionError'], .01, 0) == 0
    assert json.loads((tmp_path / 'monitor_status.json').read_text())['attempts'] == 0


def test_existing_failure_is_not_retried_on_start_or_after_manual_driver(tmp_path, monkeypatch):
    monkeypatch.setattr(M, 'gpu_snapshot', lambda uuid: pytest.fail('no GPU query expected'))
    observed = []

    def driver_lock(root):
        observed.append(1)
        return len(observed) == 1

    monkeypatch.setattr(M, 'driver_locked', driver_lock)
    (tmp_path / 'status.json').write_text('{"status":"failed","retryable":false}')
    assert M.watch(tmp_path, 'GPU-test', [sys.executable, '-c', 'raise AssertionError'], .01, 0) == 2
    assert json.loads((tmp_path / 'monitor_status.json').read_text())['attempts'] == 0


def test_waits_for_cpu_publication_before_gpu_checks(tmp_path, monkeypatch):
    prepared = tmp_path / 'cpu.json'
    prepared.write_text('{"status":"running","scoring_complete":false}')
    ticks = []
    def tick(duration):
        ticks.append(duration)
        prepared.write_text('{"status":"scoring-complete","scoring_complete":true}')
    monkeypatch.setattr(M.time, 'sleep', tick)
    def query(uuid):
        assert ticks
        return snapshot()
    monkeypatch.setattr(M, 'gpu_snapshot', query)
    command = [sys.executable, '-c',
        'from pathlib import Path; Path("status.json").write_text(\'{"status":"complete","remaining":0}\')']
    assert M.watch(tmp_path, 'GPU-test', command, .01, 0,
                   cpu_preparation_status=prepared) == 0
    events = [json.loads(x)['status'] for x in (tmp_path/'monitor_events.jsonl').read_text().splitlines()]
    assert 'waiting-for-cpu-preparation' in events
