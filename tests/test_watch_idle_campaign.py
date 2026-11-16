"""Waiting budgets use a fake clock and never query a real GPU."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'budget_monitor', Path(__file__).resolve().parents[1] / 'scripts/watch_idle_campaign.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        now = 0

        def sleep(self, seconds):
            self.now += seconds

    clock = Clock()
    monkeypatch.setattr(M.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(M.time, 'time', lambda: 1000 + clock.now)
    monkeypatch.setattr(M.time, 'sleep', clock.sleep)
    monkeypatch.setattr(M, 'driver_locked', lambda root: False)
    monkeypatch.setattr(M, 'gpu_snapshot', lambda uuid: dict(
        pids=[], processes=[], gpu_utilization=0,
        memory_utilization=0, memory_used_mib=0))
    return clock


def status(root):
    return json.loads((root / 'monitor_status.json').read_text())


def test_wait_expires_and_resume_preserves_campaign(tmp_path, clock):
    campaign = dict(status='measuring', measured_rows=3, remaining=4)
    campaign_path = tmp_path / 'status.json'
    campaign_path.write_text(json.dumps(campaign))
    for expected_time in (2, 4):
        assert M.watch(tmp_path, 'fake', ['unused'], poll_seconds=10,
                       idle_seconds=30, wait_hours=2 / 3600) == 124
        assert clock.now == expected_time
        assert status(tmp_path)['status'] == 'deadline-expired'
        assert status(tmp_path)['attempts'] == 0
        assert json.loads(campaign_path.read_text()) == campaign


def test_already_complete(tmp_path, clock):
    (tmp_path / 'status.json').write_text(json.dumps(dict(status='complete', remaining=0)))
    assert M.watch(tmp_path, 'fake', ['unused'], wait_hours=1) == 0
    assert status(tmp_path)['status'] == 'finished'


@pytest.mark.parametrize('retry', [False, True])
def test_active_child_can_pass_deadline(tmp_path, clock, monkeypatch, retry):
    launches = []

    class Child:
        pid = 999

        def poll(self):
            if clock.now < 4:
                return None
            campaign = (dict(status='interrupted-by-foreign-gpu-work', retryable=True,
                             updated_unix=1000 + clock.now)
                        if retry else dict(status='complete', remaining=0))
            (tmp_path / 'status.json').write_text(json.dumps(campaign))
            # The watcher reads campaign immediately before polling.
            return None if len(launches) == 1 and clock.now == 4 else (75 if retry else 0)

        def terminate(self):
            pytest.fail('budget must not terminate a running child')

    def launch(*args, **kwargs):
        launches.append(clock.now)
        return Child()

    monkeypatch.setattr(M.subprocess, 'Popen', launch)
    result = M.watch(tmp_path, 'fake', ['unused'], poll_seconds=1,
                     idle_seconds=0, wait_hours=2 / 3600)
    assert result == (124 if retry else 0)
    assert clock.now >= 4
    assert launches == [0]
    assert status(tmp_path)['status'] == ('deadline-expired' if retry else 'finished')


@pytest.mark.parametrize('hours', [0, -1, float('inf'), float('-inf'), float('nan')])
def test_invalid_budget(tmp_path, hours):
    with pytest.raises(ValueError, match='positive and finite'):
        M.watch(tmp_path, 'fake', ['unused'], wait_hours=hours)
