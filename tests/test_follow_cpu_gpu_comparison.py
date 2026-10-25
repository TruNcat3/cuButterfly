"""No-GPU lifecycle, frozen-input, and finalization tests for the follower."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('follow_cpu_gpu_test', SCRIPTS / 'follow_cpu_gpu_comparison.py')
follower = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(follower)
from semantic_campaign_manifest import manifest_hash


def test_proc_stat_comm_and_starttime():
    fields = ['S'] + ['0'] * 18 + ['9876'] + ['0'] * 10
    assert follower.parse_process_stat('123 (driver with ) parentheses) ' + ' '.join(fields)) == {
        'state': 'S', 'starttime': 9876}


@pytest.mark.parametrize('current,expected', [(None, 'exited'),
                                             ({'state': 'S', 'starttime': 2}, 'pid-reused'),
                                             ({'state': 'Z', 'starttime': 1}, 'exited-unreaped'),
                                             ({'state': 'S', 'starttime': 1}, 'running')])
def test_bound_process_lifecycle(current, expected):
    assert follower.driver_observation({'state': 'S', 'starttime': 1}, current) == expected


def setup_files(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'cases': []}))
    cpu_output = tmp_path / 'cpu'
    cpu_output.mkdir()
    (cpu_output / 'results.json').write_text(json.dumps({
        'identity': {'manifest_sha256': follower.cpu.semantic.digest({'cases': []})}, 'rows': []}))
    gpu = tmp_path / 'gpu.json'
    gpu.write_text(json.dumps({'status': 'complete'}))
    monkeypatch.setattr(follower, 'validate_manifest_hash', lambda _: True)
    events = []
    monkeypatch.setattr(follower.cpu, 'export_table', lambda *args: events.append('join'))
    monkeypatch.setattr(follower.gpu_review, 'review', lambda *args: events.append('review') or {
        'completed_checkpoint_cases': 2, 'expected_cases': 2,
        'rows': [{'status': 'accuracy-review-required'}]})
    return manifest, cpu_output, gpu, events


def test_only_finalize_once_after_pid_reuse(tmp_path, monkeypatch):
    manifest, output, gpu, events = setup_files(tmp_path, monkeypatch)
    identities = iter([{'state': 'S', 'starttime': 10}, {'state': 'S', 'starttime': 11}])
    monkeypatch.setattr(follower, 'process_identity', lambda _: next(identities))
    monkeypatch.setattr(follower.time, 'sleep', lambda seconds: events.append(('sleep', seconds)))
    assert follower.follow(manifest, tmp_path / 'prep', output, gpu, 123) == 0
    assert events == [('sleep', 30), 'join', 'review']
    status = json.loads((output / 'join_status.json').read_text())
    assert status['driver_starttime'] == 10
    assert status['driver_observation'] == 'pid-reused'
    assert status['status'] == 'joined-and-reviewed'
    assert status['gpu_calls'] == 0


def test_changed_cpu_results_refuses_merge(tmp_path, monkeypatch):
    manifest, output, gpu, events = setup_files(tmp_path, monkeypatch)
    identities = iter([{'state': 'S', 'starttime': 10}, None])
    monkeypatch.setattr(follower, 'process_identity', lambda _: next(identities))
    monkeypatch.setattr(follower.time, 'sleep', lambda _: (output / 'results.json').write_text('{}'))
    assert follower.follow(manifest, tmp_path / 'prep', output, gpu, 123) == 2
    assert events == []
    status = json.loads((output / 'join_status.json').read_text())
    assert status['status'] == 'failed'
    assert 'CPU results changed' in status['reason']


def test_missing_checkpoint_reports_honestly(tmp_path, monkeypatch):
    manifest, output, gpu, events = setup_files(tmp_path, monkeypatch)
    gpu.unlink()
    monkeypatch.setattr(follower, 'process_identity', lambda _: None)
    assert follower.follow(manifest, tmp_path / 'prep', output, gpu, 123) == 2
    assert events == []
    assert json.loads((output / 'join_status.json').read_text())['status'] == 'driver-ended-without-checkpoint'


def test_wrong_cpu_manifest_refuses_start(tmp_path, monkeypatch):
    manifest, output, gpu, events = setup_files(tmp_path, monkeypatch)
    (output / 'results.json').write_text(json.dumps({'identity': {'manifest_sha256': 'wrong'}}))
    assert follower.follow(manifest, tmp_path / 'prep', output, gpu, 123) == 2
    assert events == []
    assert 'another semantic manifest' in json.loads((output / 'join_status.json').read_text())['reason']


@pytest.mark.parametrize('change_source', [False, True])
def test_explicit_cpu_source_manifest_binding_and_freeze(tmp_path, monkeypatch, change_source):
    manifest, output, gpu, events = setup_files(tmp_path, monkeypatch)
    source = {'schema': 'cubutterfly-semantic-campaign-v1',
              'cases': [], 'fingerprints': {'library': 'old'}}
    source['manifest_sha256'] = manifest_hash(source)
    target = dict(source, fingerprints={'library': 'new'})
    target['manifest_sha256'] = manifest_hash(target)
    manifest.write_text(json.dumps(target))
    source_path = tmp_path / 'source.json'
    source_path.write_text(json.dumps(source))
    cpu_path = output / 'results.json'
    cpu_path.write_text(json.dumps({'identity': {'manifest_sha256': follower.cpu.semantic.digest(source)}, 'rows': []}))
    before = cpu_path.read_bytes()
    monkeypatch.setattr(follower.cpu.preparation, '_config', lambda _: ({
        'manifest_sha256': follower.cpu.semantic.digest(source),
        'source_sha256': follower.cpu.preparation._sources()}, 'config-sha'))
    if change_source:
        identities = iter([{'state': 'S', 'starttime': 10}, None])
        monkeypatch.setattr(follower, 'process_identity', lambda _: next(identities))
        monkeypatch.setattr(follower.time, 'sleep', lambda _: source_path.write_text('{}'))
    else:
        monkeypatch.setattr(follower, 'process_identity', lambda _: None)
    result = follower.follow(manifest, tmp_path / 'prep', output, gpu, 123, source_path)
    assert cpu_path.read_bytes() == before
    state = json.loads((output / 'join_status.json').read_text())
    if change_source:
        assert result == 2 and events == []
        assert 'source manifest changed' in state['reason']
    else:
        assert result == 0 and events == ['join', 'review']
        assert state['comparison_manifest_sha256'] == follower.cpu.semantic.digest(target)
        assert state['cpu_source_manifest_sha256'] == follower.cpu.semantic.digest(source)
