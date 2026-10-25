"""CPU-only queue tests isolate quarantine and fatal worker outcomes."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_async_semantic_queue as queue
from semantic_campaign_manifest import MANIFEST_SCHEMA, manifest_hash


@pytest.fixture
def setup(monkeypatch, tmp_path):
    cases = []
    for marker in range(3):
        semantics = {'operator': 0, 'marker': marker}
        cases.append({'id': queue.old.digest(semantics), 'descriptor': {'semantics': semantics},
                      'populations': ['test'], 'requested_variants': []})
    manifest = {'schema': MANIFEST_SCHEMA, 'cases': cases, 'invalid_requests': [], 'metrics': {}}
    manifest['manifest_sha256'] = manifest_hash(manifest)
    path = tmp_path / 'parent.json'
    path.write_text(json.dumps(manifest))
    args = SimpleNamespace(output=tmp_path / 'queue', library=tmp_path / 'lib.so',
        gpu_uuid='fake', compile_broker=tmp_path / 'broker', cpu_workers=2, in_flight=2)
    writes = []
    real_save = queue.save

    def save(target, data):
        if target.name == 'queue_status.json':
            writes.append(copy.deepcopy(data))
        real_save(target, data)

    monkeypatch.setattr(queue, 'save', save)
    monkeypatch.setattr(queue.continuation, 'quarantine_entry',
        lambda row, result_path: {'id': row['id'], 'status': 'correctness-failure', 'row': {'path': str(result_path)}})
    return SimpleNamespace(cases=cases, manifest=manifest, manifest_path=path, args=args,
                          state={}, writes=writes, cache=tmp_path / 'prepared')


def workers(monkeypatch, outcomes):
    results, selected_cases, commands = {}, [], []

    def run(command):
        commands.append(command)
        manifest_path = Path(command[command.index('--manifest') + 1])
        output = Path(command[command.index('--output') + 1]) / 'results.json'
        manifest = queue.continuation.read(manifest_path)
        selected_cases.append(manifest['cases'])
        outcome = outcomes[len(commands) - 1]
        rows = []
        for case, status in zip(manifest['cases'], outcome['rows']):
            rows.append({'id': case['id'], 'descriptor': case['descriptor'], 'status': status,
                         'baselines': outcome.get('baselines', []), 'error': 'test worker error'})
        results[str(output)] = ({'status': outcome['status'],
            'remaining_case_ids': [case['id'] for case in manifest['cases'][len(rows):]]}, rows, [])
        return SimpleNamespace(returncode=outcome['returncode'])

    monkeypatch.setattr(queue.subprocess, 'run', run)
    monkeypatch.setattr(queue.continuation, 'checked_report', lambda path, *args: results[str(path)])
    return selected_cases, commands


def test_one_correctness_failure_quarantines_only_that_case_then_runs_remaining(setup, monkeypatch):
    selected, commands = workers(monkeypatch, [
        {'rows': ['accuracy-review-required', 'correctness-failure'], 'status': 'stopped-investigation-required', 'returncode': 1},
        {'rows': ['pass'], 'status': 'complete', 'returncode': 0}])
    queue.run_phase('timing', setup.manifest_path, setup.cache, setup.args, setup.state, ['--library', 'fake'])
    assert selected == [setup.cases, setup.cases[2:]]
    assert all('--timing' in command for command in commands)
    assert all(command[command.index('--preparation') + 1] == str(setup.cache) for command in commands)
    phase = setup.state['phases']['timing']
    assert phase['status'] == 'complete-with-unresolved-evidence'
    assert phase['processed_cases'] == 2
    assert [entry['id'] for entry in phase['quarantine']] == [setup.cases[1]['id']]
    assert phase['shards'][0]['status'] == 'closed-with-quarantine'
    assert phase['shards'][1]['status'] == 'complete'


@pytest.mark.parametrize('outcome,match', [
    ({'rows': [], 'status': 'stopped-investigation-required', 'returncode': 1}, 'without one verified correctness failure'),
    ({'rows': ['runner-error'], 'status': 'stopped-investigation-required', 'returncode': 1}, 'unexpected measurement error'),
    ({'rows': ['pass'], 'status': 'stopped-investigation-required', 'returncode': 1,
      'baselines': [{'status': 'baseline-error'}]}, 'baseline error requires investigation')])
def test_compiler_runner_or_baseline_failure_stops_without_next_shard(setup, monkeypatch, outcome, match):
    selected, commands = workers(monkeypatch, [outcome])
    with pytest.raises(RuntimeError, match=match):
        queue.run_phase('recovery', setup.manifest_path, setup.cache, setup.args, setup.state, [])
    assert len(commands) == 1
    assert '--timing' not in commands[0]
    assert setup.state['phases']['recovery']['quarantine'] == []
    assert setup.state['phases']['recovery']['status'] == 'running'


def test_successful_worker_cannot_skip_remaining_population(setup, monkeypatch):
    selected, commands = workers(monkeypatch, [
        {'rows': ['pass'], 'status': 'complete', 'returncode': 0}])
    with pytest.raises(RuntimeError, match='success with incomplete population'):
        queue.run_phase('recovery', setup.manifest_path, setup.cache, setup.args, setup.state, [])
    assert len(commands) == 1


def test_complete_phase_never_relaunches_a_worker(setup, monkeypatch):
    setup.state['phases'] = {'timing': {'status': 'complete-with-unresolved-evidence', 'shards': [], 'quarantine': []}}

    def forbidden(*args):
        pytest.fail('completed phase must not launch a worker')

    monkeypatch.setattr(queue.subprocess, 'run', forbidden)
    queue.run_phase('timing', setup.manifest_path, setup.cache, setup.args, setup.state, [])
    assert setup.writes == []


def test_scaled_controls_keep_untimed_separate_preparation(setup, monkeypatch):
    selected, commands = workers(monkeypatch, [
        {'rows': ['pass', 'pass', 'pass'], 'status': 'complete', 'returncode': 0}])
    scaled_cache = setup.args.output / 'scaled_preparation/prepared'
    queue.run_phase('scaled_controls', setup.manifest_path, scaled_cache, setup.args, setup.state, [])
    assert '--timing' not in commands[0]
    assert commands[0][commands[0].index('--preparation') + 1] == str(scaled_cache)
    assert setup.state['phases']['scaled_controls']['processed_cases'] == 3
