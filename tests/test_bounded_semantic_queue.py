import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_bounded_semantic_queue as queue


def family_case(name, **changes):
    semantics = dict(operator=0, storage=6, compute=1, shape=[256], word_bits=None,
                     fft_type=0, length_mode=0, output_orders=[0], ntt_output_layout=0)
    semantics.update(changes)
    return {'id': name, 'descriptor': {'semantics': semantics}}


def test_family_selection_preserves_objects_and_stable_budget_fragments(tmp_path):
    cases = [family_case('a1'), family_case('b1', storage=7), family_case('a2'),
             family_case('b2', storage=7), family_case('a3')]
    completed, batches = set(), []
    while len(completed) < len(cases):
        selected = queue.ready_cases(cases, completed, {}, tmp_path, 2, True)
        assert len({queue.module_family(case) for case in selected}) == 1
        assert all(case is next(original for original in cases if original['id'] == case['id'])
                   for case in selected)
        batches.append([case['id'] for case in selected])
        completed.update(case['id'] for case in selected)
    assert batches == [['a1', 'a2'], ['b1', 'b2'], ['a3']]
    assert sum(map(len, batches)) == len(cases)


def test_family_selection_skips_blocked_family_without_scanning_other_records(tmp_path, monkeypatch):
    cases = [family_case('blocked'), family_case('b1', storage=7),
             family_case('a2'), family_case('b2', storage=7)]
    waiting = {'blocked': ['missing'], 'a2': ['unrelated']}
    (tmp_path / 'records').mkdir()
    (tmp_path / 'records/unrelated.json').write_text(json.dumps({'status': 'failed'}))
    selected = queue.ready_cases(cases, set(), waiting, tmp_path, 10, True)
    assert selected == [cases[1], cases[3]]
    assert waiting == {'blocked': ['missing'], 'a2': ['unrelated']}
    (tmp_path / 'records/missing.json').write_text(json.dumps({'status': 'ready'}))
    with pytest.raises(RuntimeError, match='compile request failed'):
        queue.ready_cases(cases, {'b1', 'b2'}, waiting, tmp_path, 10, True)


@pytest.mark.parametrize('changes', [dict(operator=2), dict(storage=7), dict(compute=2),
    dict(shape=[4096]), dict(word_bits=32), dict(fft_type=1), dict(length_mode=1)])
def test_family_key_retains_compile_dimensions(changes):
    assert queue.module_family(family_case('base')) != queue.module_family(family_case('changed', **changes))


def test_ntt_family_tracks_output_order_and_layout_only_for_ntt():
    base = family_case('base', operator=1, word_bits=64)
    for changes in (dict(output_orders=[1]), dict(ntt_output_layout=1)):
        assert queue.module_family(base) != queue.module_family(family_case('changed', operator=1,
            word_bits=64, **changes))
    assert queue.module_family(family_case('one')) == queue.module_family(family_case('two',
        output_orders=[1], ntt_output_layout=1, batch=3, inverse=1, placement=1))


def test_queue_records_grouping_option_and_requires_paired_client(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, 'source_identity', lambda: {})
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'gpu_uuid': 'cpu-test', 'phases': []}))
    output = tmp_path / 'output'
    assert queue.main(['--plan', str(plan), '--output', str(output), '--group-by-module-family']) == 0
    status = json.loads((output / 'queue_status.json').read_text())
    assert status['identity']['group_by_module_family'] is True
    assert status['revision'] == '0.9.3.4-async-experiments.3'
    plan.write_text(json.dumps({'compile_client': '/client', 'phases': []}))
    with pytest.raises(ValueError, match='must be paired'):
        queue.main(['--plan', str(plan), '--output', str(tmp_path / 'invalid')])


def test_worker_command_forwards_compile_client_and_socket(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'cases': [family_case('one')]}))
    monkeypatch.setattr(queue, 'validate_manifest_hash', lambda _: True)
    monkeypatch.setattr(queue.runner, 'validate_fingerprints', lambda *args: None)
    monkeypatch.setattr(queue.runner, 'exclusive_gpu', lambda *args: None)
    monkeypatch.setattr(queue, 'source_identity', lambda: {})
    monkeypatch.setattr(queue.evidence, 'submanifest', lambda parent, *args: parent)
    commands = []

    def capture(command, **kwargs):
        commands.append(command)
        raise RuntimeError('CPU command captured')

    monkeypatch.setattr(queue.subprocess, 'Popen', capture)
    args = SimpleNamespace(output=tmp_path / 'queue', shard_cases=128, cpu_workers=2,
                           in_flight=1, cpu_wait_seconds=10, group_by_module_family=True)
    plan = dict(library='/unused', gpu_uuid='cpu-test', compile_broker=str(tmp_path),
                compile_client='/client', compile_socket='/socket')
    with pytest.raises(RuntimeError, match='CPU command captured'):
        queue.run_phase(dict(name='phase', manifest=str(manifest), preparation='/prepared'),
                        args, plan, {'phases': {}, 'identity': {'sources': {}}})
    command = commands[0]
    assert command[command.index('--compile-client') + 1] == '/client'
    assert command[command.index('--compile-socket') + 1] == '/socket'
    assert command[command.index('--in-flight') + 1] == '1'


def test_ready_selection_never_changes_cases_or_drops_pending_requests(tmp_path):
    cases = [{'id': name, 'descriptor': {'tag': name}} for name in ('complete', 'blocked', 'ready', 'new')]
    records = tmp_path / 'records'
    records.mkdir()
    (records / 'done.json').write_text(json.dumps({'status': 'ready'}))
    waiting = {'blocked': ['missing'], 'ready': ['done']}
    selected = queue.ready_cases(cases, {'complete'}, waiting, tmp_path, 2)
    assert selected == cases[2:] and selected[0] is cases[2]
    assert waiting == {'blocked': ['missing'], 'ready': ['done']}
    (records / 'missing.json').write_text(json.dumps({'status': 'failed'}))
    with pytest.raises(RuntimeError, match='compile request failed'):
        queue.ready_cases(cases, {'complete'}, waiting, tmp_path, 2)


@pytest.mark.parametrize('code,status,remaining,rows,valid', [
    (75, 'worker-recycle-required', ['unfinished'], [], True),
    (75, 'running', ['unfinished'], [], False),
    (0, 'complete-with-gaps-and-pending-accuracy', [], [{'status': 'accuracy-review-required'}], True),
    (0, 'complete', ['unfinished'], [], False),
    (1, 'stopped-investigation-required', ['unfinished'], [{'status': 'correctness-failure'}], True),
    (1, 'stopped-investigation-required', ['unfinished'], [{'status': 'runner-error'}], False),
    (1, 'running', ['unfinished'], [], False),
])
def test_worker_exit_does_not_accept_partial_or_unknown_results(code, status, remaining, rows, valid):
    report = {'status': status, 'remaining_case_ids': remaining}
    if valid:
        queue.validate_worker_exit(code, report, rows)
    else:
        with pytest.raises(RuntimeError):
            queue.validate_worker_exit(code, report, rows)


def test_baseline_failure_blocks_even_successful_worker():
    report = {'status': 'complete', 'remaining_case_ids': []}
    with pytest.raises(RuntimeError, match='baseline'):
        queue.validate_worker_exit(0, report, [{'status': 'accuracy-review-required',
            'baselines': [{'status': 'runner-error'}]}])


def test_changed_historical_evidence_is_not_adopted(tmp_path):
    artifact = tmp_path / 'old.json'
    artifact.write_text('original')
    item = queue.evidence.evidence(artifact)
    signature = list(queue.evidence.file_signature(artifact))
    handoff = {'phases': {'one': {'evidence': [item], 'evidence_stat_signatures': {str(artifact): signature}}}}
    path = tmp_path / 'handoff.json'
    path.write_text(json.dumps(handoff))
    assert queue.verify_handoff(path) == handoff
    artifact.write_text('modified')
    with pytest.raises(ValueError, match='evidence changed'):
        queue.verify_handoff(path)
