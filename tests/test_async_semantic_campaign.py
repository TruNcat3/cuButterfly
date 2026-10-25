"""CPU fakes exercise scheduler acceptance, retry, and recovery boundaries."""
import copy
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_async_semantic_campaign as campaign


def test_completed_budget_recycles_preserving_uncommitted_cases(setup, monkeypatch):
    closed = []
    setup.args.resource_budget = campaign.resources.limits(max_cases=1)

    def steps(case, *args):
        try:
            yield {'case': case['id']}
            return row(case)
        finally:
            closed.append(case['id'])

    monkeypatch.setattr(campaign, 'execution_steps', steps)
    campaign.drive(setup.cases, setup.report, setup.args, None, setup.torch, None, Executor())
    assert setup.report['status'] == 'worker-recycle-required'
    assert [record['id'] for record in setup.report['rows']] == ['one']
    assert setup.report['remaining_case_ids'] == ['two']
    assert set(closed) == {'one', 'two'}


def test_recycle_during_selection_keeps_entire_case_uncommitted(setup, monkeypatch):
    closed = []

    def steps(case, *args):
        try:
            raise campaign.WorkerRecycle('selection VMA limit', {'vma_count': 45000})
            yield
        finally:
            closed.append(case['id'])

    monkeypatch.setattr(campaign, 'execution_steps', steps)
    campaign.drive(setup.cases, setup.report, setup.args, None, setup.torch, None, Executor())
    assert setup.report['status'] == 'worker-recycle-required'
    assert setup.report['rows'] == []
    assert setup.report['remaining_case_ids'] == ['one', 'two']
    assert closed == ['one']


def test_worker_attempt_budget_counts_selection_retries(setup, monkeypatch):
    setup.args.resource_budget = campaign.resources.limits(max_attempts=1)

    def steps(case, *args):
        raise campaign.PreparationPending(['module'])
        yield

    monkeypatch.setattr(campaign, 'execution_steps', steps)
    monkeypatch.setattr(campaign, 'request_state', lambda *args: True)
    campaign.drive(setup.cases, setup.report, setup.args, None, setup.torch, None, Executor())
    assert setup.report['status'] == 'worker-recycle-required'
    assert setup.report['rows'] == []
    assert setup.report['worker_resources']['plan_attempts'] == 1


class Context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class Future:
    def __init__(self, value, before_ready=None):
        self.value = value
        self.before_ready = before_ready

    def done(self):
        if self.before_ready is not None:
            callback, self.before_ready = self.before_ready, None
            callback()
            return False
        return True

    def result(self):
        return self.value


class Executor:
    def __init__(self, before_validation=None):
        self.before_validation = before_validation
        self.loads = 0
        self.validations = 0

    def submit(self, function, *args):
        if function is campaign.validate_payload:
            self.validations += 1
            return Future({'ok': True, 'status': 'pass'}, self.before_validation)
        self.loads += 1
        return Future({'prepared': args[-1]['id']})


@pytest.fixture
def setup(monkeypatch, tmp_path):
    cases = [{'id': name, 'descriptor': {'semantics': {}}, 'populations': []}
             for name in ('one', 'two')]
    report = {'rows': [], 'status': 'running', 'remaining_case_ids': ['one', 'two']}
    args = SimpleNamespace(output=tmp_path, preparation=tmp_path / 'prep',
        compile_broker=tmp_path / 'broker', gpu_uuid='fake', in_flight=2,
        max_prepare_attempts=4, timing=False, vkfft_library=None, gpuntt_library=None)
    checkpoints, rows, handlers = [], [], {}
    monkeypatch.setattr(campaign.old, 'exclusive_gpu', lambda *args: None)
    monkeypatch.setattr(campaign.old, 'write_report', lambda path, value: checkpoints.append(copy.deepcopy(value)))

    def write_row(directory, row):
        rows.append(copy.deepcopy(row))
        return {'id': row['id'], 'status': row['status'], 'baseline_status_counts': {}}

    monkeypatch.setattr(campaign.old, 'write_case_checkpoint', write_row)
    monkeypatch.setattr(campaign.preparation, 'record_for', lambda *args: {'status': 'ready'})
    monkeypatch.setattr(campaign.signal, 'signal', lambda sig, handler: handlers.setdefault(sig, handler))
    monkeypatch.setattr(campaign.time, 'sleep', lambda seconds: None)
    torch = SimpleNamespace(cuda=SimpleNamespace(stream=lambda stream: Context()))
    return SimpleNamespace(cases=cases, report=report, args=args, rows=rows,
                          checkpoints=checkpoints, handlers=handlers, torch=torch)


def row(case, baselines=None):
    return {'id': case['id'], 'status': 'pass', 'baselines': baselines or []}


def test_cpu_validation_pending_never_commits_terminal_result(setup, monkeypatch):
    accepted = []

    def steps(case, *args):
        token = yield {'case': case['id']}
        assert token['ok']
        accepted.append(case['id'])
        return row(case)

    monkeypatch.setattr(campaign, 'execution_steps', steps)
    inspected = []

    def before_ready():
        inspected.append(True)
        assert setup.report['rows'] == []
        assert setup.rows == []
        assert accepted == []

    executor = Executor(before_ready)
    campaign.drive(setup.cases, setup.report, setup.args, None, setup.torch, None, executor)
    assert len(inspected) == 2
    assert accepted == ['one', 'two']
    assert len(setup.rows) == 2 and setup.report['remaining_case_ids'] == []


def test_compile_miss_discards_attempt_and_retries_original_case_with_new_events(setup, monkeypatch):
    attempts, closed = [], []
    seen_blocked = []

    def steps(case, prepared, library, torch, stream, options):
        attempts.append((case, options.events_path))
        try:
            if len(attempts) == 1:
                raise campaign.PreparationPending(['candidate'])
            token = yield {'case': case['id']}
            assert token['ok']
            return row(case)
        finally:
            closed.append(options.events_path)

    def ready(directory, identifiers):
        seen_blocked.append(identifiers)
        assert setup.rows == []
        assert len(closed) == 1
        return True

    monkeypatch.setattr(campaign, 'execution_steps', steps)
    monkeypatch.setattr(campaign, 'request_state', ready)
    setup.report['remaining_case_ids'] = ['one']
    executor = Executor()
    campaign.drive(setup.cases[:1], setup.report, setup.args, None, setup.torch, None, executor)
    assert seen_blocked == [['candidate']]
    assert [path.name for _, path in attempts] == ['one-1.jsonl', 'one-2.jsonl']
    assert all(case is setup.cases[0] for case, _ in attempts)
    assert executor.loads == 2 and executor.validations == 1
    assert setup.report['preparation_attempts'] == {'one': 2}
    assert setup.report['waiting_compilation'] == {}
    assert len(setup.rows) == 1


def test_baseline_error_is_checkpointed_then_stops_investigation(setup, monkeypatch):
    def steps(case, *args):
        if False:
            yield None
        return row(case, [{'name': 'cufft', 'status': 'baseline-error'}])

    monkeypatch.setattr(campaign, 'execution_steps', steps)
    with pytest.raises(RuntimeError, match='baseline investigation required'):
        campaign.drive(setup.cases, setup.report, setup.args, None, setup.torch, None, Executor())
    assert setup.report['status'] == 'stopped-investigation-required'
    assert [entry['id'] for entry in setup.rows] == ['one']
    assert setup.report['remaining_case_ids'] == ['two']
    assert setup.checkpoints[-1]['status'] == 'stopped-investigation-required'


def test_resume_skips_committed_cases_and_advances_saved_attempt_count(setup, monkeypatch):
    setup.report['rows'] = [{'id': 'one', 'status': 'pass', 'baseline_status_counts': {}}]
    setup.report['remaining_case_ids'] = ['two']
    setup.report['preparation_attempts'] = {'two': 2}
    events = []

    def steps(case, prepared, library, torch, stream, options):
        events.append(options.events_path.name)
        yield {'case': case['id']}
        return row(case)

    monkeypatch.setattr(campaign, 'execution_steps', steps)
    executor = Executor()
    campaign.drive(setup.cases, setup.report, setup.args, None, setup.torch, None, executor)
    assert events == ['two-3.jsonl']
    assert executor.loads == 1
    assert setup.report['preparation_attempts'] == {'two': 3}
    assert [entry['id'] for entry in setup.report['rows']] == ['one', 'two']


def test_interruption_closes_suspended_attempt_without_accepting_it(setup, monkeypatch):
    closed = []

    def steps(case, *args):
        try:
            yield {'case': case['id']}
            return row(case)
        finally:
            closed.append(case['id'])

    monkeypatch.setattr(campaign, 'execution_steps', steps)

    def interrupt():
        setup.handlers[signal.SIGTERM](signal.SIGTERM, None)

    setup.report['remaining_case_ids'] = ['one']
    campaign.drive(setup.cases[:1], setup.report, setup.args, None, setup.torch, None, Executor(interrupt))
    assert setup.report['status'] == 'interrupted'
    assert setup.report['rows'] == []
    assert setup.report['remaining_case_ids'] == ['one']
    assert setup.report['preparation_attempts'] == {'one': 1}
    assert closed == ['one']


def test_failed_compile_record_never_grants_ready(tmp_path):
    import json
    records = tmp_path / 'records'
    records.mkdir()
    (records / 'candidate.json').write_text(json.dumps({'status': 'failed', 'error': 'nvcc error'}))
    with pytest.raises(campaign.PreparationFailed):
        campaign.request_state(tmp_path, ['candidate'])


def test_unsupported_candidate_is_terminal_but_pending_peer_still_blocks(tmp_path):
    import json
    records = tmp_path / 'records'
    records.mkdir()
    log = tmp_path / 'compiler.log'
    log.write_text('/mathdx/cufftdx/include/traits/detail/check_and_get_trait.hpp(94): error: static assertion failed with "This FFT configuration is not supported"\n')
    (records / 'unsupported.json').write_text(json.dumps({'status': 'failed', 'error': f'specialization compilation failed; see {log}'}))
    assert campaign.request_state(tmp_path, ['unsupported'])
    assert not campaign.request_state(tmp_path, ['unsupported', 'pending'])
    (records / 'pending.json').write_text(json.dumps({'status': 'ready'}))
    assert campaign.request_state(tmp_path, ['unsupported', 'pending'])
