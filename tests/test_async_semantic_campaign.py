"""CPU fakes exercise scheduler acceptance, retry, and recovery boundaries."""
import copy
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_async_semantic_campaign as campaign
import semantic_resource_compile_proxy as proxy


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


@pytest.mark.parametrize('guard_number, committed', [(1, []), (4, []), (5, ['one'])])
def test_foreign_gpu_guard_discards_unaccepted_work_and_resumes(setup, monkeypatch,
                                                               guard_number, committed):
    checks = 0
    closed = []

    def guard(uuid):
        nonlocal checks
        checks += 1
        if checks == guard_number:
            raise RuntimeError(f'GPU {uuid} has foreign compute processes: [123]; stopping without killing')

    def steps(case, *args):
        try:
            yield {'case': case['id']}
            return row(case)
        finally:
            closed.append(case['id'])

    monkeypatch.setattr(campaign.old, 'exclusive_gpu', guard)
    monkeypatch.setattr(campaign, 'execution_steps', steps)
    campaign.drive(setup.cases, setup.report, setup.args, None, setup.torch, None, Executor())
    assert setup.report['status'] == 'interrupted-by-foreign-gpu-work'
    assert setup.report['gpu_interruption']['gpu_uuid'] == setup.args.gpu_uuid
    assert [entry['id'] for entry in setup.report['rows']] == committed
    assert setup.report['remaining_case_ids'] == [name for name in ('one', 'two') if name not in committed]
    attempts = dict(setup.report['preparation_attempts'])
    if guard_number > 1:
        assert closed
    remaining = setup.report['remaining_case_ids'][:]

    monkeypatch.setattr(campaign.old, 'exclusive_gpu', lambda uuid: None)
    campaign.drive(setup.cases, setup.report, setup.args, None, setup.torch, None, Executor())
    assert setup.report['status'].startswith('complete')
    assert [entry['id'] for entry in setup.report['rows']] == ['one', 'two']
    for name in remaining:
        assert setup.report['preparation_attempts'][name] > attempts.get(name, 0)
    assert [entry['id'] for entry in setup.rows].count('one') == 1


@pytest.mark.parametrize('message', ['static ranking helper failed',
                                   'GPU fake query failed',
                                   'GPU another has foreign compute processes: [123]; stopping without killing',
                                   'GPU fake has foreign compute processes: []; stopping without killing'])
def test_other_guard_errors_still_require_investigation(setup, monkeypatch, message):
    def guard(uuid):
        raise RuntimeError(message)

    monkeypatch.setattr(campaign.old, 'exclusive_gpu', guard)
    def steps(*args):
        yield {}
    monkeypatch.setattr(campaign, 'execution_steps', steps)
    with pytest.raises(RuntimeError, match=message.replace('[', r'\[').replace(']', r'\]')):
        campaign.drive(setup.cases, setup.report, setup.args, None, setup.torch, None, Executor())
    assert setup.report['status'] == 'stopped-investigation-required'
    assert setup.report['rows'] == []


def test_busy_looking_runtime_error_from_execution_is_not_retryable(setup, monkeypatch):
    def steps(*args):
        raise RuntimeError('GPU fake has foreign compute processes: [123]; stopping without killing')
        yield

    monkeypatch.setattr(campaign, 'execution_steps', steps)
    with pytest.raises(RuntimeError):
        campaign.drive(setup.cases, setup.report, setup.args, None, setup.torch, None, Executor())
    assert setup.report['status'] == 'stopped-investigation-required'


def test_sigterm_inside_terminal_step_cannot_accept_result(setup, monkeypatch):
    def steps(case, *args):
        yield {}
        setup.handlers[signal.SIGTERM](signal.SIGTERM, None)
        return row(case)

    monkeypatch.setattr(campaign, 'execution_steps', steps)
    campaign.drive(setup.cases, setup.report, setup.args, None, setup.torch, None, Executor())
    assert setup.report['status'] == 'interrupted'
    assert setup.report['rows'] == []
    assert setup.report['interruption_diagnostics']['unaccepted_row']['id'] == 'one'


def test_sigterm_during_preparation_gap_has_no_active_helper(setup, monkeypatch):
    monkeypatch.setattr(campaign.preparation, 'record_for', lambda *args:
                        {'status': 'gap', 'gap_status': 'unsupported', 'reason': 'CPU gap'})

    def gap(*args):
        setup.handlers[signal.SIGTERM](signal.SIGTERM, None)
        raise campaign.old.Gap('unsupported', 'CPU gap')

    monkeypatch.setattr(campaign.preparation, 'load_case', gap)
    campaign.drive(setup.cases[:1], setup.report, setup.args, None, setup.torch, None, Executor())
    assert setup.report['status'] == 'interrupted'
    assert setup.report['rows'] == []
    assert setup.report['interruption_diagnostics']['unaccepted_row']['status'] == 'unsupported'


@pytest.mark.parametrize('stop_kind', ['foreign', 'signal'])
def test_failed_validation_and_interruption_retains_investigation(setup, monkeypatch, stop_kind):
    def steps(*args):
        yield {}

    class FailingExecutor(Executor):
        def submit(self, function, *args):
            if function is campaign.validate_payload:
                def interrupt():
                    if stop_kind == 'signal':
                        setup.handlers[signal.SIGTERM](signal.SIGTERM, None)
                    else:
                        def busy(uuid):
                            raise RuntimeError(f'GPU {uuid} has foreign compute processes: [123]; stopping without killing')
                        monkeypatch.setattr(campaign.old, 'exclusive_gpu', busy)
                class FailedFuture(Future):
                    def result(self):
                        interrupt()
                        return self.value
                return FailedFuture({'status': 'correctness-failure', 'error': 'wrong output'})
            return super().submit(function, *args)

    setup.args.in_flight = 1
    monkeypatch.setattr(campaign, 'execution_steps', steps)
    with pytest.raises(RuntimeError, match='investigation required for unaccepted errors'):
        campaign.drive(setup.cases, setup.report, setup.args, None, setup.torch, None, FailingExecutor())
    assert setup.report['status'] == 'stopped-investigation-required'
    assert setup.report['rows'] == []
    assert setup.report['interruption_diagnostics']['validation_results']['one']['error'] == 'wrong output'


@pytest.mark.parametrize('clean_exit', [False, True])
def test_real_process_group_ranking_cancellation_requires_signal_exit(tmp_path, clean_exit):
    helper = tmp_path / 'rank_static_candidates.py'
    helper.write_text('import signal, time\n' +
        ('signal.signal(signal.SIGTERM, lambda *args: exit(7))\n' if clean_exit else '') +
        'print("ready", flush=True)\nwhile True: time.sleep(1)\n')
    proxy_path = str(Path(proxy.__file__).resolve())
    worker = r'''
import json, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
from test_async_semantic_campaign import campaign, Context, Executor, row
directory, proxy_path = Path(sys.argv[2]), sys.argv[3]
campaign.HERE = directory
campaign.old.exclusive_gpu = lambda *args: None
campaign.old.write_report = lambda path, value: path.write_text(json.dumps(value))
campaign.preparation.record_for = lambda *args: {'status': 'ready'}
def steps(case, prepared, library, torch, stream, options):
    subprocess.run([sys.executable, proxy_path, str(directory / 'rank_static_candidates.py')])
    result = row(case)
    result.update(status='runtime-error', runtime_status=6,
        error='backend error: static ranking helper failed; inspect its diagnostic above')
    return result
    yield
campaign.execution_steps = steps
case = {'id': 'one', 'descriptor': {'semantics': {}}, 'populations': []}
report = {'rows': [], 'status': 'running', 'remaining_case_ids': ['one']}
args = SimpleNamespace(output=directory, preparation=directory / 'prep',
    compile_broker=directory / 'broker', gpu_uuid='fake', in_flight=1,
    max_prepare_attempts=4, timing=False, vkfft_library=None, gpuntt_library=None)
torch = SimpleNamespace(cuda=SimpleNamespace(stream=lambda stream: Context()))
campaign.drive([case], report, args, None, torch, None, Executor())
'''
    process = subprocess.Popen([sys.executable, '-c', worker, str(Path(__file__).parent),
                                str(tmp_path), proxy_path], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        assert select.select([process.stdout], [], [], 10)[0], 'helper did not start'
        assert process.stdout.readline().strip() == 'ready'
        os.killpg(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=10)
    report = json.loads((tmp_path / 'results.json').read_text())
    assert report['status'] == ('stopped-investigation-required' if clean_exit else 'interrupted')
    assert (process.returncode != 0) == clean_exit, (stdout, stderr)
    if not clean_exit:
        receipt = report['interruption_diagnostics']['ranking_helper_cancellation']
        assert receipt['termination_signal'] == signal.SIGTERM
        assert receipt['worker_pid'] == process.pid
        # Helper receipts cannot masquerade as compilation requests.
        import semantic_gpu_pipeline
        semantic_gpu_pipeline.check_preparation_events(tmp_path / 'compile_events/one-1.jsonl')
    assert report['rows'] == []
    assert report['remaining_case_ids'] == ['one']


@pytest.mark.parametrize('defect', ['token', 'worker', 'helper_pid', 'stale', 'sigkill',
                                  'no_proxy_signal', 'ordinary_exit', 'wrong_error', 'wrong_status',
                                  'compile_failure', 'baseline_failure', 'validation_failure', 'not_stopping',
                                  'different_parent_signal'])
def test_ranking_cancellation_rejects_unrelated_evidence(setup, monkeypatch, defect):
    def steps(case, prepared, library, torch, stream, options):
        started = time.time()
        identity = dict(worker_pid=os.getpid(), proxy_pid=123, helper_pid=124,
            script=str(campaign.HERE / 'rank_static_candidates.py'), started_unix=started,
            attempt_token=os.environ['CUBUTTERFLY_HELPER_ATTEMPT'])
        if defect == 'token': identity['attempt_token'] = 'another-attempt'
        if defect == 'worker': identity['worker_pid'] += 1
        if defect == 'helper_pid': identity['helper_pid'] = None
        if defect == 'stale': identity['started_unix'] -= 100
        signum = signal.SIGKILL if defect == 'sigkill' else signal.SIGTERM
        end = dict(identity, kind='cpu_helper_finished', ended_unix=time.time(),
            returncode=7 if defect == 'ordinary_exit' else -signum,
            termination_signal=signum,
            received_signals=[] if defect == 'no_proxy_signal' else
                [dict(signal=signal.SIGTERM, received_unix=started)])
        records = [dict(identity, kind='cpu_helper_started'), end]
        if defect == 'compile_failure': records.append(dict(kind='compile_failed'))
        options.events_path.parent.mkdir(parents=True, exist_ok=True)
        options.events_path.write_text(''.join(json.dumps(record) + '\n' for record in records))
        if defect == 'validation_failure':
            yield {}
        if defect != 'not_stopping':
            signum = signal.SIGINT if defect == 'different_parent_signal' else signal.SIGTERM
            setup.handlers[signum](signum, None)
        result = row(case, [{'status': 'baseline-error'}] if defect == 'baseline_failure' else [])
        result.update(status='runtime-error', runtime_status=5 if defect == 'wrong_status' else 6,
            error='other runtime failure' if defect == 'wrong_error' else
                'backend error: static ranking helper failed; inspect its diagnostic above')
        return result

    class FailedValidation(Executor):
        def submit(self, function, *args):
            if function is campaign.validate_payload:
                return Future({'status': 'correctness-failure', 'error': 'wrong output'})
            return super().submit(function, *args)

    setup.args.in_flight = 1
    monkeypatch.setattr(campaign, 'execution_steps', steps)
    with pytest.raises(RuntimeError, match='investigation required'):
        campaign.drive(setup.cases[:1], setup.report, setup.args, None, setup.torch, None, FailedValidation())
    assert setup.report['status'] == 'stopped-investigation-required'


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
