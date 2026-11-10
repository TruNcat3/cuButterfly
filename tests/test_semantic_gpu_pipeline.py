"""CPU-only checks for asynchronous GPU execution protocol boundaries."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import semantic_gpu_pipeline as gpu


class FakeStream:
    cuda_stream = 42

    def synchronize(self):
        pass


class FakeEvent:
    def __init__(self, **kwargs):
        pass

    def record(self, stream):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, end):
        return 40.0


class FakeGraph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


class GraphContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_fixed_timing_runs_all_samples_atomically_then_waits_for_all_verdicts():
    counts = {'execute': 0, 'reset': 0, 'poison': 0}
    graph = FakeGraph()
    torch = SimpleNamespace(cuda=SimpleNamespace(Event=FakeEvent,
        CUDAGraph=lambda: graph, graph=lambda *args, **kwargs: GraphContext()))

    def increment(name):
        counts[name] += 1

    def verify(phase):
        token = yield {'phase': phase}
        gpu.merge_validation({}, token)

    steps = gpu.timing_steps(torch, FakeStream(), lambda: increment('execute'),
        lambda: increment('reset'), False, verify, lambda: increment('poison'))
    assert next(steps) == {'batch_snapshots': [{'phase': f'timing-{sample}'} for sample in range(7)]}
    assert counts == {'execute': 30, 'reset': 0, 'poison': 7}
    assert graph.replays == 7
    with pytest.raises(StopIteration) as completed:
        steps.send({'batch_results': [{'ok': True, 'status': 'pass'} for _ in range(7)]})
    assert completed.value.value['samples_ms'] == [2.0] * 7
    assert counts['execute'] == 30


def test_in_place_uses_ten_warmups_and_seven_reset_single_calls():
    counts = {'execute': 0, 'reset': 0}
    torch = SimpleNamespace(cuda=SimpleNamespace(Event=FakeEvent))

    def increment(name):
        counts[name] += 1

    def verify(phase):
        result = yield phase
        gpu.merge_validation({}, result)

    steps = gpu.timing_steps(torch, FakeStream(), lambda: increment('execute'),
                            lambda: increment('reset'), True, verify)
    assert next(steps) == {'batch_snapshots': [f'timing-{sample}' for sample in range(7)]}
    assert counts == {'execute': 17, 'reset': 17}
    with pytest.raises(StopIteration) as completed:
        steps.send({'batch_results': [{'ok': True} for _ in range(7)]})
    assert counts == {'execute': 17, 'reset': 17}
    assert completed.value.value['samples_ms'] == [40.0] * 7


@pytest.mark.parametrize('failure_index', range(7))
def test_any_atomic_timing_sample_failure_rejects_timing_and_closes_all_validations(failure_index):
    closed = []

    def verify(phase):
        try:
            result = yield {'phase': phase}
            gpu.merge_validation({}, result)
        finally:
            closed.append(phase)

    torch = SimpleNamespace(cuda=SimpleNamespace(Event=FakeEvent))
    steps = gpu.timing_steps(torch, FakeStream(), lambda: None, lambda: None, True, verify)
    payload = next(steps)
    assert len(payload['batch_snapshots']) == 7
    results = [{'ok': True, 'status': 'pass'} for _ in range(7)]
    results[failure_index] = {'ok': False, 'status': 'correctness-failure', 'error': 'sample failed'}
    with pytest.raises(AssertionError, match='sample failed'):
        steps.send({'batch_results': results})
    assert sorted(closed) == [f'timing-{sample}' for sample in range(7)]


@pytest.mark.parametrize('batch', [None, {}, {'batch_results': []}, {'batch_results': [{}] * 6}])
def test_atomic_timing_requires_exactly_seven_cpu_result_tokens(batch):
    def verify(phase):
        gpu.merge_validation({}, (yield phase))

    torch = SimpleNamespace(cuda=SimpleNamespace(Event=FakeEvent))
    steps = gpu.timing_steps(torch, FakeStream(), lambda: None, lambda: None, True, verify)
    next(steps)
    with pytest.raises(RuntimeError, match='verification batch'):
        steps.send(batch)


@pytest.mark.parametrize('result', [None, {}, {'ok': True, 'status': 'pending'}, {'ok': 'yes'}])
def test_verification_must_return_completed_boolean_token(result):
    with pytest.raises(RuntimeError):
        gpu.merge_validation({}, result)


def test_failed_verification_retains_artifact_and_never_becomes_pass():
    row = {}
    with pytest.raises(AssertionError, match='padding'):
        gpu.merge_validation(row, {'ok': False, 'status': 'correctness-failure',
                                  'error': 'padding', 'error_artifact': {'path': 'test.npz'}})
    assert row['error_artifact'] == {'path': 'test.npz'}


@pytest.mark.parametrize('plan_raises', [False, True])
def test_any_missing_candidate_discards_whole_plan_even_when_plan_raises(tmp_path, plan_raises):
    events = tmp_path / 'events.jsonl'
    events.write_text(json.dumps({'kind': 'compile_pending', 'request_id': 'candidate-2'}) + '\n')
    closed = []

    class Descriptor:
        @staticmethod
        def from_json(*args):
            return GraphContext()

    def plan(*args, **kwargs):
        if plan_raises:
            raise RuntimeError('search caught or exhausted candidates')
        return SimpleNamespace(close=lambda: closed.append(True))

    cb = SimpleNamespace(Descriptor=Descriptor, Plan=plan)
    with pytest.raises(gpu.PreparationPending) as pending:
        gpu.build_plan(cb, None, {}, FakeStream(), events)
    assert pending.value.request_ids == ['candidate-2']
    assert closed == ([] if plan_raises else [True])


def test_failed_compile_wins_over_pending_events(tmp_path):
    events = tmp_path / 'events.jsonl'
    events.write_text('\n'.join(json.dumps(record) for record in [
        {'kind': 'compile_pending', 'request_id': 'one'},
        {'kind': 'compile_failed', 'request_id': 'two', 'error': 'nvcc failed'}]))
    with pytest.raises(gpu.PreparationFailed) as failed:
        gpu.check_preparation_events(events)
    assert failed.value.details[0]['error'] == 'nvcc failed'


def test_interrupted_nvcc_compile_is_retryable(tmp_path):
    events = tmp_path / 'events.jsonl'
    events.write_text(json.dumps({'kind': 'compile_failed', 'request_id': 'retry',
                                  'error': 'specialization compilation failed; nvcc: Terminated'}) + '\n')
    with pytest.raises(gpu.PreparationPending) as pending:
        gpu.check_preparation_events(events)
    assert pending.value.request_ids == ['retry']
    assert gpu._transient_compile_failure({'kind': 'compile_failed', 'error': 'nvcc: Terminated'})


def test_interrupted_nvcc_compile_log_is_retryable(tmp_path):
    compiler_log = tmp_path / 'compiler.log'
    compiler_log.write_text('Compilation terminated.\nnvcc: Terminated\n')
    record = {'kind': 'compile_failed',
              'error': f'specialization compilation failed; see {compiler_log}'}
    assert gpu._transient_compile_failure(record)


def test_unrelated_compiler_failure_is_not_retryable():
    record = {'kind': 'compile_failed', 'error': 'nvcc error: unsupported host compiler'}
    assert not gpu._transient_compile_failure(record)


def test_known_cufftdx_unsupported_candidate_is_skipped(tmp_path):
    compiler_log = tmp_path / 'compiler.log'
    compiler_log.write_text('/mathdx/cufftdx/include/traits/detail/check_and_get_trait.hpp(94): error: static assertion failed with "This FFT configuration is not supported"\n')
    events = tmp_path / 'events.jsonl'
    events.write_text(json.dumps({'kind': 'compile_failed', 'request_id': 'candidate',
                                  'compiler_log': str(compiler_log)}) + '\n')
    gpu.check_preparation_events(events)


def test_generic_name_scope_error_without_cufftdx_assertion_is_fatal(tmp_path):
    compiler_log = tmp_path / 'compiler.log'
    compiler_log.write_text("error: name followed by '::' must be a class or namespace name\n")
    events = tmp_path / 'events.jsonl'
    events.write_text(json.dumps({'kind': 'compile_failed', 'request_id': 'candidate',
                                  'compiler_log_path': str(compiler_log)}) + '\n')
    with pytest.raises(gpu.PreparationFailed):
        gpu.check_preparation_events(events)


def test_broker_error_reference_to_compiler_log_is_skipped(tmp_path):
    compiler_log = tmp_path / 'compiler.log'
    compiler_log.write_text('/mathdx/cufftdx/include/traits/detail/check_and_get_trait.hpp(94): error: static assertion failed with "This FFT configuration is not supported"\n')
    events = tmp_path / 'events.jsonl'
    events.write_text(json.dumps({'kind': 'compile_failed', 'request_id': 'candidate',
                                  'error': f'specialization compilation failed; see {compiler_log}'}) + '\n')
    gpu.check_preparation_events(events)


def test_unrelated_compile_failure_remains_fatal_with_similar_event(tmp_path):
    compiler_log = tmp_path / 'compiler.log'
    compiler_log.write_text('nvcc internal error: out of memory\n')
    events = tmp_path / 'events.jsonl'
    events.write_text(json.dumps({'kind': 'compile_failed', 'request_id': 'candidate',
                                  'compiler_log': str(compiler_log)}) + '\n')
    with pytest.raises(gpu.PreparationFailed):
        gpu.check_preparation_events(events)


def test_cufftdx_assertion_does_not_mask_unrelated_compile_error(tmp_path):
    log = tmp_path / 'compiler.log'
    log.write_text('/mathdx/cufftdx/include/traits/detail/check_and_get_trait.hpp(94): error: static assertion failed with "This FFT configuration is not supported"\n'
                   '/module.cu(4): error: expected a declaration\n')
    assert not gpu._candidate_rejection({'kind': 'compile_failed', 'compiler_log': str(log)})


def test_cufftdx_assertion_with_known_template_cascade_is_terminal_rejection(tmp_path):
    log = tmp_path / 'compiler.log'
    log.write_text('/mathdx/cufftdx/include/traits/detail/check_and_get_trait.hpp(94): error: static assertion failed with "This FFT configuration is not supported"\n'
                   '/mathdx/cufftdx/include/detail/fft_execution.hpp(729): error: expected a declaration\n')
    assert gpu._candidate_rejection({'status': 'failed', 'error': f'specialization compilation failed; see {log}'})


def test_snapshot_owns_full_bytes_and_requires_cpu_verdict():
    class Tensor:
        def __init__(self, raw):
            self.raw = raw

        def view(self, dtype):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.raw

    raw_input = np.array([1, 2, 165], dtype=np.uint8)
    raw_output = np.array([3, 4, 165], dtype=np.uint8)
    row = {}
    steps = gpu.snapshot_steps({'id': 'case'}, {'s': {'placement': 0}, 'expected': np.array([3, 4])},
        SimpleNamespace(uint8='uint8'), FakeStream(), Tensor(raw_input), Tensor(raw_output),
        raw_input.copy(), raw_output.copy(), 'cubutterfly', row, SimpleNamespace(), 'correctness')
    payload = next(steps)
    raw_input.fill(0)
    raw_output.fill(0)
    assert payload['output_raw'].tolist() == [3, 4, 165]
    assert payload['input_raw'].tolist() == [1, 2, 165]
    assert 'correctness' not in row
    with pytest.raises(StopIteration):
        steps.send({'ok': True, 'status': 'accuracy-review-required', 'correctness': {'elements': 2}})
    assert row['correctness'] == {'elements': 2}


@pytest.mark.parametrize('first_ok', [False, True])
@pytest.mark.parametrize('control', [None, 'accepted', 'missing-profile', 'timed', 'changed-mapping'])
def test_first_validation_precedes_replay_and_terminal_row(monkeypatch, first_ok, control):
    plans = []

    class Tensor:
        def __init__(self, raw):
            self.raw = np.array(raw, copy=True)

        def to(self, device):
            return self

        def clone(self):
            return Tensor(self.raw)

        def view(self, dtype):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.raw

        def copy_(self, other):
            self.raw[:] = other.raw

        def fill_(self, value):
            self.raw.fill(value)

    s = {'operator': 1, 'storage': 8, 'placement': 0}
    descriptor = {'semantics': s, 'selection': {}}

    class Descriptor:
        def __init__(self, value):
            self.value = value

        @staticmethod
        def from_json(library, value):
            return Descriptor(value)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def validate(self):
            pass

        def json(self):
            return json.dumps(self.value)

        def capabilities_json(self):
            return '{"status":"eligible"}'

    class Plan:
        def __init__(self, library, imported, **kwargs):
            self.value = imported.value
            self.executions = 0
            self.closed = False
            plans.append(self)

        def descriptor_json(self):
            return json.dumps(self.value)

        def mapping_json(self):
            return '{"kind":"changed"}' if control == 'changed-mapping' else '{"kind":"test"}'

        def selection_reason(self):
            return 'test'

        def algorithm_name(self):
            return 'test'

        def workspace_size(self):
            return 0

        def input_size(self):
            return 8

        output_size = input_size

        def execute(self, source, output):
            self.executions += 1
            output.copy_(source)

        def close(self):
            self.closed = True

    def baselines(*args):
        if False:
            yield None
        return []

    import semantic_cpu_pipeline as cpu
    monkeypatch.setitem(sys.modules, 'cubutterfly', SimpleNamespace(Descriptor=Descriptor, Plan=Plan, Error=type('CbError', (Exception,), {})))
    monkeypatch.setattr(cpu, 'prepared_physical_bytes', lambda *args: np.arange(8, dtype=np.uint8))
    monkeypatch.setattr(gpu, 'baseline_steps', baselines)
    torch = SimpleNamespace(uint8='uint8', from_numpy=Tensor,
        empty=lambda size, **kwargs: Tensor(np.zeros(size, dtype=np.uint8)),
        OutOfMemoryError=type('Oom', (Exception,), {}))
    case = {'id': 'case', 'populations': [], 'descriptor': descriptor}
    if control is not None:
        case['execution_mapping'] = {'kind': 'test'}
        case['source_evidence'] = {'row_checksum': 'original-stress-evidence'}
        if control != 'missing-profile':
            case['input_profile'] = {'name': 'scaled', 'scale': 0.25}
    prepared = {'s': s, 'source': np.array([1, 2]), 'expected': np.array([1, 2])}
    steps = gpu.execution_steps(case, prepared, None, torch, FakeStream(), SimpleNamespace(timing=control == 'timed'))
    if control in ('missing-profile', 'timed', 'changed-mapping'):
        with pytest.raises(StopIteration) as rejected:
            next(steps)
        assert rejected.value.value['status'] == ('correctness-failure' if control == 'changed-mapping' else 'runner-error')
        assert all(plan.closed and plan.executions == 0 for plan in plans)
        return
    assert next(steps)['phase'] == 'initial'
    assert len(plans) == 1 and plans[0].executions == 1
    token = {'ok': first_ok, 'status': 'pass' if first_ok else 'correctness-failure',
             'correctness': {'elements': 2}, 'error': 'bad output'}
    if first_ok:
        assert steps.send(token)['phase'] == 'mapping-replay'
        assert len(plans) == 2 and plans[1].executions == 1
    with pytest.raises(StopIteration) as completed:
        steps.send(token)
    assert completed.value.value['status'] == ('pass' if first_ok else 'correctness-failure')
    assert all(plan.closed for plan in plans)
    if control == 'accepted':
        result = completed.value.value
        assert result['mapping'] == case['execution_mapping']
        assert result['input_profile'] == case['input_profile']
        assert result['source_evidence'] == case['source_evidence']
        assert result['selection_scope'] == 'explicit original stress-test mapping; separate scaled input; no timing'
        assert all(plan.value['selection']['algorithm_policy'] == 2 for plan in plans)
        assert all(json.loads(plan.value['selection']['explicit_algorithm']) == case['execution_mapping'] for plan in plans)
