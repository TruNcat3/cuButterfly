"""CPU regression checks for graph output freshness and timing stream scope."""
import ast
from contextlib import contextmanager
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/run_semantic_campaign.py'
spec = importlib.util.spec_from_file_location('campaign_graph_raw_runner', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_run_case_verifier_rejects_finite_poison_before_error_report():
    # Execute the actual nested verifier with CPU tensor mocks; no CUDA import.
    tree = ast.parse(SOURCE.read_text())
    run_case = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_case')
    verify = next(node for node in ast.walk(run_case) if isinstance(node, ast.FunctionDef) and node.name == 'verify')
    raw = np.array([1, 2, 3, 4, 0xA5], dtype=np.uint8)
    calls = []
    context = dict(np=np, gpu_output=SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: raw)),
                   output_from_bytes=lambda s, value: value, s={'storage': 4}, row={'correctness': {}},
                   expected=raw.copy(), error_dir=None, float_verdict='error-report', output_initial=raw.copy(),
                   in_place=True, check_full=lambda *args: calls.append('reference') or {},
                   check_padding=lambda *args: None)
    exec(compile(ast.Module(body=[verify], type_ignores=[]), str(SOURCE), 'exec'), context)
    direct = raw.copy()
    context['verify'](direct)
    assert calls == ['reference']
    raw.fill(0xA5)  # A no-op graph leaves finite FP16 poison; reporting alone accepts it.
    with pytest.raises(AssertionError, match='raw output differs'):
        context['verify'](direct)
    assert calls == ['reference']
    raw[:] = direct
    raw[-1] ^= 1  # Full raw output includes padding bytes.
    with pytest.raises(AssertionError, match='raw output differs'):
        context['verify'](direct)


def test_timing_wrapper_binds_replay_and_verification_to_declared_stream():
    state = {'current': None}
    stream = SimpleNamespace(synchronize=lambda: None)

    @contextmanager
    def stream_context(value):
        previous = state['current']
        state['current'] = value
        try:
            yield
        finally:
            state['current'] = previous

    @contextmanager
    def capture(graph, stream):
        assert state['current'] is stream
        yield

    def require_stream():
        assert state['current'] is stream

    class Event:
        def __init__(self, **kwargs):
            pass
        def record(self, value):
            assert value is stream
            require_stream()
        def synchronize(self):
            require_stream()
        def elapsed_time(self, other):
            return 1.0

    torch = SimpleNamespace(cuda=SimpleNamespace(
        stream=stream_context, graph=capture, Event=Event,
        CUDAGraph=lambda: SimpleNamespace(replay=require_stream)))
    result = runner.time_execute(torch, stream, require_stream, require_stream, False,
                                 require_stream, require_stream)
    assert len(result['samples_ms']) == 7
    assert state['current'] is None
