"""CPU-only grouped FFT scheduling and runner routing checks."""
import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import numpy as np
from scipy import fft as scipy_fft

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import semantic_fft_baseline as fft
import run_semantic_campaign as runner


def semantics(**changes):
    s = dict(operator=0, fft_type=0, storage=6, compute=1, shape=[8, 8, 8, 8], batch=4,
             inverse=0, normalize_inverse=False, length_mode=0, fft_output_scale=1.0)
    s.update(changes)
    return s


@pytest.mark.parametrize('backend', ['cufft', 'vkfft'])
@pytest.mark.parametrize('shape,batch', [([8] * 4, 4), ([2] * 8, 16), ([2, 4, 2, 8], 3)])
@pytest.mark.parametrize('fp64', [False, True])
@pytest.mark.parametrize('inverse,normalized', [(0, False), (1, True), (1, False)])
@pytest.mark.parametrize('max_vendor_rank', [1, 2, 3])
def test_complete_axis_composition_and_repeated_inputs(monkeypatch, backend, shape, batch, fp64, inverse, normalized, max_vendor_rank):
    s = semantics(shape=shape, batch=batch, storage=7 if fp64 else 6,
                  compute=2 if fp64 else 1, inverse=inverse, normalize_inverse=normalized)
    dtype = torch.complex128 if fp64 else torch.complex64
    source = torch.randn((batch, *shape), dtype=dtype, generator=torch.Generator().manual_seed(19))
    saved = source.clone()
    output = torch.empty_like(source)
    calls, closed, pointers = [], [], []

    def vendor(name, api, vendor_shape, lines, is_fp64, is_inverse, stream, src, out, library):
        assert name == backend and api is torch and is_fp64 == fp64
        assert src.shape == out.shape == (lines, *vendor_shape)
        assert lines * math.prod(vendor_shape) == source.numel()
        assert stream.cuda_stream == 123 and library == 'test-vkfft.so'
        pointers.append((src.data_ptr(), out.data_ptr()))

        def execute():
            calls.append(vendor_shape)
            operation = torch.fft.ifftn if is_inverse else torch.fft.fftn
            operation(src, dim=tuple(range(1, len(vendor_shape) + 1)), out=out)

        return SimpleNamespace(__call__=execute, close=lambda: closed.append(vendor_shape))

    class CallableCore:
        def __init__(self, plan):
            self.plan = plan

        def __call__(self):
            self.plan.__call__()

        def close(self):
            self.plan.close()

    monkeypatch.setattr(fft, '_vendor_core', lambda *args: CallableCore(vendor(*args)))
    core = fft.AxisComposition(backend, torch, s, source, output,
                               SimpleNamespace(cuda_stream=123), 'test-vkfft.so',
                               max_vendor_rank=max_vendor_rank)
    metadata = copy.deepcopy(core.metadata())
    assert len(set(pointers)) == 1  # All sequential plans reuse the two packed buffers.
    assert metadata['implementation'] == 'external-axis-composition'
    assert metadata['scratch_tensor_bytes'] == 2 * source.numel() * source.element_size()
    groups = [shape[i:i + max_vendor_rank] for i in range(0, len(shape), max_vendor_rank)]
    assert [group['vendor_shape'] for group in metadata['groups']] == groups
    assert [group['vendor_rank'] for group in metadata['groups']] == list(map(len, groups))
    assert [axis for group in metadata['groups'] for axis in group['logical_axes']] == list(range(len(shape)))
    assert metadata['max_vendor_rank'] == max_vendor_rank
    assert len(metadata['included_operations']) == len(groups) * (4 if inverse else 3)
    monkeypatch.setattr(torch, 'empty_like', lambda *args, **kwargs: pytest.fail('allocation during execute'))
    for change in (False, True):
        if change:
            source.copy_(saved * complex(0.25, -0.5))
        expected_source = source.clone()
        output.fill_(complex(float('nan'), float('nan')))
        core()
        axes = tuple(range(1, len(shape) + 1))
        operation = scipy_fft.ifftn if inverse else scipy_fft.fftn
        expected = torch.from_numpy(operation(source.numpy(), axes=axes, workers=1))
        # The existing physical boundary applies request scaling after the normalized core.
        factor = source.numel() // batch if inverse and not normalized else 1
        actual = output * factor * 0.25
        expected = expected * factor * 0.25
        torch.testing.assert_close(actual, expected, atol=2e-4 if not fp64 else 1e-12,
                                   rtol=2e-5 if not fp64 else 1e-12)
        assert torch.equal(source, expected_source)
    assert calls == groups * 2
    core.close()
    core.close()
    assert closed == groups
    with pytest.raises(RuntimeError, match='closed'):
        core()


@pytest.mark.parametrize('changes,reason', [
    ({'shape': [2, 2, 2]}, 'rank above three'),
    ({'shape': [2, 2, 2, True]}, 'axis lengths'),
    ({'shape': [2**31, 2, 2, 2]}, 'axis lengths'),
    ({'batch': 2**31}, 'line batch'),
    ({'compute': 2}, 'precision'),
    ({'storage': 2}, 'precision'),
    ({'fft_type': 1}, 'C2C'),
    ({'length_mode': 1}, 'embedding'),
    ({'inverse': True}, 'canonical'),
    ({'normalize_inverse': 1}, 'canonical'),
])
def test_support_rejects_semantic_substitution(changes, reason):
    report = fft.support('cufft', semantics(**changes))
    assert not report['supported'] and reason in report['reason']


def test_vkfft_dependency_remains_explicit():
    report = fft.support('vkfft', semantics())
    assert report['status'] == 'missing-external-dependency'


@pytest.mark.parametrize('max_vendor_rank', [0, 4, True, 1.5])
def test_grouping_limit_requires_supported_native_rank(max_vendor_rank):
    report = fft.support('cufft', semantics(), max_vendor_rank=max_vendor_rank)
    assert not report['supported']
    assert 'max_vendor_rank' in report['reason']


def test_default_grouping_is_three_axes_and_checks_group_distance():
    report = fft.support('cufft', semantics())
    assert report['metadata']['max_vendor_rank'] == 3
    invalid = fft.support('cufft', semantics(shape=[2048, 2048, 2048, 2], batch=1))
    assert not invalid['supported'] and 'group point count' in invalid['reason']


def test_partial_construction_closes_prepared_plans(monkeypatch):
    closed = []

    def vendor(*args):
        if closed == ['created']:
            raise RuntimeError('second axis plan failed')
        closed.append('created')
        return SimpleNamespace(close=lambda: closed.append('closed'))

    monkeypatch.setattr(fft, '_vendor_core', vendor)
    source = torch.ones((4, 8, 8, 8, 8), dtype=torch.complex64)
    with pytest.raises(RuntimeError, match='second axis'):
        fft.AxisComposition('cufft', torch, semantics(), source, torch.empty_like(source),
                            SimpleNamespace(cuda_stream=123))
    assert closed == ['created', 'closed']


@pytest.fixture
def routed_runner():
    return runner


@pytest.mark.parametrize('backend', ['cufft', 'vkfft'])
def test_runner_routes_only_high_rank_fft(routed_runner, backend):
    contract = routed_runner.baseline_contract
    for shape in ([8] * 4, [2] * 8):
        report = contract(backend, semantics(shape=shape), 'test-vkfft.so')
        assert report['supported']
        assert report['metadata']['implementation'] == 'external-axis-composition'
    native = contract(backend, semantics(shape=[8, 8, 8]), 'test-vkfft.so')
    assert native['supported'] and native['metadata'] == {}
    assert contract(backend, semantics(operator=1), 'test-vkfft.so')['status'] == 'not-applicable'


@pytest.mark.parametrize('mode', ['sync', 'async'])
@pytest.mark.parametrize('rank', [4, 8])
@pytest.mark.parametrize('inverse,normalized', [(0, False), (1, True), (1, False)])
@pytest.mark.parametrize('placement', [0, 1])
def test_complete_boundary_routing_with_orders_padding_and_scale(
        monkeypatch, routed_runner, mode, rank, inverse, normalized, placement):
    import semantic_gpu_pipeline as gpu

    shape, batch = [2] * rank, 3
    points = math.prod(shape)
    strides = [2 * math.prod(shape[i + 1:]) for i in range(rank)]
    s = semantics(shape=shape, batch=batch, inverse=inverse, normalize_inverse=normalized,
                  fft_output_scale=0.25, placement=placement,
                  input_strides=strides, output_strides=strides,
                  input_batch_stride=points * 2 + 2, output_batch_stride=points * 2 + 2,
                  input_orders=[1] * rank, output_orders=[0] * rank)
    logical = (np.random.default_rng(19).normal(size=(batch, *shape)) +
               1j * np.random.default_rng(23).normal(size=(batch, *shape))).astype(np.complex64)
    axes = tuple(range(1, rank + 1))
    operation = scipy_fft.ifftn if inverse else scipy_fft.fftn
    expected = operation(logical, axes=axes, norm='backward' if normalized or not inverse else 'forward') * 0.25
    size = ((batch - 1) * s['input_batch_stride'] + points * 2 + 2) * 8
    input_raw = routed_runner.physical_bytes(s, 'input', logical, size)
    output_initial = input_raw.copy() if placement else np.full(size, 0xA5, np.uint8)
    original_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        if args and args[0] == 'cuda':
            args = ('cpu', *args[1:])
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', cpu_to)

    class CPUAPI:
        def empty(self, *args, **kwargs):
            kwargs['device'] = 'cpu'
            return torch.empty(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(torch, name)

    class Core:
        def __init__(self, inverse, source, output):
            self.inverse, self.source, self.output = inverse, source, output

        def __call__(self):
            operation = torch.fft.ifftn if self.inverse else torch.fft.fftn
            operation(self.source, dim=tuple(range(1, self.source.ndim)), out=self.output)

        def close(self):
            pass

    monkeypatch.setattr(fft, '_vendor_core', lambda name, api, length, lines, fp64, inv, stream, src, out, lib:
                        Core(inv, src, out))
    api, stream = CPUAPI(), SimpleNamespace(cuda_stream=123, synchronize=lambda: None)
    case = {'id': 'cpu-boundary-test', 'descriptor': {'semantics': s}}
    if mode == 'sync':
        rows = routed_runner.run_baselines(case, api, stream, logical, expected,
                                           torch.from_numpy(input_raw), output_initial, False,
                                           'test-vkfft.so', 'error-report')
    else:
        monkeypatch.setattr(gpu, 'old', routed_runner)

        def snapshot(case, prepared, api, stream, physical_input, physical_output,
                     input_initial, output_initial, name, row, options, key, phase):
            raw = physical_output.view(torch.uint8).numpy()
            actual = routed_runner.output_from_bytes(s, raw)
            np.testing.assert_allclose(actual.reshape(expected.shape), expected, atol=2e-5, rtol=2e-5)
            routed_runner.check_padding(s, raw, output_initial)
            if not placement:
                np.testing.assert_array_equal(physical_input.view(torch.uint8).numpy(), input_initial)
            row[key] = routed_runner.check_full(actual, expected, 6, 'error-report')
            if False:
                yield

        monkeypatch.setattr(gpu, 'snapshot_steps', snapshot)
        prepared = dict(s=s, input_indices=routed_runner.layout_indices(s, 'input'),
                        output_indices=routed_runner.layout_indices(s, 'output'))
        options = SimpleNamespace(vkfft_library='test-vkfft.so', timing=False, float_verdict='error-report')
        steps = gpu.baseline_steps(case, prepared, api, stream, torch.from_numpy(input_raw),
                                   input_raw, output_initial, options)
        with pytest.raises(StopIteration) as done:
            next(steps)
        rows = done.value.value
    for row in rows[:2]:
        assert row['status'] == 'accuracy-review-required', row
        assert row['category'] == 'external-axis-composition'
        assert row['metadata']['included_operations'] == row['adapter_operations']
        assert 'gather including order' in row['adapter_operations'][0]
        assert 'scatter including order' in row['adapter_operations'][-1]
        factor = 0.25 * (points if inverse and not normalized else 1)
        if factor != 1:
            assert f'GPU multiply output by {factor!r}' in row['metadata']['included_operations']
