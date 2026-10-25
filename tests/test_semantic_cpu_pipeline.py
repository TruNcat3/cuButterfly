"""CPU-only preparation and deferred full-buffer verification contracts."""
import copy
import importlib
import pathlib
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
pipeline = importlib.import_module('semantic_cpu_pipeline')
runner = pipeline.runner


def semantics(**changes):
    s = {'schema': 'cubutterfly-semantic-v1', 'operator': 0, 'storage': 6,
         'compute': 1, 'shape': [4], 'batch': 3, 'inverse': 0,
         'normalize_inverse': False, 'placement': 0, 'fft_type': 0,
         'fft_output_scale': 1.0, 'length_mode': 0, 'input_strides': [2],
         'output_strides': [3], 'input_batch_stride': 13,
         'output_batch_stride': 17, 'input_orders': [1], 'output_orders': [0],
         'ntt_input_layout': 0, 'ntt_output_layout': 0, 'modulus': None,
         'word_bits': None, 'matrices': [[]]}
    s.update(changes)
    return s


def case(s):
    return {'id': runner.digest(s), 'descriptor': {'semantics': s}, 'populations': []}


def byte_size(s, side):
    return (int(runner.layout_indices(s, side).max()) + 3) * runner.side_dtype(s, side).itemsize


@pytest.mark.parametrize('changes', [
    {}, {'storage': 4}, {'storage': 0, 'fft_type': 1},
    {'storage': 1, 'fft_type': 2, 'inverse': 1, 'normalize_inverse': True, 'input_orders': [0]},
    {'operator': 1, 'storage': 9, 'compute': 4, 'word_bits': 64,
     'modulus': 18446744069414584321},
])
def test_preparation_and_plan_sized_packing_match_frozen_runner(changes):
    s = semantics(**changes)
    prepared = pipeline.prepare_case(case(s))
    source = runner.make_source(s)
    expected = runner.reference(s, source)
    assert prepared['source'].dtype == source.dtype
    assert prepared['source'].shape == source.shape
    assert prepared['source'].tobytes() == source.tobytes()
    assert prepared['expected'].dtype == expected.dtype
    assert prepared['expected'].tobytes() == expected.tobytes()
    for side, logical in [('input', prepared['source']), ('output', prepared['expected'])]:
        size = byte_size(s, side)
        np.testing.assert_array_equal(pipeline.prepared_physical_bytes(prepared, side, logical, size),
                                      runner.physical_bytes(s, side, logical, size))


def test_import_and_preparation_do_not_import_torch():
    code = ("import sys; import semantic_cpu_pipeline as p; "
            f"p.prepare_case({case(semantics())!r}); assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, '-c', code], cwd=SCRIPTS, check=True)


def test_capability_and_element_bound_precede_source_allocation(monkeypatch):
    def forbidden(s):
        raise AssertionError('source allocation forbidden')
    monkeypatch.setattr(runner, 'make_source', forbidden)
    value = case(semantics())
    for status in ('missing-lowering', 'unknown'):
        value['capabilities'] = {'status': status}
        with pytest.raises(runner.Gap):
            pipeline.prepare_case(value)
    del value['capabilities']
    with pytest.raises(runner.Gap, match='all-elements bound'):
        pipeline.prepare_case(value, reference_max_elements=1)


def test_oracle_operation_budget_is_gap():
    s = semantics(operator=1, storage=9, compute=4, word_bits=64, modulus=17)
    with pytest.raises(runner.Gap, match='bound'):
        pipeline.prepare_case(case(s), ntt_max_operations=1)


def test_float_pending_exact_integer_and_last_element_failure():
    s = semantics(storage=2)
    expected = np.zeros((3, 4), dtype=np.float64)
    actual = np.zeros_like(expected)
    actual[-1, -1] = 1
    payload = {'s': s, 'actual': actual, 'expected': expected}
    report = pipeline.validate_snapshot(payload)
    assert report['ok'] and report['status'] == 'accuracy-review-required'
    assert report['correctness']['mismatches'] == 1
    assert not pipeline.validate_snapshot(dict(payload, float_verdict='legacy-fixed'))['ok']
    s = semantics(operator=1, storage=9)
    expected = np.array([2**63 + 7], dtype=np.uint64)
    payload = {'s': s, 'actual': expected.copy(), 'expected': expected}
    assert pipeline.validate_snapshot(payload)['status'] == 'pass'
    payload['actual'][0] += np.uint64(1)
    assert pipeline.validate_snapshot(payload)['status'] == 'correctness-failure'


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), 100.0])
def test_failed_numeric_check_preserves_lossless_artifact(tmp_path, bad):
    actual = np.array([bad], dtype=np.float32)
    payload = {'s': semantics(storage=2), 'actual': actual, 'expected': np.array([0.]),
               'float_verdict': 'legacy-fixed', 'error_dir': tmp_path,
               'case_id': 'case', 'backend': 'cubutterfly'}
    report = pipeline.validate_snapshot(payload)
    assert report['status'] == 'correctness-failure'
    with np.load(report['error_artifact']['path'], allow_pickle=False) as artifact:
        np.testing.assert_array_equal(artifact['actual'], actual)


def test_full_output_padding_input_preservation_and_owned_snapshot():
    s = semantics()
    prepared = pipeline.prepare_case(case(s))
    expected = prepared['expected']
    output_raw = runner.physical_bytes(s, 'output', expected, byte_size(s, 'output'))
    output_initial = np.full_like(output_raw, 0xA5)
    input_initial = pipeline.prepared_physical_bytes(prepared, 'input', prepared['source'], byte_size(s, 'input'))
    payload = {'s': s, 'output_raw': output_raw, 'expected': expected,
               'output_initial': output_initial, 'input_initial': input_initial,
               'input_raw': input_initial.copy(), 'detailed': False}
    owned = pipeline.capture_snapshot(**payload)
    assert pipeline.validate_snapshot(owned)['ok']
    output_raw[-1] = 0
    payload['input_raw'][-1] = 0
    assert pipeline.validate_snapshot(owned)['ok']
    assert pipeline.validate_snapshot(payload)['status'] == 'correctness-failure'
    input_bad = copy.deepcopy(owned)
    input_bad['input_raw'][-1] ^= 1
    assert 'input or padding' in pipeline.validate_snapshot(input_bad)['error']
    output_bad = copy.deepcopy(owned)
    output_bad['output_raw'][-1] ^= 1
    assert 'padding' in pipeline.validate_snapshot(output_bad)['error']


def test_storage_policy_and_missing_full_buffer_evidence_cannot_pass():
    payload = {'s': semantics(storage=9), 'storage': 2,
               'actual': np.array([1.]), 'expected': np.array([1.])}
    assert pipeline.validate_snapshot(payload)['status'] == 'runner-error'
    payload = {'s': semantics(), 'output_raw': np.zeros(16, dtype=np.uint8),
               'expected': np.zeros(12, dtype=np.complex128)}
    assert not pipeline.validate_snapshot(payload)['ok']
