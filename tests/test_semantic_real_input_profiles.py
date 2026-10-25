"""CPU-only range controls preserve real FFT semantics and input provenance."""
import copy
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
import semantic_real_input_profiles as profiles


def semantics(shape=(16,), fft_type=2):
    strides = [int(np.prod(shape[i + 1:])) for i in range(len(shape))]
    return {'operator': 0, 'storage': 0, 'compute': 0, 'length_mode': 0,
            'shape': list(shape), 'batch': 1, 'fft_type': fft_type,
            'inverse': int(fft_type == 2), 'normalize_inverse': True,
            'fft_output_scale': 1., 'placement': 0,
            'input_orders': [0] * len(shape), 'output_orders': [0] * len(shape),
            'input_strides': strides, 'output_strides': strides,
            'input_batch_stride': int(np.prod(shape)),
            'output_batch_stride': int(np.prod(shape)),
            'ntt_input_layout': 0, 'ntt_output_layout': 0}


@pytest.mark.parametrize('shape,fft_type', [((16,), 1), ((16,), 2), ((8, 16), 1),
                                         ((8, 16), 2), ((4, 8, 16), 2)])
def test_finite_quantized_inputs_and_semantics(shape, fft_type):
    s = semantics(shape, fft_type)
    s['batch'] = 2
    s['input_orders'] = [1] * len(shape)
    s['output_strides'] = [v * 3 for v in s['output_strides']]
    before = copy.deepcopy(s)
    source, metadata = profiles.prepare_input(s)
    assert s == before
    assert source.shape == (2, *profiles.runner.side_extents(s, 'input'))
    assert np.isfinite(source).all()
    expected = profiles.runner.reference(s, source)
    assert np.isfinite(expected).all()
    assert metadata['quantized_max_l1_bound'] <= 65504 / 4
    assert metadata['scale_exponent'] <= 0
    assert metadata['factor'] == 2. ** metadata['scale_exponent']
    assert metadata['reference_input'] == 'final quantized input'
    words = profiles.runner.encode_storage(source, profiles.runner.side_storage(s, 'input'))
    assert metadata['input_sha256'] == profiles._digest(words)
    np.testing.assert_array_equal(source, profiles.runner.decode_storage(words, profiles.runner.side_storage(s, 'input')))


def test_actual_large_c2r_input_is_repaired_before_quantization():
    s = semantics((1 << 20,))
    with np.errstate(over='ignore', invalid='ignore'):
        raw = profiles.runner.make_source(s)
    assert not np.isfinite(raw).all()
    source, metadata = profiles.prepare_input(s)
    assert np.isfinite(source).all()
    assert metadata['original_input_nonfinite_elements'] == np.count_nonzero(~np.isfinite(raw))
    assert metadata['original_input_sha256'] == profiles._digest(profiles.runner.encode_storage(raw, 4))
    assert np.isfinite(profiles.runner.reference(s, source)).all()
    assert metadata['scale_exponent'] < 0


def test_packed_weights_reconstruct_xd_full_spectrum_l1():
    source = np.ones((2, 3, 5), dtype=np.complex64) * (1 + 2j)
    bounds = profiles.input_l1_bounds(source, full_last_extent=8)
    assert bounds == [np.nextafter(3 * 8 * 3., np.inf)] * 2
    full = np.ones((1, 3, 8), dtype=np.complex64) * (1 + 2j)
    assert bounds[0] == profiles.input_l1_bounds(full)[0]
    with pytest.raises(ValueError):
        profiles.input_l1_bounds(source, full_last_extent=16)
    with pytest.raises(ValueError):
        profiles.input_l1_bounds(np.array([[np.inf]]))


def test_new_zero_components_report_quantization_loss(monkeypatch):
    s = semantics((4,), fft_type=1)
    source = np.array([[60000., np.finfo(np.float16).smallest_subnormal, 0., 0.]], dtype=np.float16)
    monkeypatch.setattr(profiles.runner, 'make_source', lambda contract: source.copy())
    final, metadata = profiles.prepare_input(s)
    assert metadata['scale_exponent'] == -2
    assert source[0, 1] != 0 and final[0, 1] == 0
    assert metadata['quantization_new_zero_components'] == 1


@pytest.mark.parametrize('source,expected,actual,classification', [
    ([np.inf], [np.nan], [np.nan], 'input-nonfinite'),
    ([1.], [np.nan], [np.nan], 'reference-nonfinite'),
    ([1.], [70000.], [np.inf], 'reference-exceeds-storage-range'),
    ([1.], [1.], [np.inf], 'nonfinite-within-reference-range'),
    ([1.], [1.], [1.], 'finite'),
])
def test_numeric_classification(source, expected, actual, classification):
    report = profiles.classify_numeric_failure(source, expected, actual, 0)
    assert report['classification'] == classification
    assert report['acceptance'].startswith('diagnostic only')


@pytest.mark.parametrize('field,value', [('storage', 1), ('operator', 2), ('fft_type', 0),
    ('compute', 2), ('length_mode', 1), ('shape', [3]), ('shape', [1]), ('shape', [2] * 9),
    ('batch', 0), ('batch', True), ('inverse', 0), ('normalize_inverse', 1),
    ('fft_output_scale', np.inf), ('fft_output_scale', 2.), ('input_orders', [2]),
    ('input_strides', [0]), ('output_batch_stride', 0), ('ntt_input_layout', 1)])
def test_unsupported_semantics_rejected(field, value):
    s = semantics()
    s[field] = value
    with pytest.raises(ValueError):
        profiles.prepare_input(s)
