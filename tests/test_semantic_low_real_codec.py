"""CPU verification of actual real/complex storage on real FFT boundaries."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_semantic_campaign as runner


def semantics(storage, fft_type, orders=False):
    return dict(operator=0, fft_type=fft_type, storage=storage, shape=[2, 4],
                batch=2, inverse=fft_type == 2, length_mode=0,
                input_strides=[11, 2], output_strides=[13, 2],
                input_batch_stride=40, output_batch_stride=48,
                input_orders=[int(orders), 0], output_orders=[int(orders), 0])


def physical_size(s, side):
    return (int(runner.layout_indices(s, side).max()) + 8) * runner.side_dtype(s, side).itemsize


@pytest.mark.parametrize("storage", range(4))
@pytest.mark.parametrize("fft_type", [1, 2])
def test_side_storage_and_byte_width(storage, fft_type):
    s = semantics(storage, fft_type)
    complex_side = 'output' if fft_type == 1 else 'input'
    real_side = 'input' if fft_type == 1 else 'output'
    assert runner.side_storage(s, real_side) == storage
    assert runner.side_storage(s, complex_side) == storage + 4
    assert runner.side_dtype(s, complex_side).itemsize == 2 * runner.side_dtype(s, real_side).itemsize
    with pytest.raises(ValueError, match='side must'):
        runner.side_storage(s, 'bogus')


@pytest.mark.parametrize("storage", [0, 1])
@pytest.mark.parametrize("fft_type", [1, 2])
@pytest.mark.parametrize("orders", [False, True])
def test_low_real_roundtrip_strides_orders_and_padding(storage, fft_type, orders):
    s = semantics(storage, fft_type, orders)
    for side in ('input', 'output'):
        shape = (s['batch'], *runner.side_extents(s, side))
        code = runner.side_storage(s, side)
        value = (np.arange(np.prod(shape), dtype=np.float32) / 7).reshape(shape)
        if code in (4, 5):
            complex_value = np.empty(shape, np.complex64)
            complex_value.real = value
            complex_value.imag = value / 3 - 1
            value = complex_value
        quantized = runner.decode_storage(runner.encode_storage(value, code), code)
        raw = runner.physical_bytes(s, side, value, physical_size(s, side))
        stored = raw.view(runner.side_dtype(s, side))[runner.layout_indices(s, side)]
        actual = runner.decode_storage(stored, code).reshape(shape)
        np.testing.assert_array_equal(actual, quantized)
        runner.check_padding(s, raw, np.full(raw.size, 0xA5, np.uint8), side)
        if side == 'output':
            np.testing.assert_array_equal(runner.output_from_bytes(s, raw), quantized.reshape(-1))


@pytest.mark.parametrize("storage", [0, 1])
def test_complex_components_preserve_signed_zero_and_nonfinite_independently(storage):
    s = semantics(storage, 1)
    real = np.array([0., -0., np.inf, -np.inf, np.nan, 2., -3., 4., 5., 6., 7., 8.], np.float32)
    imag = np.array([-0., 0., 1., 2., 3., np.inf, -np.inf, np.nan, 0., -0., 2., 3.], np.float32)
    value = np.empty((2, 2, 3), np.complex64)
    value.real = real.reshape(value.shape)
    value.imag = imag.reshape(value.shape)
    raw = runner.physical_bytes(s, 'output', value, physical_size(s, 'output'))
    actual = runner.output_from_bytes(s, raw)
    np.testing.assert_array_equal(np.isnan(actual.real), np.isnan(real))
    np.testing.assert_array_equal(np.isnan(actual.imag), np.isnan(imag))
    np.testing.assert_array_equal(np.isinf(actual.real), np.isinf(real))
    np.testing.assert_array_equal(np.isinf(actual.imag), np.isinf(imag))
    np.testing.assert_array_equal(np.signbit(actual.real[real == 0]), np.signbit(real[real == 0]))
    np.testing.assert_array_equal(np.signbit(actual.imag[imag == 0]), np.signbit(imag[imag == 0]))


@pytest.mark.parametrize("storage", [0, 1])
@pytest.mark.parametrize("fft_type", [1, 2])
def test_generated_source_is_exact_actual_quantized_input(storage, fft_type):
    s = semantics(storage, fft_type)
    source = runner.make_source(s)
    code = runner.side_storage(s, 'input')
    raw = runner.physical_bytes(s, 'input', source, physical_size(s, 'input'))
    stored = raw.view(runner.side_dtype(s, 'input'))[runner.layout_indices(s, 'input')]
    actual = runner.decode_storage(stored, code).reshape(source.shape)
    np.testing.assert_array_equal(source, actual)
    assert source.dtype == (np.dtype('complex64') if fft_type == 2 else
                            np.dtype('float16') if storage == 0 else np.dtype('float32'))
    if fft_type == 2:
        index = np.arange(16).reshape(2, 2, 4)
        expected = np.fft.rfftn(np.sin(index * .17), axes=(1, 2)).astype(np.complex64)
        expected = runner.decode_storage(runner.encode_storage(expected, code), code)
        np.testing.assert_array_equal(source, expected)


def test_unrelated_fp32_c2c_source_identity_preserved():
    s = dict(operator=0, storage=6, fft_type=0, batch=2, shape=[8], length_mode=0, inverse=False)
    index = np.arange(16, dtype=np.uint64).reshape(2, 8)
    expected = (np.sin(index.astype(np.float64) * .17) + (index % 13).astype(np.float64) / 17
                + 1j * np.cos(index.astype(np.float64) * .11)).astype(np.complex64)
    np.testing.assert_array_equal(runner.make_source(s), expected)


def test_invalid_complex_storage_for_real_fft_rejected():
    with pytest.raises(runner.Gap, match='real floating storage'):
        runner.side_dtype(semantics(4, 1), 'output')
