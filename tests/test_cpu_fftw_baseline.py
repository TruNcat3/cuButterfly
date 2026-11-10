"""CPU-only FFTW contract checks; floating comparisons are error reports."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "cpu_fftw_baseline", Path(__file__).resolve().parents[1] / "scripts/cpu_fftw_baseline.py")
fftw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fftw)


def semantics(shape=(2, 4), dtype=np.complex64, **changes):
    value = dict(operator=0, fft_type=0, storage=6 if dtype == np.complex64 else 7,
                 compute=1 if dtype == np.complex64 else 2, shape=list(shape), batch=3,
                 inverse=0, normalize_inverse=False, fft_output_scale=1.0,
                 length_mode=0, placement=0, input_orders=[0] * len(shape),
                 output_orders=[0] * len(shape),
                 input_strides=[int(np.prod(shape[a + 1:])) for a in range(len(shape))],
                 output_strides=[int(np.prod(shape[a + 1:])) for a in range(len(shape))],
                 input_batch_stride=int(np.prod(shape)), output_batch_stride=int(np.prod(shape)))
    value.update(changes)
    return value


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("shape", [(2, 4), (2, 2, 4)])
@pytest.mark.parametrize("inverse,normalize,scale", [(0, False, 1.0), (0, False, -2.0),
                                                      (1, False, .5), (1, True, 2.0)])
def test_exact_small_basis_dimensions_batch_scaling(dtype, shape, inverse, normalize, scale):
    # Fourth roots of unity and binary scales are exactly representable. This
    # detects dimension reversal and scale omission without an error tolerance.
    s = semantics(shape, dtype, inverse=inverse, normalize_inverse=normalize, fft_output_scale=scale)
    source = np.zeros((3, *shape), dtype=dtype)
    index = (1, 1) if len(shape) == 2 else (1, 0, 1)
    for batch in range(3):
        source[(batch, *index)] = batch + 1
    original = source.copy()
    roots = np.array([1, 1j if inverse else -1j, -1, -1j if inverse else 1j], dtype=dtype)
    expected = np.empty_like(source)
    for coord in np.ndindex(shape):
        phase = (-1) ** coord[0] * roots[coord[-1]]
        for batch in range(3):
            expected[(batch, *coord)] = (batch + 1) * phase * scale / (np.prod(shape) if normalize else 1)
    core = fftw.FFTWCore(s, source)
    try:
        np.testing.assert_array_equal(source, original)
        for _ in range(2):
            core.execute()
            np.testing.assert_array_equal(core.output, expected)
            np.testing.assert_array_equal(source, original)
        metadata = core.metadata()
        assert metadata["planner"] == "FFTW_MEASURE"
        assert metadata["planning_ms"] >= 0
        assert len(metadata["library"]["sha256"]) == 64
        assert metadata["library"]["version"]
        assert core.output.dtype == dtype
    finally:
        core.close()
        core.close()
    with pytest.raises(RuntimeError, match="closed"):
        core.execute()


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
@pytest.mark.parametrize("inverse", [0, 1])
def test_floating_error_report_only(dtype, shape, inverse):
    s = semantics(shape, dtype, inverse=inverse, normalize_inverse=bool(inverse), fft_output_scale=.75)
    rng = np.random.default_rng(17)
    source = (rng.normal(size=(3, *shape)) + 1j * rng.normal(size=(3, *shape))).astype(dtype)
    original = source.copy()
    axes = tuple(range(1, len(shape) + 1))
    expected = (np.fft.ifftn(source, axes=axes) if inverse else np.fft.fftn(source, axes=axes)) * .75
    core = fftw.FFTWCore(s, source)
    try:
        core.execute()
        assert np.isfinite(core.output).all()
        error = np.abs(core.output.astype(np.complex128) - expected.astype(np.complex128))
        print(json.dumps(dict(dtype=str(np.dtype(dtype)), shape=shape, inverse=inverse,
                              max_abs_error=float(error.max()), rms_error=float(np.sqrt(np.mean(error ** 2))))))
        np.testing.assert_array_equal(source, original)
    finally:
        core.close()


@pytest.mark.parametrize("threads", [1, 8])
def test_native_threads_and_native_compute(threads):
    s = semantics((4,), compute=0)
    source = np.zeros((3, 4), dtype=np.complex64)
    source[:, 0] = [1, 2, 3]
    core = fftw.FFTWCore(s, source, threads=threads)
    try:
        core.execute()
        np.testing.assert_array_equal(core.output, np.broadcast_to(source[:, :1], source.shape))
        assert core.metadata()["threads"] == threads
        assert "threads" in core.metadata()["threads_library"]["path"]
    finally:
        core.close()


@pytest.mark.parametrize("rank", range(1, 9))
def test_rank_one_through_eight(rank):
    shape = (2,) * rank
    source = np.zeros((3, *shape), dtype=np.complex128)
    source[(slice(None), *([0] * rank))] = [1, 2, 3]
    core = fftw.FFTWCore(semantics(shape, np.complex128, compute=0), source)
    try:
        core.execute()
        expected = np.broadcast_to(np.arange(1, 4).reshape(3, *([1] * rank)), source.shape)
        np.testing.assert_array_equal(core.output, expected)
    finally:
        core.close()


@pytest.mark.parametrize("change", [dict(operator=2), dict(fft_type=1), dict(storage=4),
                                      dict(compute=2), dict(length_mode=1), dict(placement=1),
                                      dict(input_orders=[1, 0]), dict(output_strides=[5, 1]),
                                      dict(input_batch_stride=9), dict(shape=[2] * 9),
                                      dict(batch=0), dict(fft_output_scale=float("nan")),
                                      dict(normalize_inverse=1)])
def test_unsupported_semantics_rejected(change):
    with pytest.raises(ValueError):
        fftw.FFTWCore(semantics(**change), np.zeros((3, 2, 4), dtype=np.complex64))


def test_precision_and_noncontiguous_rejected():
    with pytest.raises(ValueError, match="original complex precision"):
        fftw.FFTWCore(semantics(), np.zeros((3, 2, 4), dtype=np.complex128))
    with pytest.raises(ValueError, match="contiguous"):
        fftw.FFTWCore(semantics(), np.zeros((3, 2, 8), dtype=np.complex64)[:, :, ::2])
    with pytest.raises(ValueError, match="thread policy"):
        fftw.FFTWCore(semantics(), np.zeros((3, 2, 4), dtype=np.complex64), threads=2)
