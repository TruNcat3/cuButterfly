"""Independent CPU checks for the cooperative register-prefix FFT oracle."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fft_cooperative_oracle import (  # noqa: E402
    cooperative_fft,
    dense_fft,
    enumerate_shared_addresses,
    enumerate_write_read_pairs,
    shared_bank_diagnostic,
    shared_row,
    verify_shared_layout,
    verify_shuffle_groups,
)


def _values(r, columns=1):
    rng = np.random.RandomState(817 + r + columns)
    return rng.normal(size=(r * r, columns)) + 1j * rng.normal(size=(r * r, columns))


@pytest.mark.parametrize("r", [4, 8, 16])
@pytest.mark.parametrize("g", [1, 2, 4])
@pytest.mark.parametrize("layout", ["linear", "xor"])
def test_cooperative_transform_matches_numpy_for_requested_shapes(r, g, layout):
    if g > r:
        pytest.skip("lane group cannot exceed local FFT root")
    values = _values(r, columns=2)
    actual = cooperative_fft(values, r, g, columns=2, layout=layout)
    expected = np.fft.fft(values, axis=0)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("r", [4, 8, 16])
@pytest.mark.parametrize("g", [1, 2, 4])
def test_extreme_g_equals_r_and_inverse_normalization(r, g):
    if g > r:
        pytest.skip("lane group cannot exceed local FFT root")
    values = _values(r)
    forward = cooperative_fft(values, r, g, layout="xor")
    np.testing.assert_allclose(forward, np.fft.fft(values, axis=0), rtol=2e-11, atol=2e-11)
    inverse_unscaled = cooperative_fft(values, r, g, inverse=True, layout="xor")
    np.testing.assert_allclose(
        inverse_unscaled, np.fft.ifft(values, axis=0) * (r * r), rtol=2e-11, atol=2e-11
    )
    inverse = cooperative_fft(values, r, g, inverse=True, normalize=True, layout="xor")
    np.testing.assert_allclose(inverse, np.fft.ifft(values, axis=0), rtol=2e-11, atol=2e-11)


def test_dense_reference_uses_the_same_flattening_contract():
    r = 8
    values = _values(r, columns=3)
    expected = np.fft.fft(values, axis=0)
    np.testing.assert_allclose(dense_fft(values, r, columns=3), expected)
    np.testing.assert_allclose(
        dense_fft(values.reshape(r, r, 3), r, columns=3), expected.reshape(r, r, 3)
    )


@pytest.mark.parametrize("r", [4, 8, 16, 64])
@pytest.mark.parametrize("g", [1, 2, 4])
@pytest.mark.parametrize("layout", ["linear", "xor"])
def test_shared_address_pairing_is_bijective(r, g, layout):
    if g > r:
        pytest.skip("lane group cannot exceed local FFT root")
    report = verify_shared_layout(r, g, 4, layout=layout, complex_bytes=8)
    assert report["ok"]
    assert report["unique_addresses"] == r * r * 4
    assert report["in_range"]
    records = enumerate_write_read_pairs(r, g, 4, layout=layout, complex_bytes=8)
    assert len(records) == r * r * 4
    assert {record.address for record in records} == set(range(r * r * 4))
    assert all(record.writer_lane == record.a // (r // g) for record in records)
    assert all(record.reader_lane == record.b % g for record in records)


def test_shared_address_formula_matches_cuda_contract_boundaries():
    # FP32/C4 gives L=2; FP64/C2 has the same 32-byte vector payload.
    for complex_bytes, columns in ((8, 4), (16, 2)):
        for r, g in ((4, 4), (8, 2), (16, 4), (64, 4)):
            k = r // g
            low_bits = min(r.bit_length() - 1, (128 // (complex_bytes * columns)).bit_length() - 1)
            h = min(g.bit_length() - 1, low_bits)
            for a in range(r):
                for b in range(r):
                    mask = (1 << low_bits) - 1 if low_bits else 0
                    def ror(value):
                        if not low_bits or h == 0:
                            return value
                        low = value & mask
                        return (value & ~mask) | (low >> h) | ((low << (low_bits - h)) & mask)
                    expected = ror(b) ^ a ^ (ror((a >> k.bit_length() - 1) << (low_bits - h)) & mask)
                    assert shared_row(a, b, r, g, columns, layout="xor", complex_bytes=complex_bytes) == expected


def test_partial_warp_and_maximum_legal_lane_column_groups():
    # The first case launches only 16 active threads, so the __activemask path
    # is exercised conceptually rather than assuming a full warp.
    for r, g, columns in ((4, 2, 2), (8, 2, 16), (8, 4, 8), (8, 8, 4), (16, 16, 2)):
        report = verify_shuffle_groups(r, g, columns)
        assert report["ok"], report
    with pytest.raises(ValueError, match="at most one warp"):
        verify_shuffle_groups(8, 4, 16)


def test_bank_model_is_explicitly_conditional_and_not_a_performance_claim():
    addresses = [item.address for item in enumerate_shared_addresses(4, 2, 4, layout="xor")]
    for complex_bytes in (8, 16):
        report = shared_bank_diagnostic(addresses, complex_bytes=complex_bytes)
        assert report["diagnostic_only"]
        assert report["requires_ncu_confirmation"]
        assert report["aligned_128b_base"]
        assert report["sectors_128b"] > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"r": 6, "g": 2, "columns": 1},
        {"r": 8, "g": 3, "columns": 1},
        {"r": 8, "g": 4, "columns": 16},
    ],
)
def test_invalid_cooperative_shapes_are_rejected(kwargs):
    with pytest.raises(ValueError):
        cooperative_fft(np.zeros(64, dtype=np.complex128), **kwargs)
