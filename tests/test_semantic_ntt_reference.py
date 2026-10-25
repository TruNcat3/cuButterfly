"""Exact radix-two oracle checked against independent direct transforms."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from semantic_ntt_reference import (ReferenceBudgetExceeded, ntt_butterfly_count,
                                    primitive_power_of_two_root, radix2_ntt_reference)


def direct_root(modulus, length):
    if length == 1:
        return 1
    for candidate in range(2, 10002):
        value = pow(candidate, (modulus - 1) // length, modulus)
        if pow(value, length, modulus) == 1 and all(pow(value, exponent, modulus) != 1 for exponent in range(1, length)):
            return value
    raise AssertionError("test root not found")


def direct_transform(source, modulus, inverse=False, normalize_inverse=True):
    work = np.array(source, dtype=object, copy=True)
    for axis, length in enumerate(source.shape[1:], 1):
        root = direct_root(modulus, length)
        if inverse:
            root = pow(root, -1, modulus)
        moved = np.moveaxis(work, axis, -1).copy()
        for row in moved.reshape(-1, length):
            values = list(map(int, row))
            row[:] = [sum(value * pow(root, frequency * index, modulus)
                          for index, value in enumerate(values)) % modulus
                      for frequency in range(length)]
            if inverse and normalize_inverse:
                row[:] = [int(value) * pow(length, -1, modulus) % modulus for value in row]
        work = np.moveaxis(moved, -1, axis)
    return work


@pytest.mark.parametrize("modulus", [17, 97, 998244353, 576460756061519873])
@pytest.mark.parametrize("shape", [(1,), (8,), (2, 4), (2, 1, 4)])
@pytest.mark.parametrize("inverse,normalize", [(False, True), (True, True), (True, False)])
def test_exact_against_direct_all_elements(modulus, shape, inverse, normalize):
    count = 3 * math.prod(shape)
    values = [(modulus - 1 - index * 104729) % modulus for index in range(count)]
    source = np.array(values, dtype=np.uint64).reshape((3, *shape))
    saved = source.copy()
    actual = radix2_ntt_reference(source, modulus, inverse=inverse, normalize_inverse=normalize)
    expected = direct_transform(source, modulus, inverse, normalize)
    assert actual.shape == source.shape and actual.dtype == object
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, saved)
    assert all(type(value) is int for value in actual.flat)


@pytest.mark.parametrize("modulus", [998244353, 576460756061519873])
def test_large_length_round_trip_and_exact_64bit_products(modulus):
    source = np.array([modulus - 1 - (i % 97) for i in range(4096)], dtype=np.uint64).reshape(1, 4096)
    forward = radix2_ntt_reference(source, modulus)
    restored = radix2_ntt_reference(forward, modulus, inverse=True)
    np.testing.assert_array_equal(restored, source)
    assert int(forward[0, 0]) == sum(map(int, source.flat)) % modulus


def test_object_numpy_integer_values_are_widened_exactly():
    modulus = 576460756061519873
    source = np.array([np.uint64(modulus - 1)] * 8, dtype=object).reshape(1, 8)
    np.testing.assert_array_equal(radix2_ntt_reference(source, modulus), direct_transform(source, modulus))


def test_root_order_and_first_candidate_convention():
    for modulus in (17, 97, 998244353, 576460756061519873):
        for length in (1, 2, 4, 8, 16):
            root = primitive_power_of_two_root(modulus, length)
            assert root == direct_root(modulus, length)
            assert pow(root, length, modulus) == 1
            if length > 1:
                assert pow(root, length // 2, modulus) != 1


def test_explicit_butterfly_budget_and_unit_axes():
    assert ntt_butterfly_count([1, 1], 3) == 0
    assert ntt_butterfly_count([4096], 1) == 24576
    assert ntt_butterfly_count([4, 8], 3) == 240
    source = np.arange(8, dtype=np.uint64).reshape(1, 8)
    with pytest.raises(ReferenceBudgetExceeded, match="butterflies=12; bound=11"):
        radix2_ntt_reference(source, 17, max_butterflies=11)
    radix2_ntt_reference(source, 17, max_butterflies=12)
    np.testing.assert_array_equal(radix2_ntt_reference(np.array([[18]]), 17, max_butterflies=0), [[1]])


@pytest.mark.parametrize("source,modulus", [(np.zeros((1, 3), dtype=np.uint64), 97),
                                            (np.zeros((1, 8), dtype=np.uint64), 19),
                                            (np.zeros((1, 8), dtype=np.float64), 17),
                                            (np.zeros(8, dtype=np.uint64), 17),
                                            (np.zeros((0, 8), dtype=np.uint64), 17),
                                            (np.zeros((1, 8), dtype=np.uint64), 2)])
def test_invalid_or_unsupported_contract_rejected(source, modulus):
    with pytest.raises(ValueError):
        radix2_ntt_reference(source, modulus)
