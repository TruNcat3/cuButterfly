"""Exact separable radix-two NTT oracle using arbitrary-width Python integers."""
from __future__ import annotations

from functools import lru_cache
import math
import numbers

import numpy as np


class ReferenceBudgetExceeded(ValueError):
    """The declared radix-two butterfly work budget was exceeded."""


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _radix_two_length(n):
    n = _positive_integer(n, "axis length")
    if n & (n - 1):
        raise ValueError("radix-two NTT requires power-of-two axis lengths")
    return n


def ntt_butterfly_count(shape, batch=1):
    """Count two-input modular butterflies, excluding boundary copies/scaling."""
    shape = tuple(_radix_two_length(n) for n in shape)
    if not shape:
        raise ValueError("at least one transform axis is required")
    batch = _positive_integer(batch, "batch")
    levels = sum(n.bit_length() - 1 for n in shape)
    return batch * _product(shape) * levels // 2


@lru_cache(maxsize=128)
def primitive_power_of_two_root(modulus, n):
    """Match the direct oracle's first valid candidate in [2, 10001]."""
    modulus = _positive_integer(modulus, "modulus")
    if modulus < 3 or modulus % 2 == 0:
        raise ValueError("modulus must be an odd prime of at least 3")
    n = _radix_two_length(n)
    if n == 1:
        return 1
    if (modulus - 1) % n:
        raise ValueError("axis length does not divide modulus-1")
    for candidate in range(2, 10002):
        root = pow(candidate, (modulus - 1) // n, modulus)
        if pow(root, n, modulus) == 1 and pow(root, n // 2, modulus) != 1:
            return root
    raise ValueError("primitive root search bound reached")


@lru_cache(maxsize=32)
def _bit_reversed_indices(n):
    n = _radix_two_length(n)
    original = np.arange(n, dtype=np.intp)
    reversed_indices = np.zeros(n, dtype=np.intp)
    for _ in range(n.bit_length() - 1):
        reversed_indices = (reversed_indices << 1) | (original & 1)
        original >>= 1
    reversed_indices.flags.writeable = False
    return reversed_indices


@lru_cache(maxsize=128)
def _stage_powers(modulus, root, n, width):
    step = pow(root, n // width, modulus)
    powers = np.empty(width // 2, dtype=object)
    current = 1
    for index in range(width // 2):
        powers[index] = current
        current = current * step % modulus
    powers.flags.writeable = False
    return powers


def radix2_ntt_reference(source, modulus, *, inverse=False,
                         normalize_inverse=True, max_butterflies=None):
    """Return an exact object array shaped (batch, *transform_axes).

    The caller supplies an odd prime modulus validated by its semantic contract.
    No floating-point operations or fixed-width modular products are used.
    Unsupported non-power-of-two shapes raise ValueError for an explicit caller
    fallback. The optional budget counts radix-two butterflies rather than the
    direct oracle's quadratic summation terms.
    """
    source = np.asarray(source)
    if source.ndim < 2:
        raise ValueError("source must have batch and at least one transform axis")
    modulus = _positive_integer(modulus, "modulus")
    if modulus < 3 or modulus % 2 == 0:
        raise ValueError("modulus must be an odd prime of at least 3")
    shape = tuple(_radix_two_length(n) for n in source.shape[1:])
    batch = _positive_integer(source.shape[0], "batch")
    operations = ntt_butterfly_count(shape, batch)
    if max_butterflies is not None:
        if isinstance(max_butterflies, bool) or not isinstance(max_butterflies, numbers.Integral) or max_butterflies < 0:
            raise ValueError("max_butterflies must be a nonnegative integer")
        if operations > max_butterflies:
            raise ReferenceBudgetExceeded(f"radix-two NTT butterflies={operations}; bound={max_butterflies}")
    if source.dtype.kind not in "iuO":
        raise ValueError("NTT source requires integer values")
    if source.dtype.kind == "O" and any(isinstance(x, bool) or not isinstance(x, numbers.Integral) for x in source.flat):
        raise ValueError("NTT object source requires integer values")
    # Object ufuncs dispatch to Python integers and cannot overflow at UINT64.
    work = source.astype(object, copy=True)
    if source.dtype.kind == "O":
        # An object array may contain np.uint64 scalars, whose operators remain
        # fixed-width until explicitly converted to Python integers.
        work.flat[:] = [int(value) for value in work.flat]
    work %= modulus
    for axis, n in enumerate(shape, 1):
        if n == 1:
            continue
        root = primitive_power_of_two_root(modulus, n)
        if inverse:
            root = pow(root, -1, modulus)
        moved = np.moveaxis(work, axis, -1).copy()
        rows = moved.reshape(-1, n)
        rows[:] = rows[:, _bit_reversed_indices(n)]
        width = 2
        while width <= n:
            blocks = rows.reshape(-1, width)
            half = width // 2
            left = blocks[:, :half].copy()
            right = blocks[:, half:] * _stage_powers(modulus, root, n, width)
            right %= modulus
            blocks[:, :half] = (left + right) % modulus
            blocks[:, half:] = (left - right) % modulus
            width *= 2
        if inverse and normalize_inverse:
            rows *= pow(n, -1, modulus)
            rows %= modulus
        work = np.moveaxis(moved, -1, axis)
    return work
