"""Declared CPU reference for both LowFftOperator compute modes.

Native mode rounds each primitive RNE to FP16/BF16. FP32-compute uses
explicit unfused FP32 products and sums, narrowing each complex product,
butterfly sum/difference and normalization result to storage, as in the source.
Twiddles follow host double cos/sin -> FP32 -> storage. The caller supplies
measured multidimensional axis order. GPU FMA contraction may differ; this
reference does not claim instruction equivalence or vendor arithmetic parity.
"""
from __future__ import annotations
import math
import numpy as np


def rounded(value, storage):
    value = np.asarray(value, dtype=np.float32)
    if storage == 4:
        with np.errstate(over='ignore', invalid='ignore'):
            return value.astype(np.float16).astype(np.float32)
    if storage != 5:
        raise ValueError('expected FP16/BF16 complex storage (4/5)')
    bits = value.view(np.uint32)
    result = ((bits + np.uint32(0x7fff) + ((bits >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xffff0000)).view(np.float32)
    return np.where(np.isnan(value), np.float32(np.nan), result)


def butterfly(left, right, root, storage, normalize=False, native=True):
    q = lambda x: rounded(x, storage)
    # Last dimension contains real/imaginary components, independently rounded.
    left, right = np.asarray(left, np.float32), np.asarray(right, np.float32)
    limit = 16376. if storage == 4 else np.array(0x7e7f0000, np.uint32).view(np.float32).item()
    after = normalize & (np.max(np.abs(left), axis=-1) <= limit) & (np.max(np.abs(right), axis=-1) <= limit)
    before = normalize & ~after
    left = np.where(np.asarray(before)[..., None], q(left * .5), left)
    right = np.where(np.asarray(before)[..., None], q(right * .5), right)
    product_round = q if native else lambda x: np.asarray(x, dtype=np.float32)
    rr, ii = product_round(right[..., 0] * root[..., 0]), product_round(right[..., 1] * root[..., 1])
    ri, ir = product_round(right[..., 0] * root[..., 1]), product_round(right[..., 1] * root[..., 0])
    weighted = np.stack((q(rr - ii), q(ri + ir)), axis=-1)
    lo, hi = q(left + weighted), q(left - weighted)
    return (np.where(np.asarray(after)[..., None], q(lo * .5), lo),
            np.where(np.asarray(after)[..., None], q(hi * .5), hi))


def transform(source, semantics, axis_order=None):
    if semantics['compute'] not in (0, 1):
        raise ValueError('expected native or FP32 compute')
    shape = semantics['shape']
    if axis_order is None:
        if len(shape) != 1:
            raise ValueError('measured axis_order required for multidimensional stage arithmetic')
        axis_order = [0]
    if sorted(axis_order) != list(range(len(shape))):
        raise ValueError('invalid axis_order')
    if semantics.get('fft_output_scale', 1) != 1:
        raise ValueError('non-unit output scale adapter not modeled')
    storage = semantics['storage']
    source = np.asarray(source).reshape((semantics['batch'], *shape))
    work = np.stack((source.real, source.imag), axis=-1).astype(np.float32)
    if not np.array_equal(work, rounded(work, storage), equal_nan=True):
        raise ValueError('source must already be storage-quantized')
    for axis in axis_order:
        n = shape[axis]
        if n < 2 or n & (n - 1):
            raise ValueError('power-of-two axes >=2 required')
        bits = n.bit_length() - 1
        reverse = [int(f'{i:0{bits}b}'[::-1], 2) for i in range(n)]
        a = np.moveaxis(work, axis + 1, -2)[..., reverse, :].copy()
        for stage in range(bits):
            half = 1 << stage
            angle = [(1 if semantics['inverse'] else -1) * math.pi * j / half for j in range(half)]
            root = rounded(np.array([(math.cos(t), math.sin(t)) for t in angle], np.float32), storage)
            blocks = a.reshape((*a.shape[:-2], n // (2 * half), 2 * half, 2))
            lo, hi = butterfly(blocks[..., :half, :], blocks[..., half:, :], root, storage,
                bool(semantics['inverse'] and semantics['normalize_inverse']), native=semantics['compute'] == 0)
            blocks[..., :half, :], blocks[..., half:, :] = lo, hi
        work = np.moveaxis(a, -2, axis + 1)
    result = np.empty(source.shape, np.complex64)
    result.real, result.imag = work[..., 0], work[..., 1]
    return result
