#!/usr/bin/env python3
"""Descriptor-first conformance and explicitly bounded GPU timing campaign.

No CUDA initialization occurs at import. Missing implementations are gaps, never
passes. CPU oracle bounds and the fixed tolerances are part of resume identity.
"""
from __future__ import annotations

import argparse
import copy
import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

import numpy as np

RUNNER_REVISION = '0.9.3-bounded-exact-reference'


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result

TOLERANCES = {0: (.004, .004), 1: (.04, .04), 4: (.004, .004), 5: (.04, .04),
              2: (3e-5, 3e-5), 3: (1e-10, 1e-10), 6: (3e-5, 3e-5),
              7: (1e-10, 1e-10), 8: (0, 0), 9: (0, 0)}
DTYPES = {0: np.dtype('float16'), 1: np.dtype('uint16'), 4: np.dtype('V4'), 5: np.dtype('V4'), 2: np.dtype('float32'), 3: np.dtype('float64'),
          6: np.dtype('complex64'), 7: np.dtype('complex128'),
          8: np.dtype('uint32'), 9: np.dtype('uint64')}
PROTOCOL = {'reference_max_elements': 1048576, 'ntt_max_operations': 16777216,
            'ntt_reference': 'exact integer radix-2 butterflies; direct DFT for non-power-of-two axes',
            'ntt_operation_budget_units': 'radix-2 butterflies for all-power-of-two shapes; direct DFT terms otherwise',
            'warmup_calls': 10, 'graph_calls': 20, 'replays': 7,
            'tolerances': {str(k): list(v) for k, v in TOLERANCES.items()},
            'comparison': 'all elements; abs(a-e) <= atol+rtol*abs(e); finite required',
            'in_place_timing': 'single calls, input reset outside events',
            'excluded': ['planning', 'allocation', 'compilation', 'reference', 'warmup']}
QUANTILE_PROBABILITIES = (0, .25, .5, .75, .9, .95, .99, .999, 1)
LOG_BIN_EDGES = [0.0, *(10.0**power for power in range(-16, 17)), float('inf')]
PROTOCOL['error_distribution'] = {
    'quantiles': list(QUANTILE_PROBABILITIES),
    'quantile_rank': 'nearest rank: max(0, ceil(q*N)-1), sorted ascending; q=0 is minimum',
    'positive_log_bin_edges': [*LOG_BIN_EDGES[:-1], 'infinity'],
    'bin_convention': 'positive values only; [left,right), final bin includes infinity; zeros separate',
    'artifact': 'lossless NPZ actual/reference plus full errors and reference index partition'}


def error_arrays(actual, expected):
    a, e = np.asarray(actual).reshape(-1), np.asarray(expected).reshape(-1)
    arrays = {'actual': a, 'reference': e}
    if a.dtype.kind == 'u' and e.dtype.kind == 'u':
        return arrays  # Preserve unsigned words without subtraction or float conversion.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        absolute = np.abs(a - e)
        magnitude = np.abs(e)
        nonzero = magnitude != 0
        arrays.update(abs_error=absolute,
                      relative_error_nonzero_reference=absolute[nonzero] / magnitude[nonzero],
                      nonzero_reference_flat_indices=np.flatnonzero(nonzero),
                      abs_error_zero_reference=absolute[~nonzero],
                      zero_reference_flat_indices=np.flatnonzero(~nonzero))
    return arrays


def distribution(values):
    values = np.asarray(values).reshape(-1)
    ordered = np.sort(values)
    quantiles = {}
    for probability in QUANTILE_PROBABILITIES:
        rank = max(0, math.ceil(probability * len(ordered)) - 1)
        value = float(ordered[rank]) if len(ordered) else None
        quantiles[str(probability)] = value if value is None or math.isfinite(value) else 'infinite'
    positive = values[values > 0]
    counts, _ = np.histogram(positive, bins=LOG_BIN_EDGES)
    return {'elements': int(len(values)), 'zero_count': int(np.count_nonzero(values == 0)),
            'nonfinite_count': int(np.count_nonzero(~np.isfinite(values))), 'quantiles': quantiles,
            'positive_log_bin_counts': [int(count) for count in counts]}


def retain_error_artifact(directory, case_id, backend, actual, expected):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (case_id + '-' + backend + '.npz')
    temporary = path.with_suffix('.npz.tmp')
    arrays = error_arrays(actual, expected)
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return {'path': str(path.resolve()), 'sha256': sha256_file(path), 'format': 'lossless-numpy-npz',
            'array_count': len(arrays), 'array_elements': {name: int(value.size) for name, value in arrays.items()},
            'array_dtypes': {name: str(value.dtype) for name, value in arrays.items()}}


class Gap(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class BaselineCorrectnessFailure(AssertionError):
    def __init__(self, message, rows):
        super().__init__(message)
        self.rows = rows


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def assert_semantics(original, exported):
    if original['semantics'] != exported['semantics']:
        raise AssertionError('descriptor semantics changed during import/planning/replay')


def selection_descriptor(descriptor, timing=False, mapping=None):
    value = copy.deepcopy(descriptor)
    value['selection']['algorithm_policy'] = 2 if mapping is not None else int(timing)
    value['selection']['explicit_algorithm'] = canonical(mapping) if mapping is not None else ''
    assert_semantics(descriptor, value)
    return value


def side_extents(s, side):
    shape = list(s['shape'])
    if s['length_mode'] and ((side == 'input') == bool(s['inverse'])):
        shape = [1 << (n - 1).bit_length() for n in shape]
    if s['operator'] == 0 and ((side == 'input' and s['fft_type'] == 2) or
                               (side == 'output' and s['fft_type'] == 1)):
        shape[-1] = shape[-1] // 2 + 1
    return shape


def side_storage(s, side):
    if side not in ('input', 'output'):
        raise ValueError('side must be input or output')
    storage = s['storage']
    if storage not in DTYPES:
        raise Gap('runner-gap', 'no safe exact storage adapter for storage=' + str(storage))
    if s['operator'] == 0 and s['fft_type']:
        if storage not in (0, 1, 2, 3) or s['fft_type'] not in (1, 2):
            raise Gap('runner-gap', 'real FFT requires real floating storage and R2C/C2R type')
        complex_side = (side == 'input' and s['fft_type'] == 2) or (side == 'output' and s['fft_type'] == 1)
        if complex_side:
            storage += 4
    return storage


def side_dtype(s, side):
    return DTYPES[side_storage(s, side)]


def encode_storage(value, storage):
    """Raw low precision storage, including RNE BF16 component encoding."""
    if storage in (4, 5):
        value = np.asarray(value, dtype=np.complex64)
        components = np.stack((value.real, value.imag), axis=-1)
        if storage == 4:
            return components.astype(np.float16).view(np.dtype('V4')).reshape(value.shape)
        return encode_storage(components, 1).view(np.dtype('V4')).reshape(value.shape)
    if storage == 1:
        bits = np.asarray(value, dtype=np.float32).view(np.uint32)
        return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return np.asarray(value, dtype=DTYPES[storage])


def decode_storage(value, storage):
    value = np.asarray(value)
    if storage in (4, 5):
        components = value.copy().view(np.float16 if storage == 4 else np.uint16).reshape((*value.shape, 2))
        if storage == 5:
            components = decode_storage(components, 1)
        # Arithmetic construction (real + 1j * imag) evaluates 0 * Inf
        # and can manufacture a NaN in a different component. Copy the raw
        # component values, preserving infinities, NaNs and signed zeros.
        decoded = np.empty(value.shape, dtype=np.complex64)
        decoded.real = components[..., 0]
        decoded.imag = components[..., 1]
        return decoded
    if storage == 1:
        return (value.astype(np.uint32) << 16).view(np.float32)
    return value


def floating_range_report(actual, expected, storage):
    """All-element diagnostics only; a range finding never grants acceptance."""
    if storage in (8, 9):
        return None
    limit = (float(np.finfo(np.float16).max) if storage in (0, 4) else
             float(np.array([0x7f7f0000], dtype=np.uint32).view(np.float32)[0])
             if storage in (1, 5) else
             float(np.finfo(np.float64 if storage in (3, 7) else np.float32).max))
    a, e = np.asarray(actual).reshape(-1), np.asarray(expected).reshape(-1)
    if a.shape != e.shape:
        raise AssertionError('output shape mismatch')
    parts = [('real', a.real, e.real)]
    if np.iscomplexobj(a) or np.iscomplexobj(e):
        parts.append(('imaginary', a.imag, e.imag))
    outside = np.zeros(e.size, dtype=bool)
    actual_bad = ~np.isfinite(a)
    components = {}
    for name, av, ev in parts:
        finite = np.isfinite(ev)
        beyond = finite & (np.abs(ev) > limit)
        outside |= beyond
        components[name] = {
            'reference_nonfinite': int(np.count_nonzero(~finite)),
            'reference_exceeds_max_finite': int(np.count_nonzero(beyond)),
            'reference_max_abs_finite': float(np.abs(ev[finite]).max(initial=0)),
            'actual_nan': int(np.count_nonzero(np.isnan(av))),
            'actual_positive_infinity': int(np.count_nonzero(np.isposinf(av))),
            'actual_negative_infinity': int(np.count_nonzero(np.isneginf(av))),
        }
    return {'storage_max_finite': limit, 'elements': int(e.size), 'components': components,
            'reference_exceeds_max_finite_elements': int(np.count_nonzero(outside)),
            'actual_nonfinite_elements': int(np.count_nonzero(actual_bad)),
            'actual_nonfinite_at_reference_in_range': int(np.count_nonzero(actual_bad & ~outside & np.isfinite(e))),
            'classification': ('reference-exceeds-storage-range' if np.any(outside) else
                               'nonfinite-within-reference-range' if np.any(actual_bad) else 'finite'),
            'acceptance': 'diagnostic only; does not relax nonfinite or error checks'}


def layout_indices(s, side):
    shape = side_extents(s, side)
    coords = np.indices((s['batch'], *shape), dtype=np.int64)
    result = coords[0] * s[side + '_batch_stride']
    for axis, (n, stride, order) in enumerate(zip(shape, s[side + '_strides'], s[side + '_orders'])):
        coord = coords[axis + 1]
        if order:
            if n & (n - 1):
                raise Gap('runner-gap', 'bit order requires power of two')
            reversed_coord = np.zeros_like(coord)
            for bit in range(n.bit_length() - 1):
                reversed_coord |= ((coord >> bit) & 1) << (n.bit_length() - 2 - bit)
            coord = reversed_coord
        result = result + coord * stride
    return result.reshape(-1)


def check_full(actual, expected, storage, float_verdict='legacy-fixed', include_distribution=True):
    a, e = np.asarray(actual).reshape(-1), np.asarray(expected).reshape(-1)
    if a.shape != e.shape:
        raise AssertionError('output shape mismatch')
    if storage in (8, 9):
        dtype = DTYPES[storage]
        if a.dtype != dtype or e.dtype != dtype:
            raise AssertionError('integer validation requires exact unsigned storage dtype')
        bad = a != e
        if np.any(bad):
            index = int(np.flatnonzero(bad)[0])
            raise AssertionError(f'integer mismatch at {index}: {int(a[index])} != {int(e[index])}')
        return {'elements': int(a.size), 'exact': True, 'mismatches': 0}
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(e)):
        raise AssertionError('NaN/Inf in actual or reference')
    atol, rtol = TOLERANCES[storage]
    error = np.abs(a - e)
    if not np.all(np.isfinite(error)):
        raise AssertionError('nonfinite absolute error from finite inputs')
    limits = atol + rtol * np.abs(e)
    bad = error > limits
    if np.any(bad) and float_verdict == 'legacy-fixed':
        index = int(np.flatnonzero(bad)[0])
        raise AssertionError(f'output mismatch at {index}: error={error[index]}, limit={limits[index]}')
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        relative = np.divide(error, np.abs(e), out=np.zeros_like(error), where=np.abs(e) != 0)
    max_relative = float(relative.max(initial=0))
    max_abs = float(error.max(initial=0))
    rms_abs = max_abs * float(np.sqrt(np.mean((error / max_abs)**2))) if max_abs else 0.0
    result = {'elements': int(a.size), 'max_abs': float(error.max(initial=0)),
            'max_relative_nonzero_reference': max_relative if math.isfinite(max_relative) else 'infinite',
            'max_abs_at_zero_reference': float(error[np.abs(e) == 0].max(initial=0)), 'rms_abs': rms_abs,
            'expected_max_abs': float(np.abs(e).max(initial=0)),
            'atol': atol, 'rtol': rtol, 'mismatches': int(np.count_nonzero(bad)),
            'tolerance_role': 'historical comparison only; no acceptance threshold' if float_verdict == 'error-report' else 'fixed acceptance threshold',
            'verdict': 'accuracy-review-required' if float_verdict == 'error-report' else 'pass'}
    reference_scale = float(np.abs(e).max(initial=0))
    if reference_scale:
        with np.errstate(over='ignore', invalid='ignore'):
            l2_relative = (max_abs / reference_scale) * float(np.sqrt(np.sum((error / max_abs)**2) / np.sum((np.abs(e) / reference_scale)**2))) if max_abs else 0.0
        result['l2_relative'] = l2_relative if math.isfinite(l2_relative) else None
    else:
        result['l2_relative'] = 0.0 if max_abs == 0 else None
    if include_distribution:
        arrays = error_arrays(a, e)
        result['error_distributions'] = {
            'absolute': distribution(arrays['abs_error']),
            'relative_nonzero_reference': distribution(arrays['relative_error_nonzero_reference']),
            'absolute_zero_reference': distribution(arrays['abs_error_zero_reference'])}
    return result


def make_source(s):
    shape = (s['batch'], *side_extents(s, 'input'))
    dtype = side_dtype(s, 'input')
    storage = side_storage(s, 'input')
    if storage in (1, 4, 5):
        dtype = np.dtype('complex64' if storage in (4, 5) else 'float32')
    index = np.arange(_product(shape), dtype=np.uint64).reshape(shape)
    if dtype.kind == 'u':
        # Include values above float's exact integer range for UINT64 zeta.
        value = (index * np.uint64(104729) + np.uint64(17))
        if s['operator'] == 1:
            modulus = np.uint64(s['modulus'])
            value %= modulus
            near_modulus = modulus - np.uint64(1) - (index % np.uint64(97) % modulus)
            value = np.where(index % np.uint64(5) == 0, near_modulus, value)
        elif dtype.itemsize == 8:
            value += np.uint64(1 << 55)
        return value.astype(dtype)
    real = np.sin(index.astype(np.float64) * .17) + (index % 13).astype(np.float64) / 17
    if dtype.kind == 'c':
        real = real + 1j * np.cos(index.astype(np.float64) * .11)
    value = real.astype(dtype)
    if s['operator'] == 0 and s['fft_type'] == 2:
        # Generate a valid Hermitian packed spectrum with batch variation.
        full = list(s['shape'])
        if s['length_mode']:
            full = [1 << (n - 1).bit_length() for n in full]
        real_index = np.arange(s['batch'] * _product(full)).reshape((s['batch'], *full))
        value = np.fft.rfftn(np.sin(real_index * .17), axes=tuple(range(1, len(full) + 1))).astype(dtype)
    if storage in (0, 1, 4, 5):
        value = decode_storage(encode_storage(value, storage), storage)
    return value


def reference(s, source, max_elements=1048576, ntt_max_operations=16777216):
    points = s['batch'] * _product([1 << (n - 1).bit_length() if s['length_mode'] else n for n in s['shape']])
    if points > max_elements:
        raise Gap('validation-unavailable', f'full CPU reference requires {points} elements; bound={max_elements}')
    op, inverse = s['operator'], bool(s['inverse'])
    axes = tuple(range(1, len(s['shape']) + 1))
    if s['ntt_input_layout'] == 1 or s['ntt_output_layout'] == 1:
        raise Gap('runner-gap', 'APPT static layout storage adapter is not bound')
    full_shape = [1 << (n - 1).bit_length() if s['length_mode'] else n for n in s['shape']]
    if op == 0:
        sizes = full_shape
        # Older NumPy/MKL builds used by the CPU preparation environment only
        # accept None/ortho.  Apply the inverse scaling explicitly below.
        norm = None
        if s['fft_type'] == 1:
            work = np.fft.rfftn(source, s=sizes, axes=axes)
        elif s['fft_type'] == 2:
            work = np.fft.irfftn(source, s=sizes, axes=axes, norm=norm)
        elif inverse:
            work = np.fft.ifftn(source, s=sizes, axes=axes, norm=norm)
        else:
            work = np.fft.fftn(source, s=sizes, axes=axes)
        if inverse and not s['normalize_inverse']:
            work *= _product(sizes)
        work *= s['fft_output_scale']
    else:
        dtype = object if op == 1 else np.float64 if source.dtype.kind == 'f' else np.complex128 if source.dtype.kind == 'c' else source.dtype
        if s['length_mode'] and not inverse:
            work = np.zeros((s['batch'], *full_shape), dtype=dtype)
            work[(slice(None), *(slice(0, n) for n in s['shape']))] = source
        else:
            work = source.astype(dtype, copy=True)
        radix2_ntt = op == 1 and all(n > 0 and not n & (n - 1) for n in full_shape)
        if radix2_ntt:
            module_directory = str(Path(__file__).resolve().parent)
            if module_directory not in sys.path:
                sys.path.insert(0, module_directory)
            from semantic_ntt_reference import radix2_ntt_reference, ReferenceBudgetExceeded
            try:
                work = radix2_ntt_reference(work, s['modulus'], inverse=inverse,
                    normalize_inverse=bool(s['normalize_inverse']), max_butterflies=ntt_max_operations)
            except (ReferenceBudgetExceeded, ValueError) as error:
                raise Gap('validation-unavailable', str(error)) from error
        elif op == 1:
            operations = points * sum(full_shape)
            if operations > ntt_max_operations:
                raise Gap('validation-unavailable', f'quadratic NTT oracle operations={operations}; bound={ntt_max_operations}')
        for axis, n in enumerate([] if radix2_ntt else full_shape, 1):
            moved = np.moveaxis(work, axis, -1).copy()
            rows = moved.reshape(-1, n)
            if op == 1:
                p = s['modulus']
                if n == 1:
                    root = 1
                else:
                    # Bluestein's public non-power-of-two transform uses the
                    # square of the first primitive 2N-th root.
                    order = n if not n & (n - 1) else 2 * n
                    remaining, divisor, factors = order, 2, []
                    while divisor * divisor <= remaining:
                        if remaining % divisor == 0:
                            factors.append(divisor)
                            while remaining % divisor == 0:
                                remaining //= divisor
                        divisor += 1
                    if remaining > 1:
                        factors.append(remaining)
                    root = None
                    for candidate in range(2, 10002):
                        trial = pow(candidate, (p - 1) // order, p)
                        if (p - 1) % order == 0 and pow(trial, order, p) == 1 and all(pow(trial, order // factor, p) != 1 for factor in factors):
                            root = trial if order == n else trial * trial % p
                            break
                    if root is None:
                        raise Gap('validation-unavailable', 'primitive root search bound reached')
                if inverse:
                    root = pow(root, p - 2, p)
                for row in rows:
                    original = [int(x) for x in row]
                    row[:] = [sum(x * pow(root, j * k, p) for j, x in enumerate(original)) % p for k in range(n)]
                    if inverse and s['normalize_inverse']:
                        row[:] = [int(x) * pow(n, p - 2, p) % p for x in row]
            else:
                half, stage = 1, 0
                while half < n:
                    if n % (2 * half):
                        raise Gap('runner-gap', 'non-radix-two butterfly reference')
                    blocks = rows.reshape(-1, 2 * half)
                    left, right = blocks[:, :half].copy(), blocks[:, half:].copy()
                    if op == 2:
                        blocks[:, :half], blocks[:, half:] = left + right, left - right
                    elif op == 3:
                        blocks[:, half:] = right - left if inverse else right + left
                    elif op == 4:
                        blocks[:, :half] = left - right if inverse else left + right
                    elif op == 5:
                        matrices = s['matrices'][axis - 1]
                        matrix = np.asarray(matrices[0 if len(matrices) == 1 else stage]).reshape(2, 2)
                        if inverse:
                            matrix = np.linalg.inv(matrix)
                        blocks[:, :half] = matrix[0, 0] * left + matrix[0, 1] * right
                        blocks[:, half:] = matrix[1, 0] * left + matrix[1, 1] * right
                    else:
                        raise Gap('runner-gap', 'unknown operator')
                    half *= 2
                    stage += 1
                if op == 2 and inverse and s['normalize_inverse']:
                    rows /= n
            work = np.moveaxis(rows.reshape(moved.shape), -1, axis)
    if s['length_mode'] and inverse:
        work = work[(slice(None), *(slice(0, n) for n in side_extents(s, 'output')))]
    if side_storage(s, 'output') in (0, 1, 4, 5):
        # Reference arithmetic stays high precision; storage tolerance accounts
        # for the explicitly requested execution precision, not a relaxed rerun.
        return np.asarray(work, dtype=np.complex128 if side_storage(s, 'output') in (4, 5) else np.float64)
    if side_dtype(s, 'output').kind in ('f', 'c'):
        return np.asarray(work, dtype=np.complex128 if side_dtype(s, 'output').kind == 'c' else np.float64)
    return np.asarray(work, dtype=side_dtype(s, 'output'))


def physical_bytes(s, side, logical, size):
    dtype = side_dtype(s, side)
    storage = side_storage(s, side)
    if size % dtype.itemsize:
        raise AssertionError('plan buffer byte size does not align with side dtype')
    raw = np.full(size, 0xA5, dtype=np.uint8)
    indices = layout_indices(s, side)
    if indices.size != logical.size or int(indices.max(initial=0)) >= size // dtype.itemsize:
        raise AssertionError('public side layout exceeds plan size')
    encoded = encode_storage(logical, storage)
    raw.view(dtype)[indices] = encoded.reshape(-1)
    return raw


def output_from_bytes(s, raw):
    stored = np.asarray(raw).view(side_dtype(s, 'output'))[layout_indices(s, 'output')]
    return decode_storage(stored, side_storage(s, 'output'))


def check_padding(s, actual, initial, side='output'):
    indices = layout_indices(s, side)
    width = side_dtype(s, side).itemsize
    mask = np.ones(len(actual), dtype=bool)
    for byte in range(width):
        mask[indices * width + byte] = False
    if not np.array_equal(np.asarray(actual)[mask], np.asarray(initial)[mask]):
        raise AssertionError(side + ' padding was modified')


def time_execute(torch, stream, execute, reset, in_place, verify, poison_output=None):
    # Graph replay uses the current stream, unlike events recorded explicitly on
    # `stream`. Keep both on the supplied stream even for external callers.
    with torch.cuda.stream(stream):
        return _time_execute_current_stream(torch, stream, execute, reset, in_place,
                                            verify, poison_output)


def _time_execute_current_stream(torch, stream, execute, reset, in_place, verify, poison_output=None):
    for _ in range(10):
        if in_place:
            reset()
        execute()
    stream.synchronize()
    samples = []
    if in_place:
        for _ in range(7):
            reset()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(stream)
            execute()
            end.record(stream)
            end.synchronize()
            samples.append(start.elapsed_time(end))
            verify()
        protocol = '7 single event calls; reset enqueued before start event; 10 warmup'
    else:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(20):
                execute()
        for _ in range(7):
            if poison_output is not None:
                poison_output()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(stream)
            graph.replay()
            end.record(stream)
            end.synchronize()
            samples.append(start.elapsed_time(end) / 20)
            verify()
        protocol = '20 complete calls per CUDA graph; 7 replays; 10 warmup'
    return {'median_ms': statistics.median(samples), 'samples_ms': samples, 'protocol': protocol}


def run_case(case, library, torch, stream, timing=False, reference_max_elements=1048576,
             ntt_max_operations=16777216, vkfft_library=None, float_verdict='error-report', error_dir=None,
             gpuntt_library=None):
    """Execute one unchanged descriptor, return explicit terminal result and gaps."""
    import cubutterfly as cb
    descriptor = case['descriptor']
    s = descriptor['semantics']
    row = {'id': case['id'], 'populations': case['populations'], 'descriptor': descriptor,
           'requested_variants': case.get('requested_variants', []), 'baselines': []}
    if s['operator'] in (3, 4, 5):
        row['baseline_coverage_limitations'] = ['No selected external comparator for zeta/structured operator; public execution is compared with the complete CPU oracle only.']
    plan = replay = None
    try:
        with cb.Descriptor.from_json(library, descriptor) as imported:
            imported.validate()
            exported = json.loads(imported.json())
            assert_semantics(descriptor, exported)
            row['capabilities'] = json.loads(imported.capabilities_json())
        if row['capabilities'].get('status') == 'missing-lowering':
            raise Gap('missing-lowering', 'public capability has no executable lowering')
        if row['capabilities'].get('status') != 'eligible':
            raise Gap('runner-gap', 'unrecognized public capability status')
        # Bound CPU allocation before generating coordinates or source.
        points = s['batch'] * max(_product(side_extents(s, side)) for side in ('input', 'output'))
        if points > reference_max_elements:
            raise Gap('validation-unavailable', f'all-elements bound exceeded: {points}>{reference_max_elements}')
        source = make_source(s)
        expected = reference(s, source, reference_max_elements, ntt_max_operations)
        chosen = selection_descriptor(descriptor, timing)
        start = time.perf_counter()
        with cb.Descriptor.from_json(library, chosen) as imported:
            plan = cb.Plan(library, imported, stream=stream.cuda_stream)
        row['plan_build_ms'] = (time.perf_counter() - start) * 1000
        row['plan_descriptor'] = json.loads(plan.descriptor_json())
        assert_semantics(descriptor, row['plan_descriptor'])
        row.update(mapping=json.loads(plan.mapping_json()), selection_reason=plan.selection_reason(),
                   algorithm=plan.algorithm_name(), workspace_bytes=plan.workspace_size(),
                   input_bytes=plan.input_size(), output_bytes=plan.output_size())
        row['boundary'] = 'complete public descriptor execution'
        row['category'] = 'adapter' if row['mapping'].get('kind') in ('semantic-boundary', 'numeric-adapter') else 'native-contract'
        row['mapping_kind'] = row['mapping'].get('kind')
        row['selection_scope'] = 'public MEASURE policy; adapter child selection may exclude fixed boundary cost' if timing else 'public DEFAULT policy'
        workspace = torch.empty(max(1, plan.workspace_size()), dtype=torch.uint8, device='cuda')
        if plan.workspace_size():
            plan.set_workspace(workspace, plan.workspace_size())
        in_place = bool(s['placement'])
        input_raw = physical_bytes(s, 'input', source, plan.input_size())
        raw_size = max(plan.input_size(), plan.output_size()) if in_place else plan.input_size()
        input_initial = np.full(raw_size, 0xA5, dtype=np.uint8)
        input_initial[:len(input_raw)] = input_raw
        gpu_input = torch.from_numpy(input_initial).to('cuda')
        gpu_saved = gpu_input.clone()
        output_initial = input_initial.copy() if in_place else np.full(plan.output_size(), 0xA5, dtype=np.uint8)
        gpu_output = gpu_input if in_place else torch.from_numpy(output_initial).to('cuda')

        def reset():
            gpu_input.copy_(gpu_saved)

        def verify(raw_reference=None):
            raw = gpu_output.cpu().numpy()
            if raw_reference is not None and not np.array_equal(raw, raw_reference):
                raise AssertionError('timed execution raw output differs from the same plan direct execution')
            actual = output_from_bytes(s, raw)
            first = 'correctness' not in row
            if s['storage'] not in (8, 9) and (first or not np.all(np.isfinite(actual))):
                row['numeric_diagnostics'] = floating_range_report(actual, expected, s['storage'])
            if first and error_dir is not None:
                row['error_artifact'] = retain_error_artifact(error_dir, case['id'], 'cubutterfly', actual, expected)
            result = check_full(actual, expected, s['storage'], float_verdict, first)
            check_padding(s, raw, output_initial)
            if not in_place and not torch.equal(gpu_input, gpu_saved):
                raise AssertionError('out-of-place input or padding modified')
            return result

        plan.execute(gpu_input, gpu_output)
        stream.synchronize()
        row['correctness'] = verify()
        direct_output_raw = gpu_output.cpu().numpy().copy() if timing else None
        replay_desc = selection_descriptor(descriptor, mapping=row['mapping'])
        with cb.Descriptor.from_json(library, replay_desc) as imported:
            replay = cb.Plan(library, imported, stream=stream.cuda_stream)
        assert_semantics(descriptor, json.loads(replay.descriptor_json()))
        if json.loads(replay.mapping_json()) != row['mapping']:
            raise AssertionError('explicit mapping replay changed mapping')
        replay_workspace = torch.empty(max(1, replay.workspace_size()), dtype=torch.uint8, device='cuda')
        if replay.workspace_size():
            replay.set_workspace(replay_workspace, replay.workspace_size())
        reset()
        if not in_place:
            gpu_output.fill_(0xA5)
        replay.execute(gpu_input, gpu_output)
        stream.synchronize()
        row['mapping_replay_correctness'] = verify()
        if timing:
            row['timing'] = time_execute(torch, stream, lambda: plan.execute(gpu_input, gpu_output),
                                         reset, in_place, lambda: verify(direct_output_raw),
                                         lambda: gpu_output.fill_(0xA5))
        row['baselines'] = run_baselines(case, torch, stream, source, expected, gpu_saved,
                                         output_initial, timing, vkfft_library, float_verdict, error_dir, gpuntt_library)
        row['status'] = 'accuracy-review-required' if s['storage'] not in (8, 9) and float_verdict == 'error-report' else 'pass'
    except Gap as exc:
        row.update(status=exc.status, error=str(exc))
    except cb.Error as exc:
        # Only a runtime UNSUPPORTED result is classified as unsupported.
        row.update(status='unsupported' if exc.status == 3 else 'runtime-error', error=str(exc), runtime_status=exc.status)
    except torch.OutOfMemoryError as exc:
        row.update(status='resource-limit', error=str(exc))
    except AssertionError as exc:
        row.update(status='correctness-failure', error=str(exc))
        if isinstance(exc, BaselineCorrectnessFailure):
            row['baselines'] = exc.rows
    except Exception as exc:
        row.update(status='runner-error', error=str(exc), exception_type=type(exc).__name__)
    finally:
        if replay is not None:
            replay.close()
        if plan is not None:
            plan.close()
    return row


def baseline_contract(name, s, vkfft_library=None, gpuntt_library=None):
    """CPU-only exact contract routing, before tensor allocation or backend load."""
    domains = {'cufft': 0, 'vkfft': 0, 'daofht': 2, 'gpu-ntt': 1}
    if name in domains and domains[name] != s['operator']:
        return {'supported': False, 'status': 'not-applicable',
                'reason': 'baseline operator domain does not include this requested operator', 'metadata': {}}
    if name == 'daofht':
        from semantic_dao_baseline import support
        return support(s)
    if name == 'gpu-ntt':
        from semantic_gpuntt_baseline import DEFAULT_LIBRARY, support
        return support(s, librarypath=gpuntt_library or DEFAULT_LIBRARY)
    if name in ('cufft', 'vkfft') and len(s['shape']) > 3:
        from semantic_fft_baseline import support
        return support(name, s, vkfft_library)
    if name not in ('cufft', 'vkfft') or s['operator'] != 0 or s['fft_type'] != 0 or s['storage'] not in (6, 7) or s['length_mode']:
        return {'supported': False, 'status': 'unsupported-semantic',
                'reason': 'no bound exact baseline contract for this descriptor', 'metadata': {}}
    if s['compute'] not in ((0, 1) if s['storage'] == 6 else (0, 2)):
        return {'supported': False, 'status': 'unsupported-semantic',
                'reason': 'vendor arithmetic differs from requested compute precision', 'metadata': {}}
    if name == 'vkfft' and vkfft_library is None:
        return {'supported': False, 'status': 'missing-external-dependency',
                'reason': '--vkfft-library not supplied', 'metadata': {}}
    return {'supported': True, 'status': 'supported', 'reason': 'C2C full logical batch contract', 'metadata': {}}


def baseline_transfer_views(torch, s, physical_input, source, physical_output, output):
    if s['storage'] in (8, 9):
        width = side_dtype(s, 'input').itemsize
        return tuple(tensor.view(torch.uint8).reshape(-1, width)
                     for tensor in (physical_input, source, physical_output, output))
    return physical_input, source.reshape(-1), physical_output, output.reshape(-1)


def run_baselines(case, torch, stream, source, expected, input_saved, output_initial, timing, vkfft_library, float_verdict='error-report', error_dir=None, gpuntt_library=None):
    """Vendor cores plus complete GPU gather/scatter and placement boundaries."""
    s = case['descriptor']['semantics']
    requested = ['cufft', 'vkfft', 'daofht', 'gpu-ntt']
    results = []
    for name in requested:
        native = None
        row = {'name': name, 'status': 'missing-baseline'}
        try:
            contract = baseline_contract(name, s, vkfft_library, gpuntt_library)
            row.update(support_status=contract['status'], support_metadata=contract.get('metadata', {}))
            if not contract['supported']:
                raise Gap('not-applicable' if contract['status'] == 'not-applicable' else 'missing-baseline', contract['reason'])
            dtype_name = {0: 'float16', 1: 'bfloat16', 2: 'float32', 6: 'complex64',
                          7: 'complex128', 8: 'uint32', 9: 'uint64'}[s['storage']]
            dtype = getattr(torch, dtype_name, None)
            if dtype is None:
                row['support_status'] = 'missing-exact-storage-binding'
                raise Gap('missing-baseline', 'Torch lacks exact requested storage dtype ' + dtype_name)
            shape = (s['batch'], *s['shape'])
            src = torch.empty(shape, dtype=dtype, device='cuda')
            out = torch.empty_like(src)
            physical_input = input_saved.clone().view(dtype)
            physical_saved = physical_input.view(torch.uint8).clone()
            physical_output = physical_input if s['placement'] else torch.from_numpy(output_initial).to('cuda').view(dtype)
            in_index = torch.from_numpy(layout_indices(s, 'input')).to('cuda')
            out_index = torch.from_numpy(layout_indices(s, 'output')).to('cuda')
            contiguous = list(range(_product(shape)))
            gather = not np.array_equal(layout_indices(s, 'input'), contiguous)
            scatter = not np.array_equal(layout_indices(s, 'output'), contiguous) or bool(s['placement'])
            # Direct native pointers when the public layout is truly contiguous.
            native_source = src if gather else physical_input[:_product(shape)].reshape(shape)
            native_output = out if scatter else physical_output[:_product(shape)].reshape(shape)
            axis_composed = name in ('cufft', 'vkfft') and len(s['shape']) > 3
            if axis_composed:
                from semantic_fft_baseline import AxisComposition
                native = AxisComposition(name, torch, s, native_source, native_output, stream, vkfft_library)
                vendor_scale = _product(s['shape']) if s['inverse'] and not s['normalize_inverse'] else 1
                row['metadata'] = native.metadata()
            elif name == 'cufft':
                from diagnose_xd_fft import NativeCufft
                native = NativeCufft(torch, s['shape'], s['batch'], s['storage'] == 7,
                                     bool(s['inverse']), stream, native_source, native_output)
                vendor_scale = 1 / _product(s['shape']) if s['inverse'] and s['normalize_inverse'] else 1
            elif name == 'vkfft':
                from vkfft_xd import NativeVkfft
                native = NativeVkfft(torch, s['shape'], s['batch'], s['storage'] == 7,
                                     bool(s['inverse']), stream, native_source, native_output, vkfft_library)
                vendor_scale = _product(s['shape']) if s['inverse'] and not s['normalize_inverse'] else 1
                row['metadata'] = native.metadata()
            elif name == 'daofht':
                from semantic_dao_baseline import build_core
                native = build_core(torch, s, native_source, native_output)
                vendor_scale = 1  # Exact FWHT inverse scale is fused in the Dao core.
                row['metadata'] = native.metadata()
            else:
                from semantic_gpuntt_baseline import DEFAULT_LIBRARY, build_core
                native = build_core(torch, s, native_source, native_output, stream,
                                    librarypath=gpuntt_library or DEFAULT_LIBRARY)
                vendor_scale = 1  # Core includes the exact public inverse-N contract.
                row['metadata'] = native.metadata()
                if row['metadata'].get('caller_stream') != int(stream.cuda_stream):
                    raise AssertionError('GPU-NTT core did not bind the supplied CUDA stream')
            factor = vendor_scale * s['fft_output_scale']
            integer_transfer = s['storage'] in (8, 9)
            transfer_input, transfer_source, transfer_output, transfer_result = baseline_transfer_views(
                torch, s, physical_input, src, physical_output, out)
            operations = []
            if gather:
                operations.append('GPU byte-row index_select logical input gather preserving unsigned words' if integer_transfer else 'GPU index_select logical input gather including order')
            operations.extend(row.get('metadata', {}).get('included_operations', []))
            if factor != 1:
                operations.append(f'GPU multiply output by {factor!r}')
            if scatter:
                operations.append('GPU byte-row index_copy logical output scatter preserving unsigned words' if integer_transfer else 'GPU index_copy logical output scatter including order')
            row.update(category='adapter' if operations or s['placement'] else 'native', adapter_operations=operations,
                       placement_contract='separate scratch output permits public in-place alias' if s['placement'] else 'out-of-place')
            if axis_composed:
                row['category'] = 'external-axis-composition'
                row['metadata']['included_operations'] = operations

            def execute():
                if gather:
                    torch.index_select(transfer_input, 0, in_index, out=transfer_source)
                if name == 'cufft' and not axis_composed:
                    native.check(native.exec_fn(native.handle, ct.c_void_p(native_source.data_ptr()),
                        ct.c_void_p(native_output.data_ptr()), 1 if s['inverse'] else -1))
                else:
                    native()
                if factor != 1:
                    native_output.mul_(factor)
                if scatter:
                    transfer_output.index_copy_(0, out_index, transfer_result)

            def verify():
                raw = physical_output.view(torch.uint8).cpu().numpy()
                actual = output_from_bytes(s, raw)
                first = 'correctness' not in row
                if s['storage'] not in (8, 9) and (first or not np.all(np.isfinite(actual))):
                    row['numeric_diagnostics'] = floating_range_report(actual, expected, s['storage'])
                if first and error_dir is not None:
                    row['error_artifact'] = retain_error_artifact(error_dir, case['id'], name, actual, expected)
                result = check_full(actual, expected, s['storage'], float_verdict, first)
                check_padding(s, raw, output_initial)
                if not s['placement'] and not torch.equal(physical_input.view(torch.uint8), physical_saved):
                    raise AssertionError(name + ': out-of-place input or padding modified')
                return result

            execute()
            stream.synchronize()
            row['correctness'] = verify()
            if timing:
                row['timing'] = time_execute(torch, stream, execute, lambda: physical_input.view(torch.uint8).copy_(physical_saved), bool(s['placement']), verify,
                    lambda: physical_output.view(torch.uint8).fill_(0xA5))
            row['status'] = 'accuracy-review-required' if s['storage'] not in (8, 9) and float_verdict == 'error-report' else 'pass'
        except Gap as exc:
            row.update(status=exc.status, reason=str(exc))
        except AssertionError as exc:
            row.update(status='correctness-failure', error=str(exc))
            results.append(row)
            raise BaselineCorrectnessFailure(str(exc), results) from exc
        except torch.OutOfMemoryError as exc:
            row.update(status='resource-limit', reason=str(exc))
        except Exception as exc:
            row.update(status='baseline-error', reason=str(exc))
        finally:
            if native is not None:
                native.close()
        results.append(row)
    return results


def exclusive_gpu(uuid):
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    foreign = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) == 2 and fields[0] == uuid and int(fields[1]) != os.getpid():
            foreign.append(int(fields[1]))
    if foreign:
        raise RuntimeError(f'GPU {uuid} has foreign compute processes: {foreign}; stopping without killing')


def same_gpu_uuid(requested, observed):
    def value(identifier):
        return uuid.UUID(str(identifier).removeprefix('GPU-'))
    return value(requested) == value(observed)


def validate_resume(report, identity, output_directory=None, allowed_ids=None):
    if report.get('identity') != identity:
        raise ValueError('resume identity mismatch (manifest/library/hardware/protocol/runtime/source)')
    rows = report.get('rows', [])
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('duplicate resume case ids')
    if report.get('rows_sha256') != digest(rows):
        raise ValueError('resume rows checksum mismatch')
    for index in rows:
        if allowed_ids is not None and index['id'] not in allowed_ids:
            raise ValueError('foreign case id in resume index: ' + index['id'])
        if output_directory is not None:
            expected_path = 'rows/' + index['id'] + '.json'
            if index.get('path') != expected_path:
                raise ValueError('invalid row checkpoint path')
            path = Path(output_directory) / expected_path
            if not path.is_file():
                raise ValueError('missing row checkpoint: ' + expected_path)
            envelope = json.loads(path.read_text())
            row = envelope['row']
            if digest(row) != envelope.get('sha256') or envelope['sha256'] != index.get('sha256'):
                raise ValueError('row checkpoint checksum mismatch: ' + expected_path)
            if row.get('id') != index['id'] or row.get('status') != index['status']:
                raise ValueError('row checkpoint index mismatch')
            for implementation in [row, *row.get('baselines', [])]:
                artifact = implementation.get('error_artifact')
                if artifact is None:
                    continue
                artifact_path = Path(artifact['path'])
                if not artifact_path.is_file() or sha256_file(artifact_path) != artifact['sha256']:
                    raise ValueError('missing or corrupted error artifact: ' + str(artifact_path))
    return {row['id'] for row in rows if row['status'] in ('pass', 'accuracy-review-required', 'unsupported', 'missing-lowering', 'runner-gap', 'validation-unavailable', 'resource-limit')}


def write_report(path, report):
    report['status_summary'] = status_summary(report['rows'])
    report['rows_sha256'] = digest(report['rows'])
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_case_checkpoint(directory, row):
    directory = Path(directory)
    relative = 'rows/' + row['id'] + '.json'
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    checksum = digest(row)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'sha256': checksum, 'row': row}, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    statuses = {}
    for baseline in row.get('baselines', []):
        statuses[baseline['status']] = statuses.get(baseline['status'], 0) + 1
    return {'id': row['id'], 'status': row['status'], 'sha256': checksum, 'path': relative,
            'baseline_status_counts': statuses,
            'baseline_pending_accuracy': statuses.get('accuracy-review-required', 0),
            'baseline_gaps': sum(count for status, count in statuses.items() if status_bucket(status) == 'missing-or-unavailable'),
            'baseline_errors': sum(count for status, count in statuses.items() if status_bucket(status) == 'error')}


def status_bucket(status):
    if status == 'pass':
        return 'pass'
    if status == 'accuracy-review-required':
        return 'pending-accuracy'
    if status == 'not-applicable':
        return 'not-applicable'
    if status in ('correctness-failure', 'runtime-error', 'runner-error', 'baseline-error'):
        return 'error'
    return 'missing-or-unavailable'


def status_summary(indices):
    result = {'cases': {'pass': 0, 'pending-accuracy': 0, 'missing-or-unavailable': 0, 'error': 0, 'not-applicable': 0},
              'baselines': {'pass': 0, 'pending-accuracy': 0, 'missing-or-unavailable': 0, 'error': 0, 'not-applicable': 0},
              'case_status_counts': {}, 'baseline_status_counts': {}}
    for index in indices:
        status = index['status']
        result['cases'][status_bucket(status)] += 1
        result['case_status_counts'][status] = result['case_status_counts'].get(status, 0) + 1
        for baseline_status, count in index.get('baseline_status_counts', {}).items():
            result['baselines'][status_bucket(baseline_status)] += count
            result['baseline_status_counts'][baseline_status] = result['baseline_status_counts'].get(baseline_status, 0) + count
    return result


def completion_status(report):
    if report['remaining_case_ids']:
        return 'tranche-complete'
    counts = status_summary(report['rows'])
    gaps = report.get('invalid_request_count', 0) or any(group['missing-or-unavailable'] or group['error'] for group in (counts['cases'], counts['baselines']))
    pending = any(group['pending-accuracy'] for group in (counts['cases'], counts['baselines']))
    if gaps and pending:
        return 'complete-with-gaps-and-pending-accuracy'
    if gaps:
        return 'complete-with-gaps'
    if pending:
        return 'complete-pending-accuracy'
    return 'complete'


def validate_calibration(manifest, identity):
    """Timing calibration must bind the identical execution environment."""
    if manifest.get('schema') != 'cubutterfly-local-calibration-v1' or manifest.get('status') != 'ready-with-measured-mappings':
        raise ValueError('unsupported or incomplete calibration manifest')
    recorded = manifest.get('measurement_identity', {})
    if recorded.get('library_sha256') != identity['library_sha256']:
        raise ValueError('calibration library identity mismatch')
    if recorded.get('compile_mode') != 'research' or identity.get('compile_mode') != 'research':
        raise ValueError('calibration and campaign require matching research compile mode')
    if manifest.get('device') != identity['gpu_name'] or manifest.get('global_memory_bytes') != identity['gpu_total_memory']:
        raise ValueError('calibration hardware identity mismatch')
    if str(manifest.get('compute_capability')) != identity['compute_capability']:
        raise ValueError('calibration compute capability mismatch')
    if not manifest.get('hardware_rates_remeasured') or recorded.get('verify_batches') != 0:
        raise ValueError('calibration lacks full-batch verification or measured hardware rates')
    return {'scope': manifest.get('scope'), 'inventory_complete': manifest.get('inventory_complete'),
            'service_gaps': manifest.get('service_gaps', []),
            'cost_model_status': manifest.get('cost_model_status'),
            'model_evidence': identity.get('model_evidence'),
            'case_selection': 'each timed case uses public MEASURE; adapter child selection may exclude fixed boundary cost; reported timing includes complete execution',
            'runtime_binding': 'calibration records library/hardware; Python runtime is frozen in campaign identity'}


def bind_calibration_environment(manifest):
    if manifest.get('measurement_identity', {}).get('compile_mode') != 'research':
        raise ValueError('calibration compile mode must be research')
    mode = os.environ.get('CUBUTTERFLY_COMPILE_MODE', 'research')
    if mode != 'research':
        raise ValueError('CUBUTTERFLY_COMPILE_MODE conflicts with calibration research mode')
    registry = Path(manifest['registry_path']).resolve(strict=True)
    os.environ['CUBUTTERFLY_REGISTRY'] = str(registry)
    os.environ['CUBUTTERFLY_COMPILE_MODE'] = 'research'
    profile = Path(manifest['hardware_rates_source']).resolve(strict=True)
    cost_model = Path(manifest['cost_model']).resolve(strict=True)
    model = json.loads(cost_model.read_text())
    evidence = {key: model.get(key) for key in ('status', 'usable_for_unmeasured_ranking',
                'concurrency_calibrated', 'stage_service_status', 'limitations')}
    for key in ('validation', 'whole_plan_validation', 'overlap_validation', 'composition_validation'):
        evidence[key] = {field: value for field, value in model.get(key, {}).items()
                         if value is None or isinstance(value, (str, int, float, bool))}
    return {'registry': str(registry), 'registry_sha256': sha256_file(registry),
            'hardware_profile': str(profile), 'hardware_profile_sha256': sha256_file(profile),
            'cost_model': str(cost_model), 'cost_model_sha256': sha256_file(cost_model),
            'model_evidence': evidence,
            'compile_mode': 'research'}


def validate_fingerprints(manifest, library_path):
    fingerprints = manifest['fingerprints']
    if fingerprints['library']['sha256'] != sha256_file(library_path):
        raise ValueError('manifest library fingerprint mismatch')
    for item in fingerprints['sources']:
        if not Path(item['path']).is_file() or sha256_file(item['path']) != item['sha256']:
            raise ValueError('manifest source fingerprint mismatch: ' + item['path'])
    config = fingerprints.get('config', {})
    if 'path' in config and (not Path(config['path']).is_file() or sha256_file(config['path']) != config['sha256']):
        raise ValueError('manifest config fingerprint mismatch')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-cases', type=int, required=True, help='explicit bounded tranche size')
    parser.add_argument('--population')
    parser.add_argument('--timing', action='store_true')
    parser.add_argument('--calibration-manifest', type=Path)
    parser.add_argument('--vkfft-library', type=Path)
    parser.add_argument('--gpuntt-library', type=Path, help='prepared exact GPU-NTT bridge; defaults to audited adapter path')
    parser.add_argument('--reference-max-elements', type=int, default=1048576)
    parser.add_argument('--ntt-max-operations', type=int, default=16777216,
                        help='exact NTT budget: radix-2 butterflies, or direct DFT terms for arbitrary axes')
    parser.add_argument('--error-directory', type=Path, help='lossless first-validation artifacts; default OUTPUT/errors')
    parser.add_argument('--float-verdict', choices=('legacy-fixed', 'error-report'), default='error-report',
                        help='error-report records errors without a floating pass verdict; integers always exact')
    args = parser.parse_args(argv)
    error_dir = (args.error_directory or args.output / 'errors').resolve()
    if min(args.max_cases, args.reference_max_elements, args.ntt_max_operations) < 1:
        parser.error('tranche and oracle bounds must be positive')
    if args.timing and args.calibration_manifest is None:
        parser.error('--timing requires --calibration-manifest')
    from semantic_campaign_manifest import validate_manifest_hash
    manifest = json.loads(args.manifest.read_text())
    if not validate_manifest_hash(manifest):
        raise ValueError('manifest checksum mismatch')
    validate_fingerprints(manifest, args.library)
    if manifest.get('schema') != 'cubutterfly-semantic-campaign-v1':
        raise ValueError('unsupported campaign schema')
    cases = [case for case in manifest['cases'] if args.population is None or args.population in case['populations']]
    if not cases:
        raise ValueError('no matching cases')
    for case in cases:
        if case['id'] != digest(case['descriptor']['semantics']):
            raise ValueError('case semantic checksum mismatch: ' + case['id'])
    exclusive_gpu(args.gpu_uuid)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    import torch
    import cubutterfly as cb
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    properties = torch.cuda.get_device_properties(0)
    actual_uuid = str(getattr(properties, 'uuid', ''))
    if not same_gpu_uuid(args.gpu_uuid, actual_uuid):
        raise ValueError(f'GPU UUID mismatch: requested {args.gpu_uuid}, CUDA reports {actual_uuid}')
    protocol = dict(PROTOCOL, reference_max_elements=args.reference_max_elements, ntt_max_operations=args.ntt_max_operations,
                    float_verdict=args.float_verdict)
    identity = {'manifest_sha256': digest(manifest), 'library_sha256': sha256_file(args.library),
                'library': str(args.library.resolve()), 'gpu_uuid': args.gpu_uuid,
                'gpu_name': properties.name, 'gpu_total_memory': properties.total_memory,
                'compute_capability': f'{properties.major}.{properties.minor}',
                'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
                'python': sys.version, 'runner_sha256': sha256_file(__file__),
                'runner_revision': RUNNER_REVISION,
                'ntt_reference_sha256': sha256_file(Path(__file__).with_name('semantic_ntt_reference.py')),
                'frontend_sha256': sha256_file(cb.__file__), 'numpy': np.__version__,
                'timing': args.timing, 'population': args.population, 'protocol': protocol,
                'error_directory': str(error_dir),
                'vkfft_sha256': sha256_file(args.vkfft_library) if args.vkfft_library else None}
    limitations = []
    if args.calibration_manifest:
        calibration = json.loads(args.calibration_manifest.read_text())
        identity.update(bind_calibration_environment(calibration))
        limitations = validate_calibration(calibration, identity)
        identity['calibration_sha256'] = digest(calibration)
    else:
        registry = os.environ.get('CUBUTTERFLY_REGISTRY')
        identity.update(compile_mode=os.environ.get('CUBUTTERFLY_COMPILE_MODE'),
                        registry=registry, registry_sha256=sha256_file(registry) if registry else None)
    identity['runtime_environment'] = {key: value for key, value in sorted(os.environ.items())
                                       if key.startswith('CUBUTTERFLY_') or key in ('CUDA_VISIBLE_DEVICES', 'CUDA_MODULE_LOADING')}
    from semantic_dao_baseline import backend_inventory
    identity['dao_baseline'] = {'adapter_sha256': sha256_file(Path(__file__).with_name('semantic_dao_baseline.py')),
                                'backend_identity': backend_inventory()}
    from semantic_gpuntt_baseline import DEFAULT_LIBRARY, backend_inventory as gpuntt_inventory
    identity['gpuntt_baseline'] = {'adapter_sha256': sha256_file(Path(__file__).with_name('semantic_gpuntt_baseline.py')),
                                   'backend_identity': gpuntt_inventory(args.gpuntt_library or DEFAULT_LIBRARY)}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'results.json'
    report = {'schema': 'cubutterfly-semantic-results-v1', 'identity': identity,
              'calibration_limitations': limitations, 'manifest_metrics': manifest.get('metrics'),
              'invalid_requests_source': str(args.manifest.resolve()),
              'invalid_request_count': len(manifest.get('invalid_requests', [])),
              'rows': [], 'status': 'running'}
    completed = set()
    if args.resume:
        report = json.loads(path.read_text())
        completed = validate_resume(report, identity, args.output, {case['id'] for case in cases})
        if any(row['status'] == 'correctness-failure' for row in report['rows']):
            raise ValueError('correctness failure requires investigation before resume')
    elif path.exists():
        raise ValueError('results already exist; use --resume or another output directory')
    pending = [case for case in cases if case['id'] not in completed]
    selected = pending[:args.max_cases]
    report['remaining_case_ids'] = [case['id'] for case in pending]
    report['tranche_limit'] = args.max_cases
    write_report(path, report)
    library = cb.Library(args.library)
    stream = torch.cuda.Stream()
    try:
        with torch.cuda.stream(stream):
            for case in selected:
                exclusive_gpu(args.gpu_uuid)
                row = run_case(case, library, torch, stream, args.timing,
                               args.reference_max_elements, args.ntt_max_operations, args.vkfft_library, args.float_verdict, error_dir, args.gpuntt_library)
                # Never commit a measured pass before the post-run exclusivity
                # guard. A conflicted case remains pending for a clean rerun.
                exclusive_gpu(args.gpu_uuid)
                index = write_case_checkpoint(args.output, row)
                report['rows'] = [old for old in report['rows'] if old['id'] != row['id']] + [index]
                report['remaining_case_ids'].remove(case['id'])
                write_report(path, report)
                if row['status'] in ('correctness-failure', 'runtime-error', 'runner-error'):
                    report['status'] = 'stopped-' + row['status']
                    break
            else:
                report['status'] = completion_status(report)
    except Exception as exc:
        report.update(status='stopped', stop_reason=str(exc))
        write_report(path, report)
        raise
    write_report(path, report)
    print(json.dumps({'output': str(path), 'status': report['status'], 'rows': len(report['rows']),
                      'remaining': len(report['remaining_case_ids'])}))
    return 1 if report['status'].startswith('stopped') else 0


if __name__ == '__main__':
    raise SystemExit(main())
