"""CPU-only, separately labelled finite-range controls for FP16 real FFT.

C2R starts from the frozen generator's spectrum before FP16 quantization.
The bound covers exact intermediate magnitudes, not floating-point roundoff.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

import run_semantic_campaign as runner


PROFILE = 'real-fft-stage-l1-pow2-quarter-max-v2'
HEADROOM_FRACTION = 0.25
STORAGE_MAX = float(np.finfo(np.float16).max)


def _digest(array):
    array = np.ascontiguousarray(array)
    header = json.dumps({'dtype': array.dtype.str, 'shape': list(array.shape)},
                        sort_keys=True, separators=(',', ':')).encode('ascii')
    result = hashlib.sha256(header)
    result.update(array.tobytes())
    return result.hexdigest()


def _validate(s):
    if (s.get('operator'), s.get('storage'), s.get('length_mode')) != (0, 0, 0):
        raise ValueError('requires exact-length FP16 real FFT')
    if s.get('fft_type') not in (1, 2) or s.get('compute') not in (0, 1):
        raise ValueError('requires R2C/C2R with native or FP32 computation')
    if s.get('inverse') != int(s['fft_type'] == 2):
        raise ValueError('real FFT direction must match FFT type')
    if type(s.get('normalize_inverse')) is not bool:
        raise ValueError('normalization must be an explicit boolean')
    shape = s.get('shape')
    if (not isinstance(shape, (list, tuple)) or not 1 <= len(shape) <= 8 or
            any(type(n) is not int or n < 2 or n & (n - 1) for n in shape)):
        raise ValueError('requires one to eight power-of-two axes of length at least two')
    if type(s.get('batch')) is not int or s['batch'] < 1:
        raise ValueError('requires a positive integer batch')
    scale = s.get('fft_output_scale')
    if type(scale) not in (int, float) or not math.isfinite(scale) or abs(scale) > 1:
        raise ValueError('requires finite absolute output scale at most one')
    for side in ('input', 'output'):
        orders, strides = s.get(side + '_orders'), s.get(side + '_strides')
        if (not isinstance(orders, (list, tuple)) or len(orders) != len(shape) or
                any(type(v) is not int or v not in (0, 1) for v in orders)):
            raise ValueError('requires canonical per-axis orders')
        if (not isinstance(strides, (list, tuple)) or len(strides) != len(shape) or
                any(type(v) is not int or v <= 0 for v in strides) or
                type(s.get(side + '_batch_stride')) is not int or s[side + '_batch_stride'] <= 0):
            raise ValueError('requires positive per-axis and batch strides')
    if s.get('placement') not in (0, 1):
        raise ValueError('requires canonical placement')
    if s.get('ntt_input_layout', 0) != 0 or s.get('ntt_output_layout', 0) != 0:
        raise ValueError('NTT-specific layout is not a real FFT control')


def input_l1_bounds(source, *, full_last_extent=None):
    """Per-transform L1 bound, reconstructing omitted Hermitian frequencies.

    In xD packed storage only the last axis is compact. Its DC and Nyquist
    planes have weight one, and every interior last-axis plane has weight two.
    """
    source = np.asarray(source)
    if source.ndim < 2 or not np.isfinite(source).all():
        raise ValueError('requires finite batched logical input')
    components = np.abs(source.real.astype(np.float64))
    if np.iscomplexobj(source):
        components += np.abs(source.imag.astype(np.float64))
    if full_last_extent is not None:
        n = full_last_extent
        if type(n) is not int or n < 2 or n & (n - 1) or source.shape[-1] != n // 2 + 1:
            raise ValueError('packed input does not match full last-axis extent')
        components = components.copy()
        components[..., 1:-1] *= 2
    return [math.nextafter(value, math.inf) if value else 0.0
            for value in (math.fsum(float(x) for x in batch.flat) for batch in components)]


def prepare_input(s):
    """Return a final quantized logical input and explicit v2 provenance."""
    _validate(s)
    c2r = s['fft_type'] == 2
    if c2r:
        shape = (s['batch'], *s['shape'])
        index = np.arange(math.prod(shape)).reshape(shape)
        # Match make_source's construction and complex64 conversion exactly.
        source = np.fft.rfftn(np.sin(index * .17),
                             axes=tuple(range(1, len(shape)))).astype(np.complex64)
    else:
        source = runner.make_source(s)
    if not np.isfinite(source).all():
        raise ValueError('prequantization source contains non-finite data')
    storage = runner.side_storage(s, 'input')
    with np.errstate(over='ignore', invalid='ignore'):
        original_encoded = runner.encode_storage(source, storage)
        original = runner.decode_storage(original_encoded, storage)
    kwargs = {'full_last_extent': s['shape'][-1]} if c2r else {}
    bounds = input_l1_bounds(source, **kwargs)
    target, exponent = HEADROOM_FRACTION * STORAGE_MAX, 0
    while math.ldexp(max(bounds), exponent) > target:
        exponent -= 1
    while True:
        scaled = np.empty(source.shape, dtype=np.complex64 if c2r else np.float32)
        scaled.real = np.ldexp(source.real.astype(np.float32), exponent)
        if c2r:
            scaled.imag = np.ldexp(source.imag, exponent)
        encoded = runner.encode_storage(scaled, storage)
        final = runner.decode_storage(encoded, storage).copy()
        if not np.isfinite(final).all():
            raise ValueError('scaled quantized source contains non-finite data')
        final_bounds = input_l1_bounds(final, **kwargs)
        if max(final_bounds) <= target:
            break
        exponent -= 1
    metadata = {
        'schema': 'cubutterfly-semantic-input-profile-v2', 'profile': PROFILE,
        'profile_version': 2, 'fft_type': s['fft_type'], 'storage': storage,
        'scale_exponent': exponent, 'exponent': exponent,
        'factor': math.ldexp(1.0, exponent), 'input_sha256': _digest(encoded),
        'input_hash_encoding': 'sha256(dtype/shape JSON header + contiguous storage bytes)',
        'original_input_sha256': _digest(original_encoded),
        'original_input_nonfinite_elements': int(np.count_nonzero(~np.isfinite(original))),
        'original_input_nonfinite_components': int(np.count_nonzero(~np.isfinite(original.real))) +
            (int(np.count_nonzero(~np.isfinite(original.imag))) if c2r else 0),
        'quantization_new_zero_components': int(np.count_nonzero((scaled.real != 0) & (final.real == 0))) +
            (int(np.count_nonzero((scaled.imag != 0) & (final.imag == 0))) if c2r else 0),
        'prequantization_input_sha256': _digest(source),
        'original_batch_l1_bounds': bounds, 'original_max_l1_bound': max(bounds),
        'original_bound_input': 'finite prequantization spectrum' if c2r else
            'original quantized real input',
        'quantized_batch_l1_bounds': final_bounds, 'quantized_max_l1_bound': max(final_bounds),
        'bound_target': target, 'storage_max': STORAGE_MAX,
        'headroom_fraction': HEADROOM_FRACTION,
        'selection_basis': 'complete input only; real/imaginary component L1 sum',
        'packed_weights': 'last-axis DC/Nyquist=1, interior=2' if c2r else 'real input=1',
        'bound_role': 'exact FFT intermediate magnitude upper bound; excludes arithmetic roundoff',
        'reference_input': 'final quantized input', 'semantics_unchanged': True,
        'source_construction': 'frozen generator before FP16 spectrum quantization' if c2r else
            'frozen generator final quantized real input',
        'elements': int(final.size),
    }
    return final, metadata


def classify_numeric_failure(source, expected, actual, storage):
    """Extend the frozen range report with input/reference provenance priority."""
    source, expected = np.asarray(source), np.asarray(expected)
    report = runner.floating_range_report(actual, expected, storage)
    report['input_nonfinite_elements'] = int(np.count_nonzero(~np.isfinite(source)))
    report['reference_nonfinite_elements'] = int(np.count_nonzero(~np.isfinite(expected)))
    if report['input_nonfinite_elements']:
        report['classification'] = 'input-nonfinite'
    elif report['reference_nonfinite_elements']:
        report['classification'] = 'reference-nonfinite'
    return report
