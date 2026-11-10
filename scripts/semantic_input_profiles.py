"""CPU-only, input-derived profiles for complete quantized FFT/FWHT sources.

The scaled profiles bound exact FFT/FWHT intermediate magnitudes, not floating-
point roundoff or every implementation's internal arithmetic. No outputs are consulted.
The returned source is the input to both library execution and CPU reference.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np


RAW_PROFILE = "raw-stress"
SCALED_PROFILE = "fft-stage-l1-pow2-quarter-max"
FWHT_SCALED_PROFILE = "fwht-stage-l1-pow2-quarter-max"
HEADROOM_FRACTION = 0.25
STORAGE_MAX = float(np.finfo(np.float16).max)


def _digest(value):
    value = np.ascontiguousarray(value)
    header = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256(header)
    digest.update(value.tobytes())
    return digest.hexdigest()


def _batch_bounds(source):
    # fsum avoids losing small terms in a long transform. Round upward once
    # so the reported bound does not underestimate a rounded exact sum.
    result = []
    for batch in source:
        components = _components(batch).astype(np.float64)
        value = math.fsum(abs(float(x)) for x in components.flat)
        result.append(math.nextafter(value, math.inf) if value else 0.0)
    return result


def _components(source):
    return (np.stack((source.real, source.imag), axis=-1)
            if source.dtype.kind == "c" else source)


def scaled_profile(s):
    """Select a separately labelled control for each supported operator."""
    if s.get("operator") == 0:
        return SCALED_PROFILE
    if s.get("operator") == 2:
        return FWHT_SCALED_PROFILE
    raise ValueError("input profiles support only C2C FFT or standard FWHT")


def _validate(s, source, encode_storage, decode_storage):
    profile = scaled_profile(s)
    is_fft = profile == SCALED_PROFILE
    if s.get("fft_type") != 0:
        raise ValueError("input profiles require C2C FFT or standard FWHT (fft_type=0)")
    storage = 4 if is_fft else 0
    if s.get("storage") != storage:
        raise ValueError("input profiles require FP16 storage=" + str(storage))
    if s.get("length_mode") != 0:
        raise ValueError("input profiles require exact length (length_mode=0)")
    shape = s.get("shape")
    if (not isinstance(shape, (list, tuple)) or not shape or
            any(type(n) is not int or n < 1 or n & (n - 1) for n in shape)):
        raise ValueError("input profiles require positive power-of-two extents")
    batch = s.get("batch")
    if type(batch) is not int or batch < 1:
        raise ValueError("input profiles require a positive integer batch")
    scale = s.get("fft_output_scale")
    if type(scale) not in (int, float) or not math.isfinite(scale) or abs(scale) > 1:
        raise ValueError("input profiles require finite abs(fft_output_scale) <= 1")
    if not is_fft and scale != 1:
        raise ValueError("FWHT input controls require fft_output_scale=1; frozen reference/baseline scale contract gap")
    if s.get("inverse") not in (0, 1) or type(s.get("normalize_inverse")) is not bool:
        raise ValueError("input profiles require canonical direction and normalization")
    source = np.asarray(source)
    dtype = np.dtype("complex64" if is_fft else "float16")
    if source.dtype != dtype or source.shape != (batch, *shape):
        raise ValueError("source must be the complete " + dtype.name + " decoded FP16 input")
    if not np.isfinite(source).all():
        raise ValueError("source contains non-finite data")
    encoded = encode_storage(source, s["storage"])
    decoded = decode_storage(encoded, s["storage"])
    if source.tobytes() != decoded.tobytes():
        raise ValueError("source must already be quantized to requested storage")
    return source, encoded


def apply_input_profile(s, source, profile=RAW_PROFILE, *, encode_storage,
                        decode_storage):
    """Return ``(final_source, metadata)`` without mutating source or semantics.

    Inject the frozen runner's encode_storage/decode_storage functions. The
    largest nonpositive power-of-two exponent satisfying the input L1 bound is
    selected once, before requantization. Quarter-max is a fixed bound target,
    leaving three quarters of finite FP16 storage range unused by that bound.
    Use final_source, rather than source * factor, for the reference as well.
    """
    if profile not in (RAW_PROFILE, SCALED_PROFILE, FWHT_SCALED_PROFILE):
        raise ValueError("unknown input profile: " + str(profile))
    source, original_storage = _validate(s, source, encode_storage, decode_storage)
    if profile != RAW_PROFILE and profile != scaled_profile(s):
        raise ValueError("input profile does not match operator")
    bounds = _batch_bounds(source)
    bound = max(bounds)
    target = HEADROOM_FRACTION * STORAGE_MAX
    exponent = 0
    if profile != RAW_PROFILE:
        # An input-only decision; no output measurement or candidate scan.
        while math.ldexp(bound, exponent) > target:
            exponent -= 1
    factor = math.ldexp(1.0, exponent)
    if profile == RAW_PROFILE:
        final = source.copy()
        final_storage = original_storage.copy()
    else:
        scaled = np.empty_like(source)
        if source.dtype.kind == "c":
            scaled.real = np.ldexp(source.real, exponent)
            scaled.imag = np.ldexp(source.imag, exponent)
        else:
            scaled[:] = np.ldexp(source, exponent)
        final_storage = encode_storage(scaled, s["storage"])
        final = decode_storage(final_storage, s["storage"]).copy()
    final_bounds = _batch_bounds(final)
    if not np.isfinite(final).all():
        raise ValueError("requantized profile contains non-finite data")
    if profile != RAW_PROFILE and max(final_bounds) > target:
        raise ValueError("requantized input exceeds the fixed L1 bound target")
    original_components = _components(source)
    final_components = _components(final)
    metadata = {
        "schema": "cubutterfly-semantic-input-profile-v1",
        "profile": profile,
        "storage": s["storage"],
        "exponent": exponent,
        "factor": factor,
        "selection_basis": "complete input only; max per-transform sum(abs(real)+abs(imag))",
        "bound_role": "exact FFT stage magnitude upper bound; excludes arithmetic roundoff",
        "storage_max": STORAGE_MAX,
        "headroom_fraction": HEADROOM_FRACTION,
        "bound_target": target,
        "original_batch_l1_bounds": bounds,
        "original_max_l1_bound": bound,
        "scaled_prequantization_max_l1_bound": math.ldexp(bound, exponent),
        "quantized_batch_l1_bounds": final_bounds,
        "quantized_max_l1_bound": max(final_bounds),
        "original_input_sha256": _digest(original_storage),
        "input_sha256": _digest(final_storage),
        "input_hash_encoding": "sha256(dtype/shape JSON header + contiguous storage bytes)",
        "elements": int(source.size),
        "quantized_zero_elements": int(np.count_nonzero(final == 0)),
        "quantized_zero_components": int(np.count_nonzero(final_components == 0)),
        "newly_zero_elements": int(np.count_nonzero((source != 0) & (final == 0))),
        "newly_zero_components": int(np.count_nonzero((original_components != 0) & (final_components == 0))),
        "reference_input": "final quantized input",
    }
    if s["operator"] == 2:
        metadata.update(
            selection_basis="complete input only; max per-transform sum(abs(input))",
            bound_role="exact FWHT add/subtract stage magnitude upper bound; excludes arithmetic roundoff")
    return final, metadata
