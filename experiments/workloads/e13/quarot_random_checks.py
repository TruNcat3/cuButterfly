"""CPU-testable random-state and numerical checks for the QuaRot runner."""
from __future__ import annotations

import hashlib
import math
import numpy as np
import torch


def random_quantized_weight(out_features, in_features, generator, std=0.02):
    if in_features % 2:
        raise ValueError("packed int4 requires an even input dimension")
    weight = torch.randn(out_features, in_features, generator=generator) * std
    scales = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) / 7
    quantized = (weight / scales).round().clamp(-7, 7).to(torch.int8)
    nibble = (quantized.to(torch.int16) & 15).to(torch.uint8)
    packed = nibble[:, 0::2] | (nibble[:, 1::2] << 4)
    return packed, scales


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checked_positions(position_ids, query_length, kv_length):
    if position_ids is None:
        raise RuntimeError("RoPE requires explicit attention position_ids")
    if position_ids.ndim != 2 or position_ids.shape[-1] != query_length:
        raise RuntimeError("RoPE position_ids shape does not match query length")
    expected = torch.arange(kv_length - query_length, kv_length,
                            device=position_ids.device).expand_as(position_ids)
    if position_ids.is_cuda and torch.cuda.is_current_stream_capturing():
        torch._assert_async(torch.all(position_ids == expected),
                            "RoPE positions disagree with the unpadded cache length")
    elif not torch.equal(position_ids, expected):
        raise RuntimeError("RoPE positions disagree with the unpadded cache length")
    return position_ids


def checked_attention_positions(position_ids, query_length, cache_length, layer_idx):
    kv_length = cache_length + (query_length if layer_idx == 0 else 0)
    return checked_positions(position_ids, query_length, kv_length)


def state_summary(model):
    scales = [module.weight_scales.detach().float().cpu()
              for module in model.modules() if hasattr(module, "weight_scales")]
    values = torch.cat([value.flatten() for value in scales]) if scales else torch.empty(0)
    return {"state_tensor_count": len(model.state_dict()),
            "buffer_count": len(list(model.buffers())),
            "quantized_module_count": len(scales),
            "scale_count": values.numel(),
            "nonzero_scale_count": int(torch.count_nonzero(values)),
            "positive_finite_scale_count": int(((values > 0) & torch.isfinite(values)).sum()),
            "scale_min": float(values.min()) if values.numel() else None,
            "scale_max": float(values.max()) if values.numel() else None}


def compare_outputs(reference, actual, atol=None, rtol=None):
    if reference.shape != actual.shape:
        raise RuntimeError("output shapes differ")
    reference_nonfinite_count = int((~torch.isfinite(reference)).sum())
    actual_nonfinite_count = int((~torch.isfinite(actual)).sum())
    finite = reference_nonfinite_count == actual_nonfinite_count == 0
    result = {"finite": finite, "atol": atol, "rtol": rtol, "pass": False,
              "reference_nonfinite_count": reference_nonfinite_count,
              "actual_nonfinite_count": actual_nonfinite_count}
    if finite:
        metric_dtype = (torch.float64 if torch.float64 in (reference.dtype, actual.dtype)
                        else torch.float32)
        reference_metric = reference.to(metric_dtype)
        difference = (actual.to(metric_dtype) - reference_metric).abs()
        result.update(max_abs_error=float(difference.max()),
                      rms_error=float(difference.square().mean().sqrt()),
                      reference_max_abs=float(reference_metric.abs().max()))
        # NumPy partitions the complete array without torch.quantile's
        # 2**24-element limit; linear interpolation is exact, not sampled.
        difference_array = difference.detach().cpu().numpy()
        try:
            quantiles = np.quantile(difference_array, [0.5, 0.9, 0.99],
                                    method="linear")
        except TypeError:  # NumPy < 1.22 calls this argument ``interpolation``.
            quantiles = np.quantile(difference_array, [0.5, 0.9, 0.99],
                                    interpolation="linear")
        result["abs_error_quantiles"] = dict(zip(
            ("p50", "p90", "p99"), quantiles.tolist()))
        result["abs_error_quantiles_method"] = "numpy.quantile/full_array/linear"
        result["max_relative_error"] = float((difference / reference_metric.abs().clamp_min(1e-8)).max())
        result["relative_error_denominator_floor"] = 1e-8
        reference_l2 = float(torch.linalg.vector_norm(reference_metric, dtype=torch.float64))
        error_l2 = float(torch.linalg.vector_norm(difference, dtype=torch.float64))
        result.update(reference_l2_norm=reference_l2, error_l2_norm=error_l2,
                      relative_l2_error=(error_l2 / reference_l2 if reference_l2 > 0
                                         else 0.0 if error_l2 == 0 else None))
        if atol is not None and rtol is not None:
            result["pass"] = bool(torch.allclose(reference_metric, actual.to(metric_dtype), atol=atol, rtol=rtol))
    result["tolerance_assessed"] = atol is not None and rtol is not None
    return result
def cpu_hadamard_reference(values, hadK, K):
    """Float64 mathematical oracle, including QuaRot's unmodified K factor."""
    original_shape = values.shape
    n = original_shape[-1]
    if K <= 0 or n % K or (n // K) & (n // K - 1):
        raise ValueError("K must divide the size into power-of-two vectors")
    work = values.detach().cpu().to(torch.float64).reshape(-1, K, n // K)
    width = 1
    while width < n // K:
        blocks = work.reshape(-1, K, (n // K) // (2 * width), 2, width)
        left, right = blocks[..., 0, :], blocks[..., 1, :]
        work = torch.stack((left + right, left - right), dim=-2).reshape_as(work)
        width *= 2
    work /= math.sqrt(n)
    if K != 1:
        if hadK is None or tuple(hadK.shape) != (K, K):
            raise ValueError("missing or invalid K-factor matrix")
        work = hadK.detach().cpu().to(torch.float64) @ work
    return work.reshape(original_shape)
