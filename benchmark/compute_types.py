"""FFT reporting classes retain the complete input and arithmetic contract."""

FFT_TYPES = {"fft-h", "fft-l"}
FFT_PRECISIONS = {"fp16", "bf16", "fp32", "fp64", "fp16-fp32"}


def fft_semantics(precision, *, storage_precision=None, compute_precision=None, accumulation=None):
    if precision not in FFT_PRECISIONS:
        raise ValueError(f"FFT precision {precision!r} is unsupported; expected {sorted(FFT_PRECISIONS)}")
    mixed = precision == "fp16-fp32"
    storage = "fp32" if mixed else precision
    accum = accumulation or "native"
    if accum not in {"native", "fp32"}:
        raise ValueError(f"unsupported FFT accumulation {accum!r}")
    if not mixed and precision in {"fp32", "fp64"} and accum != "native":
        raise ValueError("explicit fp32 accumulation requires fp16/bf16 storage")
    if mixed and accum != "native":
        raise ValueError("fp16-fp32 uses native policy with intrinsic fp32 accumulation")
    compute = "fp16" if mixed else ("fp32" if accum == "fp32" else precision)
    if storage_precision is not None and storage_precision != storage:
        raise ValueError(f"precision {precision!r} requires storage_precision={storage!r}")
    if compute_precision is not None and compute_precision != compute:
        raise ValueError(f"precision/accumulation requires compute_precision={compute!r}; got {compute_precision!r}")
    return {"storage_precision": storage, "compute_precision": compute,
            "accumulation": accum, "accumulation_precision": "fp32" if mixed else compute,
            "compute_type": "fft-h" if storage in {"fp32", "fp64"} and compute in {"fp32", "fp64"} else "fft-l"}
