"""Project public mappings onto the existing standalone-module cache contract.

The fields intentionally match src/jit_module.cpp, including omitted defaults.
This is compilation preparation, not a second candidate generator or a device
legality check. Runtime-only semantics and scheduling remain on the public point.
"""
from __future__ import annotations

import json

try:
    import resident_mapping
except ImportError:  # pragma: no cover - package import fallback
    from . import resident_mapping


def _positive(mapping, field):
    value = int(mapping.get(field, 0))
    if value <= 0:
        raise ValueError(f"module preparation requires resolved {field}")
    return value


def _partition(mapping, field, log_n):
    value = mapping.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"module preparation requires resolved {field}")
    stages = [int(stage) for stage in value]
    if min(stages) <= 0 or sum(stages) != log_n:
        raise ValueError(f"invalid {field}")
    return stages


def _shared_partition(mapping, operator, log_n):
    if not mapping.get("stage_partition"):
        local = (int(mapping.get("flow_tile_log_n", 0)) or min(10, log_n)) if operator == "ntt" else _positive(mapping, "local_stages")
        mapping = dict(mapping, stage_partition=[min(local, log_n - first)
                                                 for first in range(0, log_n, local)])
    stages = _partition(mapping, "stage_partition", log_n)
    # Butterfly shared specialization compiles physical groups after explicit
    # resident edges have been folded, rather than the logical partition.
    boundaries = mapping.get("boundaries") or []
    if operator == "ntt" or not boundaries:
        return stages
    if len(boundaries) != len(stages) - 1:
        raise ValueError("shared boundary count differs from stage partition")
    physical, count = [], 0
    for index, stage in enumerate(stages):
        count += stage
        if index == len(stages) - 1 or boundaries[index].get("residency") != "fused":
            physical.append(count)
            count = 0
    return physical


def _factor_io_policies(mapping, factors):
    """Validate and preserve the per-physical-factor I/O policy axis."""
    raw = mapping.get("factor_io_policies", [])
    if raw in (None, ""):
        raw = []
    if not isinstance(raw, list):
        raise ValueError("factor_io_policies must be a list")
    if raw and len(raw) != len(factors):
        raise ValueError("factor_io_policies must match factor_partition")
    if any(value not in {"dynamic", "static-unrolled"} for value in raw):
        raise ValueError("factor_io_policies entries must be dynamic or static-unrolled")
    # [] is the canonical public default.  Preserve a non-empty list exactly;
    # the compiler, not runtime launch code, owns the selected lowering.
    return list(raw)


def _register_prefix_axes(mapping, precision):
    """Validate prefix-only register-tile implementation axes.

    ``prefix_codelet_lanes`` is deliberately validated here, before any
    backend-specific request is selected.  A non-default value must never be
    silently ignored by a lowering that does not consume cooperative prefix
    lanes.
    """
    backend = mapping.get("backend")
    core = mapping.get("fft_core")
    codelet = mapping.get("prefix_codelet", "native")
    layout = mapping.get("prefix_shared_layout", "linear")
    lanes = mapping.get("prefix_codelet_lanes", 1)
    if codelet not in {"native", "cufftdx-thread"}:
        raise ValueError("prefix_codelet must be native or cufftdx-thread")
    if layout not in {"linear", "xor"}:
        raise ValueError("prefix_shared_layout must be linear or xor")
    if isinstance(lanes, bool) or not isinstance(lanes, int):
        raise ValueError("prefix_codelet_lanes must be an integer")
    if lanes <= 0 or lanes & (lanes - 1):
        raise ValueError("prefix_codelet_lanes must be a positive power of two")
    register_prefix = backend == "online-reorder" and core == "register-tile"
    if lanes != 1 and not register_prefix:
        raise ValueError("prefix_codelet_lanes are valid only for online-reorder register-tile")
    if (codelet != "native" or layout != "linear") and not (
            register_prefix):
        raise ValueError("prefix codelet/layout are valid only for online-reorder register-tile")
    if codelet == "cufftdx-thread" and precision == "fp64":
        raise ValueError("cufftdx-thread prefix is unavailable for FP64")
    return codelet, layout, lanes


def _validate_prefix_geometry(mapping, lanes, local_stages, log_n=None):
    """Validate the cooperative prefix geometry consumed by the compiler."""
    if isinstance(local_stages, bool) or not isinstance(local_stages, int) or local_stages <= 0:
        raise ValueError("resolved local_stages must be a positive integer")
    if isinstance(mapping.get("prefix_threads"), bool) or not isinstance(mapping.get("prefix_threads"), int):
        raise ValueError("resolved prefix_threads must be an integer")
    threads = mapping["prefix_threads"]
    if threads <= 0 or threads > 1024:
        raise ValueError("module preparation requires resolved prefix_threads")
    radius = 1 << (local_stages // 2)
    if lanes > radius:
        raise ValueError("prefix_codelet_lanes must not exceed local-stage register radius")
    divisor = radius * lanes
    if threads % divisor:
        raise ValueError("prefix_threads must produce integral cooperative prefix Columns")
    columns = threads // divisor
    if columns <= 0 or columns & (columns - 1):
        raise ValueError("cooperative prefix Columns must be a positive power of two")
    if log_n is not None and columns > (1 << (log_n - local_stages)):
        raise ValueError("cooperative prefix Columns exceed the suffix workload extent")
    if lanes > 1 and lanes * columns > 32:
        raise ValueError("cooperative prefix lane group must fit within one warp")
    expected_ept = radius // lanes
    if isinstance(mapping.get("prefix_ept"), bool) or not isinstance(mapping.get("prefix_ept"), int):
        raise ValueError("resolved prefix_ept must be an integer")
    if mapping["prefix_ept"] != expected_ept:
        raise ValueError("prefix_ept must equal register radius divided by prefix_codelet_lanes")
    return radius, columns, expected_ept


def _resident_request(mapping, stages, *, backend, operator):
    parts, chunks = resident_mapping.validate_resident_axes(
        mapping, stages, backend=backend, operator=operator)
    request = {}
    # Omitted and empty outer axes are the historical lowering.  Preserve an
    # explicitly sized all-empty list because it is useful for replay audits.
    if "local_stage_partitions" in mapping and parts:
        request["local_stage_partitions"] = parts
    if "exchange_chunks" in mapping and chunks:
        request["exchange_chunks"] = chunks
    return request


def compile_request(point, policy="research"):
    """Return one canonical compiler request, or None for a linked lowering.

Only the existing JIT families are projected. Missing runtime-resolved axes
raise explicitly; guessing them would populate cache entries runtime never uses.
"""
    if policy not in {"research", "auto", "on-demand", "precompiled"}:
        raise ValueError("unknown compile policy")
    mapping = point.get("mapping_json", {})
    mapping = json.loads(mapping) if isinstance(mapping, str) else mapping
    if not isinstance(mapping, dict):
        raise ValueError("mapping_json must contain an object")
    mapping = {**point, **mapping}
    backend = mapping.get("backend")
    operator = point.get("operator", "fft")
    precision = point.get("precision", "fp32")
    log_n = int(point["logN"])
    if policy == "precompiled":
        if resident_mapping.resident_requested(mapping):
            raise ValueError("resident axes require on-demand compilation")
        return None
    # Validate even when this backend has no prefix compiler path; a
    # non-default axis must never be silently ignored by another lowering.
    _, _, lanes = _register_prefix_axes(mapping, precision)
    raw_policies = mapping.get("factor_io_policies", [])
    if raw_policies not in (None, "", []) and backend != "factor-streamed":
        raise ValueError("factor_io_policies are valid only for factor-streamed")
    if backend == "shared-iterative":
        stages = _shared_partition(mapping, operator, log_n)
        axes = _resident_request(mapping, stages, backend=backend, operator=operator)
        if policy != "research" and not resident_mapping.resident_requested(mapping):
            return None
        request = dict(backend=backend, operator=operator, precision=precision,
                       logN=log_n, stage_partition=stages)
        request.update(axes)
        if operator == "ntt":
            # Output layout belongs to the workload, never a donor mapping.
            order = point.get("output_order", point.get("placement", "natural"))
            if order in ("out-of-place", "in-place"):
                order = "natural"
            if order not in ("natural", "bit-reversed"):
                raise ValueError("shared NTT module requires natural or bit-reversed output")
            request.update(threads=int(mapping.get("threads_per_block", 0)) or 128,
                           writer_aligned=mapping.get("dataflow_layout", "hermes-xor") == "hermes-xor",
                           output_order=order)
        else:
            request.update(threads=_positive(mapping, "tile_threads"),
                           writer_aligned=mapping.get("shared_layout", "linear") == "writer-aligned",
                           accumulation=point.get("accumulation", "native"),
                           complex_multiply=mapping.get("complex_multiply", "four-mul"), batch_tile=1)
        return request
    if operator != "fft":
        return None
    if backend == "factor-streamed":
        if resident_mapping.resident_requested(mapping):
            raise ValueError("resident axes are unsupported for factor-streamed")
        mapping = dict(mapping)
        mapping["stage_partition"] = mapping.get("stage_partition") or [log_n]
        mapping["factor_partition"] = mapping.get("factor_partition") or mapping["stage_partition"]
        request_factors = _partition(mapping, "factor_partition", log_n)
        policies = _factor_io_policies(mapping, request_factors)
        request = dict(backend=backend, precision=precision, logN=log_n,
                       stage_partition=_partition(mapping, "stage_partition", log_n),
                       factor_partition=request_factors,
                       factor_ept=_positive(mapping, "factor_ept"),
                       factor_columns=_positive(mapping, "factor_columns"),
                       data_tiles_per_cta=_positive(mapping, "data_tiles_per_cta"),
                       prefetch_depth=int(mapping.get("prefetch_depth", 0)))
        if policies:
            request["factor_io_policies"] = policies
        if mapping.get("fft_core") == "register-tile":
            request["fft_core"] = "register-tile"
        elif mapping.get("fft_core") != "cufftdx-block":
            raise ValueError("factor-streamed module requires a resolved FFT core")
        slices = _positive({"factor_slices": mapping.get("factor_slices", 1)}, "factor_slices")
        if slices > 1:
            request["factor_slices"] = slices
        return request
    if backend == "online-reorder" and mapping.get("fft_core") == "register-tile":
        codelet, layout, lanes = _register_prefix_axes(mapping, precision)
        if mapping.get("stage_partition"):
            stages = _partition(mapping, "stage_partition", log_n)
            if len(stages) != 2:
                raise ValueError("register prefix/suffix requires two stage groups")
            mapping = dict(mapping, local_stages=stages[0])
        if mapping.get("prefix_threads") in (None, "", 0):
            raise ValueError("resolved prefix_threads must be an integer")
        stages = [int(mapping["local_stages"]), log_n - int(mapping["local_stages"])]
        geometry = resident_mapping.register_geometry(mapping, stages)
        request = dict(fft_core="register-tile", precision=precision, logN=log_n,
                       **{field: _positive(mapping, field) for field in
                          ("local_stages", "prefix_threads", "prefix_ept", "suffix_threads", "suffix_ept")})
        request.update(_resident_request(mapping, stages, backend=backend, operator=operator))
        if codelet != "native":
            request["prefix_codelet"] = codelet
        if layout != "linear":
            request["prefix_shared_layout"] = layout
        if lanes != 1:
            request["prefix_codelet_lanes"] = lanes
        return request
    # Non-register lowerings may carry only the default spellings.  Reject a
    # non-default request instead of silently compiling the wrong implementation.
    if (backend != "online-reorder" or mapping.get("fft_core") != "register-tile") and (
            mapping.get("prefix_codelet", "native") != "native" or
            mapping.get("prefix_shared_layout", "linear") != "linear"):
        raise ValueError("prefix codelet/layout are valid only for online-reorder register-tile")
    if resident_mapping.resident_requested(mapping):
        raise ValueError("resident axes are unsupported for this lowering")
    return None
