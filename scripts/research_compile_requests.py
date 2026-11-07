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
    # Shared specialization compiles physical groups after explicit resident
    # edges have been folded, rather than the logical partition.
    boundaries = mapping.get("boundary_mappings" if operator == "ntt" else "boundaries") or []
    if not boundaries:
        return stages
    if len(boundaries) != len(stages) - 1:
        raise ValueError("shared boundary count differs from stage partition")
    physical, count = [], 0
    if operator == "ntt":
        for boundary in boundaries:
            if (boundary.get("storage", "full-scratch") not in {"full-scratch", "resident-fused"}
                    or boundary.get("buffers", 1) != 1):
                raise ValueError("shared NTT boundaries require full-scratch/resident-fused storage with one buffer")
    for index, stage in enumerate(stages):
        count += stage
        fused = (index < len(boundaries) and
                 (boundaries[index].get("storage") == "resident-fused" if operator == "ntt"
                  else boundaries[index].get("residency") == "fused"))
        if not fused:
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
    if codelet not in {"native", "radix4", "cufftdx-thread"}:
        raise ValueError("prefix_codelet must be native, radix4 or cufftdx-thread")
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


def _radix4_uses_linked_inventory(mapping, operator, precision, log_n):
    """Validate the runtime mapping, then match the finite linked inventory."""
    if operator != "fft" or precision not in {"fp32", "fp64"}:
        raise ValueError("radix4 register prefix requires FP32/FP64 FFT")
    stages = (_partition(mapping, "stage_partition", log_n) if mapping.get("stage_partition")
              else [int(mapping.get("local_stages", 0)), log_n - int(mapping.get("local_stages", 0))])
    if len(stages) != 2:
        raise ValueError("register prefix/suffix requires two stage groups")
    mapping = dict(mapping, local_stages=stages[0], logN=log_n)
    geometry = resident_mapping.register_geometry(mapping, stages)
    if mapping.get("factor_io_policies"):
        raise ValueError("factor_io_policies are valid only for factor-streamed")
    if mapping.get("stage_overlap") and int(mapping.get("batch_tile_count", 1)) <= 0:
        raise ValueError("register-tile overlap requires a positive batch tile")
    expected = {"shared_layout": "writer-aligned",
                "cross_twiddle": "recurrence", "local_exchange": "shared",
                "direct_boundary": "direct-strided", "complex_multiply": "four-mul"}
    if any(mapping.get(key, value) != value for key, value in expected.items()):
        raise ValueError("radix4 register prefix requires writer-aligned placement, recurrence and shared exchange")
    if mapping.get("compute_unit", "auto") not in ("auto", "radix2"):
        raise ValueError("radix4 register prefix requires auto or radix2 compute unit")
    boundaries = mapping.get("boundaries")
    if boundaries and (len(boundaries) != 1 or boundaries[0].get("residency", "global-scratch") != "global-scratch"):
        raise ValueError("radix4 register prefix requires one global-scratch boundary")
    if geometry["log_a"] != geometry["log_b"] and mapping.get("prefix_shared_layout", "linear") == "xor":
        width = 16 if precision == "fp64" else 8
        bank_rows = min(geometry["b_size"], 128 // (width * geometry["prefix_columns"]))
        if geometry["prefix_lanes"] > bank_rows or geometry["second_lanes"] > bank_rows:
            raise ValueError("rectangular XOR lane digits exceed the shared transaction rows")
    if (precision != "fp32" or stages[0] != 8 or stages[1] not in (10, 12) or
            mapping.get("prefix_shared_layout", "linear") != "linear" or
            mapping.get("prefix_twiddle", "runtime") != "runtime" or
            mapping.get("prefix_codelet_lanes", 1) != 1 or
            mapping.get("suffix_core", "cufftdx-block") != "cufftdx-block" or
            resident_mapping.resident_requested(mapping)):
        return False
    pc, sc, ept = geometry["prefix_columns"], geometry["suffix_columns"], geometry["suffix_ept"]
    suffix_ok = ((ept == 8 and sc == 4) or (ept == 16 and sc in (4, 8)) if stages[1] == 10
                 else ept in (16, 32) and sc in (2, 4))
    return pc in (4, 8, 16, 32) and suffix_ok


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
    if backend == "temporal-tile" and mapping.get("fft_core") == "cufftdx-block" and precision == "fp16":
        inverse = point.get("inverse", point.get("direction") == "inverse")
        normalize = point.get("normalize_inverse", point.get("normalization", "inverse") == "inverse")
        if inverse and normalize:
            raise ValueError("FP16 local cuFFTDx normalized inverse is not exposed by the public core contract")
        if policy == "precompiled":
            raise ValueError("FP16 local cuFFTDx requires the standalone module compiler")
        try:
            from cufftdx_low_module import validate_point
        except ImportError:
            from .cufftdx_low_module import validate_point
        request = dict(backend=backend, fft_core="cufftdx-block", precision=precision,
                       accumulation=point.get("accumulation", "native"), logN=log_n,
                       tile_threads=_positive(mapping, "tile_threads"),
                       prefix_ept=_positive(mapping, "prefix_ept"), packing="half2-transform-pair-v1")
        validate_point(dict(mapping, **request, operator=operator,
                            stage_partition=mapping.get("stage_partition") or [log_n]), 0)
        return request
    suffix_core = resident_mapping.validate_suffix_core(mapping)
    prefix_twiddle = resident_mapping.validate_prefix_twiddle({**mapping, "operator": operator})
    if prefix_twiddle != "runtime" and policy == "precompiled":
        raise ValueError("prefix_twiddle=table requires on-demand compilation")
    shared_stages, group_threads = None, []
    if backend == "shared-iterative" and mapping.get("execution_group_mappings", []) != []:
        shared_stages = _shared_partition(mapping, operator, log_n)
        group_threads = resident_mapping.shared_group_threads(mapping, shared_stages, operator=operator)
    if mapping.get("prefix_codelet") == "radix4":
        _register_prefix_axes(mapping, precision)
        if _radix4_uses_linked_inventory(mapping, operator, precision, log_n):
            return None
        if policy == "precompiled":
            raise ValueError("radix4 mapping is outside the prebuilt inventory and requires on-demand compilation")
    if backend == "factor-streamed" and precision not in {"fp16", "fp32", "fp64"}:
        raise ValueError("factor-streamed supports native FP16/32/64; BF16 has no cuFFTDx lowering")
    if backend == "factor-streamed" and precision == "fp16":
        inverse = point.get("inverse", point.get("direction") == "inverse")
        normalize = point.get("normalize_inverse", point.get("normalization", "inverse") == "inverse")
        if inverse and normalize:
            raise ValueError("FP16 factor-streamed rejects normalized inverse")
        if point.get("accumulation", "native") != "native" or mapping.get("fft_core") != "cufftdx-block":
            raise ValueError("FP16 factor-streamed requires native accumulation and cuFFTDx block")
        if policy == "precompiled":
            raise ValueError("FP16 factor-streamed requires the standalone module compiler")
    if policy == "precompiled":
        if backend == "shared-iterative" and mapping.get("data_tiles_per_cta", 1) != 1:
            raise ValueError("shared multi-tile columns require on-demand compilation")
        if suffix_core != "cufftdx-block":
            raise ValueError("native suffix_core requires on-demand compilation")
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
        stages = shared_stages if shared_stages is not None else _shared_partition(mapping, operator, log_n)
        axes = _resident_request(mapping, stages, backend=backend, operator=operator)
        tiles = mapping.get("data_tiles_per_cta", 1)
        if type(tiles) is not int or tiles < 1 or tiles & (tiles - 1):
            raise ValueError("shared data_tiles_per_cta must be a positive power of two")
        if any(tiles > 1 << (log_n - stage) for stage in stages):
            raise ValueError("shared data tiles exceed the independent tiles in an execution group")
        if tiles > 1 and (operator == "ntt" or resident_mapping.resident_requested(mapping)):
            raise ValueError("multi-tile shared columns require the generic iterative mapping")
        if policy != "research" and tiles == 1 and not resident_mapping.resident_requested(mapping) and not group_threads:
            return None
        batch_tile = 1 if mapping.get("stage_overlap", False) else int(mapping.get("batch_tile_count", 1))
        if not 1 <= batch_tile <= 4:
            raise ValueError("shared bulk batch tile count must be in [1,4]")
        request = dict(backend=backend, operator=operator, precision=precision,
                       logN=log_n, stage_partition=stages, batch_tile=batch_tile)
        request.update(axes)
        if group_threads:
            request["group_threads"] = group_threads
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
                           complex_multiply=mapping.get("complex_multiply", "four-mul"))
            if tiles != 1:
                request["data_tiles_per_cta"] = tiles
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
        if precision == "fp16":
            if request["factor_columns"] < 2 or request["factor_columns"] > 32 or request["factor_columns"] & (request["factor_columns"] - 1):
                raise ValueError("FP16 factor_columns must be an even power of two in [2,32]")
            request.update(accumulation="native", packing="half2-transform-pair-v1")
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
        if suffix_core != "cufftdx-block":
            request["suffix_core"] = suffix_core
        if codelet != "native":
            request["prefix_codelet"] = codelet
        if layout != "linear":
            request["prefix_shared_layout"] = layout
        if prefix_twiddle != "runtime":
            request["prefix_twiddle"] = prefix_twiddle
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
