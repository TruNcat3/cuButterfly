#!/usr/bin/env python3
"""Measured, mechanism-separated service curves for physical stage groups.

The stage probe deliberately measures one physical execution group at a time.
Its duration is therefore a complete kernel duration (including the launch
that produced that group), rather than a term to which a second launch cost
should be added.  Curves are indexed by the normalized work waveform
``K = work_blocks`` (with ``grid_ctas`` as its legacy alias).  Batch size and
the benchmark's absolute load are never part of a curve key; address geometry
and implementation/resource choices are.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
import pathlib
import re
import statistics
from typing import Any, Iterable

try:
    import resident_mapping
except ImportError:  # pragma: no cover - package import fallback
    from . import resident_mapping


SCHEMA = "cubutterfly-stage-probe-v1"
PROBE_SCHEMA = SCHEMA
PROFILE_SCHEMA = "cubutterfly-stage-service-profile-v1"
VERSION = "measured-stage-service-v5"
UNKNOWN = "unknown"

_IDENTITY_FIELDS = (
    "device", "compute_capability", "global_memory_bytes", "memory_bytes",
    "gpu_uuid", "uuid", "runtime_fingerprint",
)

_MISSING = object()


def _number(value: Any, default: float | None = None) -> float | None:
    """Parse numbers emitted by JSON and CSV paths without treating zero as missing."""
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _integerish(value: Any) -> Any:
    parsed = _number(value)
    if parsed is None:
        return UNKNOWN if value is None or value == "" else _text(value)
    if parsed.is_integer():
        return int(parsed)
    return parsed


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower().replace("_", "-")
    return str(value).strip().lower()


def _canonical(value: Any) -> Any:
    """Create JSON-stable key material while normalizing numeric strings."""
    if value is UNKNOWN or value is None or value == "":
        return UNKNOWN
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return UNKNOWN
        return int(value) if float(value).is_integer() else float(value)
    if isinstance(value, str):
        # CSV integer identities (notably a 64-bit NTT modulus) must not pass
        # through float: values beyond 2**53 otherwise select another curve.
        if re.fullmatch(r"[+-]?[0-9]+", value.strip()):
            return int(value)
        numeric = _number(value)
        if numeric is not None:
            return int(numeric) if numeric.is_integer() else numeric
        return _text(value)
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    return _text(value)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "measured", "correct"}
    return bool(value)


def _mapping(sample: dict[str, Any]) -> dict[str, Any]:
    value = sample.get("mapping_json", {})
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _first(source: dict[str, Any], names: Iterable[str], default: Any = UNKNOWN) -> Any:
    for name in names:
        if name in source and source[name] not in (None, ""):
            return source[name]
    return default


def _sample_value(sample: dict[str, Any], mapping: dict[str, Any], names: Iterable[str],
                  default: Any = UNKNOWN) -> Any:
    value = _first(sample, names, _MISSING)
    if value is not _MISSING:
        return value
    return _first(mapping, names, default)


def _normalized_exchange(value: Any) -> Any:
    if value is UNKNOWN:
        return UNKNOWN
    # ExchangePolicy is serialized as a string by the normal probe.  Keep an
    # integer enum stable as well; guessing a name from an integer could pool
    # two incompatible builds.
    if isinstance(value, str):
        return _text(value)
    return _canonical(value)


def _normalized_direction(value: Any) -> str:
    """Normalize probe/CSV direction spellings without guessing other enums."""
    if value is UNKNOWN:
        return "forward"
    if isinstance(value, bool):
        return "inverse" if value else "forward"
    numeric = _number(value)
    if numeric is not None and numeric in (0, 1):
        return "inverse" if numeric else "forward"
    text = _text(value)
    if text in {"0", "false", "fwd", "forward"}:
        return "forward"
    if text in {"1", "true", "bwd", "backward", "inverse", "reverse"}:
        return "inverse"
    return text


def _stage_index(group: dict[str, Any]) -> int | None:
    value = group.get("index")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _selected_mapping_state(mapping: dict[str, Any], sample: dict[str, Any] | None = None,
                            group: dict[str, Any] | None = None) -> dict[str, Any]:
    """Retain mapping/template knobs without putting the complete mapping in a key."""
    # These are discrete implementation choices.  In particular, do not add
    # batch/grid keys here: K is the only load axis of a service curve.
    names = (
        "backend", "dispatch", "fft_core", "core", "compute_unit", "local_exchange",
        "exchange", "shared_layout", "cross_twiddle", "twiddle", "twiddle_mode",
        "fused_twiddle", "needed_twiddle", "neededtwiddle", "twiddle_count", "twiddle_stride",
        "direct_boundary", "boundary_layout", "boundary_storage",
        "reorder_columns", "data_tiles_per_cta", "prefetch_depth", "factor_ept", "factor_columns",
    )
    merged = dict(mapping)
    if sample and not (mapping.get("schema_version") and mapping.get("kind")):
        merged.update({name: sample[name] for name in names if name in sample})
        # Strategy axes are intentionally handled below per physical group,
        # but legacy probe samples may carry them only at top level.
        for name in ("prefix_codelet", "prefix_shared_layout", "factor_io_policies"):
            if name in sample and name not in merged:
                merged[name] = sample[name]
    # A serialized, resolved mapping owns its implementation axes. NTT CSV
    # columns such as cross_twiddle=fused and empty boundary_storage describe
    # execution, but are not extra mapping choices absent from the probe JSON.
    # Importing those labels would split the very same physical curve by format.
    state = {name: _canonical(merged[name]) for name in names if name in merged}
    # Per-stage implementation choices belong to the physical group, not to
    # every curve produced by the complete mapping.  This permits validated
    # service reuse for an unchanged suffix when only the prefix codelet or a
    # different factor's I/O policy changes.
    index = _stage_index(group or {})
    backend = _text(merged.get("backend", ""))
    if backend == "online-reorder" and merged.get("fft_core") == "register-tile" and index == 0:
        # The compiler and runtime both treat omitted prefix axes as the
        # historical native/linear lowering.  Materialize those defaults in
        # the stage key so legacy mappings and explicit defaults share one
        # measured service curve, while non-default choices remain distinct.
        state["prefix_codelet"] = _canonical(merged.get("prefix_codelet", "native"))
        state["prefix_shared_layout"] = _canonical(
            merged.get("prefix_shared_layout", "linear"))
        # Cooperative prefix lanes are a group-local implementation choice.
        # Keep the historical key spelling for G=1 so omitted and explicit
        # defaults reuse existing curves; only true cooperative groups split
        # the prefix service identity.
        lanes = _sample_value(sample or {}, mapping, ("prefix_codelet_lanes",), 1)
        if _canonical(lanes) != 1:
            state["prefix_codelet_lanes"] = _canonical(lanes)
    if backend == "factor-streamed":
        policies = merged.get("factor_io_policies", [])
        if not isinstance(policies, list):
            policies = []
        policy = policies[index] if index is not None and index < len(policies) else "dynamic"
        state["factor_io_policy"] = _canonical(policy)
    return state


def _resource_state(group: dict[str, Any]) -> dict[str, Any]:
    """Encode absent resource fields as unknown, never as a known zero."""
    fields = (
        "live_shared_bytes", "dynamic_shared_bytes", "compiler_registers_per_thread", "compiler_local_bytes_per_thread",
        "compiler_resources_known", "compiler_local_resources_known", "compiler_registers_known",
        "compiler_local_known",
    )
    result = {field: _canonical(group[field]) if field in group and group[field] not in (None, "") else UNKNOWN
              for field in fields}
    for field in ("compiler_resources_known", "compiler_local_resources_known",
                  "compiler_registers_known", "compiler_local_known"):
        if field in group and group[field] not in (None, ""):
            result[field] = _truthy(group[field])
    # A compiler commonly emits a numeric zero together with a false known
    # flag when compilation did not expose resources.  Preserve that state as
    # unknown rather than making zero look like a measured allocation.
    if not (_truthy(group.get("compiler_resources_known", False)) or
            _truthy(group.get("compiler_registers_known", False))):
        result["compiler_registers_per_thread"] = UNKNOWN
    if not (_truthy(group.get("compiler_local_resources_known", False)) or
            _truthy(group.get("compiler_local_known", False))):
        result["compiler_local_bytes_per_thread"] = UNKNOWN
    return result


def _actual_kernel_state(group: dict[str, Any]) -> Any:
    """Retain non-load resources for composite physical kernel groups.

    A normal one-kernel capture is already represented by the flat group
    fields, so presence of a one-item ``actual_kernels`` list must not split
    otherwise identical curves.  For a composite lowering, each physical
    kernel's resource shape is part of the mechanism identity; grid/load and
    timing fields are deliberately omitted.
    """
    kernels = group.get("actual_kernels")
    if not isinstance(kernels, list) or len(kernels) <= 1:
        return UNKNOWN
    fields = (
        "threads", "dynamic_shared_bytes", "live_shared_bytes",
        "compiler_registers_per_thread", "compiler_local_bytes_per_thread",
        "compiler_resources_known", "compiler_local_resources_known",
        "compiler_registers_known", "compiler_local_known", "core", "codelet", "io_policy", "exchange",
        "shared_layout", "elements_per_thread", "EPT", "ept", "units_per_cta",
        "exchange_chunk", "local_stage_partition",
        "data_space", "data_time", "stage_space", "stage_time", "kernel_launch_count",
    )
    normalized = []
    for kernel in kernels:
        if not isinstance(kernel, dict):
            normalized.append(UNKNOWN)
            continue
        item = {}
        for field in fields:
            if field not in kernel or kernel[field] in (None, ""):
                item[field] = UNKNOWN
            elif field in {"compiler_resources_known", "compiler_local_resources_known",
                           "compiler_registers_known", "compiler_local_known"}:
                item[field] = _truthy(kernel[field])
            elif field == "exchange":
                item[field] = _normalized_exchange(kernel[field])
            else:
                item[field] = _canonical(kernel[field])
        normalized.append(item)
        # A composite group's aggregate known flag may be false even when
        # stream-capture exposed real per-kernel compiler resources.  Only an
        # explicit per-kernel known=false may erase a numeric zero; absent
        # flags leave present capture values intact.
        register_flags = ("compiler_resources_known", "compiler_registers_known")
        local_flags = ("compiler_local_resources_known", "compiler_local_known")
        if any(field in kernel for field in register_flags) and not any(
                _truthy(kernel.get(field)) for field in register_flags):
            item["compiler_registers_per_thread"] = UNKNOWN
        if any(field in kernel for field in local_flags) and not any(
                _truthy(kernel.get(field)) for field in local_flags):
            item["compiler_local_bytes_per_thread"] = UNKNOWN
    return normalized


def _group_shape(group: dict[str, Any], sample: dict[str, Any] | None = None) -> dict[str, Any]:
    """Discrete group template/resource shape, excluding the load axis."""
    fields = (
        "first_stage", "stage_count", "stage_space", "stage_time", "data_space", "data_time",
        "batch_space",
        "threads", "units_per_cta", "core", "backend",
        "exchange", "exchange_chunk", "local_stage_partition", "shared_layout",
        "data_tiles_per_cta", "prefetch_depth", "local_exchange",
        "remaining_bits", "local_logN", "local_points", "codelet", "io_policy",
    )
    shape = {}
    for field in fields:
        value = group.get(field)
        if value in (None, "") and sample is not None:
            mapping = _mapping(sample)
            index = _stage_index(group) or 0
            if field == "exchange_chunk":
                value = resident_mapping.group_axis(mapping, "exchange_chunk", index, 0)
            elif field == "local_stage_partition":
                value = resident_mapping.group_axis(mapping, "local_stage_partition", index, [])
        if value not in (None, ""):
            shape[field] = _normalized_exchange(value) if field == "exchange" else _canonical(value)
        else:
            shape[field] = UNKNOWN
    shape["elements_per_thread"] = _canonical(_first(group, ("elements_per_thread", "EPT", "ept")))
    # local_exchange is usually carried by the benchmark mapping rather than
    # the C++ ExecutionGroup; use the physical sample value when the group
    # omitted its alias.
    # launch_count is a legacy spelling for one group's physical kernel count;
    # normalize the alias so producer spelling cannot split a curve.
    shape["kernel_launch_count"] = _canonical(_first(group, ("kernel_launch_count", "launch_count")))
    # ExecutionGroup.batch_time is ceil(total batch / batch_space), not a
    # kernel template parameter. Its work is already represented by K. Keeping
    # it in this key would turn every measured load into a one-point curve.
    return shape


def curve_key(sample: dict[str, Any], group: dict[str, Any]) -> str:
    """Return a stable key for one mechanism and resource regime.

    ``mapping_json`` is parsed only for selected implementation state.  This
    intentionally preserves address geometry while avoiding accidental lookup
    keys based on a whole mapping or on batch/grid load.
    """
    sample = sample or {}
    group = group or {}
    mapping = _mapping(sample)
    raw_operator = _sample_value(sample, mapping, ("operator",))
    operator = _text(raw_operator) if raw_operator is not UNKNOWN else (
        "ntt" if _sample_value(sample, mapping, ("word_bits",)) is not UNKNOWN else "fft")
    raw_precision = _sample_value(sample, mapping, ("precision",))
    if raw_precision is UNKNOWN:
        word_bits = _sample_value(sample, mapping, ("word_bits",))
        raw_precision = (f"word{_integerish(word_bits)}" if operator == "ntt" and word_bits is not UNKNOWN
                         else ("word64" if operator == "ntt" else "fp32"))
    raw_direction = _sample_value(sample, mapping, ("direction",))
    raw_inverse = _sample_value(sample, mapping, ("inverse",))
    if raw_direction is UNKNOWN:
        raw_direction = "inverse" if raw_inverse is not UNKNOWN and _truthy(raw_inverse) else "forward"
    direction = _normalized_direction(raw_direction)
    inverse = direction in {"inverse", "backward", "reverse"}
    raw_normalization = _sample_value(sample, mapping, ("normalization",))
    normalization = "none" if raw_normalization is UNKNOWN else _text(raw_normalization)
    if direction == "forward" or operator not in {"fft", "fwht"}:
        normalization = "none"
    raw_modulus = _sample_value(sample, mapping, ("modulus",))
    # Butterfly operators do not have a numeric modulus.  CSV paths often
    # emit modulus=0 while JSON paths omit it; make those explicit aliases.
    modulus = 0 if operator != "ntt" or raw_modulus is UNKNOWN or _number(raw_modulus, 0) == 0 else raw_modulus
    raw_stage_matrix = _sample_value(sample, mapping, ("stage_matrix", "stage_matrices"), "")
    raw_input_order = _sample_value(sample, mapping, ("input_order",))
    raw_output_order = _sample_value(sample, mapping, ("output_order",))
    raw_placement = _sample_value(sample, mapping, ("placement",), UNKNOWN)
    if operator == "ntt":
        raw_input_order = "natural" if raw_input_order is UNKNOWN else raw_input_order
        raw_output_order = (raw_output_order if raw_output_order is not UNKNOWN
                            else (raw_placement if raw_placement is not UNKNOWN else "natural"))
        raw_placement = raw_output_order
    raw_word_bits = _sample_value(sample, mapping, ("word_bits",))
    if operator == "ntt" and raw_word_bits is UNKNOWN and isinstance(raw_precision, str):
        precision_match = raw_precision.lower().replace("_", "")
        raw_word_bits = precision_match[4:] if precision_match.startswith("word") else UNKNOWN
    semantics = {
        "operator": _canonical(operator), "precision": _canonical(raw_precision),
        "accumulation": _canonical(_sample_value(sample, mapping, ("accumulation",), "native")),
        "direction": _canonical(direction), "normalization": _canonical(normalization),
        "placement": _canonical(raw_placement),
        "inverse": inverse, "modulus": _canonical(modulus),
        "word_bits": _canonical(raw_word_bits) if operator == "ntt" else UNKNOWN,
        "value_type": _canonical(_sample_value(sample, mapping, ("value_type",))),
        "complex_values": _canonical(_sample_value(sample, mapping, ("complex_values",))),
        "stage_matrix": _canonical(raw_stage_matrix),
    }
    raw_log_n = _sample_value(sample, mapping, ("logN", "log_n"))
    raw_n = _sample_value(sample, mapping, ("N",))
    log_n = _number(raw_log_n)
    n = (2 ** int(log_n)) if log_n is not None and log_n >= 0 and log_n <= 62 else _number(raw_n)
    element_stride = _number(_sample_value(sample, mapping, ("element_stride",)), 1)
    if element_stride is None or element_stride <= 0:
        element_stride = 1
    extent = ((n - 1) * element_stride + 1) if n is not None and n > 0 else None
    raw_batch_stride = _number(_sample_value(sample, mapping, ("batch_stride",)))
    batch_stride = extent if raw_batch_stride is None or raw_batch_stride <= 0 else raw_batch_stride
    geometry = {
        "logN": _canonical(raw_log_n), "N": UNKNOWN if raw_log_n is not UNKNOWN else _canonical(raw_n),
        "element_stride": _canonical(element_stride), "batch_stride": _canonical(batch_stride),
        "input_order": _canonical(raw_input_order),
        "output_order": _canonical(raw_output_order),
        "address_layout": _canonical(_sample_value(sample, mapping, ("address_layout",))),
        "layout": _canonical(_sample_value(sample, mapping, ("layout",))),
        "dimension_order": _canonical(_sample_value(sample, mapping, ("dimension_order",))),
        "remaining_bits": _canonical(_sample_value(sample, mapping, ("remaining_bits",))),
        "local_logN": _canonical(_sample_value(sample, mapping, ("local_logN", "local_log_n"))),
    }

    # Prefer the physical group fields when present, then the resolved mapping
    # fields, and finally the ordinary benchmark sample fields.
    mechanism = {
        "backend": _canonical(_first(group, ("backend",),
                                     _sample_value(sample, mapping, ("backend",)))),
        "core": _canonical(_first(group, ("core",),
                                   _sample_value(sample, mapping, ("core", "fft_core")))),
        "fft_core": _canonical(_first(group, ("fft_core", "core"),
                                       _sample_value(sample, mapping, ("fft_core", "core")))),
        "compute_unit": _canonical(_first(group, ("compute_unit",),
                                           _sample_value(sample, mapping, ("compute_unit",)))),
        "exchange": _normalized_exchange(_first(group, ("exchange",),
                                                   _sample_value(sample, mapping, ("exchange",)))),
        # Keep local_exchange separate.  The legacy primitive key omitted it,
        # which pooled FWHT register and shared-memory mechanisms.
        "local_exchange": _canonical(_first(group, ("local_exchange",),
                                               _sample_value(sample, mapping, ("local_exchange",)))),
        "shared_layout": _canonical(_first(group, ("shared_layout",),
                                             _sample_value(sample, mapping, ("shared_layout",)))),
        "mapping_state": _selected_mapping_state(mapping, sample, group),
    }
    shape = _group_shape(group, sample)
    if shape.get("local_exchange") == UNKNOWN:
        shape["local_exchange"] = _canonical(_sample_value(sample, mapping, ("local_exchange",)))
    stage_index = _stage_index(group)
    backend = _text(_sample_value(sample, mapping, ("backend",), ""))
    if shape.get("codelet") == UNKNOWN:
        if backend == "online-reorder" and mapping.get("fft_core") == "register-tile" and stage_index == 0:
            shape["codelet"] = _canonical(_sample_value(sample, mapping, ("prefix_codelet",), "native"))
        else:
            shape["codelet"] = _canonical(_first(group, ("codelet",), "native"))
    if shape.get("io_policy") == UNKNOWN:
        if backend == "factor-streamed":
            policies = _sample_value(sample, mapping, ("factor_io_policies",), [])
            if not isinstance(policies, list):
                policies = []
            shape["io_policy"] = _canonical(
                policies[stage_index] if stage_index is not None and stage_index < len(policies) else "dynamic")
        else:
            shape["io_policy"] = "dynamic"
    payload = {
        "version": VERSION,
        "semantic": semantics,
        "address_geometry": geometry,
        "mechanism": mechanism,
        "group_shape": shape,
        "resources": {**_resource_state(group), "actual_kernels": _actual_kernel_state(group)},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def shape_key(sample: dict[str, Any], group: dict[str, Any]) -> str:
    """Return the non-resource identity for safe resource metadata lookup.

    This helper is intentionally not used by :func:`predict_group`: an
    unknown resource regime has no coverage.  A caller that has exactly one
    known resource regime for this shape may use this key to attach measured
    compiler metadata before doing the final strict prediction.
    """
    payload = json.loads(curve_key(sample, group))
    payload.pop("resources", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def resource_regime_key(sample: dict[str, Any], group: dict[str, Any]) -> str:
    """Return only normalized compiler/live-resource state for diagnostics."""
    # This projection depends only on the group. Rebuilding the complete
    # semantic/address/mechanism key twice per prediction is unnecessary.
    group = group or {}
    resources = {**_resource_state(group), "actual_kernels": _actual_kernel_state(group)}
    return json.dumps(resources, sort_keys=True, separators=(",", ":"))


@lru_cache(maxsize=8)
def _resource_candidates_index(curve_keys: tuple[Any, ...]) -> dict[str, tuple[str, ...]]:
    """Index serialized curve keys by shape without retaining a profile object.

    The tuple of curve keys is an immutable snapshot of the profile's curve
    set.  A changed set therefore gets a distinct cache entry automatically,
    while the cached values remain immutable serialized resource regimes.
    """
    indexed: dict[str, list[str]] = {}
    for key in curve_keys:
        try:
            payload = json.loads(key)
        except (TypeError, ValueError):
            continue
        resources = payload.get("resources")
        payload.pop("resources", None)
        candidate_shape = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if isinstance(resources, dict):
            regime_key = json.dumps(resources, sort_keys=True, separators=(",", ":"))
            indexed.setdefault(candidate_shape, []).append(regime_key)
    return {shape: tuple(regimes) for shape, regimes in indexed.items()}


def resolve_resources(sample: dict[str, Any], group: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Copy resources only from a unique measured regime of the same shape.

    A projected descriptor may know the mechanism and local dimensions before
    compilation exposes registers/shared memory.  It must not match a random
    curve in that state.  This helper is a conservative bridge: all candidate
    curves must have the same :func:`shape_key`, and their normalized resource
    dictionaries must collapse to one regime with at least one known field.
    The caller still performs the normal strict :func:`predict_group` lookup.
    """
    body = _profile_body(profile)
    if not isinstance(body, dict) or not body.get("curves"):
        return dict(group or {})
    target_shape = shape_key(sample or {}, group or {})
    curves = body.get("curves", {})
    candidate_regimes = _resource_candidates_index(tuple(curves.keys())).get(target_shape, ())
    if not candidate_regimes:
        return dict(group or {})
    if len(set(candidate_regimes)) != 1:
        return dict(group or {})
    resources = json.loads(candidate_regimes[0])
    known_fields = {field: value for field, value in resources.items() if value != UNKNOWN}
    if not known_fields:
        return dict(group or {})
    resolved = dict(group or {})
    for field, value in known_fields.items():
        # The resource dictionary has canonical values.  Keep JSON scalar
        # types acceptable to curve_key/predict_group (numeric strings are
        # normalized there as well).
        resolved[field] = value
    return resolved


def _waveform(group: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    work_blocks = _number(group.get("work_blocks"))
    grid_ctas = _number(group.get("grid_ctas"))
    if work_blocks is None:
        # A stage probe may use the existing field as its work count when the
        # driver did not duplicate it under work_blocks.
        work_blocks = _number(group.get("work"))
    if work_blocks is None:
        work_blocks = grid_ctas
    if grid_ctas is None:
        # Capture records may emit only the new work_blocks spelling.  It is
        # the same physical grid quantity, so the legacy alias is recoverable.
        grid_ctas = work_blocks
    if work_blocks is None or grid_ctas is None or work_blocks <= 0 or grid_ctas <= 0:
        return None, work_blocks, grid_ctas
    # ``work_blocks`` is the normalized physical work waveform.  The probe
    # contract emits it as the actual grid count; older descriptors call the
    # same quantity grid_ctas.  Do not divide the aliases or all endpoints
    # would collapse to K=1.
    return work_blocks, work_blocks, grid_ctas


def _positive_trials(values: Any, *, context: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{context} requires positive trial_kernel_ms values")
    result = []
    for value in values:
        parsed = _number(value)
        if parsed is None or parsed <= 0:
            raise ValueError(f"{context} has non-positive or non-finite trial_kernel_ms")
        result.append(parsed)
    return result


def _identity_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    identity = {}
    for field in _IDENTITY_FIELDS:
        value = sample.get(field)
        if value not in (None, ""):
            identity[field] = str(value)
    hardware = sample.get("hardware")
    if isinstance(hardware, dict):
        for field in _IDENTITY_FIELDS:
            value = hardware.get(field)
            if value not in (None, "") and field not in identity:
                identity[field] = str(value)
    return identity


def _flatten_identity(identity: dict[str, Any] | None) -> dict[str, Any]:
    if not identity:
        return {}
    result = {}
    for field in _IDENTITY_FIELDS:
        value = identity.get(field)
        if value in (None, "") and isinstance(identity.get("hardware"), dict):
            value = identity["hardware"].get(field)
        if value not in (None, ""):
            result[field] = str(value)
    # Runtime/device identity supplied by a caller may use a custom field;
    # retain it in the cache hash without using it as a curve key.
    for field, value in identity.items():
        if field != "hardware" and field not in result and value not in (None, ""):
            result[str(field)] = _canonical(value)
    return result


def _quality(sample: dict[str, Any], group: dict[str, Any], trials: list[float]) -> dict[str, Any]:
    quality = {
        "trial_count": len(trials),
        "measurement_exclusive_gpu": _truthy(
            group.get("measurement_exclusive_gpu", sample.get("measurement_exclusive_gpu", False))),
    }
    for field in ("warmup", "repeat", "trials"):
        parsed = _number(group.get(field, sample.get(field)))
        if parsed is not None:
            quality[field] = int(parsed) if parsed.is_integer() else parsed
    # More raw repeats and an explicitly exclusive measurement are stronger
    # aliases.  Keep this score in the artifact so merging is auditable.
    quality["score"] = (
        (1000000 if quality["measurement_exclusive_gpu"] else 0)
        + quality["trial_count"] * 1000
        + int(quality.get("repeat", 0))
        + int(quality.get("warmup", 0))
    )
    return quality


def _actual_load_key(k: float, work_blocks: float, grid_ctas: float) -> tuple[Any, ...]:
    # Aliases are merged only when they describe the same measured load.  A
    # different absolute load with the same K remains auditable as an alias,
    # but cannot silently inflate the replicate count of an endpoint.
    return (k, work_blocks, grid_ctas)


def _record_list(records: Any) -> list[dict[str, Any]]:
    if isinstance(records, dict):
        if isinstance(records.get("records"), list):
            return records["records"]
        if isinstance(records.get("probes"), list):
            return records["probes"]
        if records.get("schema") == SCHEMA:
            return [records]
    if records is None:
        return []
    return list(records)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _train_prediction(curve: dict[str, Any], k: float) -> tuple[float | None, bool, bool, str]:
    points = _representative_points(curve.get("train_points", []))
    if not points:
        return None, False, False, "no-independent-training-measurements"
    exact = [point for point in points if math.isclose(point["k"], k, rel_tol=1e-12, abs_tol=1e-12)]
    if exact:
        return exact[0]["kernel_ms"], True, False, "exact-independent-training-measurement"
    if k < points[0]["k"] or k > points[-1]["k"]:
        return None, False, True, "outside-measured-K-hull"
    for left, right in zip(points, points[1:]):
        if left["k"] <= k <= right["k"]:
            if math.isclose(right["k"], left["k"], rel_tol=1e-12, abs_tol=1e-12):
                return left["kernel_ms"], True, False, "duplicate-K-endpoint"
            ratio = (k - left["k"]) / (right["k"] - left["k"])
            value = left["kernel_ms"] + ratio * (right["kernel_ms"] - left["kernel_ms"])
            return value, True, False, "piecewise-linear-independent-training"
    return None, False, True, "outside-measured-K-hull"


def _validation_exact(curve: dict[str, Any], k: float) -> tuple[float | None, bool]:
    exact = [point for point in _representative_points(curve.get("validation_points", []))
             if math.isclose(point["k"], k, rel_tol=1e-12, abs_tol=1e-12)]
    if not exact:
        return None, False
    # A validation exact value is allowed as evidence, but it never enters the
    # train hull or the interpolation endpoints.
    return exact[0]["kernel_ms"], True


def _validation_report(profile: dict[str, Any]) -> dict[str, Any]:
    checks = []
    for key, curve in sorted(profile.get("curves", {}).items()):
        for point in curve.get("validation_points", []):
            predicted, covered, extrapolated, reason = _train_prediction(curve, point["k"])
            error = abs(predicted / point["kernel_ms"] - 1.0) if covered and predicted else None
            checks.append({
                "curve_key": key, "k": point["k"], "observed_kernel_ms": point["kernel_ms"],
                "predicted_kernel_ms": predicted, "covered": covered,
                "extrapolated": extrapolated, "relative_error": error, "reason": reason,
            })
    covered_errors = [item["relative_error"] for item in checks if item["relative_error"] is not None]
    count = len(checks)
    covered_count = len(covered_errors)
    median_error = statistics.median(covered_errors) if covered_errors else None
    p90_error = _percentile(covered_errors, .90)
    thresholds = {"heldout_median_relative_error": .10, "heldout_p90_relative_error": .20}
    passed = bool(count and covered_count == count and median_error is not None and p90_error is not None
                  and median_error <= thresholds["heldout_median_relative_error"]
                  and p90_error <= thresholds["heldout_p90_relative_error"])
    if not count:
        status = "insufficient-holdout"
    elif covered_count != count:
        status = "warning-uncovered-holdout"
    else:
        status = "passed" if passed else "warning-accuracy"
    return {
        "status": status,
        "passed": passed,
        "heldout_count": count,
        "heldout_covered_count": covered_count,
        "heldout_uncovered_count": count - covered_count,
        "heldout_median_relative_error": median_error,
        "heldout_p90_relative_error": p90_error,
        # Short aliases make the report convenient for existing audit tools.
        "median_relative_error": median_error,
        "p90_relative_error": p90_error,
        "pass_thresholds": thresholds,
        "checks": checks,
    }


def _representative_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose the strongest actual-load alias for interpolation at repeated K."""
    chosen: list[dict[str, Any]] = []
    for point in sorted(points, key=lambda item: item["k"]):
        match = next((item for item in chosen
                      if math.isclose(item["k"], point["k"], rel_tol=1e-12, abs_tol=1e-12)), None)
        if match is None:
            chosen.append(point)
            continue
        old_score = (match.get("quality") or {}).get("score", 0)
        new_score = (point.get("quality") or {}).get("score", 0)
        if new_score > old_score:
            chosen[chosen.index(match)] = point
    return chosen


def build_profile(records: Any, identity: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a measured service profile from stage probe records.

    Validation groups are retained for independent reporting, but only train
    groups define interpolation hulls.  Duplicate aliases with identical
    actual load are merged by concatenating their raw positive trials.
    """
    rows = _record_list(records)
    explicit_identity = _flatten_identity(identity)
    observed_identity: dict[str, Any] = {}
    groups_seen = groups_used = validation_groups = 0
    skipped = []
    buckets: dict[tuple[str, str, tuple[Any, ...]], dict[str, Any]] = {}

    for record_index, record in enumerate(rows):
        if not isinstance(record, dict):
            skipped.append({"record": record_index, "reason": "record-is-not-an-object"})
            continue
        if record.get("schema") != SCHEMA:
            skipped.append({"record": record_index, "reason": "unsupported-schema"})
            continue
        if record.get("status") != "measured":
            skipped.append({"record": record_index, "reason": "record-not-measured"})
            continue
        if not _truthy(record.get("correct")):
            skipped.append({"record": record_index, "reason": "record-not-correct"})
            continue
        sample = record.get("sample")
        if not isinstance(sample, dict):
            skipped.append({"record": record_index, "reason": "missing-sample"})
            continue
        row_identity = _identity_from_sample(sample)
        for field, value in row_identity.items():
            if field in observed_identity and str(observed_identity[field]) != str(value):
                raise ValueError(f"stage service records disagree on identity: {field}")
            observed_identity[field] = value
            if field in explicit_identity and str(explicit_identity[field]) != str(value):
                raise ValueError(f"stage service identity mismatch: {field}")
        group_rows = record.get("groups")
        if not isinstance(group_rows, list):
            skipped.append({"record": record_index, "reason": "missing-groups"})
            continue
        record_role = record.get("role", "train")
        for group_index, group in enumerate(group_rows):
            groups_seen += 1
            if not isinstance(group, dict):
                skipped.append({"record": record_index, "group": group_index, "reason": "group-is-not-an-object"})
                continue
            if not _truthy(group.get("independent")):
                skipped.append({"record": record_index, "group": group_index, "reason": "group-not-independent"})
                continue
            role = str(group.get("role", record_role)).strip().lower()
            if role not in {"train", "validation", "holdout"}:
                raise ValueError(f"stage service group has unsupported role: {role}")
            role = "validation" if role == "holdout" else role
            k, work_blocks, grid_ctas = _waveform(group)
            if k is None:
                skipped.append({"record": record_index, "group": group_index, "reason": "invalid-workwave"})
                continue
            trials = _positive_trials(group.get("trial_kernel_ms"),
                                      context=f"record {record_index} group {group_index}")
            key = curve_key(sample, group)
            load_key = _actual_load_key(k, work_blocks, grid_ctas)
            bucket_key = (key, role, load_key)
            entry = buckets.setdefault(bucket_key, {
                "key": key, "role": role, "k": k, "work_blocks": work_blocks,
                "grid_ctas": grid_ctas, "trial_kernel_ms": [], "aliases": [],
                "quality": None,
            })
            entry["trial_kernel_ms"].extend(trials)
            quality = _quality(sample, group, trials)
            if entry["quality"] is None or quality["score"] > entry["quality"]["score"]:
                entry["quality"] = quality
            entry["aliases"].append({
                "record": record_index, "group": group_index, "index": group.get("index", group_index),
                "work_blocks": work_blocks, "grid_ctas": grid_ctas, "trial_count": len(trials),
                "quality": quality,
            })
            groups_used += 1
            if role == "validation":
                validation_groups += 1

    final_identity = dict(observed_identity)
    final_identity.update(explicit_identity)
    # Hardware/runtime identity is case-sensitive (device names and build
    # fingerprints commonly are).  Do not apply the lower-casing key
    # normalization used for semantic curve fields.
    canonical_identity = {key: (value.strip() if isinstance(value, str) else value)
                          for key, value in sorted(final_identity.items())}
    curves: dict[str, dict[str, Any]] = {}
    for entry in buckets.values():
        trial_values = entry["trial_kernel_ms"]
        point = {
            "k": entry["k"], "kernel_ms": statistics.median(trial_values),
            "trial_kernel_ms": list(trial_values), "trial_count": len(trial_values),
            "work_blocks": entry["work_blocks"], "grid_ctas": entry["grid_ctas"],
            "quality": entry["quality"], "aliases": entry["aliases"],
        }
        curve = curves.setdefault(entry["key"], {
            "key": entry["key"], "train_points": [], "validation_points": [],
            "points": [], "k_hull": None,
        })
        curve["train_points" if entry["role"] == "train" else "validation_points"].append(point)
    for curve in curves.values():
        # Points at the same K but different absolute loads remain visible in
        # the profile.  Prediction picks the strongest quality alias without
        # pretending those distinct loads were replicate measurements.
        curve["train_points"].sort(key=lambda point: point["k"])
        curve["validation_points"].sort(key=lambda point: point["k"])
        # Keep one public points alias for callers that predate role-aware
        # profiles.  It intentionally exposes training points only.
        curve["points"] = list(curve["train_points"])
        if curve["train_points"]:
            curve["k_hull"] = [curve["train_points"][0]["k"], curve["train_points"][-1]["k"]]

    profile = {
        "schema": PROFILE_SCHEMA, "version": VERSION, "identity": canonical_identity,
        "cache_identity": hashlib.sha256(json.dumps(canonical_identity, sort_keys=True,
                                                       separators=(",", ":")).encode()).hexdigest(),
        "curves": curves, "curve_count": len(curves), "groups_seen": groups_seen,
        "groups_used": groups_used, "training_groups": sum(len(c["train_points"]) for c in curves.values()),
        "validation_groups": validation_groups,
        "skipped": skipped, "source": "independent physical stage probes",
        "training_policy": "role=train only; validation points never define interpolation hull",
        "interpolation": "piecewise-linear in K=work_blocks (grid_ctas alias) within training hull",
    }
    profile["validation"] = _validation_report(profile)
    profile["validation_status"] = profile["validation"]["status"]
    if not profile["training_groups"]:
        profile["status"] = "insufficient-stage-service-training"
    elif profile["validation"]["passed"]:
        profile["status"] = "calibrated-stage-service"
    else:
        profile["status"] = "calibrated-stage-service-warning"
    profile["profile_status"] = profile["status"]
    return profile


def _profile_body(profile: dict[str, Any]) -> dict[str, Any]:
    nested = profile.get("stage_service") if isinstance(profile, dict) else None
    if isinstance(nested, dict) and nested.get("curves") is not None:
        return nested
    return profile


def _identity_matches(sample: dict[str, Any], profile: dict[str, Any]) -> tuple[bool, str | None]:
    expected = profile.get("identity", {})
    actual = _identity_from_sample(sample)
    for field, expected_value in expected.items():
        observed = actual.get(field, sample.get(field))
        if observed not in (None, "") and str(observed) != str(expected_value):
            return False, f"profile-identity-mismatch:{field}"
    return True, None


def predict_group(sample: dict[str, Any], group: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Predict one group with exact/interpolated measured evidence only."""
    body = _profile_body(profile)
    key = curve_key(sample or {}, group or {})
    result = {
        "kernel_ms": None, "covered": False, "extrapolated": False,
        "source": "measured-stage-service", "reason": "unknown-curve-key", "curve_key": key,
    }
    if not isinstance(body, dict) or not body.get("curves"):
        result["source"] = "unavailable"
        result["reason"] = "empty-stage-service-profile"
        return result
    identity_ok, identity_reason = _identity_matches(sample or {}, body)
    if not identity_ok:
        result["source"] = "unavailable"
        result["reason"] = identity_reason
        return result
    k, _, _ = _waveform(group or {})
    if k is None:
        result["source"] = "unavailable"
        result["reason"] = "invalid-workwave"
        return result
    curve = body.get("curves", {}).get(key)
    if not curve:
        result["reason"] = "unknown-primitive-or-resource-regime"
        return result
    value, covered, extrapolated, reason = _train_prediction(curve, k)
    result.update(kernel_ms=value, covered=covered, extrapolated=extrapolated, reason=reason)
    role = str((group or {}).get("role", "")).lower()
    validation_value, validation_exact = _validation_exact(curve, k)
    if role in {"validation", "holdout"} and validation_exact:
        # An exact validation query may return its independently observed
        # value.  It is explicitly tagged and never changes train points.
        result.update(kernel_ms=validation_value, covered=True, extrapolated=False,
                      source="independent-validation-exact",
                      validation_kernel_ms=validation_value,
                      reason="exact-independent-validation-measurement")
    elif covered:
        result["source"] = "measured-exact" if reason.startswith("exact") else "measured-piecewise-linear"
    elif extrapolated:
        result["source"] = "measured-stage-service"
    return result


def validation_report(profile: dict[str, Any]) -> dict[str, Any]:
    """Return the role-aware holdout report for a built or cached profile."""
    body = _profile_body(profile)
    if "validation" in body and isinstance(body["validation"], dict):
        return body["validation"]
    return _validation_report(body)


def profile_status(profile: dict[str, Any]) -> dict[str, Any]:
    """Expose a compact status suitable for hardware-profile audit output."""
    body = _profile_body(profile)
    validation = validation_report(body)
    return {
        "status": body.get("status", "insufficient-stage-service-training"),
        "profile_status": body.get("profile_status", body.get("status")),
        "validation_status": body.get("validation_status", validation.get("status")),
        "curve_count": len(body.get("curves", {})),
        "training_groups": body.get("training_groups", 0),
        "validation_groups": body.get("validation_groups", 0),
        "heldout_median_relative_error": validation.get("heldout_median_relative_error"),
        "heldout_p90_relative_error": validation.get("heldout_p90_relative_error"),
        "heldout_covered_count": validation.get("heldout_covered_count", 0),
        "heldout_count": validation.get("heldout_count", 0),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=pathlib.Path, required=True,
                        help="JSON list or object containing stage probe records")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--identity", type=pathlib.Path)
    args = parser.parse_args()
    records = json.loads(args.records.read_text())
    identity = json.loads(args.identity.read_text()) if args.identity else None
    profile = build_profile(records, identity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(profile, indent=2) + "\n")
    temporary.replace(args.output)
    print(f"stage service profile: {profile['status']}; {profile['curve_count']} curves")


if __name__ == "__main__":
    _main()
