"""Bounded, semantics-preserving evolutionary search proposals.

This module deliberately sits between the calibration model and the runtime.
It proposes complete public mapping points, but it does not describe, compile,
validate, or time them.  A mapping is therefore allowed to be syntactically
well formed while still being rejected by the runtime for a particular device.

The search is a neighbourhood search rather than a Cartesian product: every
candidate changes one structural or tuning axis from its parent.  This keeps
the work proportional to the number of axes and to the length of a partition.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import multiprocessing
import os
import threading
from typing import Any, Callable, Iterable, Mapping, Sequence

try:  # scripts are imported directly by the existing calibration tools
    from calibration_space import candidate_id, mapping_point
except ImportError:  # pragma: no cover - useful when imported as a package
    from .calibration_space import candidate_id, mapping_point
try:
    from research_compile_requests import _shared_partition
except ImportError:  # pragma: no cover - useful when imported as a package
    from .research_compile_requests import _shared_partition
try:
    import resident_mapping
except ImportError:  # pragma: no cover - useful when imported as a package
    from . import resident_mapping


__all__ = ["neighbors", "evolutionary_candidates"]


# Keep an unusually long partition from turning one parent into an unbounded
# stream.  Normal mappings have far fewer axes than this; the cap is only a
# last-resort guard and does not influence the candidate values themselves.
MAX_NEIGHBORS = 64


# Fields copied by calibration_space.mapping_point are projections of a
# mapping, not workload semantics.  Removing them before calling
# mapping_point avoids stale top-level values after a mutation.
_MAPPING_FIELDS = {
    "schema_version",
    "kind",
    "backend",
    "compute_unit",
    "fft_core",
    "complex_multiply",
    "cross_twiddle",
    "cross_twiddle_placement",
    "direct_boundary",
    "local_exchange",
    "shared_layout",
    "stage_handoff",
    "hierarchical_core",
    "dataflow_layout",
    "dataflow_state_mode",
    "modular_multiply",
    "packet_readiness_mode",
    "packet_compute_layout",
    "stage_partition",
    "stages_per_decomposition",
    "execution_stage_partition",
    "factor_partition",
    "stage_overlap",
    "factor_overlap",
    "batch_tile_count",
    "tile_threads",
    "threads_per_block",
    "local_stages",
    "reorder_columns",
    "stage_space",
    "warp_stages",
    "pipeline_warps",
    "flow_tile_log_n",
    "flow_tile_log",
    "data_space",
    "data_time",
    "role_stages",
    "target_ctas_per_sm",
    "token_interleave",
    "pipeline_buffers",
    "n1_log",
    "rows_per_block",
    "ready_window",
    "packet_fold_wave_barriers",
    "profile_appt_roles",
    "factor_ept",
    "factor_columns",
    "data_tiles_per_cta",
    "prefetch_depth",
    "factor_slices",
    "prefix_threads",
    "suffix_threads",
    "prefix_ept",
    "suffix_ept",
    "prefix_codelet_lanes",
    "prefix_units_per_cta",
    "suffix_units_per_cta",
    "local_stage_partitions",
    "exchange_chunks",
    "prefix_codelet",
    "prefix_shared_layout",
    "factor_io_policies",
    "units_per_cta",
    "coefficient_reuse_stages",
    "cta_weight",
    "producer_weight",
    "tail_weight",
    "writer_weight",
    "fragment_width",
    "writer_tiles_per_cta",
    "data_time_role_mask",
    "segment_threads",
    "segment_ept",
    "segment_cores",
    "group_threads",
    "group_ept",
    "group_cores",
    "boundary_twiddles",
    "boundary_layouts",
    "boundary_residencies",
    "mapping_json",
}


# Measurement/selection annotations are not part of a workload point.  The
# caller remains the owner of measured IDs and scores.
_NON_SEMANTIC_FIELDS = {
    "candidate_id",
    "kernel_ms",
    "median_kernel_ms",
    "runtime_fingerprint",
    "correct",
    "status",
    "selection_reason",
    "selection_method",
    "score",
    "predicted_kernel_ms",
    "prediction",
    "samples",
    "trials",
    "configuration",
    "execution_groups_json",
    "execution_group_count",
    "decomposition_count",
}


_BUTTERFLY_BACKENDS = {
    "temporal-tile",
    "hierarchical",
    "online-reorder",
    "warp-hybrid",
    "stage-pipeline",
    "shared-iterative",
    "factor-streamed",
}

_NTT_BACKENDS = {
    "baseline",
    "tile256",
    "hybrid2d",
    "compact-stage",
    "stage-pipeline",
    "hybrid-dataflow",
    "hierarchical-barrier",
    "hierarchical-dataflow",
    "shared-iterative",
}

_BUTTERFLY_CORE_CHOICES = {
    "temporal-tile": ("scalar", "thread-dft8", "cta-dft8", "wmma-dft8", "cufftdx-direct"),
    "hierarchical": ("scalar", "cufftdx-block"),
    "online-reorder": ("cufftdx-block", "cufftdx-resident", "register-tile"),
    "shared-iterative": ("scalar",),
    "factor-streamed": ("cufftdx-block", "register-tile"),
    "stage-pipeline": ("scalar",),
    "warp-hybrid": ("scalar",),
}

_SHARED_LAYOUTS = ("linear", "xor-swizzle", "writer-aligned")
_NTT_ENUM_CHOICES = {
    "compute_unit": ("radix2", "radix4", "radix8"),
    "stage_handoff": ("atomic", "named-barrier"),
    "hierarchical_core": ("dataflow-radix4", "hybrid2d-radix4"),
    "dataflow_layout": ("hermes-xor", "linear"),
    "dataflow_state_mode": ("inplace", "ping-pong"),
    "cross_twiddle_placement": ("first", "second", "fused", "fused-barrett"),
    "modular_multiply": ("shoup", "barrett"),
    "packet_readiness_mode": ("per-packet", "wave-bitmap"),
    "packet_compute_layout": ("interleaved-rows", "warp-rows"),
}

_BOOL_FIELDS = (
    "stage_overlap",
    "factor_overlap",
    "packet_fold_wave_barriers",
    "profile_appt_roles",
)

_NUMERIC_AXIS_ORDER = (
    "tile_threads",
    "threads_per_block",
    "prefix_threads",
    "suffix_threads",
    "prefix_ept",
    "suffix_ept",
    "factor_ept",
    "factor_columns",
    "units_per_cta",
    "data_space",
    "data_time",
    "role_stages",
    "stage_space",
    "flow_tile_log_n",
    "local_stages",
    "reorder_columns",
    "warp_stages",
    "pipeline_warps",
    "pipeline_buffers",
    "target_ctas_per_sm",
    "token_interleave",
    "n1_log",
    "rows_per_block",
    "ready_window",
    "data_tiles_per_cta",
    "prefetch_depth",
    "factor_slices",
    "batch_tile_count",
    "coefficient_reuse_stages",
    "cta_weight",
    "producer_weight",
    "tail_weight",
    "writer_weight",
    "fragment_width",
    "writer_tiles_per_cta",
    "data_time_role_mask",
)

_ENUM_AXIS_ORDER = (
    "compute_unit",
    "fft_core",
    "complex_multiply",
    "cross_twiddle",
    "cross_twiddle_placement",
    "direct_boundary",
    "local_exchange",
    "shared_layout",
    "stage_handoff",
    "hierarchical_core",
    "dataflow_layout",
    "dataflow_state_mode",
    "modular_multiply",
    "packet_readiness_mode",
    "packet_compute_layout",
    "prefix_codelet",
    "prefix_shared_layout",
)

_DERIVED_MAPPING_FIELDS = {
    "segment_mappings",
    "subgraph_mappings",
    "execution_group_mappings",
    "boundary_mappings",
    "boundaries",
    "segment_threads",
    "segment_ept",
    "segment_cores",
    "group_threads",
    "group_ept",
    "group_cores",
    "boundary_twiddles",
    "boundary_layouts",
    "boundary_residencies",
    "stages_per_decomposition",
    "execution_stage_partition",
}

_DERIVED_SENSITIVE_AXES = {
    "compute_unit",
    "fft_core",
    "complex_multiply",
    "cross_twiddle",
    "cross_twiddle_placement",
    "direct_boundary",
    "local_exchange",
    "shared_layout",
    "stage_handoff",
    "hierarchical_core",
    "dataflow_layout",
    "dataflow_state_mode",
    "modular_multiply",
    "packet_readiness_mode",
    "packet_compute_layout",
    "tile_threads",
    "threads_per_block",
    "prefix_threads",
    "suffix_threads",
    "prefix_ept",
    "suffix_ept",
    "prefix_codelet_lanes",
    "prefix_codelet",
    "prefix_shared_layout",
    "local_stage_partitions",
    "exchange_chunks",
    "factor_io_policies",
    "factor_ept",
    "factor_columns",
    "units_per_cta",
    "data_space",
    "data_time",
    "role_stages",
    "stage_space",
    "flow_tile_log_n",
    "local_stages",
    "reorder_columns",
    "warp_stages",
    "pipeline_warps",
    "pipeline_buffers",
    "target_ctas_per_sm",
    "token_interleave",
    "n1_log",
    "rows_per_block",
    "ready_window",
    "data_tiles_per_cta",
    "prefetch_depth",
    "factor_slices",
    "coefficient_reuse_stages",
    "cta_weight",
    "producer_weight",
    "tail_weight",
    "writer_weight",
    "fragment_width",
    "writer_tiles_per_cta",
    "data_time_role_mask",
}


def _as_int(value: Any) -> int | None:
    """Return a strict positive integer representation when possible."""

    if isinstance(value, bool):
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if isinstance(value, float) and value != integer:
        return None
    if isinstance(value, str) and str(integer) != value.strip():
        return None
    return integer


def _bool_value(value: Any) -> bool | int | str:
    """Toggle a JSON boolean while retaining legacy 0/1/string spelling."""

    if isinstance(value, bool):
        return not value
    if isinstance(value, int) and value in (0, 1):
        return 1 - value
    if isinstance(value, str) and value.strip().lower() in {"0", "1", "true", "false"}:
        lowered = value.strip().lower()
        if lowered in {"1", "true"}:
            return "false" if lowered == "true" else "0"
        return "true" if lowered == "false" else "1"
    return not bool(value)


def _mapping_from_point(point: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(point, Mapping):
        raise TypeError("point must be a mapping")
    raw = point.get("mapping_json")
    if isinstance(raw, str):
        try:
            mapping = json.loads(raw)
        except (TypeError, ValueError) as error:
            raise ValueError("point mapping_json must be valid JSON") from error
    elif isinstance(raw, Mapping):
        # Accepting this is convenient for callers constructing a point, while
        # all returned points still use the public string representation.
        mapping = copy.deepcopy(dict(raw))
    else:
        raise ValueError("point requires mapping_json containing a v1 mapping")
    if not isinstance(mapping, dict) or mapping.get("schema_version") != 1:
        raise ValueError("point mapping_json must contain schema_version=1")
    if not isinstance(mapping.get("backend"), str) or not mapping["backend"]:
        raise ValueError("point mapping_json must contain a backend")
    return mapping


def _workload(point: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in point.items()
        if key not in _MAPPING_FIELDS and key not in _NON_SEMANTIC_FIELDS
    }


def _operator(point: Mapping[str, Any], mapping: Mapping[str, Any]) -> str:
    operator = str(point.get("operator", ""))
    if operator == "ntt" or mapping.get("kind") == "ntt":
        return "ntt"
    return "butterfly"


def _log_n(point: Mapping[str, Any]) -> int | None:
    for key in ("logN", "log_n"):
        if key in point:
            value = _as_int(point[key])
            if value is not None and value > 0:
                return value
    if "N" in point:
        value = _as_int(point["N"])
        if value is not None and value > 1 and value & (value - 1) == 0:
            return value.bit_length() - 1
    return None


def _partition(value: Any, delimiter: str = "x") -> list[int] | None:
    if isinstance(value, (tuple, list)):
        values = list(value)
    elif isinstance(value, str):
        values = value.split(delimiter)
    else:
        return None
    result = []
    for item in values:
        number = _as_int(item)
        if number is None or number < 1:
            return None
        result.append(number)
    return result or None


def _stage_partition(point: Mapping[str, Any], mapping: Mapping[str, Any]) -> list[int] | None:
    value = mapping.get("stage_partition")
    if value is None:
        value = mapping.get("stages_per_decomposition")
    delimiter = "+" if str(mapping.get("kind", "")) == "ntt" else "x"
    result = _partition(value, delimiter)
    log_n = _log_n(point)
    if result is None or log_n is None or sum(result) != log_n:
        return None
    return result


def _physical_stage_partition(point: Mapping[str, Any], mapping: Mapping[str, Any]) -> list[int] | None:
    """Resolve shared butterfly resident groups after fused boundaries."""
    logical = _stage_partition(point, mapping)
    if logical is None or mapping.get("backend") != "shared-iterative":
        return logical
    operator = ("ntt" if point.get("operator") == "ntt" or mapping.get("kind") == "ntt"
                else str(point.get("operator", mapping.get("operator", "fft"))))
    try:
        return _shared_partition({**point, **mapping, "stage_partition": logical}, operator,
                                 _log_n(point) or sum(logical))
    except (TypeError, ValueError, KeyError):
        return None


def _macro_endpoints(partition: Sequence[int]) -> set[int]:
    total = 0
    result = set()
    for stages in partition[:-1]:
        total += stages
        result.add(total)
    return result


def _valid_partition(partition: Sequence[int], total: int | None) -> bool:
    return bool(partition) and all(isinstance(value, int) and value > 0 for value in partition) and (
        total is None or sum(partition) == total
    )


def _factor_partition(point: Mapping[str, Any], mapping: Mapping[str, Any], stages: Sequence[int] | None) -> list[int] | None:
    result = _partition(mapping.get("factor_partition"), "x")
    log_n = _log_n(point)
    if result is None or log_n is None or sum(result) != log_n:
        return None
    if stages is not None:
        endpoints = _macro_endpoints(stages)
        cumulative = 0
        for factor in result[:-1]:
            cumulative += factor
            if cumulative in endpoints:
                continue
        # Every macro endpoint must be materialized by a factor endpoint.  A
        # factor may not straddle an endpoint, which is the relevant check.
        cumulative = 0
        factor_endpoints = set()
        for factor in result[:-1]:
            cumulative += factor
            factor_endpoints.add(cumulative)
        if not endpoints <= factor_endpoints:
            return None
    return result


def _resize_boundary_policy(entries: Any, old_partition: Sequence[int], new_partition: Sequence[int], operation: tuple[str, int] | None) -> list[Any] | None:
    """Resize explicit boundary policies without inventing an empty policy."""

    if not isinstance(entries, list) or not entries:
        return None
    old_count = max(0, len(old_partition) - 1)
    new_count = max(0, len(new_partition) - 1)
    if new_count == 0:
        return []
    source = entries[:old_count]
    if not source:
        return None
    kind, index = operation or ("move", -1)
    resized: list[Any] = []
    for boundary in range(new_count):
        if kind == "split":
            if boundary < index:
                source_index = boundary
            elif boundary == index:
                source_index = min(index, len(source) - 1)
            else:
                source_index = min(boundary - 1, len(source) - 1)
        elif kind == "merge":
            if boundary < index:
                source_index = boundary
            else:
                source_index = min(boundary + 1, len(source) - 1)
        else:
            source_index = min(boundary, len(source) - 1)
        resized.append(copy.deepcopy(source[source_index]))
    return resized


def _reset_derived(mapping: dict[str, Any], old_partition: Sequence[int], new_partition: Sequence[int], operation: tuple[str, int] | None, *, preserve_boundaries: bool = True) -> None:
    explicit_boundaries: dict[str, list[Any] | None] = {}
    for key in ("boundaries", "boundary_mappings"):
        if key in mapping:
            explicit_boundaries[key] = _resize_boundary_policy(mapping[key], old_partition, new_partition, operation)

    for key in _DERIVED_MAPPING_FIELDS:
        mapping.pop(key, None)

    if preserve_boundaries:
        for key, value in explicit_boundaries.items():
            # An empty serialized boundary array is generated metadata.  Drop
            # it so the runtime can derive the correct count for the mutation.
            if value:
                mapping[key] = value


def _register_prefix_geometry(point: Mapping[str, Any], mapping: Mapping[str, Any],
                              partition: Sequence[int] | None = None) -> tuple[int, int, int, int] | None:
    """Return ``(G, R, Columns, EPT)`` for a valid register prefix.

    Search proposals are allowed to be rejected by the runtime, but a lane
    mutation must not manufacture a truncated column count or stale EPT.  This
    helper therefore returns ``None`` for an incomplete/invalid parent and is
    used only to construct geometry-preserving neighbours.
    """
    if mapping.get("backend") != "online-reorder" or mapping.get("fft_core") != "register-tile":
        return None
    stages = list(partition) if partition is not None else _stage_partition(point, mapping)
    if not stages or len(stages) < 1:
        return None
    merged = {**point, **mapping}
    merged.setdefault("local_stages", stages[0])
    try:
        geometry = resident_mapping.register_geometry(merged, stages)
    except (TypeError, ValueError, KeyError):
        return None
    return (int(geometry["prefix_lanes"]), int(geometry["a_size"]),
            int(geometry["prefix_columns"]), int(geometry["prefix_ept"]))


def _revalidate_register_prefix_geometry(mapping: dict[str, Any], point: Mapping[str, Any],
                                         old_partition: Sequence[int],
                                         new_partition: Sequence[int]) -> None:
    """Keep a register prefix's lane geometry legal after a stage edit."""
    if mapping.get("backend") != "online-reorder" or mapping.get("fft_core") != "register-tile":
        return
    # The standalone register lowering has exactly one prefix and one suffix.
    # Other structural proposals remain visible to the search/runtime contract,
    # but there is no prefix geometry to repair until that shape is restored.
    if len(new_partition) != 2 or len(old_partition) < 1:
        return
    old = _register_prefix_geometry(point, mapping, old_partition)
    if old is None:
        return
    old_lanes, _, columns, _ = old
    new_local = _as_int(new_partition[0])
    if new_local is None or new_local <= 0:
        return
    radius = 1 << (new_local // 2)
    legal = []
    for exponent in range(radius.bit_length()):
        lanes = 1 << exponent
        if lanes > radius:
            break
        if lanes > 1 and lanes * columns > 32:
            continue
        if radius * lanes * columns > 1024:
            continue
        legal.append(lanes)
    if not legal:
        return
    lanes = old_lanes if old_lanes in legal else max(value for value in legal if value <= old_lanes)
    mapping["prefix_threads"] = radius * lanes * columns
    mapping["prefix_ept"] = radius // lanes
    if lanes != 1 or "prefix_codelet_lanes" in mapping:
        mapping["prefix_codelet_lanes"] = lanes
    else:
        mapping.pop("prefix_codelet_lanes", None)


def _structural_mapping(point: Mapping[str, Any], base: Mapping[str, Any], partition: Sequence[int], old_partition: Sequence[int], operation: tuple[str, int], *, factor_partition: Sequence[int] | None = None) -> dict[str, Any]:
    mapping = copy.deepcopy(dict(base))
    mapping["stage_partition"] = list(partition)
    # A factor partition is tied to macro endpoints.  A stage edit changes
    # those endpoints, so let runtime resolution choose a compatible physical
    # factorization unless a caller explicitly supplies a new one.
    if mapping.get("backend") == "factor-streamed":
        if factor_partition is None:
            mapping.pop("factor_partition", None)
        else:
            mapping["factor_partition"] = list(factor_partition)
    _reset_derived(mapping, old_partition, partition, operation)
    _revalidate_register_prefix_geometry(mapping, point, old_partition, partition)
    return mapping


def _append_candidate(out: list[dict[str, Any]], seen: set[str], point: Mapping[str, Any], mapping: Mapping[str, Any]) -> None:
    if len(out) >= MAX_NEIGHBORS:
        return
    try:
        stages = (_physical_stage_partition(point, mapping)
                  if mapping.get("backend") == "shared-iterative"
                  else _stage_partition(point, mapping))
        backend = mapping.get("backend")
        operator = _operator(point, mapping)
        if stages and backend == "shared-iterative":
            resident_mapping.validate_resident_axes(mapping, stages,
                                                    backend=backend, operator=operator)
        elif (stages and backend == "online-reorder" and
              mapping.get("fft_core") == "register-tile"):
            merged = {**point, **mapping}
            if mapping.get("stage_partition"):
                merged["local_stages"] = stages[0]
            resident_mapping.register_geometry(merged, stages)
        elif resident_mapping.resident_requested(mapping):
            raise ValueError("resident axes are unsupported for this lowering")
        candidate = mapping_point(_workload(point), dict(mapping))
        identifier = candidate_id(candidate)
    except (TypeError, ValueError, KeyError):
        # Candidate generation must remain a proposal mechanism.  A malformed
        # optional axis is skipped; the runtime remains the legality oracle.
        return
    if identifier in seen:
        return
    seen.add(identifier)
    out.append(candidate)


def _numeric_values(field: str, current: int, point: Mapping[str, Any], mapping: Mapping[str, Any]) -> list[int]:
    log_n = _log_n(point) or max(current, 1)
    if field in {"tile_threads", "threads_per_block", "prefix_threads", "suffix_threads"}:
        choices = (32, 64, 128, 256, 512, 1024)
    elif field in {"prefix_ept", "suffix_ept", "factor_ept"}:
        choices = (1, 2, 4, 8, 16, 32, 64)
    elif field in {"factor_columns", "units_per_cta", "pipeline_warps", "pipeline_buffers"}:
        choices = (1, 2, 4, 8, 16, 32)
    elif field in {"data_tiles_per_cta", "prefetch_depth", "factor_slices", "batch_tile_count"}:
        choices = (0, 1, 2, 4, 8, 16, 32)
    elif field in {"local_stages", "stage_space", "flow_tile_log_n", "n1_log", "rows_per_block", "role_stages", "warp_stages"}:
        choices = tuple(range(1, min(log_n, 32) + 1))
    else:
        choices = (1, 2, 4, 8, 16, 32, 64, 128, 256)

    values: list[int] = []
    # Discrete launch axes use known legal choices.  Stage/data-time axes also
    # get arithmetic neighbours because they are ordered positive dimensions.
    discrete = field in {
        "tile_threads", "threads_per_block", "prefix_threads", "suffix_threads",
        "prefix_ept", "suffix_ept", "factor_ept", "factor_columns", "units_per_cta",
        "pipeline_warps", "pipeline_buffers", "data_tiles_per_cta", "prefetch_depth",
        "factor_slices", "batch_tile_count",
    }
    candidates = choices if discrete else (current - 1, current + 1, current // 2, current * 2, *choices)
    for value in candidates:
        if value < 0:
            continue
        if field not in {"prefetch_depth", "data_tiles_per_cta", "factor_slices", "batch_tile_count"} and value < 1:
            continue
        if field in {"local_stages", "stage_space", "flow_tile_log_n", "n1_log", "rows_per_block", "role_stages", "warp_stages"} and value > log_n:
            continue
        if field == "batch_tile_count":
            batch = _as_int(point.get("batch", 1)) or 1
            if value < 1 or value > batch:
                continue
        if field == "prefetch_depth" and value > 8:
            continue
        if field in {"factor_slices", "data_tiles_per_cta"} and value < 1:
            continue
        if value != current and value not in values:
            values.append(value)
    return values


def _axis_mapping(point: Mapping[str, Any], mapping: Mapping[str, Any], field: str, value: Any) -> dict[str, Any]:
    mutated = copy.deepcopy(dict(mapping))
    mutated[field] = value
    if field in _DERIVED_SENSITIVE_AXES:
        stages = _stage_partition(point, mapping) or []
        _reset_derived(mutated, stages, stages, ("move", -1))
    return mutated


def _enum_values(field: str, current: Any, backend: str, operator: str) -> list[Any]:
    if field in {"prefix_codelet", "prefix_shared_layout"}:
        # These choices describe only the online register-tile prefix.  Do
        # not propose them for imported or generic lowerings where the axes
        # are not consumed by the compiler.
        if backend != "online-reorder" or str(current) not in {
                "native", "cufftdx-thread", "linear", "xor"}:
            return []
        if field == "prefix_codelet":
            return [value for value in ("native", "cufftdx-thread") if value != current]
        return [value for value in ("linear", "xor") if value != current]
    if field == "fft_core":
        choices = _BUTTERFLY_CORE_CHOICES.get(backend, (str(current),))
    elif field == "shared_layout":
        if backend == "factor-streamed":
            choices = ("writer-aligned",)
        elif backend == "shared-iterative":
            choices = ("linear", "writer-aligned")
        else:
            choices = _SHARED_LAYOUTS
    elif field == "compute_unit":
        choices = ("radix2", "radix4", "radix8")
    elif field == "complex_multiply":
        choices = ("four-mul", "gauss3")
    elif field == "cross_twiddle":
        choices = ("table", "recurrence")
    elif field == "direct_boundary":
        choices = ("direct-strided", "tiled-transpose", "prefix-tiled-transpose")
    elif field == "local_exchange":
        choices = ("shared", "warp-register")
    else:
        choices = _NTT_ENUM_CHOICES.get(field, (current,))
    return [value for value in choices if value != current]


def _factor_io_policy_neighbors(point: Mapping[str, Any], mapping: Mapping[str, Any],
                                stages: Sequence[int] | None, factors: Sequence[int]) -> Iterable[dict[str, Any]]:
    """Toggle one physical factor's I/O lowering without losing stage identity.

    An empty policy list is the public default (all dynamic).  Once a policy
    axis is explored, retain an explicit per-factor list even when the result
    happens to contain only ``dynamic`` entries: collapsing it back to ``[]``
    would erase the mutation from candidate identity and from calibration
    provenance.
    """
    if mapping.get("backend") != "factor-streamed":
        return
    raw = mapping.get("factor_io_policies", [])
    if raw in (None, ""):
        raw = []
    if not isinstance(raw, list):
        return
    if raw and len(raw) != len(factors):
        return
    current = list(raw) if raw else ["dynamic"] * len(factors)
    if any(value not in {"dynamic", "static-unrolled"} for value in current):
        return
    for index, value in enumerate(current):
        mutated = copy.deepcopy(dict(mapping))
        policies = list(current)
        policies[index] = "static-unrolled" if value == "dynamic" else "dynamic"
        mutated["factor_io_policies"] = policies
        # The physical group descriptors are compiler-derived.  Keep the
        # factor partition and all other per-stage choices intact while
        # forcing a fresh descriptor for this policy variant.
        if stages is not None:
            _reset_derived(mutated, stages, stages, ("move", -1))
        yield mutated


def _resident_partition_neighbors(entry: Sequence[int], *, limit: int = 8) -> list[list[int]]:
    """Return one-operation resident-unit neighbours for one physical group.

    A candidate changes one leaf or one adjacent pair only.  The per-operation
    caps keep a long stage group from dominating the evolutionary population,
    while retaining several representatives of every legal operation class.
    """
    if not entry or any(value < 1 or value > 5 for value in entry):
        return []
    split: list[list[int]] = []
    merge: list[list[int]] = []
    move: list[list[int]] = []

    for index, stages in enumerate(entry):
        for cut in range(1, stages):
            split.append(list(entry[:index]) + [cut, stages - cut] + list(entry[index + 1:]))
    for index in range(len(entry) - 1):
        if entry[index] + entry[index + 1] <= 5:
            merge.append(list(entry[:index]) + [entry[index] + entry[index + 1]] +
                         list(entry[index + 2:]))
    for index, (left, right) in enumerate(zip(entry, entry[1:])):
        if left > 1 and right < 5:
            move.append(list(entry[:index]) + [left - 1, right + 1] + list(entry[index + 2:]))
        if right > 1 and left < 5:
            move.append(list(entry[:index]) + [left + 1, right - 1] + list(entry[index + 2:]))

    result: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for variants in (split[:limit], merge[:limit], move[:limit]):
        for variant in variants:
            key = tuple(variant)
            if key not in seen:
                seen.add(key)
                result.append(variant)
    return result


def _resident_axis_neighbors(point: Mapping[str, Any], mapping: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Propose bounded resident-factor and output-chunk variants."""
    backend = mapping.get("backend")
    stages = (_physical_stage_partition(point, mapping)
              if backend == "shared-iterative" else _stage_partition(point, mapping))
    if not stages:
        return
    if backend == "shared-iterative":
        try:
            parts, chunks = resident_mapping.normalize_axes(mapping)
        except ValueError:
            return
        base_parts = ([list(entry) for entry in parts] if parts else
                      [[] for _ in stages])
        base_chunks = list(chunks) if chunks else [0] * len(stages)
        seen: set[tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]] = set()

        def candidate(candidate_parts: Sequence[Sequence[int]],
                      candidate_chunks: Sequence[int] = base_chunks) -> dict[str, Any] | None:
            key = (tuple(tuple(entry) for entry in candidate_parts), tuple(candidate_chunks))
            if key in seen:
                return None
            seen.add(key)
            mutated = copy.deepcopy(dict(mapping))
            mutated["local_stage_partitions"] = [list(entry) for entry in candidate_parts]
            mutated["exchange_chunks"] = list(candidate_chunks)
            return mutated

        # An omitted or all-empty axis starts from the representative seed.
        # Build local neighbours from that seed too, so a search can reach
        # mixed resident partitions without changing the global stage split.
        if not any(base_parts):
            seeded = resident_mapping.resident_partition_seed(stages)
            seeded_candidate = candidate(seeded)
            if seeded_candidate is not None:
                yield seeded_candidate
            base_parts = seeded

        # Enable one previously opaque physical group at a time.  This keeps
        # the outer list explicit while avoiding a Cartesian product.
        for index, entry in enumerate(base_parts):
            if entry:
                continue
            seeded = resident_mapping.resident_partition_seed([stages[index]])[0]
            variant = [list(item) for item in base_parts]
            variant[index] = seeded
            seeded_candidate = candidate(variant)
            if seeded_candidate is not None:
                yield seeded_candidate

        # Split/merge/move only one physical group in each proposal.
        for index, entry in enumerate(base_parts):
            for neighbour in _resident_partition_neighbors(entry):
                variant = [list(item) for item in base_parts]
                variant[index] = neighbour
                neighbour_candidate = candidate(variant)
                if neighbour_candidate is not None:
                    yield neighbour_candidate

        if any(base_parts):
            disabled = candidate([[] for _ in stages])
            if disabled is not None:
                yield disabled
        return
    if backend != "online-reorder" or mapping.get("fft_core") != "register-tile" or len(stages) != 2:
        return
    merged = {**point, **mapping}
    merged.setdefault("local_stages", stages[0])
    try:
        geometry = resident_mapping.register_geometry(merged, stages)
    except (TypeError, ValueError, KeyError):
        return
    local = int(geometry["local"])
    current = geometry["local_stage_partitions"]
    current_prefix = current[0] if current else []
    current_a = int(current_prefix[0]) if current_prefix else local // 2
    factor_logs: list[int] = []

    def add_factor(log_a: int) -> None:
        if 0 < log_a < local and log_a not in factor_logs:
            factor_logs.append(log_a)

    # Keep the parent's geometry close by, then cover all cheap legal factors
    # for small prefixes.  Runtime validation still filters warp/EPT limits.
    add_factor(current_a)
    add_factor(current_a - 1)
    add_factor(current_a + 1)
    add_factor(local // 2)
    add_factor(1)
    add_factor(local - 1)
    if local <= 11:
        for log_a in range(1, local):
            add_factor(log_a)

    seen = set()
    columns = int(geometry["prefix_columns"])
    base_ept = int(geometry["prefix_ept"])
    for log_a in factor_logs:
        log_b = local - log_a
        if (log_a, log_b) in seen:
            continue
        seen.add((log_a, log_b))
        first = 1 << log_a
        second = 1 << log_b
        ept_candidates = [1 << exponent for exponent in range(min(log_a, log_b) + 1)]
        base_ept_log = base_ept.bit_length() - 1
        ept_candidates.sort(key=lambda value: (abs(value.bit_length() - 1 - base_ept_log), -value))
        column_candidates = []
        candidate_columns = columns
        while candidate_columns:
            column_candidates.append(candidate_columns)
            candidate_columns //= 2
        selected = None
        for ept in ept_candidates:
            lanes = first // ept
            for candidate_columns in column_candidates:
                prefix_threads = (1 << local) * candidate_columns // ept
                # CUDA permits smaller blocks, but resident prefix codelets
                # are warp-oriented; keep proposals in the launcher's normal
                # full-warp range while retaining the parent column ceiling.
                if prefix_threads < 32 or prefix_threads > 1024:
                    continue
                mutated = copy.deepcopy(dict(mapping))
                mutated["local_stage_partitions"] = [[log_a, log_b], []]
                mutated["prefix_ept"] = ept
                mutated["prefix_threads"] = prefix_threads
                if lanes == 1 and "prefix_codelet_lanes" not in mutated:
                    mutated.pop("prefix_codelet_lanes", None)
                else:
                    mutated["prefix_codelet_lanes"] = lanes
                try:
                    resident_mapping.register_geometry(
                        {**point, **mutated, "local_stages": local}, stages)
                except (TypeError, ValueError, KeyError):
                    continue
                selected = mutated
                break
            if selected is not None:
                break
        if selected is not None:
            yield selected

    suffix_ept = int(geometry["suffix_ept"])
    suffix_columns = int(geometry["suffix_columns"])
    if suffix_columns > 1:
        current_chunks = geometry["exchange_chunks"]
        current_chunk = int(current_chunks[1]) if current_chunks else 0
        for chunk in (0, 1, 2, 4, 8, 16, 32):
            if chunk == current_chunk:
                continue
            if chunk and (chunk > suffix_ept or suffix_ept % chunk or chunk & (chunk - 1)):
                continue
            mutated = copy.deepcopy(dict(mapping))
            mutated["exchange_chunks"] = [0, chunk]
            yield mutated


def _factor_legal(mapping: Mapping[str, Any], factors: Sequence[int], stages: Sequence[int] | None, point: Mapping[str, Any]) -> bool:
    if not factors or any(value < 1 for value in factors):
        return False
    log_n = _log_n(point)
    if log_n is None or sum(factors) != log_n:
        return False
    if stages is not None and not _macro_endpoints(stages) <= set(itertools.accumulate(factors)):
        return False
    if mapping.get("fft_core") == "register-tile":
        # The native core currently uses one common EPT and requires even-log
        # factors.  Keep proposals in that legal family; runtime still checks
        # resource limits and exact codelet availability.
        if len(set(factors)) != 1 or factors[0] % 2:
            return False
    return True


def _native_factor_ept(mapping: dict[str, Any], factors: Sequence[int]) -> None:
    if mapping.get("fft_core") == "register-tile" and factors and factors[0] % 2 == 0:
        mapping["factor_ept"] = 1 << (factors[0] // 2)


def _prefix_lane_neighbors(point: Mapping[str, Any], mapping: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Vary cooperative prefix lanes while preserving physical Columns."""
    stages = _stage_partition(point, mapping)
    geometry = _register_prefix_geometry(point, mapping, stages)
    if geometry is None:
        return
    current, radius, columns, _ = geometry
    for exponent in range(radius.bit_length()):
        lanes = 1 << exponent
        if lanes == current or (lanes > 1 and lanes * columns > 32):
            continue
        mutated = copy.deepcopy(dict(mapping))
        mutated["prefix_codelet_lanes"] = lanes
        mutated["prefix_threads"] = radius * lanes * columns
        mutated["prefix_ept"] = radius // lanes
        if mutated["prefix_threads"] > 1024:
            continue
        _reset_derived(mutated, stages or (), stages or (), ("move", -1))
        yield mutated


def _stage_neighbors(point: Mapping[str, Any], mapping: Mapping[str, Any], stages: Sequence[int]) -> Iterable[dict[str, Any]]:
    total = sum(stages)
    for index, value in enumerate(stages):
        if value >= 2:
            split = value // 2
            if split == value:
                split = value - 1
            new = list(stages[:index]) + [split, value - split] + list(stages[index + 1:])
            yield _structural_mapping(point, mapping, new, stages, ("split", index))
    for index in range(len(stages) - 1):
        new = list(stages[:index]) + [stages[index] + stages[index + 1]] + list(stages[index + 2:])
        yield _structural_mapping(point, mapping, new, stages, ("merge", index))
    for index in range(len(stages) - 1):
        if stages[index] > 1:
            new = list(stages)
            new[index] -= 1
            new[index + 1] += 1
            yield _structural_mapping(point, mapping, new, stages, ("move", index))
        if stages[index + 1] > 1:
            new = list(stages)
            new[index] += 1
            new[index + 1] -= 1
            yield _structural_mapping(point, mapping, new, stages, ("move", index))


def _factor_neighbors(point: Mapping[str, Any], mapping: Mapping[str, Any], stages: Sequence[int] | None, factors: Sequence[int]) -> Iterable[dict[str, Any]]:
    endpoints = _macro_endpoints(stages or ())

    def candidate(new_factors: Sequence[int]) -> dict[str, Any]:
        mutated = copy.deepcopy(dict(mapping))
        mutated["factor_partition"] = list(new_factors)
        # Factor boundaries are structural launches too.  The macro partition
        # is unchanged, so explicit non-empty boundary policies can be kept,
        # while all derived unit/group descriptors are rebuilt by the runtime.
        if stages is not None:
            _reset_derived(mutated, stages, stages, ("move", -1))
        return mutated

    for index, value in enumerate(factors):
        if value >= 2:
            split = value // 2
            new = list(factors[:index]) + [split, value - split] + list(factors[index + 1:])
            if _factor_legal(mapping, new, stages, point):
                yield candidate(new)
    for index in range(len(factors) - 1):
        boundary = sum(factors[:index + 1])
        if boundary in endpoints:
            continue
        new = list(factors[:index]) + [factors[index] + factors[index + 1]] + list(factors[index + 2:])
        if _factor_legal(mapping, new, stages, point):
            yield candidate(new)
    running = 0
    for index in range(len(factors) - 1):
        running += factors[index]
        if running in endpoints:
            continue
        if factors[index] > 1:
            new = list(factors)
            new[index] -= 1
            new[index + 1] += 1
            if _factor_legal(mapping, new, stages, point):
                yield candidate(new)
        if factors[index + 1] > 1:
            new = list(factors)
            new[index] += 1
            new[index + 1] -= 1
            if _factor_legal(mapping, new, stages, point):
                yield candidate(new)

    # Native register tiles have equal factors, so count changes are the useful
    # local mutation.  Try only nearby counts; no composition enumeration.
    if mapping.get("fft_core") == "register-tile" and len(set(factors)) == 1:
        total = sum(factors)
        for count in (len(factors) - 2, len(factors) - 1, len(factors) + 1, len(factors) + 2):
            if count < 1 or total % count:
                continue
            new = [total // count] * count
            if _factor_legal(mapping, new, stages, point):
                mutated = candidate(new)
                _native_factor_ept(mutated, new)
                yield mutated


def _structural_candidates(point: Mapping[str, Any], mapping: Mapping[str, Any], operator: str) -> Iterable[dict[str, Any]]:
    stages = _stage_partition(point, mapping)
    if stages:
        yield from _stage_neighbors(point, mapping, stages)
    if mapping.get("backend") == "factor-streamed":
        factors = _factor_partition(point, mapping, stages)
        if factors:
            yield from _factor_neighbors(point, mapping, stages, factors)
            yield from _factor_io_policy_neighbors(point, mapping, stages, factors)


def _strategy_candidates(point: Mapping[str, Any], mapping: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield implementation-strategy mutations before structural neighbours.

    Structural partition neighbourhoods can be large.  Keeping these
    compiler/codelet axes first guarantees a bounded search still sees each
    physically distinct lowering at least once.
    """
    backend = str(mapping.get("backend", ""))
    yield from _resident_axis_neighbors(point, mapping)
    if backend == "online-reorder" and mapping.get("fft_core") == "register-tile":
        yield from _prefix_lane_neighbors(point, mapping)
        codelet = mapping.get("prefix_codelet", "native")
        layout = mapping.get("prefix_shared_layout", "linear")
        if codelet == "native" and point.get("precision") != "fp64":
            mutated = copy.deepcopy(dict(mapping)); mutated["prefix_codelet"] = "cufftdx-thread"
            _reset_derived(mutated, _stage_partition(point, mapping) or (),
                           _stage_partition(point, mapping) or (), ("move", -1))
            yield mutated
        elif codelet == "cufftdx-thread":
            mutated = copy.deepcopy(dict(mapping)); mutated["prefix_codelet"] = "native"
            _reset_derived(mutated, _stage_partition(point, mapping) or (),
                           _stage_partition(point, mapping) or (), ("move", -1))
            yield mutated
        if layout == "linear":
            mutated = copy.deepcopy(dict(mapping)); mutated["prefix_shared_layout"] = "xor"
            _reset_derived(mutated, _stage_partition(point, mapping) or (),
                           _stage_partition(point, mapping) or (), ("move", -1))
            yield mutated
        elif layout == "xor":
            mutated = copy.deepcopy(dict(mapping)); mutated["prefix_shared_layout"] = "linear"
            _reset_derived(mutated, _stage_partition(point, mapping) or (),
                           _stage_partition(point, mapping) or (), ("move", -1))
            yield mutated
    if backend == "factor-streamed":
        stages = _stage_partition(point, mapping)
        factors = _factor_partition(point, mapping, stages)
        if factors:
            yield from _factor_io_policy_neighbors(point, mapping, stages, factors)


def neighbors(point: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a deterministic bounded neighbourhood of ``point``.

    ``point`` must contain a public ``mapping_json`` string carrying a complete
    schema-v1 mapping.  Workload semantics (including NTT modulus, direction,
    orders, and strides) are copied unchanged.  The returned points are
    syntactic proposals only; no runtime or hardware assumptions are made.
    """

    mapping = _mapping_from_point(point)
    operator = _operator(point, mapping)
    backend = str(mapping.get("backend"))
    result: list[dict[str, Any]] = []
    seen: set[str] = set()

    for strategy in _strategy_candidates(point, mapping):
        _append_candidate(result, seen, point, strategy)
        if len(result) >= MAX_NEIGHBORS:
            return result

    for structural in _structural_candidates(point, mapping, operator):
        _append_candidate(result, seen, point, structural)
        if len(result) >= MAX_NEIGHBORS:
            return result

    # Change one independent numeric axis at a time.  This is intentionally
    # after structural representatives so long partitions cannot hide all
    # stage/factor neighbourhoods behind launch-axis variants.
    for field in _NUMERIC_AXIS_ORDER:
        if field not in mapping:
            continue
        if (backend == "online-reorder" and mapping.get("fft_core") == "register-tile"
                and field in {"prefix_threads", "prefix_ept"}):
            # These two fields are coupled to G and Columns.  Their legal
            # alternatives are emitted by _prefix_lane_neighbors instead of
            # being mutated independently into a truncated register shape.
            continue
        current = _as_int(mapping[field])
        if current is None:
            continue
        for value in _numeric_values(field, current, point, mapping)[:3]:
            mutated = _axis_mapping(point, mapping, field, value)
            if backend == "factor-streamed" and field == "factor_ept":
                factors = _factor_partition(point, mapping, _stage_partition(point, mapping))
                if factors and mapping.get("fft_core") == "register-tile" and value != (1 << (factors[0] // 2)):
                    continue
            _append_candidate(result, seen, point, mutated)
            if len(result) >= MAX_NEIGHBORS:
                return result

    for field in _ENUM_AXIS_ORDER:
        if field not in mapping:
            continue
        current = mapping[field]
        if field == "fft_core" and operator == "ntt":
            continue
        for value in _enum_values(field, current, backend, operator):
            mutated = _axis_mapping(point, mapping, field, value)
            if field in {"prefix_codelet", "prefix_shared_layout"}:
                if backend != "online-reorder" or mapping.get("fft_core") != "register-tile":
                    continue
                if field == "prefix_codelet" and value == "cufftdx-thread" and point.get("precision") == "fp64":
                    continue
            if backend == "factor-streamed" and field == "fft_core":
                factors = _factor_partition(point, mapping, _stage_partition(point, mapping))
                if factors and not _factor_legal(mutated, factors, _stage_partition(point, mapping), point):
                    continue
                if value == "register-tile":
                    _native_factor_ept(mutated, factors or ())
            _append_candidate(result, seen, point, mutated)
            if len(result) >= MAX_NEIGHBORS:
                return result

    for field in _BOOL_FIELDS:
        if field not in mapping:
            continue
        if field == "stage_overlap":
            stages = _stage_partition(point, mapping)
            batch = _as_int(point.get("batch", 1)) or 1
            if not stages or len(stages) < 2 or batch < 2:
                continue
        if field == "factor_overlap" and backend != "factor-streamed":
            continue
        mutated = copy.deepcopy(mapping)
        mutated[field] = _bool_value(mapping[field])
        _append_candidate(result, seen, point, mutated)
        if len(result) >= MAX_NEIGHBORS:
            return result

    return result


def _family_key(point: Mapping[str, Any]) -> tuple[str, str, str, str]:
    try:
        mapping = _mapping_from_point(point)
    except (TypeError, ValueError):
        return (str(point.get("operator", "")), str(point.get("backend", "")), "", "")
    lane = "1"
    if mapping.get("backend") == "online-reorder" and mapping.get("fft_core") == "register-tile":
        lane = str(mapping.get("prefix_codelet_lanes", 1))
    return (
        str(point.get("operator", mapping.get("kind", ""))),
        str(mapping.get("backend", "")),
        str(mapping.get("fft_core", mapping.get("compute_unit", ""))),
        lane,
    )


def _id(point: Mapping[str, Any]) -> str:
    try:
        canonical = mapping_point(_workload(point), _mapping_from_point(point))
    except (TypeError, ValueError, KeyError):
        canonical = dict(point)
    return candidate_id(canonical)


def _score_key(score: Callable[[dict[str, Any]], Any] | None, point: dict[str, Any]) -> tuple[int, int, float | str]:
    if score is None:
        return (1, 1, "")
    try:
        value = score(point)
    except Exception:  # model feedback is advisory; preserve deterministic search
        return (1, 1, "")
    if isinstance(value, bool):
        return (0, 0, float(value))
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return (0, 0, float(value))
    return (0, 1, str(value))


_WORKER_SCORE = None


def _initialize_score_worker(score):
    global _WORKER_SCORE
    _WORKER_SCORE = score
    # Scoring uses frozen CPU descriptors and service curves only.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _worker_score_key(point):
    return _score_key(_WORKER_SCORE, point)


def _inventory_score_keys(points, score):
    """Optionally score a frozen inventory in forked CPU workers.

    Opt-in only: the supplied scorer must be pure. Each pool inherits this
    call's model and descriptors, and is discarded before either can change.
    Ordered map retains the serial stable-sort and tie-breaking contract.
    """
    workers = int(os.environ.get("CUBUTTERFLY_SEARCH_SCORE_WORKERS", "1"))
    if workers < 1:
        raise ValueError("CUBUTTERFLY_SEARCH_SCORE_WORKERS must be positive")
    if (workers == 1 or score is None or len(points) < 128
            or "fork" not in multiprocessing.get_all_start_methods()
            or threading.active_count() != 1):
        return [_score_key(score, point) for point in points]
    workers = min(workers, len(points), os.cpu_count() or 1)
    context = multiprocessing.get_context("fork")
    # Pool's context manager terminates outstanding work on interruption;
    # no timing or partially scored selection is committed here.
    with context.Pool(workers, initializer=_initialize_score_worker, initargs=(score,)) as pool:
        return pool.map(_worker_score_key, points, chunksize=32)


def evolutionary_candidates(
    population: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    measured_ids: set[str],
    score: Callable[[dict[str, Any]], Any],
    limit: int = 64,
) -> list[dict[str, Any]]:
    """Build the next bounded candidate set from measured winners and seeds.

    ``population`` is already ordered by measured fitness; its first several
    entries are treated as winners without inventing or recomputing latency.
    ``inventory`` is ranked only through the caller-provided ``score`` model.
    Lower scores are preferred.  Exact inventory seeds and their neighbourhoods
    are both eligible, while measured IDs and duplicate points are removed.
    """

    if limit <= 0:
        return []
    if not isinstance(population, list) or not isinstance(inventory, list):
        raise TypeError("population and inventory must be lists")
    measured = set(measured_ids or ())
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set(measured)

    valid_population: list[dict[str, Any]] = []
    for parent in population:
        try:
            _mapping_from_point(parent)
            valid_population.append(parent)
            selected_ids.add(_id(parent))
        except (TypeError, ValueError):
            continue

    valid_inventory: list[dict[str, Any]] = []
    for seed in inventory:
        try:
            _mapping_from_point(seed)
            valid_inventory.append(seed)
        except (TypeError, ValueError):
            continue
    score_keys = _inventory_score_keys(valid_inventory, score)
    ranked_inventory = [seed for _, seed in sorted(
        zip(score_keys, valid_inventory), key=lambda item: (item[0], _id(item[1])))]

    # A small winner pool is enough to preserve several measured basins while
    # keeping the number of generated neighbourhoods bounded by ``limit``.
    winner_count = min(len(valid_population), max(2, min(8, limit // 4 or 1)))
    sources: list[tuple[dict[str, Any], bool]] = [(parent, False) for parent in valid_population[:winner_count]]
    seed_count = min(len(ranked_inventory), max(2, min(8, limit // 4 or 1)))
    sources.extend((seed, True) for seed in ranked_inventory[:seed_count])

    streams: list[list[dict[str, Any]]] = []
    for source, is_seed in sources:
        stream: list[dict[str, Any]] = []
        if is_seed:
            # Inventory entries may carry legacy top-level projections or JSON
            # whitespace.  Normalize exact seeds through the same public
            # mapping_point path as generated neighbours.
            stream.append(mapping_point(_workload(source), _mapping_from_point(source)))
        stream.extend(neighbors(source))
        streams.append(stream)

    # Family-diversity pass: choose one candidate from each source family before
    # filling the remaining budget in deterministic round-robin order.
    family_seen: set[tuple[str, str, str]] = set()
    for stream in streams:
        for candidate in stream:
            family = _family_key(candidate)
            if family in family_seen:
                continue
            identifier = _id(candidate)
            if identifier in selected_ids:
                continue
            family_seen.add(family)
            selected_ids.add(identifier)
            selected.append(candidate)
            break
        if len(selected) >= limit:
            return selected[:limit]

    position = 0
    while len(selected) < limit:
        exhausted = True
        for stream in streams:
            if position >= len(stream):
                continue
            exhausted = False
            candidate = stream[position]
            identifier = _id(candidate)
            if identifier not in selected_ids:
                selected_ids.add(identifier)
                selected.append(candidate)
                if len(selected) >= limit:
                    break
        if exhausted:
            break
        position += 1
    return selected[:limit]
