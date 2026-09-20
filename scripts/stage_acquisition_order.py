"""Deterministic acquisition ordering for physical-stage calibration points.

This module only changes the order in which points are visited.  A
SharedIterative point can move ahead of another point when it introduces a
new physical template fingerprint, but no point is removed and no coverage is
claimed.  All other backends retain their complete static identity and are
never pooled by this helper.
"""

from __future__ import annotations

import json
import heapq
import math
import re
from collections.abc import Iterable, Mapping
from typing import Any


_MISSING = object()
_SHARED_BACKENDS = {"shared-iterative", "shared_iterative"}
_FUSED_BOUNDARIES = {"fused", "resident-fused", "resident_fused"}

def _text(value: Any) -> str:
    return str(value).strip().lower().replace("_", "-")


def _number(value: Any) -> int | float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        # Preserve serialized uint64 fields (for example an NTT modulus)
        # without sending them through an imprecise binary float.
        if re.fullmatch(r"[+-]?\d+", text):
            try:
                return int(text)
            except ValueError:
                return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _canonical(value: Any) -> Any:
    """Return JSON-stable, hashable input without guessing missing values."""
    if value is _MISSING:
        return "<missing>"
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, set):
        return sorted((_canonical(item) for item in value), key=_json_key)
    if isinstance(value, str):
        text = value.strip()
        numeric = _number(text)
        return numeric if numeric is not None else text
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return repr(value)


def _json_key(value: Any) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), default=repr)


def _mapping(point: Mapping[str, Any]) -> dict[str, Any]:
    """Decode a serialized mapping without allowing it to mutate the point."""
    value = point.get("mapping_json", _MISSING)
    if isinstance(value, Mapping):
        return dict(value)
    if value is _MISSING or value in (None, ""):
        return {}
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return dict(decoded) if isinstance(decoded, Mapping) else {}


def _value(point: Mapping[str, Any], mapping: Mapping[str, Any], *names: str) -> Any:
    """Read serialized mapping fields first, then point-level fallbacks."""
    for source in (mapping, point):
        for name in names:
            if name in source and source[name] not in (None, ""):
                return source[name]
    return _MISSING


def _backend(point: Mapping[str, Any], mapping: Mapping[str, Any]) -> str:
    value = _value(point, mapping, "backend")
    return _text(value) if value is not _MISSING else "<missing>"


def _partition(value: Any) -> tuple[int, ...] | None:
    if value is _MISSING or value is None or value == "":
        return None
    if isinstance(value, str):
        value = value.replace("+", ",").split(",")
    if not isinstance(value, (list, tuple)):
        return None
    result: list[int] = []
    for item in value:
        parsed = _number(item)
        if parsed is None or isinstance(parsed, float) or parsed <= 0:
            return None
        result.append(int(parsed))
    return tuple(result) if result else None


def _log_n(point: Mapping[str, Any], mapping: Mapping[str, Any]) -> Any:
    value = _value(point, mapping, "logN", "log_n")
    if value is not _MISSING:
        parsed = _number(value)
        return parsed if parsed is not None else _canonical(value)
    # N is accepted by some calibration point producers.  Only derive logN
    # for an exact power of two; an ambiguous length stays distinct.
    value = _value(point, mapping, "N", "n")
    parsed = _number(value)
    if parsed is not None and isinstance(parsed, int) and parsed > 0 and parsed & (parsed - 1) == 0:
        return parsed.bit_length() - 1
    return "<missing>"


def _effective_partition(point: Mapping[str, Any], mapping: Mapping[str, Any], backend: str,
                        log_n: Any) -> tuple[int, ...] | None:
    partition = _partition(_value(point, mapping, "stage_partition", "stages_per_decomposition"))
    if partition is None:
        if backend in _SHARED_BACKENDS:
            # NTT resolves only flow_tile_log_n; Butterfly resolves only
            # local_stages.  Both lower to positive chunks of logN.
            is_ntt = (_text(_value(point, mapping, "kind")) == "ntt" or
                      _text(_value(point, mapping, "operator")) == "ntt")
            local = (_value(point, mapping, "flow_tile_log_n") if is_ntt else
                     _value(point, mapping, "local_stages"))
            local_number = _number(local)
            log_number = _number(log_n)
            if ((local is _MISSING or (is_ntt and (local_number is None or local_number <= 0))) and
                    log_number is not None and log_number > 0):
                # ButterflyConfig defaults local_stages to 10; NTT's shared
                # resolver defaults flow_tile_log_n to min(10, logN).
                local_number = min(10, int(log_number))
            if local_number is not None and log_number is not None and local_number > 0 and log_number > 0:
                remaining = int(log_number)
                tile = int(local_number)
                chunks: list[int] = []
                while remaining:
                    stages = min(remaining, tile)
                    chunks.append(stages)
                    remaining -= stages
                partition = tuple(chunks)
    if partition is None:
        return None

    # Only Butterfly SharedIterative supports logical segments joined by a
    # fused boundary.  NTT SharedIterative rejects all boundary mappings.
    if backend not in _SHARED_BACKENDS:
        return partition
    if (_text(_value(point, mapping, "kind")) == "ntt" or
            _text(_value(point, mapping, "operator")) == "ntt"):
        return partition
    boundaries = _value(point, mapping, "boundaries")
    if boundaries is _MISSING or not isinstance(boundaries, (list, tuple)):
        boundaries = ()
    physical: list[int] = []
    accumulated = 0
    for index, stages in enumerate(partition):
        accumulated += stages
        fused = False
        if index < len(partition) - 1 and index < len(boundaries):
            boundary = boundaries[index]
            if isinstance(boundary, Mapping):
                boundary = boundary.get("residency", _MISSING)
            fused = _text(boundary) in _FUSED_BOUNDARIES
        if not fused:
            physical.append(accumulated)
            accumulated = 0
    # An invalid/malformed boundary list must not silently erase the final
    # group.  Valid plans always end with zero accumulated stages here.
    if accumulated:
        physical.append(accumulated)
    return tuple(physical)


def _writer_aligned(point: Mapping[str, Any], mapping: Mapping[str, Any]) -> Any:
    value = _value(point, mapping, "writer_aligned")
    if value is not _MISSING:
        if isinstance(value, str):
            return _text(value) in {"1", "true", "yes", "on", "writer-aligned"}
        return bool(value)
    layout = _value(point, mapping, "shared_layout", "dataflow_layout")
    if layout is _MISSING:
        return "<missing>"
    return _text(layout) in {"writer-aligned", "hermes-xor", "hermesxor", "xor-swizzle"}


def _threads(point: Mapping[str, Any], mapping: Mapping[str, Any]) -> Any:
    value = _value(point, mapping, "threads", "tile_threads", "threads_per_block")
    parsed = _number(value)
    return parsed if parsed is not None else _canonical(value)


def _shared_template_material(point: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[int, ...]] | None:
    if not isinstance(point, Mapping):
        return None
    mapping = _mapping(point)
    backend = _backend(point, mapping)
    if backend not in _SHARED_BACKENDS:
        return None
    log_n = _log_n(point, mapping)
    partition = _effective_partition(point, mapping, backend, log_n)
    if partition is None:
        return None
    operator = _value(point, mapping, "operator")
    precision = _value(point, mapping, "precision")
    accumulation = _value(point, mapping, "accumulation")
    complex_multiply = _value(point, mapping, "complex_multiply")
    kind = _value(point, mapping, "kind")
    word_bits = _value(point, mapping, "word_bits")
    if (_text(operator) == "ntt" or _text(kind) == "ntt") and precision is _MISSING and word_bits is not _MISSING:
        parsed_word_bits = _number(word_bits)
        precision = (f"word{int(parsed_word_bits)}" if parsed_word_bits is not None
                     else f"word{_canonical(word_bits)}")
    explicit_batch_tile = _value(point, mapping, "batch_tile")
    batch_tile = 1 if explicit_batch_tile is _MISSING else _number(explicit_batch_tile)
    if batch_tile is None:
        batch_tile = _canonical(explicit_batch_tile)
    # Keep build/device provenance in the fingerprint.  Compiler version and
    # template bytes are folded into runtime_fingerprint by the probe; an
    # explicitly supplied compile fingerprint is retained as well.
    provenance = {
        name: _value(point, mapping, name)
        for name in ("runtime_fingerprint", "compile_fingerprint", "compute_capability", "sm", "gpu_uuid", "device")
    }
    payload = {
        "backend": backend,
        "kind": kind,
        "operator": operator,
        "precision": precision,
        "word_bits": word_bits,
        "accumulation": accumulation,
        "complex_multiply": complex_multiply,
        "logN": log_n,
        "threads": _threads(point, mapping),
        "writer_aligned": _writer_aligned(point, mapping),
        "batch_tile": batch_tile,
        "provenance": provenance,
    }
    return payload, partition


def shared_template_key(point: Mapping[str, Any]) -> str | None:
    """Return the conservative actual shared JIT module fingerprint.

    The C++ JIT adapter supplies operator/precision/accumulation, logN, the
    *physical* partition, threads and writer layout to ``render_shared``.
    ``batch_tile`` is included only when explicitly supplied because the
    current adapter hard-codes it to one; ``batch_tile_count`` is a runtime
    pipeline policy and is intentionally excluded.
    """
    material = _shared_template_material(point)
    if material is None:
        return None
    payload, partition = material
    return _json_key({**payload, "physical_partition": partition})


def shared_stage_template_keys(point: Mapping[str, Any]) -> tuple[str, ...]:
    """Return one physical kernel fingerprint per ``(first_stage, stages)``.

    A complete module's ordered partition remains part of
    :func:`shared_template_key`, while this finer identity reflects the
    static arguments used by each ``shared_iterative_kernel`` specialization.
    It is suitable only for acquisition ordering; it does not make whole-plan
    correctness, workspace, or composition interchangeable.
    """
    material = _shared_template_material(point)
    if material is None:
        return ()
    payload, partition = material
    first = 0
    result = []
    for stages in partition:
        result.append(_json_key({**payload, "first_stage": first, "stage_count": stages}))
        first += stages
    return tuple(result)


def static_identity_key(point: Any) -> str:
    """Return a complete conservative identity for non-SharedIterative data."""
    if not isinstance(point, Mapping):
        return _json_key(point)
    # Do not use this identity to merge points.  It is solely a stable sort
    # key, and retaining every field keeps unknown backend axes isolated.
    return _json_key(dict(point))


def _workload_rank(point: Any) -> tuple[int | float, int | float, str]:
    if not isinstance(point, Mapping):
        return (0, 0, "")
    mapping = _mapping(point)
    batch = _number(_value(point, mapping, "batch"))
    if batch is _MISSING or batch is None:
        batch = 1
    length = _number(_value(point, mapping, "N", "n"))
    if length is None:
        log_n = _number(_value(point, mapping, "logN", "log_n"))
        length = (1 << int(log_n)) if log_n is not None and int(log_n) >= 0 else 0
    if length is None:
        length = 0
    # The product is the useful work amount; the remaining fields make ties
    # deterministic without relying on object comparison.
    return (batch * length, batch, _json_key(point))


def order_points(points: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return all points in deterministic representative-first order.

    One index pass identifies complete module groups and their
    maximum-workload representative.  A lazy-gain heap then emits
    representatives that still add at least one not-yet-seen physical stage
    key before representatives with no new stage key.  A second pass emits
    every non-representative exactly once.  No backend is considered covered
    by another backend or by a representative.
    """
    records = list(points)
    shared_groups: dict[str, dict[str, Any]] = {}
    shared_representatives: set[int] = set()
    non_shared: list[tuple[str, int, Mapping[str, Any]]] = []
    point_template_keys: list[str | None] = [None] * len(records)

    for index, point in enumerate(records):
        key = shared_template_key(point)
        stage_keys = shared_stage_template_keys(point)
        if key is None or not stage_keys:
            non_shared.append((static_identity_key(point), index, point))
            continue
        point_template_keys[index] = key
        group = shared_groups.get(key)
        if group is None:
            shared_groups[key] = {"first": index, "representative": index,
                                  "workload": _workload_rank(point),
                                  "stage_keys": stage_keys}
            continue
        workload = _workload_rank(point)
        if workload[:2] > group["workload"][:2]:
            group["representative"] = index
            group["workload"] = workload

    representative_order = sorted(shared_groups.values(), key=lambda item: item["first"])
    stage_to_groups: dict[str, list[int]] = {}
    gains: list[int] = []
    heap: list[tuple[int, int, int]] = []
    for group_id, group in enumerate(representative_order):
        group["stage_keys"] = tuple(dict.fromkeys(group["stage_keys"]))
        keys = set(group["stage_keys"])
        gains.append(len(keys))
        for stage_key in group["stage_keys"]:
            stage_to_groups.setdefault(stage_key, []).append(group_id)
        heapq.heappush(heap, (-int(len(keys) > 0), group["first"], group_id))

    # Each stage-to-group incidence is decremented at most once, when that
    # stage key first becomes covered.  Positive-gain representatives stay
    # ahead of zero-gain representatives, while each bucket remains stable
    # by first occurrence.  Stale heap entries are discarded using the
    # incrementally maintained gain, keeping the complete traversal
    # subquadratic in the number of points.
    covered: set[str] = set()
    selected_groups: set[int] = set()
    selected_order: list[int] = []
    while heap:
        priority, _, group_id = heapq.heappop(heap)
        if group_id in selected_groups:
            continue
        current = representative_order[group_id]
        actual_priority = -int(gains[group_id] > 0)
        if actual_priority != priority:
            heapq.heappush(heap, (actual_priority, current["first"], group_id))
            continue
        selected_groups.add(group_id)
        selected_order.append(group_id)
        newly_covered = [stage_key for stage_key in current["stage_keys"] if stage_key not in covered]
        covered.update(newly_covered)
        for stage_key in newly_covered:
            for other_id in stage_to_groups.get(stage_key, ()):
                if other_id in selected_groups:
                    continue
                gains[other_id] -= 1
                heapq.heappush(heap, (-int(gains[other_id] > 0),
                                      representative_order[other_id]["first"], other_id))

    for group_id in selected_order:
        shared_representatives.add(representative_order[group_id]["representative"])
    result: list[Mapping[str, Any]] = [records[representative_order[group_id]["representative"]]
                                       for group_id in selected_order]

    # Keep the remaining Shared points in source order.  Non-Shared points are
    # sorted only by their complete static identity, with source index as a
    # deterministic tie breaker; neither path is deduplicated.
    for index, point in enumerate(records):
        if index not in shared_representatives and point_template_keys[index] is not None:
            result.append(point)
    for _, _, point in sorted(non_shared, key=lambda item: (item[0], item[1])):
        result.append(point)
    return result


__all__ = ["order_points", "shared_stage_template_keys", "shared_template_key", "static_identity_key"]
