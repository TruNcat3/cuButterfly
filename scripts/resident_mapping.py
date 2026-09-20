"""Shared validation and projections for resident mapping axes.

The public axes are intentionally small:
``local_stage_partitions`` is indexed by physical execution group and
``exchange_chunks`` is the output-ownership chunk for that group.  Empty
outer axes and empty entries retain the historical lowering.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if str(parsed) != str(value).strip() and not isinstance(value, int):
        try:
            if float(value) != parsed:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be an integer") from error
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return parsed


def _power_of_two(value: int) -> bool:
    return value > 0 and not value & (value - 1)


def _decode(value: Any, name: str) -> Any:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a JSON array") from error
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an array")
    return value


def normalize_axes(mapping: Mapping[str, Any]) -> tuple[list[list[int]], list[int]]:
    """Return canonical resident axes without applying backend legality."""
    raw_parts = _decode(mapping.get("local_stage_partitions"), "local_stage_partitions")
    parts: list[list[int]] = []
    for index, entry in enumerate(raw_parts):
        if not isinstance(entry, (list, tuple)):
            raise ValueError(f"local_stage_partitions[{index}] must be an array")
        parts.append([_int(value, f"local_stage_partitions[{index}]", minimum=0)
                      for value in entry])
    raw_chunks = _decode(mapping.get("exchange_chunks"), "exchange_chunks")
    chunks = [_int(value, f"exchange_chunks[{index}]", minimum=0)
              for index, value in enumerate(raw_chunks)]
    return parts, chunks


def resident_requested(mapping: Mapping[str, Any]) -> bool:
    parts, chunks = normalize_axes(mapping)
    return any(parts) or any(chunks)


def _physical_stages(stages: Sequence[int]) -> list[int]:
    result = []
    for index, value in enumerate(stages):
        result.append(_int(value, f"physical stage {index}", minimum=1))
    if not result:
        raise ValueError("physical stage partition must not be empty")
    return result


def _register_logs(parts: list[list[int]], local: int) -> tuple[int, int]:
    if parts:
        if len(parts) != 2:
            raise ValueError("resident axes must have one entry per register group")
        prefix = parts[0]
        if parts[1]:
            raise ValueError("opaque suffix core does not expose its local stage partition")
        if prefix:
            if len(prefix) != 2 or not prefix[0] or not prefix[1] or sum(prefix) != local:
                raise ValueError("register prefix needs two positive local factors covering local_stages")
            return prefix[0], prefix[1]
    if local & 1:
        raise ValueError("odd register prefix needs an explicit local stage partition")
    return local // 2, local // 2


def validate_resident_axes(mapping: Mapping[str, Any], stages: Sequence[int], *,
                           backend: str | None = None,
                           operator: str | None = None) -> tuple[list[list[int]], list[int]]:
    """Validate axes against resolved physical stages and return them.

    This checks the lowering-independent shape and the shared resident rules.
    Register FFT launch geometry is checked by :func:`register_geometry`.
    """
    physical = _physical_stages(stages)
    parts, chunks = normalize_axes(mapping)
    if parts and len(parts) != len(physical):
        raise ValueError("local_stage_partitions must have one entry per physical execution group")
    if chunks and len(chunks) != len(physical):
        raise ValueError("exchange_chunks must have one entry per physical execution group")
    if backend == "shared-iterative":
        if any(chunks):
            raise ValueError("shared iterative output already has final ownership; exchange chunks are unsupported")
        for index, entry in enumerate(parts):
            if not entry:
                continue
            if any(value < 1 or value > 5 for value in entry) or sum(entry) != physical[index]:
                raise ValueError("shared resident processing units require 1..5 stages covering the physical group")
    if backend == "online-reorder" and operator == "fft":
        if len(physical) != 2:
            raise ValueError("register FFT resident axes require two physical groups")
        _register_logs(parts, physical[0])
    elif backend not in (None, "shared-iterative") and resident_requested(mapping):
        raise ValueError(f"resident axes are unsupported for backend {backend}")
    return parts, chunks


def register_geometry(mapping: Mapping[str, Any], stages: Sequence[int] | None = None) -> dict[str, int | list[list[int]] | list[int]]:
    """Resolve rectangular register-prefix and grouped-suffix geometry."""
    total = _int(mapping.get("logN"), "logN", minimum=1)
    local = _int(mapping.get("local_stages"), "local_stages", minimum=2)
    suffix = total - local
    if local > 12 or suffix < 3 or suffix > 14 or total > 26:
        raise ValueError("register FFT template requires local prefix [2,12], suffix [3,14], total <=26")
    physical = list(stages) if stages is not None else [local, suffix]
    if len(physical) != 2 or _int(physical[0], "prefix stage", minimum=1) != local:
        raise ValueError("register FFT stage partition must contain local prefix and suffix")
    parts, chunks = validate_resident_axes(mapping, physical, backend="online-reorder", operator="fft")
    log_a, log_b = _register_logs(parts, local)
    a_size, b_size = 1 << log_a, 1 << log_b
    lanes = _int(mapping.get("prefix_codelet_lanes", 1), "prefix_codelet_lanes", minimum=1)
    if not _power_of_two(lanes) or lanes > a_size or a_size % lanes:
        raise ValueError("prefix_codelet_lanes must be a power of two dividing the first local factor")
    prefix_ept = _int(mapping.get("prefix_ept"), "prefix_ept", minimum=1)
    if prefix_ept != a_size // lanes or prefix_ept > b_size or b_size % prefix_ept:
        raise ValueError("prefix EPT must equal first factor size divided by prefix_codelet_lanes and divide the second factor")
    prefix_threads = _int(mapping.get("prefix_threads"), "prefix_threads", minimum=1)
    if prefix_threads > 1024 or prefix_threads * prefix_ept % (1 << local):
        raise ValueError("prefix threads must produce integral rectangular columns")
    columns = prefix_threads * prefix_ept // (1 << local)
    if not _power_of_two(columns) or columns > (1 << suffix):
        raise ValueError("register prefix columns must be a power of two dividing the suffix extent")
    second_lanes = b_size // prefix_ept
    if (lanes > 1 and lanes * columns > 32) or (second_lanes > 1 and second_lanes * columns > 32):
        raise ValueError("rectangular prefix lane groups exceed one warp")

    suffix_ept = _int(mapping.get("suffix_ept"), "suffix_ept", minimum=1)
    if not _power_of_two(suffix_ept) or suffix_ept > (1 << suffix):
        raise ValueError("suffix EPT must be a power of two within the suffix extent")
    suffix_threads = _int(mapping.get("suffix_threads"), "suffix_threads", minimum=1)
    suffix_product = suffix_threads * suffix_ept
    suffix_size = 1 << suffix
    if suffix_product % suffix_size:
        raise ValueError("suffix threads and EPT must produce integral columns")
    suffix_columns = suffix_product // suffix_size
    if not _power_of_two(suffix_columns) or suffix_columns > 16 or suffix_columns > (1 << local):
        raise ValueError("grouped suffix columns must be a power of two <=16 dividing the prefix extent")
    chunk = chunks[1] if chunks else 0
    if chunks and chunks[0]:
        raise ValueError("prefix exchange chunk must be zero")
    if chunk and (suffix_columns == 1 or not _power_of_two(chunk) or
                  chunk > suffix_ept or suffix_ept % chunk):
        raise ValueError("suffix exchange chunk must divide EPT; single-column exchange uses zero")
    return {
        "total": total, "local": local, "suffix": suffix,
        "log_a": log_a, "log_b": log_b, "a_size": a_size, "b_size": b_size,
        "prefix_ept": prefix_ept, "prefix_threads": prefix_threads,
        "prefix_lanes": lanes, "second_lanes": second_lanes, "prefix_columns": columns,
        "suffix_ept": suffix_ept, "suffix_threads": suffix_threads,
        "suffix_columns": suffix_columns, "suffix_chunk": chunk,
        "local_stage_partitions": parts, "exchange_chunks": chunks,
    }


def group_axis(mapping: Mapping[str, Any], name: str, index: int, default: Any) -> Any:
    """Return one group-local axis, treating omitted/empty entries as default."""
    parts, chunks = normalize_axes(mapping) if name in {
        "local_stage_partition", "exchange_chunk"} else ([], [])
    values = parts if name == "local_stage_partition" else chunks
    if index < len(values) and values[index]:
        return values[index]
    return default


def resident_partition_seed(stages: Sequence[int], width: int = 5) -> list[list[int]]:
    """Build a bounded shared-resident seed with no global stage changes."""
    if width < 1:
        raise ValueError("resident partition width must be positive")
    result = []
    for index, stage in enumerate(_physical_stages(stages)):
        remaining = stage
        entry = []
        while remaining:
            piece = min(width, remaining)
            entry.append(piece)
            remaining -= piece
        result.append(entry)
    return result
