#!/usr/bin/env python3
"""Resolve serial service views for a batch-ring execution plan.

An overlapped plan changes the number of transforms submitted to each physical
stage launch.  Its stage descriptor therefore cannot be used as a service
curve directly: ``batch_space``/``grid_ctas`` describe the tile, while a final
partial tile has a different resolved launch shape.  This module asks the
existing plan/StageProbe resolver for those two serial views.  It only builds
descriptors; it never launches a timed benchmark or fabricates resource data.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Callable


SCHEMA = "cubutterfly-stage-composition-v1"

_SUPPORTED_BACKENDS = {"shared-iterative", "online-reorder"}
_SUPPORTED_ONLINE_CORES = {"cufftdx-block", "register-tile"}
_POLICY_FIELDS = {"stage_overlap", "batch_tile_count", "factor_overlap"}
_GROUP_SHAPE_FIELDS = ("first_stage", "stage_count", "core", "threads",
                       "exchange_chunk", "local_stage_partition")
_RESOURCE_FIELDS = (
    "live_shared_bytes", "dynamic_shared_bytes", "compiler_registers_per_thread",
    "compiler_local_bytes_per_thread", "compiler_resources_known",
    "compiler_local_resources_known", "compiler_registers_known",
    "compiler_local_known",
)
_DERIVED_SAMPLE_FIELDS = {
    "execution_groups", "execution_groups_json", "groups", "actual_kernels",
    "workspace_bytes", "plan_allocation_bytes", "plan_bytes", "plan_workspace_bytes",
    "hardware", "free_memory_bytes", "free_bytes", "memory_free_bytes",
    "available_memory_bytes", "probe_payload_buffers", "runtime_fingerprint",
    "descriptor_source", "status", "reason", "correct", "kernel_ms",
    "median_kernel_ms", "trial_kernel_ms", "plan_trial_kernel_ms", "samples",
    "execution_boundaries", "execution_group_count", "decomposition_count", "N",
}


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "measured", "correct"}
    return bool(value)


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _scalar(value: Any) -> Any:
    """Normalize only comparison scalars; preserve descriptor values verbatim."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        # Python integers already retain arbitrary precision.  In particular,
        # do not round large NTT moduli through the float compatibility path.
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        # Do not route large NTT integers (notably 64-bit moduli in aliases)
        # through float, which would make two distinct values compare equal.
        return int(value.strip())
    parsed = _number(value)
    if parsed is not None and parsed.is_integer():
        return int(parsed)
    if parsed is not None:
        return parsed
    if isinstance(value, str):
        return value.strip().lower().replace("_", "-")
    return value


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value in (None, ""):
        return {}
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return dict(decoded) if isinstance(decoded, dict) else {}


def _mapping_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return _mapping(sample.get("mapping_json", sample.get("mapping", {})))


def _policy(sample: dict[str, Any], name: str) -> bool:
    # A resolved top-level field is authoritative.  Older mapping-only points
    # carry the same policy in mapping_json, so retain that fallback.
    if name in sample:
        return _truthy(sample[name])
    return _truthy(_mapping_from_sample(sample).get(name, False))


def _sample(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        return {}
    nested = document.get("sample")
    if isinstance(nested, dict):
        return dict(nested)
    return dict(document)


def _groups(document: Any, sample: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if isinstance(document, dict) and isinstance(document.get("groups"), list):
        return [dict(group) for group in document["groups"] if isinstance(group, dict)]
    source = sample if sample is not None else _sample(document)
    value = source.get("execution_groups_json")
    if isinstance(value, list):
        return [dict(group) for group in value if isinstance(group, dict)]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            decoded = None
        if isinstance(decoded, list):
            return [dict(group) for group in decoded if isinstance(group, dict)]
    if isinstance(document, dict) and isinstance(document.get("execution_groups"), list):
        return [dict(group) for group in document["execution_groups"] if isinstance(group, dict)]
    return []


def _status(document: Any) -> str:
    if not isinstance(document, dict):
        return "unavailable"
    return str(document.get("status", "unavailable")).strip().lower()


def _runtime_fingerprint(document: Any, sample: dict[str, Any]) -> Any:
    if isinstance(document, dict) and document.get("runtime_fingerprint") not in (None, ""):
        return document["runtime_fingerprint"]
    return sample.get("runtime_fingerprint")


def _identity_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    """Drop only batch composition policy from a resolved mapping."""
    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return _scalar(value)

    return {key: normalize(value) for key, value in mapping.items()
            if key not in _POLICY_FIELDS}


def _serial_point(sample: dict[str, Any], batch: int) -> dict[str, Any]:
    """Copy a resolved sample and disable only the batch-ring policy."""
    result = {key: value for key, value in sample.items()
              if key not in _DERIVED_SAMPLE_FIELDS}
    result.pop("stage_service_projection", None)
    result["batch"] = int(batch)
    result["stage_overlap"] = False
    result["factor_overlap"] = False
    result["batch_tile_count"] = 1
    mapping = _mapping_from_sample(sample)
    mapping.update({"stage_overlap": False, "factor_overlap": False, "batch_tile_count": 1})
    result["mapping_json"] = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    return result


def _group_shape(group: dict[str, Any]) -> tuple[Any, ...] | None:
    values = []
    for field in _GROUP_SHAPE_FIELDS:
        value = group.get(field)
        if value in (None, "") and field == "core":
            value = group.get("fft_core")
        if value in (None, ""):
            if field == "exchange_chunk":
                value = 0
            elif field == "local_stage_partition":
                value = []
            else:
                return None
        values.append(_scalar(value))
    return tuple(values)


def _kernel_resources(group: dict[str, Any]) -> dict[str, Any] | None:
    """Return stable resources, including one-kernel capture aliases.

    A false compiler-known flag makes the corresponding numeric value
    unknown.  This lets a serial describe add evidence that was unavailable in
    the overlapped source without requiring the source and serial schemas to
    expose identical optional fields.
    """
    kernels = group.get("actual_kernels")
    kernel = kernels[0] if isinstance(kernels, list) and len(kernels) == 1 and isinstance(kernels[0], dict) else {}
    values: dict[str, Any] = {}
    resource_known = group.get("compiler_resources_known", kernel.get("compiler_resources_known"))
    register_known = group.get("compiler_registers_known", kernel.get("compiler_registers_known"))
    local_known = group.get("compiler_local_resources_known", kernel.get("compiler_local_resources_known"))
    local_alias_known = group.get("compiler_local_known", kernel.get("compiler_local_known"))
    resources_are_known = (_truthy(resource_known) or _truthy(register_known))
    local_is_known = (_truthy(local_known) or _truthy(local_alias_known))
    for field in ("live_shared_bytes", "dynamic_shared_bytes", "compiler_registers_per_thread",
                  "compiler_local_bytes_per_thread"):
        value = group.get(field, None)
        if value in (None, "") and field in kernel:
            value = kernel[field]
        if value in (None, ""):
            continue
        if field == "compiler_registers_per_thread" and not resources_are_known:
            continue
        if field == "compiler_local_bytes_per_thread" and not local_is_known:
            continue
        values[field] = _scalar(value)
    if resources_are_known:
        values["compiler_resources_known"] = True
    if local_is_known:
        values["compiler_local_resources_known"] = True
    if isinstance(kernels, list) and len(kernels) > 1:
        # Composite captures are not a single physical service view.  Keep
        # their presence explicit instead of comparing only aggregate fields.
        values["actual_kernel_count"] = len(kernels)
    return values or None


def _view_payload(document: Any, requested: dict[str, Any], batch: int) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    status = _status(document)
    if status not in {"resolved", "measured"}:
        reason = "describe-status-" + (status or "unavailable")
        if isinstance(document, dict) and document.get("reason"):
            reason += ":" + str(document["reason"])
        return {}, [], reason
    actual = _sample(document)
    groups = _groups(document, actual)
    if not actual:
        return {}, [], "describe-missing-sample"
    if not groups:
        return {}, [], "describe-missing-groups"
    # A describe response may omit semantic aliases.  Fill only omissions from
    # the request; values emitted by the resolver remain authoritative.
    actual_batch = actual.get("batch", document.get("batch") if isinstance(document, dict) else None)
    if actual_batch not in (None, ""):
        parsed_batch = _number(actual_batch)
        if parsed_batch is None or not parsed_batch.is_integer() or int(parsed_batch) != int(batch):
            return {}, [], "actual-batch-mismatch"
    merged = dict(requested)
    merged.update(actual)
    # Missing batch is an omission in older describe responses; an explicit
    # conflicting batch was rejected above and is never masked.
    merged["batch"] = int(batch)
    if _policy(merged, "stage_overlap") or _policy(merged, "factor_overlap"):
        return {}, [], "serial-descriptor-retained-overlap"
    merged["stage_overlap"] = False
    merged["factor_overlap"] = False
    merged["batch_tile_count"] = 1
    mapping = _mapping_from_sample(merged)
    mapping.update({"stage_overlap": False, "factor_overlap": False, "batch_tile_count": 1})
    merged["mapping_json"] = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    return merged, groups, None


def _invalid_result(reason: str, *, status: str = "unavailable", tile_batch: int | None = None,
                    tail_batch: int | None = None, tile_count: int | None = None) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": status,
        "full": None,
        "tail": None,
        "tile_batch": tile_batch,
        "tail_batch": tail_batch,
        "tile_count": tile_count,
        "reason": reason,
    }


def _supported(sample: dict[str, Any], groups: list[dict[str, Any]]) -> tuple[bool, str | None]:
    backend = str(sample.get("backend", _mapping_from_sample(sample).get("backend", ""))).strip().lower().replace("_", "-")
    operator = str(sample.get("operator", _mapping_from_sample(sample).get("operator", "fft"))).strip().lower()
    mapping = _mapping_from_sample(sample)
    core = sample.get("fft_core", mapping.get("fft_core"))
    core = str(core).strip().lower().replace("_", "-") if core not in (None, "") else ""
    if backend == "factor-streamed" or _policy(sample, "factor_overlap"):
        return False, "factor-overlap-composition-unsupported"
    if backend == "shared-iterative":
        if len(groups) < 2:
            return False, "batch-ring-requires-multiple-groups"
        return True, None
    if backend == "online-reorder":
        if operator != "fft" or len(groups) != 2 or core not in _SUPPORTED_ONLINE_CORES:
            return False, "online-reorder-batch-ring-lowering-unsupported"
        return True, None
    return False, "batch-ring-backend-unsupported:" + (backend or "unknown")


def _check_identity(source_sample: dict[str, Any], source_groups: list[dict[str, Any]],
                    views: list[tuple[dict[str, Any], list[dict[str, Any]], str]],
                    source_runtime: Any) -> str | None:
    fingerprints = [source_runtime]
    fingerprints.extend(runtime for _, _, runtime in views)
    if any(value in (None, "") for value in fingerprints):
        return "runtime-fingerprint-unavailable"
    if any(str(value) != str(source_runtime) for value in fingerprints):
        return "runtime-fingerprint-mismatch"
    source_shapes = [_group_shape(group) for group in source_groups]
    if any(shape is None for shape in source_shapes):
        return "source-group-shape-unavailable"
    source_resources = [_kernel_resources(group) or {} for group in source_groups]
    source_mapping = _identity_mapping(_mapping_from_sample(source_sample))
    actual_resource_sets: list[dict[str, Any]] = []
    for sample, groups, _ in views:
        if len(groups) != len(source_groups):
            return "group-count-mismatch"
        if _identity_mapping(_mapping_from_sample(sample)) != source_mapping:
            return "resolved-mapping-identity-mismatch"
        view_resources: dict[str, Any] = {}
        for index, (source, target) in enumerate(zip(source_groups, groups)):
            if _group_shape(target) != source_shapes[index]:
                return f"group-shape-mismatch:{index}"
            resources = _kernel_resources(target)
            # The service model needs actual compiler resources for every
            # serial view.  Source-only unknown fields are intentionally not a
            # failure and are never copied into the view.
            if resources is None or "compiler_registers_per_thread" not in resources:
                return f"serial-resource-identity-unavailable:{index}"
            for field, value in source_resources[index].items():
                if field in {"compiler_resources_known", "compiler_local_resources_known"}:
                    continue
                if field in resources and resources[field] != value:
                    return f"resource-identity-mismatch:{index}:{field}"
            for field, value in resources.items():
                if field not in {"compiler_resources_known", "compiler_local_resources_known"}:
                    view_resources[f"{index}:{field}"] = value
        actual_resource_sets.append(view_resources)
    if len(actual_resource_sets) > 1 and actual_resource_sets[0] != actual_resource_sets[1]:
        return "serial-resource-identity-mismatch"
    return None


def resolve_batch_services(point: dict[str, Any], description: dict[str, Any],
                           describe: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    """Resolve exact serial tile and tail descriptors for a batch-ring point.

    ``describe`` must return the normal stage-probe JSON object for the point
    it receives.  The callback is invoked once for the full tile and, when
    needed, once for the remainder.  A resolved result means both descriptors
    have the same physical mapping/group/resource identity as the original
    overlap descriptor; no projected metadata is copied into either view.
    """
    if not isinstance(point, dict):
        return _invalid_result("point-is-not-an-object")
    requested = dict(point)
    point_mapping = _mapping_from_sample(requested)
    stage_overlap = _policy(requested, "stage_overlap")
    factor_overlap = _policy(requested, "factor_overlap")
    if not stage_overlap:
        batch = _number(requested.get("batch"))
        return {
            "schema": SCHEMA, "status": "not-applicable", "full": None, "tail": None,
            "tile_batch": int(batch) if batch is not None and batch > 0 else None,
            "tail_batch": 0, "tile_count": 1, "reason": "stage-overlap-disabled",
        }
    batch_number = _number(requested.get("batch"))
    tile_number = _number(requested.get("batch_tile_count", point_mapping.get("batch_tile_count")))
    if batch_number is None or not batch_number.is_integer() or batch_number <= 0:
        return _invalid_result("invalid-batch")
    if tile_number is None or not tile_number.is_integer() or tile_number <= 0:
        return _invalid_result("invalid-batch-tile-count")
    batch = int(batch_number)
    tile = min(batch, int(tile_number))
    tail = batch % tile
    tile_count = (batch + tile - 1) // tile

    source_sample = _sample(description)
    source_groups = _groups(description, source_sample)
    source_status = _status(description)
    # Validation may have a bulk timing record while its requested point is an
    # overlap plan.  In that case obtain the resolved overlap sample first so
    # its canonical mapping, fingerprint and physical group identities remain
    # the source for both serial views.
    if (source_status not in {"resolved", "measured"} or not source_sample or not source_groups
            or not _policy(source_sample, "stage_overlap")):
        try:
            original = describe(dict(requested))
        except Exception as error:
            return _invalid_result("original-describe-failed:" + str(error), tile_batch=tile,
                                   tail_batch=tail, tile_count=tile_count)
        source_sample = _sample(original)
        source_groups = _groups(original, source_sample)
        source_status = _status(original)
    if source_status not in {"resolved", "measured"} or not source_sample or not source_groups:
        return _invalid_result("original-descriptor-unavailable", tile_batch=tile,
                               tail_batch=tail, tile_count=tile_count)
    supported, reason = _supported(source_sample, source_groups)
    if factor_overlap or _policy(source_sample, "factor_overlap"):
        supported, reason = False, "factor-overlap-composition-unsupported"
    if not supported:
        return _invalid_result(reason or "batch-ring-lowering-unsupported", status="unsupported",
                               tile_batch=tile, tail_batch=tail, tile_count=tile_count)
    source_runtime = _runtime_fingerprint(description, source_sample)
    if source_runtime in (None, ""):
        # A fallback describe may carry the fingerprint at the response level.
        source_runtime = source_sample.get("runtime_fingerprint")

    views: list[tuple[dict[str, Any], list[dict[str, Any]], str]] = []
    output_views: list[dict[str, Any] | None] = []
    for batch_value in (tile, tail if tail else None):
        if batch_value is None:
            output_views.append(None)
            continue
        serial = _serial_point(source_sample, batch_value)
        try:
            response = describe(serial)
        except Exception as error:
            return _invalid_result(f"describe-failed-batch-{batch_value}:{error}", tile_batch=tile,
                                   tail_batch=tail, tile_count=tile_count)
        view_sample, view_groups, view_reason = _view_payload(response, serial, batch_value)
        if view_reason:
            return _invalid_result(f"batch-{batch_value}:{view_reason}", tile_batch=tile,
                                   tail_batch=tail, tile_count=tile_count)
        runtime = _runtime_fingerprint(response, view_sample)
        views.append((view_sample, view_groups, runtime))
        output_views.append({"sample": view_sample, "groups": view_groups})

    identity_reason = _check_identity(source_sample, source_groups, views, source_runtime)
    if identity_reason:
        return _invalid_result(identity_reason, tile_batch=tile, tail_batch=tail, tile_count=tile_count)
    return {
        "schema": SCHEMA,
        "status": "resolved",
        "full": output_views[0],
        "tail": output_views[1],
        "tile_batch": tile,
        "tail_batch": tail,
        "tile_count": tile_count,
        "runtime_fingerprint": source_runtime,
        "reason": "resolved-from-batch-ring-source",
    }


__all__ = ["SCHEMA", "resolve_batch_services"]
