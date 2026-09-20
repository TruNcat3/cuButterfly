#!/usr/bin/env python3
"""Adaptive calibration of physical execution-group service curves.

The CUDA stage microbenchmark is intentionally a very small protocol surface:
the driver supplies one already executable point and receives a JSON document
with resolved groups and timings.  This module owns the durable orchestration
around that probe.  It does not compile mappings, wait for a GPU, or invent
lowerings that were not supplied by the caller.

``calibrate`` is usable from the installation orchestrator, but all external
effects are injected through ``run`` and ``exclusive``.  That keeps recovery
and the adaptive policy testable on a CPU-only host.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import pathlib
import statistics
import time
from collections import OrderedDict
from typing import Any, Callable, Iterable

try:
    import resident_mapping
except ImportError:  # pragma: no cover - package import fallback
    from . import resident_mapping


SCHEMA = "cubutterfly-stage-calibration-v1"
PROBE_SCHEMA = "cubutterfly-stage-probe-v1"
VERSION = "adaptive-group-sampling-v1"
DEFAULT_MAX_POINTS = 33

_LOAD_FIELDS = {
    "batch", "batch_tile_count", "work_blocks", "grid_ctas", "load", "waves",
}
_POINT_METADATA_FIELDS = {
    "name", "id", "candidate_id", "record_id", "kernel_ms", "median_kernel_ms",
    "trial_kernel_ms", "correct", "status", "reason", "runtime_fingerprint",
    "execution_groups_json", "execution_boundaries", "workspace_bytes",
    "descriptor_source", "hardware", "groups", "pairs", "plan_trial_kernel_ms",
}


class CalibrationUnavailable(RuntimeError):
    """Raised by callers that require a usable stage-service profile."""


def _finite_number(value: Any, *, name: str, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number) or (positive and number <= 0):
        relation = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {relation}")
    return number


def _as_int(value: Any, *, name: str, minimum: int = 0) -> int:
    number = _finite_number(value, name=name)
    if not number.is_integer() or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(number)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "correct", "measured"}
    return bool(value)


def _looks_interrupted(error: BaseException) -> bool:
    if isinstance(error, (KeyboardInterrupt, InterruptedError, TimeoutError)):
        return True
    text = str(error).strip().lower()
    return any(token in text for token in ("interrupt", "pause", "stop", "cancel"))


def _canonical(value: Any) -> Any:
    """Return deterministic JSON key material without accepting NaN/Inf."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("point values must be finite")
        return int(value) if float(value).is_integer() else float(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    return str(value)


def _json_key(value: Any) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _decode_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return value
    return decoded


def _normal_point(point: Any) -> dict[str, Any]:
    if not isinstance(point, dict):
        raise ValueError("each stage calibration point must be an object")
    result = dict(point)
    if "operator" not in result:
        result["operator"] = "fft"
    if "precision" not in result:
        result["precision"] = "fp32"
    if "logN" not in result and "log_n" not in result and "N" not in result:
        raise ValueError("stage calibration point requires logN (or log_n/N)")
    if "batch" not in result:
        result["batch"] = 1
    result["batch"] = _as_int(result["batch"], name="point.batch", minimum=1)
    if "mapping_json" not in result:
        raise ValueError("stage calibration point requires mapping_json")
    mapping = _decode_mapping(result["mapping_json"])
    if not isinstance(mapping, dict):
        raise ValueError("point.mapping_json must encode an object")
    result["mapping_json"] = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    # Validate every supplied numeric field that the probe may consume.  A
    # NaN in a metadata field must not poison cache identity or command JSON.
    _canonical(result)
    return result


def _point_number(point: dict[str, Any], name: str, default: int = 0) -> int:
    value = point.get(name, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return int(parsed) if math.isfinite(parsed) and parsed.is_integer() else default


def _bulk_service_point(point: dict[str, Any]) -> dict[str, Any]:
    """Make the explicitly serial bulk copy used for service calibration.

    Overlap and tiled-batch policy are composition axes.  They must not create
    a second copy of an otherwise identical physical service curve.  Keep the
    serialized mapping and top-level point synchronized whenever a mapping
    carries these policy fields.
    """
    result = dict(point)
    result["stage_overlap"] = 0
    result["batch_tile_count"] = 1
    result["factor_overlap"] = 0
    mapping = _decode_mapping(result.get("mapping_json", {}))
    if isinstance(mapping, dict):
        mapping = dict(mapping)
        for name, value in (("stage_overlap", 0), ("batch_tile_count", 1),
                            ("factor_overlap", 0)):
            # The copied point is itself a complete probe request.  Emit the
            # policy fields even when the source mapping omitted them so that
            # top-level and serialized configuration cannot disagree.
            mapping[name] = value
        result["mapping_json"] = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    return result


def _service_points(points: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate bulk service points from unverified composition requests."""
    service: list[dict[str, Any]] = []
    composition: list[dict[str, Any]] = []
    seen_composition: set[str] = set()
    for point in points:
        mapping = _decode_mapping(point.get("mapping_json", {}))
        mapping = mapping if isinstance(mapping, dict) else {}
        stage_overlap = _truthy(point.get("stage_overlap", mapping.get("stage_overlap", False)))
        factor_overlap = _truthy(point.get("factor_overlap", mapping.get("factor_overlap", False)))
        factor_slices = _point_number(point, "factor_slices",
                                      _point_number(mapping, "factor_slices", 1))
        if factor_slices > 1 and not factor_overlap:
            key = _json_key(point)
            if key not in seen_composition:
                composition.append({"point": point,
                                    "reason": "factor-slices-not-service-equivalent",
                                    "tested": False})
                seen_composition.add(key)
            continue
        if stage_overlap or factor_overlap:
            bulk = _bulk_service_point(point)
            key = _json_key(point)
            if key not in seen_composition:
                composition.append({"point": point, "bulk_point": bulk,
                                    "reason": "overlap-composition-not-service-curve",
                                    "tested": False})
                seen_composition.add(key)
            service.append(bulk)
        else:
            service.append(point)
    return service, composition


def _point_static_key(point: dict[str, Any]) -> str:
    """Key mappings before a plan is resolved, excluding load-only metadata."""
    mapping = _decode_mapping(point.get("mapping_json", {}))
    payload = {}
    for key, value in point.items():
        if key in _LOAD_FIELDS or key in _POINT_METADATA_FIELDS:
            continue
        payload[key] = value
    payload["mapping_json"] = mapping
    return _json_key(payload)


def _profile_hardware(profile: dict[str, Any]) -> dict[str, Any]:
    hardware = profile.get("hardware") if isinstance(profile.get("hardware"), dict) else {}
    merged = dict(hardware)
    merged.update({key: value for key, value in profile.items()
                   if key not in {"hardware", "capabilities"} and value not in (None, "")})
    if "global_memory_bytes" not in merged and "memory_bytes" in merged:
        merged["global_memory_bytes"] = merged["memory_bytes"]
    if "memory_bytes" not in merged and "global_memory_bytes" in merged:
        merged["memory_bytes"] = merged["global_memory_bytes"]
    return merged


def _profile_identity(profile: dict[str, Any]) -> dict[str, Any]:
    hardware = _profile_hardware(profile)
    identity: dict[str, Any] = {}
    for field in (
        "device", "compute_capability", "global_memory_bytes", "memory_bytes",
        "sm_count", "gpu_uuid", "uuid", "runtime_fingerprint",
    ):
        if hardware.get(field) not in (None, ""):
            identity[field] = hardware[field]
    compile_policy = profile.get("compile_policy", profile.get("compile_mode"))
    if compile_policy in (None, ""):
        compile_policy = os.environ.get("CUBUTTERFLY_COMPILE_MODE", "research")
    identity["compile_policy"] = str(compile_policy)
    return _canonical(identity)


def _binary_sha256(binary: pathlib.Path | str) -> str:
    path = pathlib.Path(binary)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise FileNotFoundError(f"stage probe binary was not readable: {path}") from error
    return hashlib.sha256(payload).hexdigest()


def _protocol(*, mode: str, warmup: int, repeat: int, trials: int,
              max_points: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    protocol = {
        "warmup": int(warmup), "repeat": int(repeat), "trials": int(trials),
        "max_points": int(max_points), "mode": mode, "driver_version": VERSION,
    }
    if extra:
        for key, value in extra.items():
            if value is not None:
                protocol[str(key)] = _canonical(value)
    return protocol


def _cache_identity(binary: pathlib.Path | str, profile: dict[str, Any], protocol: dict[str, Any],
                    points_hash: str) -> dict[str, Any]:
    identity = _profile_identity(profile)
    # The point inventory is intentionally absent.  The installation
    # orchestrator may append another finite candidate/workload on resume;
    # changing that request must not discard compatible raw measurements.
    measurement_protocol = {key: value for key, value in protocol.items()
                            if key not in {"mode", "max_points", "budget", "max_seconds",
                                           "wall_budget_seconds", "time_budget"}}
    identity.update({"binary_sha256": _binary_sha256(binary), "protocol": measurement_protocol})
    return identity


def _points_hash(points: Iterable[dict[str, Any]]) -> str:
    payload = [_canonical(point) for point in points]
    return hashlib.sha256(_json_key(payload).encode()).hexdigest()


def probe_command(binary: pathlib.Path | str, point: dict[str, Any], *, warmup: int = 10,
                  repeat: int = 20, trials: int = 3, describe_only: bool = False) -> list[str]:
    """Build the exact stage microbenchmark argv used by ``calibrate``."""
    command = [str(binary), "--point-json", json.dumps(point, sort_keys=True,
                                                         separators=(",", ":")),
               "--warmup", str(int(warmup)), "--repeat", str(int(repeat)),
               "--trials", str(int(trials))]
    if describe_only:
        command.append("--describe-only")
    return command


def _result_stdout(result: Any) -> str:
    if isinstance(result, bytes):
        return result.decode()
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return json.dumps(result)
    stdout = getattr(result, "stdout", None)
    if stdout is None:
        raise ValueError("stage probe runner returned no stdout")
    if isinstance(stdout, bytes):
        return stdout.decode()
    return str(stdout)


def _parse_json_output(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        document = result
    else:
        text = _result_stdout(result).strip()
        if not text:
            raise ValueError("stage probe returned empty stdout")
        try:
            document = json.loads(text)
        except json.JSONDecodeError:
            # A benchmark may print a short informational line before its one
            # JSON result.  Parse the last complete JSON object without
            # accepting arbitrary non-JSON data as a measurement.
            document = None
            decoder = json.JSONDecoder()
            for offset, character in enumerate(text):
                if character != "{":
                    continue
                try:
                    candidate, end = decoder.raw_decode(text[offset:])
                except json.JSONDecodeError:
                    continue
                if end == len(text) - offset or not text[offset + end:].strip():
                    document = candidate
                    break
            if document is None:
                raise ValueError("stage probe stdout was not JSON")
    if not isinstance(document, dict):
        raise ValueError("stage probe result must be a JSON object")
    return document


def _invoke(binary: pathlib.Path | str, point: dict[str, Any], run: Callable[..., Any],
            exclusive: Callable[[], Any], *, warmup: int, repeat: int, trials: int,
            describe_only: bool) -> dict[str, Any]:
    """Run once while making no attempt to wait for exclusive GPU access."""
    command = probe_command(binary, point, warmup=warmup, repeat=repeat, trials=trials,
                            describe_only=describe_only)
    try:
        available = exclusive()
    except Exception as error:
        # A checker may report a busy GPU by raising rather than returning
        # false.  Stop this invocation immediately; retry/reservation policy
        # belongs to the caller and is never implemented as an internal wait.
        raise InterruptedError(f"gpu-exclusivity-check-failed: {error}") from error
    if available is False:
        raise InterruptedError("gpu-not-exclusive")
    result: dict[str, Any]
    post_error: BaseException | None = None
    try:
        runner_result = run(command)
        try:
            result = _parse_json_output(runner_result)
        except ValueError:
            # The install orchestrator deliberately invokes the probe with
            # check=False so a rejected plan can be represented as a normal
            # inventory omission.  Preserve the process diagnostics instead
            # of turning that expected non-zero result into a driver error.
            returncode = getattr(runner_result, "returncode", None)
            if returncode not in (None, 0):
                stderr = getattr(runner_result, "stderr", None)
                stdout = getattr(runner_result, "stdout", None)
                detail = stderr or stdout or f"stage probe exited with status {returncode}"
                result = {"schema": PROBE_SCHEMA, "status": "unavailable",
                          "correct": False, "reason": str(detail)}
            else:
                raise
    finally:
        # Existing callers use the second check to release/verify a bounded
        # reservation.  It is intentionally a check, never a sleep/retry.
        try:
            post_available = exclusive()
            if post_available is False:
                post_error = RuntimeError("post-exclusivity-check-returned-false")
        except Exception as error:
            post_error = error
    if post_error is not None:
        # Retain the complete raw probe result for audit, but make it ineligible
        # for model fitting.  A post-run exclusivity failure means that another
        # process may have mixed with this timing.
        result = dict(result)
        result["status"] = "contaminated"
        result["correct"] = False
        result["reason"] = f"post-exclusivity-check-failed: {post_error}"
    return result


def _group_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    groups = raw.get("groups")
    if isinstance(groups, list):
        return [dict(group) for group in groups if isinstance(group, dict)]
    groups = raw.get("execution_groups_json")
    if isinstance(groups, str):
        try:
            groups = json.loads(groups)
        except (TypeError, ValueError):
            groups = None
    if isinstance(groups, list):
        return [dict(group) for group in groups if isinstance(group, dict)]
    return []


def _merge_sample(point: dict[str, Any], raw: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    sample = dict(point)
    if isinstance(raw.get("sample"), dict):
        # Probe-confirmed fields win, while point semantics that are absent
        # from the compact sample remain available to model key generation.
        sample.update(raw["sample"])
    hardware = raw.get("hardware")
    if isinstance(hardware, dict):
        sample.setdefault("hardware", dict(hardware))
        for field in ("device", "compute_capability", "global_memory_bytes", "memory_bytes",
                      "sm_count", "gpu_uuid", "uuid"):
            if field in hardware:
                sample.setdefault(field, hardware[field])
    for field in ("runtime_fingerprint", "descriptor_source", "workspace_bytes",
                  "free_memory_bytes", "probe_payload_buffers", "plan_allocation_bytes",
                  "plan_bytes", "plan_workspace_bytes"):
        if field in raw:
            sample.setdefault(field, raw[field])
    if isinstance(sample.get("hardware"), dict):
        # The current probe reports free memory at the result top level.  Keep
        # it under hardware as well so all residency/memory consumers see the
        # same captured limit.
        sample["hardware"] = dict(sample["hardware"])
        for field in ("free_memory_bytes", "memory_free_bytes", "available_memory_bytes",
                      "free_bytes", "global_memory_bytes", "memory_bytes", "sm_count",
                      "max_blocks_per_sm", "max_threads_per_sm", "registers_per_sm",
                      "shared_bytes_per_sm"):
            if field in raw and field not in sample["hardware"]:
                sample["hardware"][field] = raw[field]
    # Protocol values are measurement provenance, not mapping axes.
    return sample


def _normal_group(group: dict[str, Any], index: int, *, sample: dict[str, Any],
                  raw: dict[str, Any], measurement_exclusive: bool) -> dict[str, Any]:
    normalized = dict(group)
    normalized.setdefault("index", index)
    mapping = _decode_mapping(sample.get("mapping_json", {}))
    if isinstance(mapping, dict):
        try:
            parts, chunks = resident_mapping.normalize_axes(mapping)
        except ValueError:
            parts, chunks = [], []
        if "local_stage_partition" not in normalized and index < len(parts):
            normalized["local_stage_partition"] = list(parts[index])
        if "exchange_chunk" not in normalized and index < len(chunks):
            normalized["exchange_chunk"] = chunks[index]
    if normalized.get("work_blocks") in (None, "") and "work" in normalized:
        normalized["work_blocks"] = normalized["work"]
    if normalized.get("work_blocks") in (None, "") and "grid_ctas" in normalized:
        normalized["work_blocks"] = normalized["grid_ctas"]
    # Missing independence is deliberately unavailable.  A group whose
    # timing cannot be separated must never become a stage-service curve.
    normalized["independent"] = _truthy(normalized.get("independent", False))
    normalized.setdefault("measurement_exclusive_gpu", measurement_exclusive)
    if "trial_kernel_ms" in normalized:
        values = normalized["trial_kernel_ms"]
        if isinstance(values, (list, tuple)):
            cleaned = []
            for value in values:
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(parsed) and parsed > 0:
                    cleaned.append(parsed)
            normalized["trial_kernel_ms"] = cleaned
    for field in ("warmup", "repeat", "trials"):
        if field not in normalized and field in raw:
            normalized[field] = raw[field]
        if field not in normalized and field in sample:
            normalized[field] = sample[field]
    return normalized


def _normalize_description(point: dict[str, Any], raw: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    status = str(raw.get("status", "unavailable")).strip().lower()
    sample = _merge_sample(point, raw, profile)
    groups = [_normal_group(group, index, sample=sample, raw=raw, measurement_exclusive=False)
              for index, group in enumerate(_group_rows(raw))]
    hardware = raw.get("hardware") if isinstance(raw.get("hardware"), dict) else {}
    hardware = dict(hardware)
    for field in ("free_memory_bytes", "memory_free_bytes", "available_memory_bytes",
                  "free_bytes", "global_memory_bytes", "memory_bytes", "sm_count",
                  "max_blocks_per_sm", "max_threads_per_sm", "registers_per_sm",
                  "shared_bytes_per_sm"):
        if field in raw and field not in hardware:
            hardware[field] = raw[field]
    return {
        "schema": PROBE_SCHEMA, "status": "resolved" if status == "resolved" else status,
        "point": point, "sample": sample, "groups": groups,
        "execution_boundaries": raw.get("execution_boundaries", []),
        "workspace_bytes": raw.get("workspace_bytes"),
        "runtime_fingerprint": raw.get("runtime_fingerprint"),
        "hardware": hardware, "raw": raw,
    }


def _normalize_measurement(point: dict[str, Any], raw: dict[str, Any], profile: dict[str, Any],
                           *, role: str, candidate_key: str, batch: int,
                           measurement_exclusive: bool, warmup: int, repeat: int,
                           trials: int) -> dict[str, Any]:
    status = str(raw.get("status", "unavailable")).strip().lower()
    sample = _merge_sample(point, raw, profile)
    sample.setdefault("batch", batch)
    sample["warmup"] = warmup
    sample["repeat"] = repeat
    sample["trials"] = trials
    groups = [_normal_group(group, index, sample=sample, raw=raw,
                            measurement_exclusive=measurement_exclusive)
              for index, group in enumerate(_group_rows(raw))]
    for group in groups:
        # The driver owns holdout assignment.  Do not let a stale role echoed
        # by a probe turn a validation timing into training data.
        group["role"] = role
    record_status = "measured" if status == "measured" else status
    record = {
        "schema": PROBE_SCHEMA, "status": record_status,
        "correct": _truthy(raw.get("correct", False)), "sample": sample,
        "groups": groups, "pairs": raw.get("pairs", []),
        "plan_trial_kernel_ms": raw.get("plan_trial_kernel_ms", []),
        "raw": raw,
        "candidate_key": candidate_key, "batch": batch, "role": role,
        "measurement_exclusive_gpu": measurement_exclusive,
        "protocol": {"warmup": warmup, "repeat": repeat, "trials": trials},
    }
    for field in ("runtime_fingerprint", "hardware", "reason", "command", "workspace_bytes"):
        if field in raw:
            record[field] = raw[field]
    return record


def _model_module() -> Any:
    try:
        return importlib.import_module("stage_service_model")
    except ImportError as error:
        raise RuntimeError("stage_service_model.py is required for stage calibration") from error


def _build_model(records: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
    model = _model_module()
    identity = _profile_identity(profile)
    return model.build_profile(records, identity=identity)


def _model_curve_key(sample: dict[str, Any], group: dict[str, Any]) -> str:
    model = _model_module()
    return str(model.curve_key(sample, group))


def _candidate_from_description(static_key: str, description: dict[str, Any],
                                profile: dict[str, Any]) -> dict[str, Any]:
    """Create one candidate state from a resolved executable point."""
    point = description["point"]
    actual_keys = []
    for group in description.get("groups", []):
        if _truthy(group.get("independent")):
            try:
                actual_keys.append(_model_curve_key(description["sample"], group))
            except Exception:
                # The model will retain the raw group as unavailable.  Keep a
                # static identity so an unsupported descriptor can still be
                # checkpointed and reported instead of silently disappearing.
                continue
    actual_key = _json_key(sorted(actual_keys)) if actual_keys else static_key
    requested = int(point["batch"])
    memory_max, memory_reason = _memory_bound_batch(point, description, profile, requested)
    loads, omitted = _residency_batches(point, description, profile, requested, memory_max)
    if memory_reason:
        omitted.append(memory_reason)
    minimum = max(1, int(point.get("min_batch", 1)))
    if memory_max is not None:
        loads = [load for load in loads if load <= memory_max]
    # Reserve an interior integer for the first adaptive holdout.  Without
    # this reservation a small range such as 1..8 would consume every legal
    # load as a seed and could never produce the independent validation point
    # required by the model's certification gate.
    reserved_holdout = None
    upper = requested if memory_max is None else min(requested, memory_max)
    if upper > minimum:
        midpoint = (minimum + upper) // 2
        if minimum < midpoint < upper:
            reserved_holdout = midpoint
            loads = [load for load in loads if load != reserved_holdout]
    available_memory = _available_memory_bytes(profile, description)
    return {
        "key": actual_key, "static_key": static_key, "actual_key": actual_key,
        "point": point, "description": description,
        "requested_batch": requested, "minimum_batch": minimum,
        "legal_min_batch": minimum,
        "legal_max_batch": min(requested, memory_max) if memory_max is not None else requested,
        "memory_bound_batch": memory_max, "loads": loads,
        "memory_available_bytes": available_memory,
        "memory_limit_bytes": available_memory * 0.90 if available_memory is not None else None,
        "memory_requested_bytes": _estimated_memory_bytes(point, requested, description, profile),
        "seed_batches": list(loads), "reserved_holdout": reserved_holdout,
        "adaptive_batches": [], "intervals": {}, "omitted": omitted,
        "aliases": [static_key],
    }


def _merge_candidate(existing: dict[str, Any], incoming: dict[str, Any], profile: dict[str, Any]) -> None:
    """Merge an alias/new requested maximum without losing measured state."""
    old_requested = int(existing.get("requested_batch", 1))
    incoming_requested = int(incoming.get("requested_batch", 1))
    existing["requested_batch"] = max(old_requested, incoming_requested)
    existing["minimum_batch"] = min(int(existing.get("minimum_batch", 1)),
                                    int(incoming.get("minimum_batch", 1)))
    existing["legal_min_batch"] = existing["minimum_batch"]
    existing["aliases"] = sorted(set(existing.get("aliases", []) + incoming.get("aliases", [])))
    if incoming_requested > old_requested:
        existing["point"] = incoming["point"]
        existing["description"] = incoming["description"]
        existing["static_key"] = incoming["static_key"]
    # Recompute the legal/seed universe using the largest requirement.  The
    # already measured batches remain in the candidate's state and are never
    # rerun.
    description = existing.get("description")
    point = existing.get("point")
    if isinstance(description, dict) and isinstance(point, dict):
        memory_max, memory_reason = _memory_bound_batch(
            point, description, profile, int(existing["requested_batch"]))
        existing["memory_bound_batch"] = memory_max
        existing["legal_max_batch"] = (min(int(existing["requested_batch"]), int(memory_max))
                                         if memory_max is not None else int(existing["requested_batch"]))
        merged_loads = sorted(set(existing.get("loads", [])) | set(incoming.get("loads", [])))
        if memory_max is not None:
            merged_loads = [load for load in merged_loads if int(load) <= int(memory_max)]
        existing["loads"] = merged_loads
        if existing.get("reserved_holdout") is None:
            existing["reserved_holdout"] = incoming.get("reserved_holdout")
        if existing.get("reserved_holdout") is not None:
            existing["loads"] = [load for load in existing["loads"]
                                  if int(load) != int(existing["reserved_holdout"])]
        if memory_reason and memory_reason not in existing.setdefault("omitted", []):
            existing["omitted"].append(memory_reason)
        available_memory = _available_memory_bytes(profile, description)
        existing["memory_available_bytes"] = available_memory
        existing["memory_limit_bytes"] = (available_memory * 0.90
                                           if available_memory is not None else None)
        existing["memory_requested_bytes"] = _estimated_memory_bytes(
            point, int(existing["requested_batch"]), description, profile)
    existing["seed_batches"] = sorted(set(existing.get("seed_batches", [])) |
                                      set(incoming.get("seed_batches", [])))
    for reason in incoming.get("omitted", []):
        if reason not in existing.setdefault("omitted", []):
            existing["omitted"].append(reason)


def _hardware_number(profile: dict[str, Any], names: Iterable[str]) -> float | None:
    hardware = _profile_hardware(profile)
    for name in names:
        if name not in hardware or hardware[name] in (None, ""):
            continue
        try:
            parsed = float(hardware[name])
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed > 0:
            return parsed
    return None


def _description_hardware(profile: dict[str, Any], description: dict[str, Any]) -> dict[str, Any]:
    """Combine target profile limits with probe-captured runtime limits."""
    hardware = _profile_hardware(profile)
    captured = description.get("hardware") if isinstance(description.get("hardware"), dict) else {}
    merged = dict(hardware)
    merged.update(captured)
    raw = description.get("raw") if isinstance(description.get("raw"), dict) else {}
    for field in ("free_memory_bytes", "memory_free_bytes", "available_memory_bytes",
                  "free_bytes", "global_memory_bytes", "memory_bytes", "sm_count",
                  "max_blocks_per_sm", "max_threads_per_sm", "registers_per_sm",
                  "shared_bytes_per_sm"):
        if field in raw and field not in merged:
            merged[field] = raw[field]
    if "global_memory_bytes" not in merged and "memory_bytes" in merged:
        merged["global_memory_bytes"] = merged["memory_bytes"]
    if "memory_bytes" not in merged and "global_memory_bytes" in merged:
        merged["memory_bytes"] = merged["global_memory_bytes"]
    return merged


def _available_memory_bytes(profile: dict[str, Any], description: dict[str, Any]) -> float | None:
    hardware = _description_hardware(profile, description)
    values = []
    for name in ("free_memory_bytes", "memory_free_bytes", "available_memory_bytes", "free_bytes"):
        value = hardware.get(name)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed > 0:
            values.append(parsed)
    for name in ("global_memory_bytes", "memory_bytes"):
        value = hardware.get(name)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed > 0:
            values.append(parsed)
    return min(values) if values else None


def _value_number(mapping: dict[str, Any], names: Iterable[str]) -> float | None:
    for name in names:
        value = mapping.get(name)
        if value in (None, ""):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed > 0:
            return parsed
    return None


def _estimated_memory_bytes(point: dict[str, Any], batch: int, description: dict[str, Any],
                            profile: dict[str, Any]) -> float | None:
    capacity = _available_memory_bytes(profile, description)
    if capacity is None:
        return None
    try:
        log_n = int(point.get("logN", point.get("log_n")))
    except (TypeError, ValueError):
        try:
            log_n = int(math.log2(float(point["N"])))
        except (KeyError, TypeError, ValueError):
            return None
    if log_n < 0 or log_n > 62:
        return None
    precision = str(point.get("precision", "fp32")).lower()
    width = {"fp16": 2, "bf16": 2, "fp32": 4, "fp64": 8,
             "word32": 4, "word64": 8, "uint32": 4, "uint64": 8,
             "u32": 4, "u64": 8, "zeta32": 4, "zeta64": 8,
             "mod32": 4, "mod64": 8}.get(precision)
    if width is None:
        return None
    operator = str(point.get("operator", "fft")).lower()
    values_per_element = 2 if operator in {"fft", "complex-fft", "cfft"} else 1
    # Reserve all payload copies that the plan/runtime may touch.  ``count``
    # is an optional explicit plan multiplicity; absent data uses one logical
    # transform, while the +5 term covers input/output/scratch/transpose and
    # allocator slack.  This is deliberately conservative for search gating.
    n = 1 << log_n
    try:
        element_stride = max(1, int(point.get("element_stride", 1)))
    except (TypeError, ValueError):
        element_stride = 1
    try:
        default_batch_stride = n * element_stride
        batch_stride = max(default_batch_stride, int(point.get("batch_stride", default_batch_stride)))
    except (TypeError, ValueError):
        batch_stride = n * element_stride
    # ``data_size`` follows the highest addressed element for strided
    # batches, not merely N * batch.  This keeps large-stride points from
    # being admitted under an optimistic contiguous estimate.
    addressed_elements = (batch - 1) * batch_stride + n * element_stride
    payload = float(addressed_elements) * width * values_per_element
    count = _point_number(point, "count", 0)
    if count <= 0:
        count = len(description.get("groups", []))
    if count <= 0:
        payload_buffers = _point_number(point, "probe_payload_buffers", 0)
        count = max(1, payload_buffers - 4) if payload_buffers > 4 else 1
    count = max(1, count)
    total = (count + 5.0) * payload
    workspace = description.get("workspace_bytes")
    raw = description.get("raw", {}) if isinstance(description.get("raw"), dict) else {}
    plan_bytes = raw.get("plan_allocation_bytes", raw.get("plan_bytes", raw.get("plan_workspace_bytes")))
    # Workspace and plan allocations are independent of the endpoint payload
    # multiplicity.  Keep them additive and unscaled; scaling either term by
    # batch would reject valid loads while still being an approximation.
    for value in (workspace, plan_bytes):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed > 0:
            total += parsed
    return total


def _memory_bound_batch(point: dict[str, Any], description: dict[str, Any], profile: dict[str, Any],
                        requested_max: int) -> tuple[int | None, str | None]:
    capacity = _available_memory_bytes(profile, description)
    if capacity is None:
        return requested_max, "memory-bound-unknown"
    # Leave a 10% guard band.  The omitted request remains explicit in the
    # coverage report instead of making a potentially OOM point look complete.
    limit = capacity * 0.90
    if _estimated_memory_bytes(point, 1, description, profile) is None:
        return None, "memory-bound-unknown"
    low, high = 1, max(1, requested_max)
    if _estimated_memory_bytes(point, low, description, profile) > limit:
        return 0, "minimum-load-exceeds-memory-bound"
    while low < high:
        middle = (low + high + 1) // 2
        if _estimated_memory_bytes(point, middle, description, profile) <= limit:
            low = middle
        else:
            high = middle - 1
    return low, None


def _residency_batches(point: dict[str, Any], description: dict[str, Any], profile: dict[str, Any],
                       requested_max: int, memory_max: int | None) -> tuple[list[int], list[str]]:
    minimum = _as_int(point.get("min_batch", 1), name="min_batch", minimum=1)
    maximum = requested_max if memory_max is None else min(requested_max, memory_max)
    loads = {minimum}
    omitted: list[str] = []
    hardware = _description_hardware(profile, description)
    sm_count = None
    try:
        candidate_sm_count = float(hardware.get("sm_count"))
        if math.isfinite(candidate_sm_count) and candidate_sm_count > 0:
            sm_count = candidate_sm_count
    except (TypeError, ValueError):
        pass
    default_resident = _value_number(hardware, ("max_blocks_per_sm", "blocks_per_sm"))
    targets = (1, 2, 4, 8)
    base_batch = max(1, int(description["point"].get("batch", 1)))
    for group in description.get("groups", []):
        grid = _value_number(group, ("grid_ctas", "work_blocks", "work"))
        if grid is None:
            continue
        group_threads = _value_number(group, ("threads", "threads_per_block"))
        shared = _value_number(group, ("live_shared_bytes", "shared_bytes", "shared_bytes_per_block"))
        registers = _value_number(group, ("compiler_registers_per_thread", "registers_per_thread"))
        limits = []
        explicit_resident = _value_number(group, ("resident_ctas_per_sm", "resident_blocks_per_sm",
                                                   "active_ctas_per_sm", "blocks_per_sm"))
        if explicit_resident:
            limits.append(explicit_resident)
        if default_resident:
            limits.append(default_resident)
        if group_threads:
            thread_limit = _value_number(hardware, ("max_threads_per_sm", "threads_per_sm"))
            if thread_limit:
                limits.append(thread_limit / group_threads)
        if shared:
            shared_limit = _value_number(hardware, ("shared_bytes_per_sm",))
            if shared_limit:
                limits.append(shared_limit / shared)
        if registers and group_threads:
            register_limit = _value_number(hardware, ("registers_per_sm", "registers_per_sm_bytes"))
            if register_limit:
                limits.append(register_limit / (registers * group_threads))
        resident = min(limits) if limits else 1.0
        resident = max(1.0, resident)
        if sm_count is None:
            continue
        capacity = max(1.0, sm_count * resident)
        for waves in targets:
            candidate = int(math.ceil(base_batch * waves * capacity / grid))
            for delta in (-1, 0, 1):
                nearby = candidate + delta
                if minimum <= nearby <= maximum:
                    loads.add(nearby)
    if requested_max > maximum:
        omitted.append(f"requested-batch-{requested_max}-exceeds-memory-bound-{maximum}")
    if maximum >= minimum:
        loads.add(maximum)
    else:
        omitted.append("requested-range-below-minimum-batch")
    return sorted(loads), omitted


def _candidate_midpoint(loads: list[int], measured: set[int], *, minimum: int, maximum: int) -> int | None:
    available = sorted({int(value) for value in loads if minimum <= int(value) <= maximum})
    known = sorted({minimum, maximum, *measured})
    if len(known) < 2:
        return None
    # Largest logarithmic gap prioritizes low-load residency transitions while
    # retaining an integer batch contract.  Ties are deterministic.
    gaps = []
    for left, right in zip(known, known[1:]):
        if right - left <= 1:
            continue
        gap = math.log1p(right) - math.log1p(left)
        gaps.append((-gap, left, right))
    for _, left, right in sorted(gaps):
        midpoint = (left + right) // 2
        if midpoint <= left:
            midpoint = left + 1
        if midpoint >= right:
            midpoint = right - 1
        if midpoint not in measured and minimum <= midpoint <= maximum:
            return midpoint
        for candidate in available:
            if left < candidate < right and candidate not in measured:
                return candidate
    return None


def _candidate_training_batches(candidate: dict[str, Any], records: list[dict[str, Any]]) -> list[int]:
    batches = set()
    for record in records:
        if record.get("candidate_key") != candidate.get("key") or record.get("status") != "measured":
            continue
        if any(_truthy(group.get("independent")) and
               str(group.get("role", record.get("role", "train"))).lower() == "train"
               for group in record.get("groups", [])):
            if "batch" in record:
                batches.add(int(record["batch"]))
    return sorted(batches)


def _next_interval(candidate: dict[str, Any], records: list[dict[str, Any]]) -> tuple[int, int, int] | None:
    """Choose an unvalidated integer midpoint between training endpoints."""
    training = _candidate_training_batches(candidate, records)
    if len(training) < 2:
        return None
    states = {str(key): str(value) for key, value in candidate.get("intervals", {}).items()}
    options = []
    for left, right in zip(training, training[1:]):
        key = f"{left}:{right}"
        if right - left <= 1 or states.get(key) in {"passed", "exact"}:
            continue
        midpoint = (left + right) // 2
        if midpoint <= left or midpoint >= right:
            continue
        options.append((-(math.log1p(right) - math.log1p(left)), left, right, midpoint))
    if not options:
        return None
    _, left, right, midpoint = sorted(options)[0]
    return left, right, midpoint


def _validation_interval_state(validation: dict[str, Any]) -> str:
    """Reduce one holdout's per-group checks to an interval state."""
    checks = validation.get("checks", []) if isinstance(validation, dict) else []
    if not checks:
        return "uncovered"
    if any(item.get("promote") for item in checks if isinstance(item, dict)):
        return "failed"
    if all(isinstance(item, dict) and item.get("status") == "passed" for item in checks):
        return "passed"
    return "uncovered"


def _interval_coverage(candidate: dict[str, Any], records: list[dict[str, Any]],
                       *, max_points: int = DEFAULT_MAX_POINTS) -> dict[str, Any]:
    """Report every current training interval, including unresolved ones.

    A failed interval disappears as soon as its holdout is promoted to a
    training endpoint and is replaced by the two child intervals.  This makes
    the checkpoint describe the actual adaptive tree rather than claiming
    that one successful midpoint covers the whole curve.
    """
    training = _candidate_training_batches(candidate, records)
    states = {str(key): str(value) for key, value in candidate.get("intervals", {}).items()}
    intervals = []
    for left, right in zip(training, training[1:]):
        key = f"{left}:{right}"
        if right - left <= 1:
            state = "exact"
        else:
            state = states.get(key, "unvalidated")
        intervals.append({"left": left, "right": right, "state": state,
                          "midpoint": (left + right) // 2 if right - left > 1 else None})
    legal_min = int(candidate.get("legal_min_batch", candidate.get("minimum_batch", 1)))
    legal_max = candidate.get("legal_max_batch", candidate.get("requested_batch"))
    try:
        legal_max = int(legal_max)
    except (TypeError, ValueError):
        legal_max = legal_min - 1
    measured = {int(record["batch"]) for record in records
                if record.get("candidate_key") == candidate.get("key")
                and record.get("status") == "measured"
                and _truthy(record.get("correct", False))
                and "batch" in record}
    span = legal_max - legal_min + 1
    exact_discrete = (0 < span <= max_points and len(measured) == span and
                      all(value in measured for value in range(legal_min, legal_max + 1)))
    unresolved = [item for item in intervals if item["state"] not in {"passed", "exact"}]
    validated = bool(intervals) and not unresolved
    if exact_discrete:
        # An exact integer inventory needs no interpolation claim.  Keep this
        # distinct from holdout validation so the installation gate can still
        # require an independent holdout when it needs one.
        validation_mode = "exact-only"
        validation_complete = True
    elif validated:
        validation_mode = "validated"
        validation_complete = True
    else:
        validation_mode = "unvalidated"
        validation_complete = False
    return {"intervals": intervals, "training_batches": training,
            "unvalidated_intervals": unresolved,
            "exact_discrete_load_coverage": exact_discrete,
            "validation_mode": validation_mode,
            "validation_complete": validation_complete}


def _trial_mad_relative(values: Any) -> float:
    if not isinstance(values, (list, tuple)) or not values:
        return 0.0
    numbers = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0:
            numbers.append(number)
    if not numbers:
        return 0.0
    median = statistics.median(numbers)
    return statistics.median([abs(number - median) for number in numbers]) / median


def _prediction(training: list[tuple[float, float]], k: float) -> float | None:
    training = sorted(training)
    if not training:
        return None
    exact = [value for point, value in training if point == k]
    if exact:
        return exact[0]
    if k < training[0][0] or k > training[-1][0]:
        return None
    for (left_k, left_v), (right_k, right_v) in zip(training, training[1:]):
        if left_k <= k <= right_k:
            if right_k == left_k:
                return left_v
            fraction = (k - left_k) / (right_k - left_k)
            return left_v + fraction * (right_v - left_v)
    return None


def _group_k(group: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    try:
        # ``work_blocks`` is the physical service-load coordinate.  Older
        # probe records only emitted ``grid_ctas``; that field is an alias,
        # not a denominator.  Dividing the two would collapse every curve to
        # an occupancy ratio and make batch-load calibration disappear.
        work_value = group.get("work_blocks")
        if work_value in (None, ""):
            work_value = group.get("work", group.get("grid_ctas"))
        grid_value = group.get("grid_ctas")
        if grid_value in (None, ""):
            grid_value = work_value
        work = float(work_value)
        grid = float(grid_value)
    except (TypeError, ValueError):
        return None, None, None
    if not math.isfinite(work) or not math.isfinite(grid) or work <= 0 or grid <= 0:
        return None, work, grid
    return work, work, grid


def _adaptive_validation(records: list[dict[str, Any]], record: dict[str, Any]) -> dict[str, Any]:
    """Assess a just-measured holdout using training records only."""
    checks = []
    if (record.get("status") != "measured" or
            not _truthy(record.get("correct", False))):
        record["validation"] = {
            "checks": [{"group": None, "status": "unavailable",
                        "reason": record.get("reason", "record-not-measured")}],
            "failed_groups": [], "promoted": False,
        }
        return record["validation"]
    for index, group in enumerate(record.get("groups", [])):
        if not _truthy(group.get("independent")):
            checks.append({"group": index, "status": "unavailable", "reason": "not-independent"})
            continue
        values = group.get("trial_kernel_ms", [])
        if not values:
            checks.append({"group": index, "status": "unavailable", "reason": "missing-trials"})
            continue
        key = _model_curve_key(record.get("sample", {}), group)
        training = []
        for prior in records:
            if prior is record or prior.get("status") != "measured" or not _truthy(prior.get("correct", False)):
                continue
            for prior_group in prior.get("groups", []):
                if (not _truthy(prior_group.get("independent")) or
                        str(prior_group.get("role", prior.get("role", "train"))).lower() != "train"):
                    continue
                if _model_curve_key(prior.get("sample", {}), prior_group) != key:
                    continue
                prior_k, _, _ = _group_k(prior_group)
                prior_values = prior_group.get("trial_kernel_ms", [])
                if prior_k is not None and prior_values:
                    training.append((prior_k, statistics.median(float(value) for value in prior_values)))
        k, _, _ = _group_k(group)
        try:
            observed = statistics.median(float(value) for value in values)
        except (TypeError, ValueError, statistics.StatisticsError):
            checks.append({"group": index, "status": "unavailable", "reason": "invalid-trials"})
            continue
        predicted = _prediction(training, k) if k is not None else None
        mad = _trial_mad_relative(values)
        error = abs(observed - predicted) / observed if predicted is not None and observed > 0 else None
        failed = error is not None and error > 0.10 + mad
        checks.append({"group": index, "curve_key": key, "k": k,
                       "observed_kernel_ms": observed, "predicted_kernel_ms": predicted,
                       "trial_mad_relative": mad, "relative_error": error,
                       "status": "failed" if failed else ("passed" if error is not None else "uncovered"),
                       "promote": bool(failed)})
        if failed:
            group["role"] = "train"
    record["validation"] = {"checks": checks,
                            "failed_groups": [item["group"] for item in checks if item.get("promote")],
                            "promoted": any(item.get("promote") for item in checks)}
    return record["validation"]


def _record_groups_for_candidate(records: list[dict[str, Any]], candidate_key: str) -> list[dict[str, Any]]:
    groups = []
    for record in records:
        if record.get("candidate_key") == candidate_key and record.get("status") == "measured":
            groups.extend(record.get("groups", []))
    return groups


def _service_mapping(sample: dict[str, Any]) -> dict[str, Any]:
    mapping = _decode_mapping(sample.get("mapping_json", {}))
    return mapping if isinstance(mapping, dict) else {}


def _service_value(sample: dict[str, Any], name: str, default: Any = None) -> Any:
    if name in sample and sample[name] not in (None, ""):
        return sample[name]
    return _service_mapping(sample).get(name, default)


def _service_policy_values(sample: dict[str, Any], name: str) -> list[Any]:
    mapping = _service_mapping(sample)
    values = []
    for source in (sample, mapping):
        if name in source and source[name] not in (None, ""):
            values.append(source[name])
    return values


def _service_integer(value: Any) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer() or number < 0:
        return None
    return int(number)


def _service_group_integer(group: dict[str, Any], name: str) -> int | None:
    return _service_integer(group.get(name))


def _service_text(value: Any) -> str:
    return str(value).strip().lower().replace("_", "-") if value not in (None, "") else ""


def _service_prefix_lanes(sample: dict[str, Any]) -> int | None:
    """Return strict cooperative prefix lanes, canonicalizing omitted to one."""
    value = _service_value(sample, "prefix_codelet_lanes", 1)
    if isinstance(value, bool):
        return None
    lanes = _service_integer(value)
    if lanes is None or lanes <= 0 or lanes & (lanes - 1):
        return None
    return lanes


def _register_prefix_shape(sample: dict[str, Any], stages: int, threads: int,
                           ept: int) -> tuple[int, int, int] | None:
    """Validate ``(G, Columns, units_per_cta)`` for a register prefix."""
    mapping = _service_mapping(sample)
    try:
        parts, _ = resident_mapping.normalize_axes(mapping)
    except ValueError:
        return None
    if parts and parts[0]:
        if len(parts[0]) != 2 or sum(parts[0]) != stages:
            return None
        log_a, log_b = parts[0]
    else:
        if stages % 2:
            return None
        log_a = log_b = stages // 2
    lanes = _service_prefix_lanes(sample)
    if lanes is None:
        return None
    first = 1 << log_a
    second = 1 << log_b
    if lanes > first or first % lanes or threads > 1024:
        return None
    if ept != first // lanes or ept > second or second % ept:
        return None
    local_n = 1 << stages
    product = threads * ept
    if threads <= 0 or product % local_n:
        return None
    columns = product // local_n
    second_lanes = second // ept
    if columns <= 0 or columns & (columns - 1):
        return None
    if lanes * columns > 32 or second_lanes * columns > 32:
        return None
    return lanes, columns, product // local_n


def _online_group_resources_known(group: dict[str, Any]) -> bool:
    if (not _truthy(group.get("compiler_resources_known")) or
            not _truthy(group.get("compiler_local_resources_known"))):
        return False
    return all(_service_group_integer(group, field) is not None for field in (
        "compiler_registers_per_thread", "compiler_local_bytes_per_thread",
        "live_shared_bytes", "dynamic_shared_bytes",
    ))


def _online_single_kernel_known(group: dict[str, Any]) -> bool:
    if group.get("resource_query_error") not in (None, ""):
        return False
    if _service_group_integer(group, "kernel_launch_count") != 1:
        return False
    kernels = group.get("actual_kernels")
    if not isinstance(kernels, list) or len(kernels) != 1 or not isinstance(kernels[0], dict):
        return False
    kernel = kernels[0]
    if (not _truthy(kernel.get("compiler_resources_known")) or
            not _truthy(kernel.get("compiler_local_resources_known"))):
        return False
    if not _online_group_resources_known(group):
        return False
    for field in ("grid_ctas", "threads", "dynamic_shared_bytes", "live_shared_bytes",
                  "compiler_registers_per_thread", "compiler_local_bytes_per_thread"):
        group_value = _service_group_integer(group, field)
        kernel_value = _service_integer(kernel.get(field))
        if group_value is None or kernel_value is None or group_value != kernel_value:
            return False
    return True


def _online_service_signature(description: dict[str, Any], group: dict[str, Any]) -> str | None:
    """Return a reuse key only for a verified two-group online FFT."""
    sample = description.get("sample", {})
    groups = description.get("groups", [])
    if description.get("status") != "resolved" or not isinstance(groups, list) or len(groups) != 2:
        return None
    if (_service_text(_service_value(sample, "operator")) != "fft" or
            _service_text(_service_value(sample, "backend")) != "online-reorder"):
        return None
    fft_core = _service_text(_service_value(sample, "fft_core"))
    if fft_core not in {"cufftdx-block", "register-tile"}:
        return None
    lanes = _service_prefix_lanes(sample)
    if lanes is None or (fft_core != "register-tile" and lanes != 1):
        return None
    if any(_truthy(value) for value in _service_policy_values(sample, "stage_overlap")):
        return None
    if any(_truthy(value) for value in _service_policy_values(sample, "factor_overlap")):
        return None
    for name in ("batch_tile_count", "factor_slices"):
        values = _service_policy_values(sample, name)
        if any(_service_integer(value) != 1 for value in values):
            return None

    point = description.get("point", {})
    batch = _service_integer(point.get("batch"))
    if batch is None:
        batch = _service_integer(_service_value(sample, "batch"))
    sample_batch = _service_integer(_service_value(sample, "batch"))
    if batch is None or batch <= 0 or (sample_batch is not None and sample_batch != batch):
        return None
    log_n = _service_integer(_service_value(sample, "logN", _service_value(sample, "log_n")))
    if log_n is None or log_n <= 0 or log_n >= 63:
        return None

    intervals = []
    for item in groups:
        first = _service_group_integer(item, "first_stage")
        stages = _service_group_integer(item, "stage_count")
        if (first is None or stages is None or stages <= 0 or first + stages > log_n or
                not _online_single_kernel_known(item)):
            return None
        intervals.append((first, stages))
    intervals.sort()
    if (intervals[0][0] != 0 or intervals[0][0] + intervals[0][1] != intervals[1][0] or
            intervals[1][0] + intervals[1][1] != log_n):
        return None

    first = _service_group_integer(group, "first_stage")
    stages = _service_group_integer(group, "stage_count")
    if first is None or stages is None or (first, stages) not in intervals:
        return None
    position = 0 if first == 0 else 1
    expected_core = ("register-tile" if position == 0 else "cufftdx-block") \
        if fft_core == "register-tile" else "cufftdx-block"
    if _service_text(group.get("core")) != expected_core:
        return None
    # New register-prefix strategies are physical codelet/layout choices.
    # Validate them when the descriptor exposes the fields, while retaining
    # compatibility with older probe records that only reported ``core``.
    requested_codelet = _service_text(_service_value(sample, "prefix_codelet", "native"))
    if "codelet" in group:
        expected_codelet = requested_codelet if position == 0 else "native"
        if _service_text(group.get("codelet")) != expected_codelet:
            return None
    requested_layout = _service_text(_service_value(sample, "prefix_shared_layout", "linear"))
    if position == 0 and "shared_layout" in group:
        # The descriptor is authoritative when it carries a dedicated prefix
        # layout.  Older descriptors may still expose the boundary layout;
        # only reject an explicit new-axis mismatch.
        descriptor_layout = _service_text(group.get("prefix_shared_layout", group.get("shared_layout")))
        if "prefix_shared_layout" in group and descriptor_layout != requested_layout:
            return None
    if _service_group_integer(group, "batch_space") != 1:
        return None
    if not _online_single_kernel_known(group):
        return None

    if fft_core == "register-tile":
        mapping = _service_mapping(sample)
        geometry_mapping = {**sample, **mapping, "logN": log_n,
                            "local_stages": intervals[0][1]}
        # Older stage probes did not serialize the public resident axes and
        # used a broader legacy register-tile EPT convention.  Enforce the
        # rectangular/chunk geometry only when the new fields are present;
        # current C++ descriptors always carry both (possibly empty) arrays.
        if ("local_stage_partitions" in geometry_mapping or
                "exchange_chunks" in geometry_mapping):
            try:
                geometry = resident_mapping.register_geometry(
                    geometry_mapping, [intervals[0][1], intervals[1][1]])
            except (TypeError, ValueError, KeyError):
                return None
            expected_chunk = int(geometry["suffix_chunk"]) if position == 1 else 0
            reported_chunk = _service_group_integer(group, "exchange_chunk")
            if reported_chunk is not None and reported_chunk != expected_chunk:
                return None

    threads = _service_group_integer(group, "threads")
    ept = _service_group_integer(group, "elements_per_thread")
    if threads is None or ept is None or threads <= 0 or ept <= 0:
        return None
    numerator = batch * (1 << (log_n - stages))
    if fft_core == "register-tile":
        if position == 0:
            shape = _register_prefix_shape(sample, stages, threads, ept)
            if shape is None:
                return None
            _, _, units = shape
            requested_threads = _service_integer(_service_value(sample, "prefix_threads"))
            requested_ept = _service_integer(_service_value(sample, "prefix_ept"))
            if ((requested_threads is not None and requested_threads != threads) or
                    (requested_ept is not None and requested_ept != ept)):
                return None
        else:
            divisor = 1 << stages
            product = threads * ept
            if product % divisor:
                return None
            units = product // divisor
        reported_units = _service_group_integer(group, "units_per_cta")
        if reported_units is not None and reported_units != units:
            return None
        if units <= 0 or numerator % units:
            return None
        expected = numerator // units
    else:
        divisor = 1 << stages
        product = threads * ept
        if product < divisor or product % divisor:
            return None
        units = product // divisor
        expected = (numerator + units - 1) // units
    work = _service_group_integer(group, "work_blocks")
    grid = _service_group_integer(group, "grid_ctas")
    if work is None or grid is None or work != expected or grid != expected:
        return None
    return _model_curve_key(sample, group)


def _service_record_matches(description: dict[str, Any], record: dict[str, Any],
                            expected_keys: list[str]) -> bool:
    sample = record.get("sample", {})
    groups = record.get("groups", [])
    record_description = {"status": "resolved", "sample": sample,
                          "point": {"batch": record.get("batch", sample.get("batch"))},
                          "groups": groups}
    keys = [_shared_service_signature(record_description, group) for group in groups]
    return len(keys) == len(expected_keys) and keys == expected_keys


def _shared_service_signature(description: dict[str, Any], group: dict[str, Any]) -> str | None:
    """Keep the legacy SharedIterative contract and add verified online FFTs."""
    sample = description.get("sample", {})
    if _service_text(_service_value(sample, "backend")) == "online-reorder":
        return _online_service_signature(description, group)
    if (description.get("status") != "resolved" or sample.get("backend") != "shared-iterative"
            or not _truthy(group.get("independent"))
            or not _truthy(group.get("compiler_resources_known"))
            or int(group.get("batch_space", 0)) != 1
            or int(group.get("kernel_launch_count", 0)) != 1):
        return None
    log_n = int(sample.get("logN", 0))
    stages = int(group.get("stage_count", 0))
    batch = int(description.get("point", {}).get("batch", 0))
    if not 0 < stages <= log_n or batch <= 0:
        return None
    expected = batch * (1 << (log_n - stages))
    if group.get("work_blocks") != expected or group.get("grid_ctas") != expected:
        return None
    return _model_curve_key(sample, group)


def _reusable_shared_services(candidate: dict[str, Any], candidates: list[dict[str, Any]],
                              records: list[dict[str, Any]], *, max_points: int) -> list[dict[str, Any]]:
    """Reference independently validated physical curves, never synthetic timings.

    Whole-plan correctness and composition still need the public-plan gate.
    This only avoids resampling an identical physical stage/load interval.
    """
    description = candidate.get("description", {})
    groups = description.get("groups", [])
    target_keys = [_shared_service_signature(description, group) for group in groups]
    if not target_keys or any(key is None for key in target_keys) or candidate.get("omitted"):
        return []
    required = set(target_keys)
    sources = {}
    for source in candidates:
        if source is candidate or source.get("service_reuse") or source.get("omitted"):
            continue
        if (source.get("minimum_batch", 1) > candidate.get("minimum_batch", 1) or
                source.get("requested_batch", 0) < candidate.get("requested_batch", 1)):
            continue
        intervals = _interval_coverage(source, records, max_points=max_points)
        if not intervals["validation_complete"] or intervals["validation_mode"] != "validated":
            continue
        source_description = source.get("description", {})
        source_groups = source_description.get("groups", [])
        source_keys = [_shared_service_signature(source_description, group) for group in source_groups]
        if not source_keys or any(key is None for key in source_keys):
            continue
        evidence = [record for record in records if record.get("candidate_key") == source["key"]
                    and record.get("status") == "measured" and _truthy(record.get("correct"))
                    and _truthy(record.get("measurement_exclusive_gpu"))]
        if _service_text(_service_value(source_description.get("sample", {}), "backend")) == "online-reorder":
            if any(not _service_record_matches(source_description, record, source_keys)
                   for record in evidence):
                continue
        batches = {int(record["batch"]) for record in evidence}
        if not {int(source["minimum_batch"]), int(source["requested_batch"])} <= batches:
            continue
        for group, key in zip(source_groups, source_keys):
            if key not in required or key in sources:
                continue
            sources[key] = {"curve_key": key, "source_candidate_key": source["key"],
                            "minimum_batch": int(source["minimum_batch"]),
                            "maximum_batch": int(source["requested_batch"]),
                            "validation_mode": "validated",
                            "validation_intervals": intervals["intervals"]}
        if required <= sources.keys():
            return [dict(sources[key], group_index=index) for index, key in enumerate(target_keys)]
    return []


def _coverage(candidates: list[dict[str, Any]], records: list[dict[str, Any]], *, mode: str,
              max_points: int, budget: int | None, status: str,
              model: dict[str, Any] | None = None,
              composition_requests: list[dict[str, Any]] | None = None,
              include_composition_requests: bool = True) -> dict[str, Any]:
    curves: dict[str, dict[str, Any]] = OrderedDict()
    non_executable: list[dict[str, Any]] = []
    adapter_gaps: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_records = [record for record in records
                             if record.get("candidate_key") == candidate["key"]]
        measured_candidate_records = [record for record in candidate_records
                                      if record.get("status") == "measured"]
        description = candidate.get("description", {})
        if not candidate_records and str(description.get("status", "")).lower() != "resolved":
            # A rejected plan is a non-executable inventory item, not a
            # missing service curve.  Keep it visible without making every
            # otherwise valid full calibration fail.
            non_executable.append({"candidate_key": candidate["key"],
                                   "reason": description.get("raw", {}).get("reason",
                                                                               description.get("status", "unavailable")),
                                   "point": candidate.get("point")})
            continue
        measured_batches = sorted({int(record["batch"]) for record in candidate_records
                                   if record.get("status") == "measured"
                                   and _truthy(record.get("correct", False))
                                   and "batch" in record})
        independent_keys = set()
        unavailable_groups = sum(1 for group in description.get("groups", [])
                                 if not _truthy(group.get("independent")))
        roles: dict[str, set[int]] = {}
        actual_keys = set()
        coordinates: dict[str, set[float]] = {}
        for record in measured_candidate_records:
            for group in record.get("groups", []):
                if not _truthy(group.get("independent")):
                    unavailable_groups += 1
                    continue
                try:
                    key = _model_curve_key(record.get("sample", {}), group)
                except Exception:
                    key = "unresolved"
                actual_keys.add(key)
                independent_keys.add(key)
                k, _, _ = _group_k(group)
                if k is not None:
                    coordinates.setdefault(key, set()).add(float(k))
                roles.setdefault(str(group.get("role", record.get("role", "train"))), set()).add(int(record["batch"]))
        if (str(description.get("status", "")).lower() == "resolved" and
                description.get("groups") and not independent_keys and not candidate.get("service_reuse")):
            adapter_gaps.append({"candidate_key": candidate["key"],
                                 "reason": "resolved-groups-not-independent"})
        interval_info = _interval_coverage(candidate, records, max_points=max_points)
        curve_key = sorted(actual_keys)[0] if len(actual_keys) == 1 else candidate.get("actual_key", candidate["key"])
        entry = curves.setdefault(curve_key, {
            "candidate_key": candidate["key"], "actual_curve_keys": sorted(actual_keys),
            "requested_batch": candidate["requested_batch"],
            "minimum_batch": candidate["minimum_batch"],
            "memory_bound_batch": candidate.get("memory_bound_batch"),
            "memory_available_bytes": candidate.get("memory_available_bytes"),
            "memory_limit_bytes": candidate.get("memory_limit_bytes"),
            "memory_requested_bytes": candidate.get("memory_requested_bytes"),
            "seed_batches": candidate.get("seed_batches", []),
            "reserved_holdout": candidate.get("reserved_holdout"),
            "adaptive_batches": candidate.get("adaptive_batches", []),
            "measured_batches": measured_batches, "coordinates": {
                key: sorted(values) for key, values in coordinates.items()
            }, "training_batches": sorted(roles.get("train", set())),
            "validation_batches": sorted(roles.get("validation", set())),
            "promoted_batches": sorted(set(candidate.get("promoted_batches", [])) |
                                       roles.get("promoted", set())),
            "validation_intervals": interval_info["intervals"],
            "unvalidated_intervals": interval_info["unvalidated_intervals"],
            "validation_mode": interval_info["validation_mode"],
            "validation_complete": interval_info["validation_complete"],
            "exact_discrete_load_coverage": interval_info["exact_discrete_load_coverage"],
            "independent_curve_count": len(independent_keys),
            "unavailable_group_count": unavailable_groups,
            "omitted": list(candidate.get("omitted", [])),
            "point_count": max((len(values) for values in coordinates.values()), default=0),
            "measured_load_count": len(measured_batches), "max_points": max_points,
        })
        if entry["measured_batches"]:
            entry["measured_load_count"] = len(entry["measured_batches"])
        if candidate.get("service_reuse"):
            reuse = candidate["service_reuse"]
            entry.update(service_reuse=reuse, service_coverage_source="validated-physical-stage-reuse",
                         actual_curve_keys=sorted({item["curve_key"] for item in reuse}),
                         independent_curve_count=len(reuse), validation_complete=True,
                         validation_mode="reused-validated", unvalidated_intervals=[])
    for entry in curves.values():
        requested = int(entry["requested_batch"])
        entry["requested_batch_measured"] = requested in entry["measured_batches"]
        entry["one_point_only"] = entry["point_count"] < 2
        entry["complete"] = bool(entry["requested_batch_measured"] and not entry["omitted"] and
                                  entry["point_count"] >= 2 and entry["independent_curve_count"] and
                                  not entry["unavailable_group_count"] and
                                  entry["validation_complete"])
        if entry.get("service_reuse"):
            entry["complete"] = True
            entry["one_point_only"] = False
    one_point = [key for key, entry in curves.items() if entry["one_point_only"]]
    omitted = sum(len(entry["omitted"]) for entry in curves.values())
    unsupported = sum(entry["unavailable_group_count"] for entry in curves.values())
    validation_complete = bool(curves) and all(entry["validation_complete"] for entry in curves.values())
    complete = bool(curves) and all(entry["complete"] for entry in curves.values()) and not adapter_gaps
    if model and not model.get("training_groups"):
        complete = False
    result = {
        "schema": "cubutterfly-stage-coverage-v1", "version": VERSION,
        "mode": mode, "status": status, "curves": curves,
        "curve_count": len(curves), "records": len(records),
        "measured_records": sum(record.get("status") == "measured" for record in records),
        "measured_points": sum(entry["point_count"] for entry in curves.values()),
        "omitted_count": omitted, "unsupported_group_count": unsupported,
        "one_point_curves": one_point, "coverage_complete": complete,
        "validation_complete": validation_complete,
        "validation_modes": sorted({entry["validation_mode"] for entry in curves.values()}),
        "budget": budget, "max_points_per_curve": max_points,
        "composition_request_count": len(composition_requests or []),
        "non_executable": non_executable, "non_executable_count": len(non_executable),
        "adapter_gaps": adapter_gaps, "adapter_gap_count": len(adapter_gaps),
        "service_reused_candidates": sum(bool(candidate.get("service_reuse")) for candidate in candidates),
        "limitations": (["requested loads omitted by memory bound"] if omitted else []) +
                       (["one-point-only curves cannot establish a service curve"] if one_point else []) +
                       (["some execution groups are not independently measurable"] if unsupported else []) +
                       (["resolved plans have no independently measurable groups"] if adapter_gaps else []) +
                       (["some training intervals lack independent midpoint validation"]
                        if any(entry["unvalidated_intervals"] for entry in curves.values()) else []) +
                       (["exact discrete loads are covered without an interpolation holdout"]
                        if any(entry["exact_discrete_load_coverage"] for entry in curves.values()) else []),
    }
    if include_composition_requests:
        result["composition_requests"] = list(composition_requests or [])
    return result


def _atomic_write(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


COMPOSITION_SIDECAR_SCHEMA = "cubutterfly-stage-composition-audit-v1"


def _composition_digest(composition_requests: Iterable[dict[str, Any]]) -> str:
    """Hash the canonical audit list so each sidecar can remain immutable."""
    return _points_hash(composition_requests)


def _composition_sidecar_path(path: pathlib.Path, digest: str) -> pathlib.Path:
    return path.with_name(f"{path.name}.composition-{digest}.json")


def _composition_sidecar_ref(path: pathlib.Path,
                             composition_requests: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Write one content-addressed composition sidecar, never replacing it."""
    if not composition_requests:
        return None
    digest = _composition_digest(composition_requests)
    sidecar = _composition_sidecar_path(path, digest)
    if not sidecar.exists():
        _atomic_write(sidecar, {
            "schema": COMPOSITION_SIDECAR_SCHEMA,
            "version": VERSION,
            "sha256": digest,
            "count": len(composition_requests),
            "composition_requests": composition_requests,
        })
    return {"schema": COMPOSITION_SIDECAR_SCHEMA, "version": VERSION,
            "path": sidecar.name, "sha256": digest, "count": len(composition_requests)}


def _load_composition_sidecar(path: pathlib.Path, reference: Any) -> list[dict[str, Any]]:
    """Load and verify a content-addressed sidecar referenced by a checkpoint."""
    if not isinstance(reference, dict):
        raise ValueError("stage calibration composition sidecar reference must be an object")
    if reference.get("schema") not in (None, COMPOSITION_SIDECAR_SCHEMA):
        raise ValueError("unsupported stage calibration composition sidecar schema")
    name = reference.get("path")
    if not isinstance(name, str) or not name or pathlib.Path(name).name != name:
        raise ValueError("stage calibration composition sidecar path must be a filename")
    sidecar = path.parent / name
    try:
        document = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"stage calibration composition sidecar is not valid JSON: {sidecar}") from error
    if not isinstance(document, dict) or document.get("schema") != COMPOSITION_SIDECAR_SCHEMA:
        raise ValueError("unsupported stage calibration composition sidecar schema")
    composition_requests = document.get("composition_requests")
    if not isinstance(composition_requests, list) or any(
            not isinstance(request, dict) for request in composition_requests):
        raise ValueError("stage calibration composition sidecar requests must be objects")
    digest = _composition_digest(composition_requests)
    expected = reference.get("sha256")
    if not isinstance(expected, str) or digest != expected or document.get("sha256") != digest:
        raise ValueError("stage calibration composition sidecar hash mismatch")
    expected_count = reference.get("count", document.get("count"))
    if expected_count is not None and int(expected_count) != len(composition_requests):
        raise ValueError("stage calibration composition sidecar count mismatch")
    return composition_requests


def _checkpoint_composition_requests(path: pathlib.Path, document: dict[str, Any]) -> list[dict[str, Any]]:
    """Read new sidecars while retaining support for legacy embedded lists."""
    reference = document.get("composition_requests_ref")
    if reference is not None:
        return _load_composition_sidecar(path, reference)
    requests = document.get("composition_requests")
    if requests is None:
        coverage = document.get("coverage")
        requests = coverage.get("composition_requests") if isinstance(coverage, dict) else None
    if requests is None:
        coverage = document.get("coverage")
        count = document.get("composition_request_count")
        if count is None and isinstance(coverage, dict):
            count = coverage.get("composition_request_count")
        if count not in (None, 0):
            raise ValueError("stage calibration composition sidecar reference is missing")
        return []
    if not isinstance(requests, list) or any(not isinstance(request, dict) for request in requests):
        raise ValueError("stage calibration composition requests must be objects")
    return list(requests)


def _checkpoint_write(path: pathlib.Path, payload: dict[str, Any], *,
                      composition_requests_ref: dict[str, Any] | None = None) -> None:
    """Persist a compact checkpoint and keep the large composition audit external."""
    composition_requests = payload.get("composition_requests", [])
    if composition_requests is None:
        composition_requests = []
    if not isinstance(composition_requests, list) or any(
            not isinstance(request, dict) for request in composition_requests):
        raise ValueError("stage calibration composition requests must be objects")
    compact = dict(payload)
    from stage_service_model import VERSION as curve_model_version
    compact["curve_model_version"] = curve_model_version
    compact.pop("composition_requests", None)
    compact["composition_request_count"] = len(composition_requests)
    reference = composition_requests_ref
    if reference is not None:
        if (reference.get("count") != len(composition_requests) or
                (not composition_requests and reference)):
            raise ValueError("stage calibration composition sidecar reference is stale")
    else:
        reference = _composition_sidecar_ref(path, composition_requests)
    if reference is None:
        compact.pop("composition_requests_ref", None)
    else:
        compact["composition_requests_ref"] = reference
    coverage = compact.get("coverage")
    if isinstance(coverage, dict):
        coverage = dict(coverage)
        coverage.pop("composition_requests", None)
        coverage["composition_request_count"] = len(composition_requests)
        compact["coverage"] = coverage
    _atomic_write(path, compact)


def _restore_composition_coverage(coverage: dict[str, Any],
                                   composition_requests: list[dict[str, Any]]) -> dict[str, Any]:
    """Restore the public coverage audit list after reading a compact checkpoint."""
    result = dict(coverage)
    result["composition_requests"] = list(composition_requests)
    result["composition_request_count"] = len(composition_requests)
    return result


def _load_checkpoint(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"stage calibration checkpoint is not valid JSON: {path}") from error
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("unsupported stage calibration checkpoint schema")
    document["composition_requests"] = _checkpoint_composition_requests(path, document)
    if document.get("acquisition_inventory_ref"):
        from stage_checkpoint_import import _inventory_values
        _inventory_values(path, document)
    return document


def _refresh_curve_model_state(checkpoint: dict[str, Any], profile: dict[str, Any]) -> None:
    """Recompute model-derived keys/validation without discarding GPU trials."""
    from stage_service_model import VERSION as curve_model_version
    if checkpoint.get("curve_model_version") == curve_model_version:
        return
    candidates = {}
    remap = {}
    for old in checkpoint.get("candidates", []):
        fresh = _candidate_from_description(old["static_key"], old["description"], profile)
        fresh["aliases"] = list(old.get("aliases", fresh["aliases"]))
        remap[old["key"]] = fresh["key"]
        if fresh["key"] in candidates:
            _merge_candidate(candidates[fresh["key"]], fresh, profile)
        else:
            candidates[fresh["key"]] = fresh
    records = checkpoint.get("records", [])
    for record in records:
        record["candidate_key"] = remap.get(record.get("candidate_key"), record.get("candidate_key"))
        record.pop("validation", None)
        for group in record.get("groups", []):
            group["role"] = record.get("role", "train")
    # Reassess holdouts in their original order: a formerly promoted point
    # becomes training again only if it still fails under the current model.
    for record in records:
        if record.get("role") != "validation":
            continue
        validation = _adaptive_validation(records, record)
        candidate = candidates.get(record.get("candidate_key"))
        if candidate is not None:
            candidate["adaptive_batches"].append(int(record["batch"]))
            if validation.get("promoted"):
                candidate.setdefault("promoted_batches", []).append(int(record["batch"]))
    checkpoint["candidates"] = list(candidates.values())
    checkpoint["curve_model_version"] = curve_model_version
    checkpoint["model"] = None
    checkpoint["status"] = "model-rebuild-required"


def _acquisition_inventory(path: pathlib.Path, checkpoint: dict[str, Any] | None,
                           static_keys: Iterable[str]) -> tuple[set[str], dict[str, Any]]:
    """Persist the complete request universe once, including unvisited plans."""
    hashes = {hashlib.sha256(key.encode()).hexdigest() for key in static_keys}
    reference = (checkpoint or {}).get("acquisition_inventory_ref")
    if reference:
        prior_path = path.parent / reference["path"]
        prior = json.loads(prior_path.read_text())
        if (_points_hash(prior) != reference["sha256"] or
                not isinstance(prior, list) or len(prior) != reference["count"]):
            raise ValueError("stage acquisition inventory sidecar mismatch")
        hashes.update(prior)
    ordered = sorted(hashes)
    digest = _points_hash(ordered)
    sidecar = path.with_name(f"{path.stem}.inventory-{digest}.json")
    if not sidecar.exists():
        temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(ordered) + "\n")
        temporary.replace(sidecar)
    return hashes, {"path": sidecar.name, "sha256": digest, "count": len(ordered)}


def _overall_status(candidates: list[dict[str, Any]], records: list[dict[str, Any]], coverage: dict[str, Any],
                    model: dict[str, Any], *, mode: str, interrupted: bool = False,
                    budget_exhausted: bool = False) -> str:
    if interrupted:
        return "interrupted"
    if budget_exhausted:
        return "incomplete-budget"
    if coverage.get("pending_description_count", 0):
        return "incomplete-inventory"
    if not candidates or not records:
        return "incomplete-unsupported"
    if not coverage.get("curve_count") or not model.get("training_groups"):
        return "incomplete-unsupported"
    if coverage.get("one_point_curves"):
        return "incomplete-one-point-only"
    if coverage.get("omitted_count") or not coverage.get("coverage_complete"):
        return "incomplete"
    if not coverage.get("validation_complete"):
        return "incomplete-validation"
    if not model.get("validation", {}).get("passed"):
        # A finite integer load inventory can be fully measured without a
        # spare holdout.  It is complete as an exact lookup artifact, but is
        # intentionally not install-validated (see _result).
        if not all(mode == "exact-only" for mode in coverage.get("validation_modes", [])):
            return "incomplete-validation"
    # Historical post-exclusivity failures stay in the raw journal for audit,
    # but a later clean retry is allowed to complete the curve.  Other
    # non-measured rows (for example a resolved plan that returned unavailable)
    # remain a coverage failure.
    if any(record.get("status") not in {"measured", "contaminated"} for record in records):
        return "incomplete-unsupported"
    return "complete" if mode == "full" else "complete-bounded"


def _result(model: dict[str, Any], coverage: dict[str, Any], status: str,
            *, composition_requests: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return the model profile itself, with orchestration status alongside it.

    Raw probe rows and describe documents belong to the checkpoint envelope.
    Keeping them out of this object is important because the installation
    profile is copied into the runtime registry and should remain a compact,
    model-shaped artifact.
    """
    result = dict(model)
    result["calibration_status"] = status
    result["coverage"] = coverage
    result["composition_requests"] = list(composition_requests or coverage.get("composition_requests", []))
    validation = model.get("validation", {})
    result["validated"] = bool(
        status in {"complete", "complete-bounded"}
        and coverage.get("coverage_complete")
        and not coverage.get("unsupported_group_count")
        and not coverage.get("adapter_gap_count")
        and not coverage.get("omitted_count")
        and not coverage.get("one_point_curves")
        and bool(validation.get("passed"))
    )
    return result


def calibrate(binary: pathlib.Path | str, path: pathlib.Path | str, profile: dict[str, Any],
              points: Iterable[dict[str, Any]], run: Callable[..., Any],
              exclusive: Callable[[], Any], mode: str = "full", warmup: int = 10,
              repeat: int = 20, trials: int = 3, max_points: int | None = None,
              **kwargs: Any) -> dict[str, Any]:
    """Measure executable mappings and build an adaptive service profile.

    An explicit ``max_points`` caps distinct loads per probe configuration.
    Full-mode defaults reserve boundary seeds, independent midpoints and one
    refinement of each midpoint interval, with a floor of DEFAULT_MAX_POINTS.
    Optional bounded controls include
    ``budget``/``max_probes`` (global probe count) and ``max_seconds`` (wall
    time).  Wall time is consulted only in ``mode='bounded'`` unless
    ``enforce_wall_budget=True`` is supplied; full calibration has no implicit
    timeout.  Any runner or GPU reservation implementation remains the
    caller's responsibility.
    """
    if mode not in {"full", "bounded"}:
        raise ValueError("mode must be 'full' or 'bounded'")
    if not isinstance(profile, dict):
        raise ValueError("profile must be an object")
    warmup = _as_int(warmup, name="warmup", minimum=0)
    repeat = _as_int(repeat, name="repeat", minimum=1)
    trials = _as_int(trials, name="trials", minimum=1)
    automatic_point_limit = max_points is None
    max_points = _as_int(DEFAULT_MAX_POINTS if automatic_point_limit else max_points,
                         name="max_points", minimum=1)
    normalized_points = [_normal_point(point) for point in points]
    service_points, new_composition_requests = _service_points(normalized_points)
    path = pathlib.Path(path)
    binary = pathlib.Path(binary)
    extras = {
        "compile_policy": kwargs.get("compile_policy"),
        "protocol_label": kwargs.get("protocol_label"),
    }
    protocol = _protocol(mode=mode, warmup=warmup, repeat=repeat, trials=trials,
                         max_points=max_points, extra=extras)
    identity = _cache_identity(binary, profile, protocol, _points_hash(normalized_points))
    budget_value = kwargs.get("budget", kwargs.get("max_probes", kwargs.get("max_total_points")))
    budget = None if budget_value in (None, 0) else _as_int(budget_value, name="budget", minimum=1)
    wall_value = kwargs.get("max_seconds", kwargs.get("wall_budget_seconds", kwargs.get("time_budget")))
    wall_seconds = None if wall_value in (None, 0) else _finite_number(wall_value, name="max_seconds", positive=True)
    enforce_wall = bool(kwargs.get("enforce_wall_budget", mode == "bounded"))

    import_paths = kwargs.get("import_checkpoints", [])
    if import_paths:
        from stage_checkpoint_import import import_stage_checkpoints
        import_stage_checkpoints(path, import_paths, binary=binary, profile=profile,
                                 protocol=protocol)
    checkpoint = _load_checkpoint(path)
    if checkpoint is not None:
        old_identity = checkpoint.get("identity", {})
        if old_identity != identity:
            raise ValueError("stage calibration checkpoint identity/protocol mismatch")
        _refresh_curve_model_state(checkpoint, profile)
        records = list(checkpoint.get("records", []))
        descriptions = dict(checkpoint.get("descriptions", {}))
        candidates = list(checkpoint.get("candidates", []))
        composition_requests = list(checkpoint.get("composition_requests", []))
        known_compositions = {_json_key(item.get("point", item)) for item in composition_requests}
        previous_composition_keys = set(known_compositions)
        for request in new_composition_requests:
            key = _json_key(request.get("point", request))
            if key not in known_compositions:
                composition_requests.append(request)
                known_compositions.add(key)
        known_static_keys = set(descriptions)
        has_new_points = any(
            _point_static_key(point) not in known_static_keys or
            int(point["batch"]) > int(descriptions[_point_static_key(point)].get("point", {}).get("batch", 1))
            for point in service_points
        )
        has_new_compositions = any(_json_key(item.get("point", item)) not in previous_composition_keys
                                   for item in new_composition_requests)
        if (not has_new_points and not has_new_compositions and
                not checkpoint.get("coverage", {}).get("pending_description_count", 0) and
                checkpoint.get("status") in {"complete", "complete-bounded"} and checkpoint.get("model")):
            model = checkpoint["model"]
            coverage = _restore_composition_coverage(
                checkpoint.get("coverage", {}), composition_requests)
            cached_status = checkpoint.get("calibration_status", checkpoint["status"])
            # Mode and point caps govern orchestration, not measurement
            # identity.  Reusing a complete bounded artifact from a full call
            # should therefore not expose the old mode/status to the caller.
            status = "complete" if mode == "full" else "complete-bounded"
            if cached_status not in {"complete", "complete-bounded"}:
                status = cached_status
            coverage["mode"] = mode
            coverage["status"] = status
            coverage["max_points_per_curve"] = (coverage.get("max_points_per_curve", max_points)
                                                  if automatic_point_limit else max_points)
            coverage["budget"] = budget
            return _result(model, coverage, status,
                           composition_requests=composition_requests)
    else:
        records, descriptions, candidates = [], {}, []
        composition_requests = list(new_composition_requests)

    # Keep a stable, deduplicated audit list even when the same overlap point
    # is replayed in a cumulative invocation.
    known_compositions = {_json_key(item.get("point", item)) for item in composition_requests}
    for request in new_composition_requests:
        key = _json_key(request.get("point", request))
        if key not in known_compositions:
            composition_requests.append(request)
            known_compositions.add(key)

    # The composition audit can contain tens of thousands of requests.  Build
    # its immutable sidecar reference once after merging this invocation's
    # requests, then reuse it for every checkpoint write below.
    composition_requests_ref = _composition_sidecar_ref(path, composition_requests)

    # Order acquisition by new physical templates, but keep every requested
    # point. Only actual descriptors/resources may establish curve coverage.
    from stage_acquisition_order import order_points
    static_points: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for point in service_points:
        key = _point_static_key(point)
        previous = static_points.get(key)
        if previous is None or int(point["batch"]) > int(previous["batch"]):
            static_points[key] = point
    static_points = OrderedDict((_point_static_key(point), point)
                                for point in order_points(list(static_points.values())))
    requested_hashes, inventory_ref = _acquisition_inventory(
        path, checkpoint, set(static_points) | set(descriptions))

    def acquisition_coverage(coverage: dict[str, Any]) -> dict[str, Any]:
        examined = {hashlib.sha256(key.encode()).hexdigest() for key in descriptions}
        pending = len(requested_hashes - examined)
        coverage.update(requested_static_count=len(requested_hashes),
                        described_static_count=len(requested_hashes & examined),
                        pending_description_count=pending,
                        acquisition_policy="interleaved-physical-template-first-v1")
        if pending:
            coverage["coverage_complete"] = False
            coverage["validation_complete"] = False
        return coverage

    def write_checkpoint(payload: dict[str, Any]) -> None:
        payload["acquisition_inventory_ref"] = inventory_ref
        acquisition_coverage(payload.setdefault("coverage", {}))
        _checkpoint_write(path, payload,
                          composition_requests_ref=composition_requests_ref)

    by_actual: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for candidate in candidates:
        by_actual.setdefault(candidate.get("actual_key", candidate.get("key")), candidate)
    for static_key, description in descriptions.items():
        incoming = _candidate_from_description(static_key, description, profile)
        existing = by_actual.get(incoming["actual_key"])
        if existing is None:
            by_actual[incoming["actual_key"]] = incoming
        else:
            _merge_candidate(existing, incoming, profile)
    candidates = list(by_actual.values())
    started = time.monotonic()
    interrupted = False
    budget_exhausted = False

    def acquisition_budget_exhausted() -> bool:
        return bool((budget is not None and
                     sum(record.get("status") == "measured" for record in records) >= budget) or
                    (enforce_wall and wall_seconds is not None and
                     time.monotonic() - started >= wall_seconds))

    def candidate_stream():
        """Finish a configuration before compiling the next whole plan."""
        nonlocal budget_exhausted
        yield from list(candidates)
        for static_key, point in static_points.items():
            if (static_key in descriptions and
                    str(descriptions[static_key].get("status", "")).lower() == "resolved"):
                existing_description = descriptions[static_key]
                old_point = existing_description.get("point", {})
                if int(point["batch"]) <= int(old_point.get("batch", 1)):
                    continue
                # A larger cumulative load can change grid geometry, resource
                # limits, or workspace allocation.  Re-resolve it once and
                # replace the complete description; existing timed records
                # remain in the journal and are reused when their physical
                # curve key still matches the refreshed shape.
            if acquisition_budget_exhausted():
                budget_exhausted = True
                return
            raw = _invoke(binary, point, run, exclusive, warmup=warmup, repeat=repeat,
                          trials=trials, describe_only=True)
            if str(raw.get("status", "")).lower() == "contaminated":
                # A describe result is not a durable plan identity when the
                # post-check found another process.  Preserve the raw attempt
                # in the checkpoint, but leave the description uncached so a
                # resume retries the same point instead of treating it as a
                # permanently non-executable mapping.
                records.append(_normalize_measurement(
                    point, raw, profile, role="train", candidate_key=static_key,
                    batch=int(point["batch"]), measurement_exclusive=False,
                    warmup=warmup, repeat=repeat, trials=trials))
                raise InterruptedError("contaminated stage describe")
            descriptions[static_key] = _normalize_description(point, raw, profile)
            incoming = _candidate_from_description(static_key, descriptions[static_key], profile)
            existing = by_actual.get(incoming["actual_key"])
            if existing is None:
                existing = incoming
                by_actual[incoming["actual_key"]] = existing
                candidates.append(existing)
            else:
                _merge_candidate(existing, incoming, profile)
            # Persist a useful resume point even when the first timed sample
            # cannot be started because the GPU is currently unavailable.
            provisional = _coverage(candidates, records, mode=mode, max_points=max_points,
                                    budget=budget, status="running", model=None,
                                    composition_requests=composition_requests)
            write_checkpoint({"schema": SCHEMA, "version": VERSION, "status": "running",
                                 "identity": identity, "protocol": protocol,
                                 "records": records, "raw_records": records,
                                 "descriptions": descriptions, "candidates": candidates,
                                 "composition_requests": composition_requests,
                                 "coverage": provisional})
            yield existing

    write_checkpoint({"schema": SCHEMA, "version": VERSION, "status": "running",
                         "identity": identity, "protocol": protocol,
                             "records": records, "raw_records": records,
                             "descriptions": descriptions, "candidates": candidates,
                             "composition_requests": composition_requests,
                             "coverage": _coverage(candidates, records, mode=mode,
                                                    max_points=max_points, budget=budget,
                                                    status="running",
                                                    composition_requests=composition_requests)})
    try:
        for candidate in candidate_stream():
            candidate.setdefault("adaptive_batches", [])
            candidate.setdefault("seed_batches", candidate.get("loads", []))
            candidate.setdefault("intervals", {})
            candidate.setdefault("omitted", [])
            if automatic_point_limit and mode == "full":
                seeds = sorted(set(candidate["seed_batches"]))
                intervals = sum(right - left > 1 for left, right in zip(seeds, seeds[1:]))
                max_points = max(max_points, len(seeds) + 3 * intervals)
                protocol["max_points"] = max_points
            description = candidate.get("description") or descriptions.get(candidate.get("static_key"))
            if not isinstance(description, dict):
                candidate.setdefault("omitted", []).append("missing-description")
                continue
            if str(description.get("status", "")).lower() != "resolved":
                # Plan construction failures are inventory omissions.  They
                # are recorded by coverage and must never be sent to the
                # timed probe as if a plan had been resolved.
                reason = description.get("raw", {}).get("reason", description.get("status", "unavailable"))
                marker = f"non-executable:{reason}"
                if marker not in candidate.setdefault("omitted", []):
                    candidate["omitted"].append(marker)
                continue
            candidate.pop("service_reuse", None)
            if not any(record.get("candidate_key") == candidate["key"] for record in records):
                reused = _reusable_shared_services(candidate, candidates, records, max_points=max_points)
                if reused:
                    candidate["service_reuse"] = reused
                    continue
            point = dict(candidate["point"])
            seen_batches = {int(record["batch"]) for record in records
                            if record.get("candidate_key") == candidate["key"] and
                            "batch" in record and record.get("status") != "contaminated"}
            measured_batches = {int(record["batch"]) for record in records
                                if record.get("candidate_key") == candidate["key"] and
                                "batch" in record and record.get("status") == "measured" and
                                _truthy(record.get("correct", False))}
            candidate_unavailable = False
            # Do not re-run a point whose result was checkpointed, regardless
            # of whether that result was measured or explicitly unavailable.
            for batch in list(candidate.get("loads", [])):
                if len(seen_batches) >= max_points or batch in seen_batches:
                    continue
                if budget is not None and sum(record.get("status") == "measured" for record in records) >= budget:
                    budget_exhausted = True
                    break
                if enforce_wall and wall_seconds is not None and time.monotonic() - started >= wall_seconds:
                    budget_exhausted = True
                    break
                variant = dict(point, batch=int(batch))
                if "batch_tile_count" in variant:
                    variant["batch_tile_count"] = min(int(variant["batch_tile_count"]), int(batch))
                raw = _invoke(binary, variant, run, exclusive, warmup=warmup, repeat=repeat,
                              trials=trials, describe_only=False)
                measured_exclusive = str(raw.get("status", "")).lower() == "measured"
                record = _normalize_measurement(variant, raw, profile, role="train",
                                                candidate_key=candidate["key"], batch=int(batch),
                                                measurement_exclusive=measured_exclusive,
                                                warmup=warmup, repeat=repeat, trials=trials)
                records.append(record)
                if record.get("status") != "contaminated":
                    seen_batches.add(int(batch))
                if record.get("status") == "measured" and record.get("correct"):
                    measured_batches.add(int(batch))
                write_checkpoint({"schema": SCHEMA, "version": VERSION, "status": "running",
                                     "identity": identity, "protocol": protocol,
                                     "records": records, "raw_records": records,
                                     "descriptions": descriptions, "candidates": candidates,
                                     "composition_requests": composition_requests,
                                     "coverage": _coverage(candidates, records, mode=mode,
                                                            max_points=max_points, budget=budget,
                                                            status="running",
                                                            composition_requests=composition_requests)})
                if record.get("status") == "contaminated":
                    # The raw row is checkpointed above, but this calibration
                    # pass must stop so the contaminated load can be retried.
                    raise InterruptedError("contaminated stage measurement")
                if record.get("status") != "measured" or not record.get("correct"):
                    candidate.setdefault("omitted", []).append(f"batch-{batch}-unavailable")
                    candidate_unavailable = True
                    break
            # Adaptive midpoint validation uses only points already classified
            # as training.  A failed holdout is promoted before another
            # midpoint is selected; successful holdouts never enter the hull.
            while not candidate_unavailable and len(seen_batches) < max_points:
                if budget is not None and sum(record.get("status") == "measured" for record in records) >= budget:
                    budget_exhausted = True
                    break
                if enforce_wall and wall_seconds is not None and time.monotonic() - started >= wall_seconds:
                    budget_exhausted = True
                    break
                interval = _next_interval(candidate, records)
                if interval is None:
                    break
                left, right, midpoint = interval
                interval_key = f"{left}:{right}"
                if midpoint in seen_batches:
                    # This can occur after resuming a checkpoint written by an
                    # older driver.  Reuse its recorded validation decision;
                    # never execute the same load solely to repair metadata.
                    prior = next((item for item in reversed(records)
                                  if item.get("candidate_key") == candidate["key"]
                                  and item.get("batch") == midpoint
                                  and item.get("status") == "measured"), None)
                    state = _validation_interval_state(prior.get("validation", {}) if prior else {})
                    candidate.setdefault("intervals", {})[interval_key] = state
                    if state == "uncovered":
                        candidate_unavailable = True
                    continue
                variant = dict(point, batch=int(midpoint))
                if "batch_tile_count" in variant:
                    variant["batch_tile_count"] = min(int(variant["batch_tile_count"]), int(midpoint))
                raw = _invoke(binary, variant, run, exclusive, warmup=warmup, repeat=repeat,
                              trials=trials, describe_only=False)
                record = _normalize_measurement(variant, raw, profile, role="validation",
                                                candidate_key=candidate["key"], batch=int(midpoint),
                                                measurement_exclusive=str(raw.get("status", "")).lower() == "measured",
                                                warmup=warmup, repeat=repeat, trials=trials)
                records.append(record)
                if record.get("status") != "contaminated":
                    seen_batches.add(int(midpoint))
                if record.get("status") == "measured" and record.get("correct"):
                    measured_batches.add(int(midpoint))
                candidate.setdefault("adaptive_batches", []).append(int(midpoint))
                validation = _adaptive_validation(records, record)
                interval_state = _validation_interval_state(validation)
                candidate.setdefault("intervals", {})[interval_key] = interval_state
                if validation.get("promoted"):
                    candidate.setdefault("promoted_batches", []).append(int(midpoint))
                    # A promoted point is training for failed groups only.  A
                    # future model build sees the per-group role directly.
                write_checkpoint({"schema": SCHEMA, "version": VERSION, "status": "running",
                                     "identity": identity, "protocol": protocol,
                                     "records": records, "raw_records": records,
                                     "descriptions": descriptions, "candidates": candidates,
                                     "composition_requests": composition_requests,
                                     "coverage": _coverage(candidates, records, mode=mode,
                                                            max_points=max_points, budget=budget,
                                                            status="running",
                                                            composition_requests=composition_requests)})
                if record.get("status") == "contaminated":
                    # The contaminated row is retained above, but it cannot
                    # close this interval or be counted toward the point cap.
                    raise InterruptedError("contaminated stage validation")
                if interval_state == "uncovered":
                    candidate_unavailable = True
                    break
    except (KeyboardInterrupt, InterruptedError):
        interrupted = True
    except Exception as error:
        if not _looks_interrupted(error):
            raise
        interrupted = True

    model = _build_model(records, profile)
    provisional_status = _overall_status(candidates, records,
                                         acquisition_coverage(_coverage(candidates, records, mode=mode,
                                                   max_points=max_points, budget=budget,
                                                   status="running", model=model,
                                                   composition_requests=composition_requests)),
                                         model, mode=mode, interrupted=interrupted,
                                         budget_exhausted=budget_exhausted)
    coverage = acquisition_coverage(_coverage(candidates, records, mode=mode, max_points=max_points,
                         budget=budget, status=provisional_status, model=model,
                         composition_requests=composition_requests))
    status = _overall_status(candidates, records, coverage, model, mode=mode,
                             interrupted=interrupted, budget_exhausted=budget_exhausted)
    result = _result(model, coverage, status, composition_requests=composition_requests)
    write_checkpoint({"schema": SCHEMA, "version": VERSION, "status": "complete" if status in {
                             "complete", "complete-bounded"} else status,
                       "calibration_status": status, "identity": identity,
                       "protocol": protocol, "records": records,
                       "raw_records": records, "descriptions": descriptions,
                       "candidates": candidates, "composition_requests": composition_requests,
                       "coverage": coverage,
                       "model": model})
    return result


def _load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON: {path}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", "--output", dest="checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--profile", type=pathlib.Path, required=True)
    parser.add_argument("--points", type=pathlib.Path, required=True,
                        help="JSON list or object containing a points list")
    parser.add_argument("--mode", choices=("full", "bounded"), default="full")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-points", type=int,
                        help="explicit load cap per probe configuration; full default reserves validation/refinement slots")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--import-checkpoint", type=pathlib.Path, action="append", default=[],
                        help="reuse compatible stage trials; repeat for multiple checkpoints")
    args = parser.parse_args(argv)
    profile = _load_json(args.profile)
    document = _load_json(args.points)
    points = document.get("points", document) if isinstance(document, dict) else document
    if not isinstance(points, list):
        raise ValueError("points JSON must be a list or an object containing points")
    import subprocess
    from run_comprehensive_suite import require_exclusive_gpu

    def run(command: list[str]) -> Any:
        return subprocess.run(command, text=True, capture_output=True, check=False)

    def exclusive() -> bool:
        # The standalone CLI has the same reservation contract as the install
        # orchestrator.  ``require_exclusive_gpu`` raises immediately when a
        # compute client is active; the driver checkpoints and returns so the
        # user can resume later instead of silently mixing timings.
        require_exclusive_gpu()
        return True

    result = calibrate(args.binary, args.checkpoint, profile, points, run, exclusive,
                       mode=args.mode, warmup=args.warmup, repeat=args.repeat,
                       trials=args.trials, max_points=args.max_points,
                       budget=args.budget, max_seconds=args.max_seconds,
                       import_checkpoints=args.import_checkpoint)
    calibration_status = result.get("calibration_status", result.get("status"))
    print(json.dumps({"status": calibration_status, "coverage": result["coverage"],
                      "profile_status": result.get("profile_status")}, sort_keys=True))
    return 0 if calibration_status in {"complete", "complete-bounded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
