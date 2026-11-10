#!/usr/bin/env python3
"""Merge compatible stage-service checkpoint journals on the CPU.

This module is deliberately a small recovery helper.  It imports already
measured records into a target checkpoint, but never runs a probe and never
turns a subset of an inventory into a globally complete calibration.  The
stage calibration module is imported lazily so importing this helper has no
runtime/model side effects.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import pathlib
from collections import OrderedDict
from typing import Any, Iterable


def _calibration_module() -> Any:
    """Load the stage driver only when a merge actually needs it."""
    return importlib.import_module("stage_service_calibration")


def _copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _canonical_key(value: Any) -> str:
    calibration = _calibration_module()
    return calibration._json_key(value)


def _document_identity(document: dict[str, Any], label: str) -> dict[str, Any]:
    identity = document.get("identity")
    if not isinstance(identity, dict):
        raise ValueError(f"stage calibration checkpoint {label} has no identity")
    return identity


def _validate_identity(document: dict[str, Any], expected_identity: dict[str, Any],
                       label: str) -> None:
    if _document_identity(document, label) != expected_identity:
        raise ValueError("stage calibration checkpoint identity/protocol mismatch")


def _validate_document(document: Any, label: str) -> dict[str, Any]:
    calibration = _calibration_module()
    if not isinstance(document, dict) or document.get("schema") != calibration.SCHEMA:
        raise ValueError(f"unsupported stage calibration checkpoint schema: {label}")
    return document


def _inventory_values(path: pathlib.Path, document: dict[str, Any]) -> set[str]:
    """Load and verify the acquisition inventory referenced by a checkpoint.

    The orchestration sidecar is intentionally just a JSON list of static-key
    SHA-256 strings.  It is distinct from the composition sidecar, so the
    normal stage loader leaves it untouched and this importer validates it
    explicitly before merging.
    """
    reference = document.get("acquisition_inventory_ref")
    if reference is None:
        return set()
    if not isinstance(reference, dict):
        raise ValueError("stage acquisition inventory reference must be an object")
    name = reference.get("path")
    if not isinstance(name, str) or not name or pathlib.Path(name).name != name:
        raise ValueError("stage acquisition inventory sidecar path must be a filename")
    sidecar = path.parent / name
    try:
        values = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"stage acquisition inventory sidecar is not valid JSON: {sidecar}") from error
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError("stage acquisition inventory sidecar must contain string hashes")
    if values != sorted(set(values)):
        raise ValueError("stage acquisition inventory sidecar must be sorted and deduplicated")
    if any(len(value) != hashlib.sha256(b"").digest_size * 2 or
           any(character not in "0123456789abcdef" for character in value)
           for value in values):
        raise ValueError("stage acquisition inventory sidecar contains an invalid hash")
    calibration = _calibration_module()
    digest = calibration._points_hash(values)
    if digest != reference.get("sha256"):
        raise ValueError("stage acquisition inventory sidecar hash mismatch")
    try:
        expected_count = int(reference.get("count"))
    except (TypeError, ValueError) as error:
        raise ValueError("stage acquisition inventory sidecar count is invalid") from error
    if expected_count != len(values):
        raise ValueError("stage acquisition inventory sidecar count mismatch")
    return set(values)


def _inventory_sidecar(path: pathlib.Path, values: Iterable[str]) -> dict[str, Any]:
    """Write one content-addressed acquisition inventory next to ``path``."""
    calibration = _calibration_module()
    ordered = sorted(set(values))
    digest = calibration._points_hash(ordered)
    sidecar = path.with_name(f"{path.stem}.inventory-{digest}.json")
    if not sidecar.exists():
        # Reuse the stage driver's atomic writer; its JSON implementation is
        # intentionally payload-agnostic even though checkpoints are objects.
        calibration._atomic_write(sidecar, ordered)
    return {"path": sidecar.name, "sha256": digest, "count": len(ordered)}


def _rows(document: dict[str, Any], field: str) -> list[dict[str, Any]]:
    value = document.get(field)
    if value is None and field == "records":
        value = document.get("raw_records")
    if value is None and field == "raw_records":
        value = document.get("records")
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"stage calibration checkpoint {field} must be a list of objects")
    return [_copy(row) for row in value]


def _measurement_key(row: dict[str, Any]) -> str:
    """Canonicalize measured content while ignoring derived assignment fields."""
    # ``candidate_key`` and holdout role are orchestration/model state.  An
    # older checkpoint can assign the same probe output to a different key
    # after a curve-model refresh; that must not duplicate its timing.  Keep
    # the raw probe payload plus normalized measurement fields so independent
    # trials remain distinct whenever their timing/provenance differs.
    payload = {
        "raw": row.get("raw"),
        "status": row.get("status"),
        "correct": row.get("correct"),
        "sample": row.get("sample"),
        "groups": row.get("groups"),
        "pairs": row.get("pairs"),
        "plan_trial_kernel_ms": row.get("plan_trial_kernel_ms"),
    }
    return _canonical_key(payload)


def _append_unique_rows(destination: list[dict[str, Any]], rows: Iterable[dict[str, Any]]) -> None:
    """Append content-identical rows once, retaining all distinct trials.

    Raw probe payload and normalized timing/group fields are the fingerprint;
    derived candidate assignment is deliberately excluded.  Thus repeated
    copies of one checkpoint row collapse, while a separate trial (even at the
    same load) is retained when it carries any different measurement/provenance.
    """
    seen = {_measurement_key(row) for row in destination}
    for row in rows:
        key = _measurement_key(row)
        if key in seen:
            continue
        destination.append(_copy(row))
        seen.add(key)


def _description_batch(description: dict[str, Any]) -> int:
    point = description.get("point")
    if not isinstance(point, dict):
        return 0
    try:
        return int(point.get("batch", 0))
    except (TypeError, ValueError):
        return 0


def _merge_descriptions(documents: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Keep one highest-load description per static mapping identity."""
    merged: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for document in documents:
        descriptions = document.get("descriptions", {})
        if descriptions is None:
            continue
        if not isinstance(descriptions, dict):
            raise ValueError("stage calibration checkpoint descriptions must be an object")
        for static_key, description in descriptions.items():
            if not isinstance(description, dict):
                raise ValueError("stage calibration checkpoint descriptions must contain objects")
            key = str(static_key)
            old = merged.get(key)
            # A larger requested batch is the extension of the same static
            # point; its resolved group geometry is the useful description.
            # Equal-size captures remain deterministic: the target/earlier
            # document wins, preserving its original raw description.
            if old is None or _description_batch(description) > _description_batch(old):
                merged[key] = _copy(description)
    return dict(merged)


def _candidate_key(candidate: dict[str, Any], fallback: str) -> str:
    value = candidate.get("key")
    return str(value) if value not in (None, "") else fallback


def _candidate_static_key(candidate: dict[str, Any], descriptions: dict[str, dict[str, Any]]) -> str:
    value = candidate.get("static_key")
    if value not in (None, ""):
        return str(value)
    description = candidate.get("description")
    if isinstance(description, dict):
        for key, known in descriptions.items():
            if _canonical_key(description) == _canonical_key(known):
                return key
    return _candidate_key(candidate, "")


def _merge_candidate_metadata(existing: dict[str, Any], incoming: dict[str, Any]) -> None:
    """Merge journal metadata without touching measured records."""
    for field in ("aliases", "loads", "seed_batches", "adaptive_batches", "promoted_batches"):
        old = existing.get(field, [])
        new = incoming.get(field, [])
        if isinstance(old, list) and isinstance(new, list):
            values = []
            for value in old + new:
                if value not in values:
                    values.append(_copy(value))
            try:
                values.sort()
            except TypeError:
                pass
            existing[field] = values
    old_omitted = existing.get("omitted", [])
    new_omitted = incoming.get("omitted", [])
    if isinstance(old_omitted, list) and isinstance(new_omitted, list):
        existing["omitted"] = list(dict.fromkeys(old_omitted + new_omitted))
    old_intervals = existing.get("intervals")
    new_intervals = incoming.get("intervals")
    if isinstance(old_intervals, dict) and isinstance(new_intervals, dict):
        merged = dict(old_intervals)
        for key, value in new_intervals.items():
            merged.setdefault(key, _copy(value))
        existing["intervals"] = merged

    old_requested = existing.get("requested_batch")
    new_requested = incoming.get("requested_batch")
    try:
        if new_requested is not None and (old_requested is None or int(new_requested) > int(old_requested)):
            existing["requested_batch"] = _copy(new_requested)
            for field in ("point", "description", "legal_max_batch", "memory_bound_batch",
                          "memory_requested_bytes"):
                if field in incoming:
                    existing[field] = _copy(incoming[field])
    except (TypeError, ValueError):
        pass


def _merge_candidates(documents: Iterable[dict[str, Any]],
                      descriptions: dict[str, dict[str, Any]],
                      profile: dict[str, Any] | None) -> list[dict[str, Any]]:
    candidates: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for document in documents:
        value = document.get("candidates", [])
        if value is None:
            continue
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise ValueError("stage calibration checkpoint candidates must be a list of objects")
        for candidate in value:
            static_key = _candidate_static_key(candidate, descriptions)
            item = _copy(candidate)
            if static_key in descriptions:
                item["static_key"] = static_key
                item["description"] = _copy(descriptions[static_key])
                point = descriptions[static_key].get("point")
                if isinstance(point, dict):
                    item["point"] = _copy(point)
            key = _candidate_key(item, static_key)
            old = candidates.get(key)
            if old is None:
                candidates[key] = item
            else:
                _merge_candidate_metadata(old, item)

    # A checkpoint may have descriptions added by an extension before its
    # candidate list was persisted.  Reconstruct those candidate states using
    # the existing driver contract when a profile is available.
    if profile is not None:
        calibration = _calibration_module()
        for static_key, description in descriptions.items():
            if any(str(item.get("static_key")) == static_key for item in candidates.values()):
                continue
            candidate = calibration._candidate_from_description(static_key, description, profile)
            candidates.setdefault(_candidate_key(candidate, static_key), _copy(candidate))
    return list(candidates.values())


def _composition_requests(documents: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for document in documents:
        requests = document.get("composition_requests", [])
        if requests is None:
            continue
        if not isinstance(requests, list) or any(not isinstance(item, dict) for item in requests):
            raise ValueError("stage calibration composition requests must be objects")
        for request in requests:
            point = request.get("point", request)
            key = _canonical_key(point)
            if key in seen:
                continue
            merged.append(_copy(request))
            seen.add(key)
    return merged


def _protocol_value(protocol: dict[str, Any], field: str, default: Any) -> Any:
    value = protocol.get(field, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _curve_model_version() -> str:
    model = importlib.import_module("stage_service_model")
    return str(model.VERSION)


def _incomplete_coverage(candidates: list[dict[str, Any]], records: list[dict[str, Any]],
                         protocol: dict[str, Any], profile: dict[str, Any] | None,
                         composition_requests: list[dict[str, Any]]) -> dict[str, Any]:
    calibration = _calibration_module()
    mode = str(protocol.get("mode", "full"))
    max_points = max(1, _protocol_value(protocol, "max_points", calibration.DEFAULT_MAX_POINTS))
    model = None
    if profile is not None:
        model = calibration._build_model(records, profile)
        coverage = calibration._coverage(
            candidates, records, mode=mode, max_points=max_points, budget=None,
            status="running", model=model, composition_requests=composition_requests,
        )
    else:
        coverage = {
            "schema": "cubutterfly-stage-coverage-v1", "version": calibration.VERSION,
            "mode": mode, "status": "running", "curves": {}, "curve_count": 0,
            "records": len(records), "measured_records": sum(row.get("status") == "measured"
                                                               for row in records),
            "measured_points": 0, "omitted_count": 0, "unsupported_group_count": 0,
            "one_point_curves": [], "coverage_complete": False,
            "validation_complete": False, "validation_modes": [], "budget": None,
            "max_points_per_curve": max_points, "non_executable": [],
            "non_executable_count": 0, "adapter_gaps": [], "adapter_gap_count": 0,
            "limitations": [],
        }
    coverage = _copy(coverage)
    coverage["status"] = "incomplete-import"
    coverage["coverage_complete"] = False
    coverage["validation_complete"] = False
    coverage["composition_request_count"] = len(composition_requests)
    coverage["composition_requests"] = _copy(composition_requests)
    limitations = coverage.setdefault("limitations", [])
    marker = "imported records remain a subset until target inventory is calibrated"
    if marker not in limitations:
        limitations.append(marker)
    return coverage


def merge_checkpoint_documents(target: dict[str, Any] | None,
                               sources: Iterable[dict[str, Any]], *,
                               expected_identity: dict[str, Any],
                               profile: dict[str, Any] | None = None,
                               protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a compatible checkpoint made from ``target`` and ``sources``.

    Every document must have exactly ``expected_identity``.  The merge is
    content-deduplicating, preserves independent records, and marks the
    resulting artifact incomplete so a caller must still run the target
    inventory.  ``profile`` enables rebuilding candidate/coverage state using
    the existing calibration helper; omitting it keeps the operation purely
    structural for tests and tooling.
    """
    calibration = _calibration_module()
    source_list = list(sources)
    documents: list[dict[str, Any]] = []
    if target is not None:
        documents.append(_validate_document(target, "target"))
    documents.extend(_validate_document(document, f"source-{index}")
                     for index, document in enumerate(source_list))
    if not documents:
        raise ValueError("at least one stage calibration checkpoint is required")
    for index, document in enumerate(documents):
        _validate_identity(document, expected_identity, "target" if index == 0 and target is not None
                           else f"source-{index if target is None else index - 1}")

    selected_protocol = _copy(protocol if protocol is not None else documents[0].get("protocol", {}))
    if not isinstance(selected_protocol, dict):
        raise ValueError("stage calibration checkpoint protocol must be an object")

    merged = _copy(target if target is not None else documents[0])
    records: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    for document in documents:
        _append_unique_rows(records, _rows(document, "records"))
        _append_unique_rows(raw_records, _rows(document, "raw_records"))
    descriptions = _merge_descriptions(documents)
    composition_requests = _composition_requests(documents)
    candidates = _merge_candidates(documents, descriptions, profile)

    merged.update({
        "schema": calibration.SCHEMA,
        "version": documents[0].get("version", calibration.VERSION),
        "identity": _copy(expected_identity),
        "protocol": selected_protocol,
        "records": records,
        "raw_records": raw_records,
        "descriptions": descriptions,
        "candidates": candidates,
        "composition_requests": composition_requests,
        "model": None,
        "status": "running",
        "calibration_status": "incomplete-import",
    })

    needs_refresh = False
    if profile is not None:
        current_curve_version = _curve_model_version()
        needs_refresh = any(
            document.get("curve_model_version") != current_curve_version
            for document in documents
        )
    if profile is not None and needs_refresh:
        # Force the existing refresh contract to remap records from older
        # curve-key versions/aliases without rewriting any timing arrays.
        merged["curve_model_version"] = None
        calibration._refresh_curve_model_state(merged, profile)
        candidates = list(merged.get("candidates", []))
    merged["coverage"] = _incomplete_coverage(
        candidates, records, selected_protocol, profile, composition_requests,
    )
    # Refresh may have replaced the candidate list in-place, but it deliberately
    # leaves the status non-complete so a subset cannot certify global coverage.
    merged["status"] = "running"
    merged["calibration_status"] = "incomplete-import"
    return merged


def _expected_identity(binary: pathlib.Path | str, profile: dict[str, Any],
                       protocol: dict[str, Any]) -> dict[str, Any]:
    calibration = _calibration_module()
    # points_hash is intentionally excluded by the existing cache identity
    # contract; an empty value keeps this helper independent of target points.
    return calibration._cache_identity(binary, profile, protocol, "")


def _load_checkpoint_file(path: pathlib.Path) -> tuple[dict[str, Any], set[str]]:
    calibration = _calibration_module()
    document = calibration._load_checkpoint(path)
    if document is None:
        raise FileNotFoundError(f"stage calibration checkpoint was not found: {path}")
    return document, _inventory_values(path, document)


def merge_stage_checkpoints(target: pathlib.Path | str,
                            sources: Iterable[pathlib.Path | str], *,
                            binary: pathlib.Path | str,
                            profile: dict[str, Any],
                            protocol: dict[str, Any],
                            write: bool = True) -> dict[str, Any]:
    """Merge compatible checkpoint files into ``target`` and optionally write it."""
    calibration = _calibration_module()
    target_path = pathlib.Path(target)
    source_paths = [pathlib.Path(path) for path in sources]
    target_document = None
    target_inventory: set[str] = set()
    if target_path.exists():
        target_document, target_inventory = _load_checkpoint_file(target_path)
    source_documents = []
    source_inventory: set[str] = set()
    for path in source_paths:
        document, inventory = _load_checkpoint_file(path)
        source_documents.append(document)
        source_inventory.update(inventory)
    if target_document is None and not source_documents:
        raise ValueError("at least one stage calibration checkpoint is required")

    identity = _expected_identity(binary, profile, protocol)
    merged = merge_checkpoint_documents(
        target_document, source_documents, expected_identity=identity,
        profile=profile, protocol=protocol,
    )
    inventory = target_inventory | source_inventory
    if inventory:
        described = {
            hashlib.sha256(str(static_key).encode()).hexdigest()
            for static_key in merged.get("descriptions", {})
        }
        coverage = merged.setdefault("coverage", {})
        if isinstance(coverage, dict):
            coverage["requested_static_count"] = len(inventory)
            coverage["described_static_count"] = len(inventory & described)
            coverage["pending_description_count"] = len(inventory - described)
            if coverage["pending_description_count"]:
                coverage["coverage_complete"] = False
                coverage["validation_complete"] = False
        if write:
            merged["acquisition_inventory_ref"] = _inventory_sidecar(target_path, inventory)
        elif target_document is None:
            merged.pop("acquisition_inventory_ref", None)
    else:
        merged.pop("acquisition_inventory_ref", None)
    if write:
        calibration._checkpoint_write(target_path, merged)
    return merged


def import_stage_checkpoints(target: pathlib.Path | str,
                             sources: Iterable[pathlib.Path | str], *,
                             binary: pathlib.Path | str,
                             profile: dict[str, Any],
                             protocol: dict[str, Any],
                             write: bool = True) -> dict[str, Any]:
    """Backward-compatible descriptive alias for :func:`merge_stage_checkpoints`."""
    return merge_stage_checkpoints(target, sources, binary=binary, profile=profile,
                                   protocol=protocol, write=write)


__all__ = [
    "import_stage_checkpoints",
    "merge_checkpoint_documents",
    "merge_stage_checkpoints",
]
