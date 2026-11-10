#!/usr/bin/env python3
"""Summarize an all-operator xD run against its declared key population.

The runner records rows, but a row count is not a coverage proof: a missing
case can be hidden by a duplicate or by an unexpected configuration.  This
module reconstructs the requested Cartesian population from the manifest and
reports those conditions explicitly.  It only reads the run directory; no
CUDA or benchmark process is started.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Tuple, Union


Key = Tuple[str, str, Tuple[int, ...], Union[int, str], Union[bool, None], str]
SemanticKey = Tuple[str, str, Tuple[int, ...], Union[int, str], Union[bool, None]]


def _is_true(value: Any) -> bool:
    return value is True or value == 1 or str(value).strip().lower() in {"true", "1"}


def _is_false(value: Any) -> bool:
    return value is False or value == 0 or str(value).strip().lower() in {"false", "0"}


def _canonical_operator(value: Any, aliases: Mapping[str, Any]) -> str:
    """Resolve declared semantic aliases without counting a second operator."""
    current = str(value)
    seen: set[str] = set()
    while current in aliases and current not in seen:
        seen.add(current)
        current = str(aliases[current])
    return current


def _canonical_rows(rows: Iterable[Mapping[str, Any]], aliases: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["operator"] = _canonical_operator(copied.get("operator", ""), aliases)
        result.append(copied)
    return result


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _normalise_shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        text = value.strip().replace(",", "x")
        parts = [part for part in text.split("x") if part]
        return tuple(int(part) for part in parts)
    if isinstance(value, (list, tuple)):
        return tuple(int(part) for part in value)
    raise ValueError(f"unsupported shape {value!r}")


def _normalise_batch(value: Any) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _normalise_direction(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"forward", "fwd", "false", "0", "normal"}:
        return False
    if text in {"inverse", "inv", "backward", "true", "1"}:
        return True
    return None


def _direction_name(inverse: bool | None) -> str:
    if inverse is True:
        return "inverse"
    if inverse is False:
        return "forward"
    return "unknown"


def _normalise_configuration(value: Any) -> str:
    if isinstance(value, Mapping):
        for field in ("name", "id", "configuration"):
            if field in value:
                return str(value[field])
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _row_key(row: Mapping[str, Any]) -> Key:
    return (
        str(row.get("operator", "")),
        str(row.get("precision", "")),
        _normalise_shape(row.get("shape", [])),
        _normalise_batch(row.get("batch", "")),
        _normalise_direction(row.get("inverse", row.get("direction"))),
        _normalise_configuration(row.get("configuration", "")),
    )


def _semantic_key(row: Mapping[str, Any]) -> SemanticKey:
    key = _row_key(row)
    return key[:5]


def _key_record(key: Key) -> dict[str, Any]:
    operator, precision, shape, batch, inverse, configuration = key
    return {
        "operator": operator,
        "precision": precision,
        "shape": list(shape),
        "batch": batch,
        "inverse": inverse,
        "direction": _direction_name(inverse),
        "configuration": configuration,
    }


def _semantic_record(key: SemanticKey) -> dict[str, Any]:
    operator, precision, shape, batch, inverse = key
    return {
        "operator": operator,
        "precision": precision,
        "shape": list(shape),
        "batch": batch,
        "inverse": inverse,
        "direction": _direction_name(inverse),
    }


def _key_sort_value(key: Key | SemanticKey) -> tuple[Any, ...]:
    # JSON text gives stable ordering even when malformed rows mix numeric and
    # string batch values.
    return tuple(json.dumps(value, sort_keys=True) for value in key)


def _sorted_key_records(keys: Iterable[Key]) -> list[dict[str, Any]]:
    return [_key_record(key) for key in sorted(keys, key=_key_sort_value)]


def _configuration_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        if any(field in value for field in ("name", "id", "configuration")):
            return [_normalise_configuration(value)]
        return [str(name) for name in value]
    if isinstance(value, (list, tuple, set)):
        names: list[str] = []
        for item in value:
            names.extend(_configuration_names(item))
        return _unique(names)
    return [str(value)]


def _requested_configurations(
    manifest: Mapping[str, Any], rows: Iterable[Mapping[str, Any]]
) -> tuple[list[str], dict[str, Any]]:
    # New manifests may use either spelling.  A nested request object is also
    # accepted so the summarizer can consume staged manifests.
    for field in (
        "requested_configurations",
        "configurations",
        "requested_configuration",
        "configuration",
    ):
        if field in manifest and manifest[field] is not None:
            names = _unique(_configuration_names(manifest[field]))
            return names, {
                "source": f"manifest.{field}",
                "inferred": False,
                "claim": (
                    "Configurations explicitly requested by this manifest; "
                    "the report is complete only for that declared population."
                ),
            }
    requested = manifest.get("requested")
    if isinstance(requested, Mapping) and "configurations" in requested:
        names = _unique(_configuration_names(requested["configurations"]))
        return names, {
            "source": "manifest.requested.configurations",
            "inferred": False,
            "claim": (
                "Configurations explicitly requested by this manifest; "
                "the report is complete only for that declared population."
            ),
        }

    names = _unique(
        _normalise_configuration(row.get("configuration", ""))
        for row in rows
        if "configuration" in row
    )
    return names, {
        "source": "rows.configuration",
        "inferred": True,
        "claim": (
            "Configurations were inferred from distinct row values because "
            "the manifest omitted configurations; external completeness is "
            "unknown and is not claimed."
        ),
    }


def _precision_names(value: Any, operator: str, manifest: Mapping[str, Any]) -> list[str]:
    if isinstance(value, Mapping):
        if "precisions" in value or "precision" in value:
            value = value.get("precisions", value.get("precision", []))
        else:
            value = value.get(operator, value.get("all", []))
    if value is None:
        value = []
    if isinstance(value, (str, int, float)):
        value = [value]
    names = _unique(str(item) for item in value)
    return names


def _operator_specs(
    manifest: Mapping[str, Any], aliases: Mapping[str, Any] | None = None
) -> list[tuple[str, list[str]]]:
    raw = manifest.get("operators", [])
    global_precisions = manifest.get("precisions", [])
    entries: list[Any]
    if isinstance(raw, Mapping):
        entries = [{"name": name, "precisions": value} for name, value in raw.items()]
    elif isinstance(raw, (list, tuple)):
        entries = list(raw)
    elif raw:
        entries = [raw]
    else:
        entries = []

    specs: list[tuple[str, list[str]]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for entry in entries:
        if isinstance(entry, Mapping):
            name = entry.get("name", entry.get("operator", ""))
            precisions = entry.get("precisions", entry.get("precision"))
            if precisions is None:
                precisions = global_precisions
        else:
            name = entry
            precisions = global_precisions
        name_text = _canonical_operator(name, aliases or {})
        values = _precision_names(precisions, name_text, manifest)
        identity = (name_text, tuple(values))
        if name_text and identity not in seen:
            seen.add(identity)
            specs.append((name_text, values))
    return specs


def _manifest_expected_keys(
    manifest: Mapping[str, Any], configurations: Iterable[str]
) -> list[Key]:
    shapes = [_normalise_shape(shape) for shape in manifest.get("shapes", [])]
    batches = [_normalise_batch(batch) for batch in manifest.get("batches", [])]
    directions = [_normalise_direction(direction) for direction in manifest.get("directions", [])]
    configs = list(configurations)
    keys: list[Key] = []
    seen: set[Key] = set()
    aliases = manifest.get("semantic_aliases", {})
    if not isinstance(aliases, Mapping):
        aliases = {}
    for operator, precisions in _operator_specs(manifest, aliases):
        for precision in precisions:
            for shape in shapes:
                for batch in batches:
                    for inverse in directions:
                        for configuration in configs:
                            key = (operator, precision, shape, batch, inverse, configuration)
                            if key not in seen:
                                seen.add(key)
                                keys.append(key)
    return keys


def _row_class(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", "")).strip().lower().replace("-", "_")
    if status in {"failed", "error", "backend_error", "invalid"}:
        return "failed"
    if _is_false(row.get("correct")):
        return "failed"
    if status in {"unavailable", "skipped", "not_run"}:
        return "unavailable"
    # A timing-only row is intentionally not a correctness pass.  The runner
    # emits status=passed only after a true correctness result.
    latency = _safe_float(row.get("kernel_ms"))
    if status == "passed" and _is_true(row.get("correct")) and latency is not None and latency > 0:
        return "verified"
    return "unverified"


def _diagnose_keys(expected_keys: Iterable[Key], rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    expected = set(expected_keys)
    observed: Counter[Key] = Counter()
    row_keys: list[Key] = []
    for row in rows:
        try:
            key = _row_key(row)
        except (TypeError, ValueError):
            # Keep malformed rows visible as unexpected without allowing one
            # bad record to hide the rest of the population.
            key = ("<invalid>", "<invalid>", tuple(), "<invalid>", None, "<invalid>")
        row_keys.append(key)
        observed[key] += 1

    observed_set = set(observed)
    missing = expected - observed_set
    unexpected = observed_set - expected
    duplicates = {key: count for key, count in observed.items() if count > 1}

    state_keys: dict[str, set[Key]] = defaultdict(set)
    state_counts: Counter[str] = Counter()
    for row, key in zip(rows, row_keys):
        state = _row_class(row)
        state_keys[state].add(key)
        state_counts[state] += 1

    failed = state_keys["failed"] & expected
    unavailable = state_keys["unavailable"] & expected
    unverified = state_keys["unverified"] & expected
    verified = state_keys["verified"] & expected
    population_complete = bool(expected) and not (missing or unexpected or duplicates)
    execution_complete = population_complete and not (failed or unavailable)
    verified_complete = execution_complete and not unverified and len(verified) == len(expected)
    if not expected:
        status = "no-expected-population"
    elif not population_complete:
        status = "incomplete-key-population"
    elif failed:
        status = "failed"
    elif unavailable:
        status = "unavailable"
    elif unverified:
        status = "unverified"
    else:
        status = "verified-complete"

    duplicate_records = [
        {"key": _key_record(key), "count": count}
        for key, count in sorted(duplicates.items(), key=lambda item: _key_sort_value(item[0]))
    ]
    result: dict[str, Any] = {
        "expected": len(expected),
        "observed": len(rows),
        "unique_observed": len(observed_set),
        "missing": _sorted_key_records(missing),
        "unexpected": _sorted_key_records(unexpected),
        "duplicates": duplicate_records,
        "duplicate": duplicate_records,
        "failed": _sorted_key_records(failed),
        "unavailable": _sorted_key_records(unavailable),
        "unverified": _sorted_key_records(unverified),
        "verified": _sorted_key_records(verified),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "duplicate_count": len(duplicates),
        "failed_count": len(failed),
        "unavailable_count": len(unavailable),
        "unverified_count": len(unverified),
        "verified_count": len(verified),
        "population_complete": population_complete,
        "execution_complete": execution_complete,
        "verified_complete": verified_complete,
        "complete": verified_complete,
        "status": status,
        "state_counts": dict(state_counts),
    }
    return result


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _median(values: Iterable[float]) -> float | None:
    samples = [float(value) for value in values if _safe_float(value) is not None]
    return float(statistics.median(samples)) if samples else None


def _completed_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return rows eligible for performance claims, never timing-only rows."""
    return [row for row in rows if _row_class(row) == "verified"]


def _row_counter(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field, "")) for row in rows))


def _unique_rows_by_semantic_key(
    rows: Iterable[Mapping[str, Any]], configuration: str
) -> dict[SemanticKey, Mapping[str, Any]]:
    grouped: dict[SemanticKey, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if _normalise_configuration(row.get("configuration", "")) != configuration:
            continue
        try:
            key = _semantic_key(row)
        except (TypeError, ValueError):
            continue
        grouped[key].append(row)
    # Duplicate keys are invalid evidence.  Excluding all copies prevents a
    # resumed/repeated cell from silently influencing a control median.
    return {key: values[0] for key, values in grouped.items() if len(values) == 1}


def _performance_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Keep only unique, correctness-verified rows for timing statistics."""
    grouped: dict[Key, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            grouped[_row_key(row)].append(row)
        except (TypeError, ValueError):
            continue
    return [values[0] for values in grouped.values()
            if len(values) == 1 and _row_class(values[0]) == "verified"]


SEMANTIC_CONTRACT_FIELDS = (
    "modulus", "modulus_bits", "normalization", "placement", "output_order",
    "element_stride", "batch_stride", "direction", "inverse", "matrix",
)


def _semantic_conflicts(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Find duplicate population keys that disagree on declared semantics."""
    grouped: dict[Key, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            grouped[_row_key(row)].append(row)
        except (TypeError, ValueError):
            continue
    conflicts: list[dict[str, Any]] = []
    for key, values in grouped.items():
        differing: dict[str, list[str]] = {}
        for field in SEMANTIC_CONTRACT_FIELDS:
            present = {json.dumps(row.get(field), sort_keys=True) for row in values
                       if field in row and row.get(field) not in (None, "")}
            if len(present) > 1:
                differing[field] = sorted(present)
        if differing:
            conflicts.append({"key": _key_record(key), "fields": differing})
    return sorted(conflicts, key=lambda item: _key_sort_value(_row_key(item["key"])))


def _command_value(command: Any, option: str) -> str | None:
    if not isinstance(command, (list, tuple)):
        return None
    for index, item in enumerate(command[:-1]):
        if str(item) == option:
            return str(command[index + 1])
    return None


def _same_path(left: Any, right: Any) -> bool:
    if left in (None, "") or right in (None, ""):
        return True
    try:
        return Path(str(left)).resolve() == Path(str(right)).resolve()
    except (OSError, RuntimeError):
        return str(left) == str(right)


def _runtime_issues(manifest: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reject rows that identify a different executable, build, or protocol."""
    timing = manifest.get("timing", {})
    if not isinstance(timing, Mapping):
        timing = {}
    expected_identity = manifest.get("runtime_fingerprint")
    if expected_identity is None and isinstance(manifest.get("runtime"), Mapping):
        expected_identity = manifest["runtime"].get("fingerprint")
    observed_identities = {
        field: {str(row.get(field)).strip() for row in rows if row.get(field) not in (None, "")}
        for field in ("runtime_fingerprint", "build_fingerprint", "library_sha256")
    }
    mixed_identity_fields = {field for field, values in observed_identities.items() if len(values) > 1}
    issues: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        reasons: list[str] = []
        for field in ("library_sha256", "bench_sha256", "build_fingerprint", "compile_mode"):
            expected = manifest.get(field)
            actual = row.get(field)
            if expected not in (None, "") and actual not in (None, "") and str(actual) != str(expected):
                reasons.append(f"{field}={actual!r} (expected {expected!r})")
        for field in sorted(mixed_identity_fields):
            if row.get(field) not in (None, ""):
                reasons.append(f"mixed {field} identities in one run")
        actual_identity = row.get("runtime_fingerprint")
        if expected_identity not in (None, "") and actual_identity not in (None, "") \
                and str(actual_identity) != str(expected_identity):
            reasons.append(f"runtime_fingerprint={actual_identity!r} (expected {expected_identity!r})")

        command = row.get("command")
        expected_bench = manifest.get("bench")
        if isinstance(command, (list, tuple)) and command and expected_bench not in (None, "") \
                and not _same_path(command[0], expected_bench):
            reasons.append(f"bench={command[0]!r} (expected {expected_bench!r})")
        expected_options = {
            "--warmup": timing.get("warmup"),
            "--repeat": timing.get("repeat"),
            "--trials": timing.get("trials"),
            "--policy": manifest.get("policy"),
            "--execution-mode": "xd",
        }
        if str(row.get("operator", "")) == "ntt":
            expected_options["--modulus"] = manifest.get("modulus")
        if str(row.get("operator", "")) == "structured-2x2":
            expected_options["--matrix"] = manifest.get("matrix")
        for option, expected in expected_options.items():
            actual = _command_value(command, option)
            if expected in (None, "") or actual is None:
                continue
            if str(actual) != str(expected):
                reasons.append(f"{option}={actual!r} (expected {expected!r})")
        if reasons:
            try:
                key = _key_record(_row_key(row))
            except (TypeError, ValueError):
                key = {"operator": str(row.get("operator", ""))}
            issues.append({"row": index, "key": key, "reasons": reasons})
    return issues


def _correctness_evidence(root: Path, manifest: Mapping[str, Any], rows: list[Mapping[str, Any]],
                          expected_keys: Iterable[Key]) -> dict[str, Any]:
    """Load the optional correctness pass separately from timing status."""
    candidates = (
        root.parent / f"{root.name}_correctness.json",
        root / "correctness.json",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    expected_ids = {str(row.get("id", "")) for row in rows if row.get("id") not in (None, "")}
    if path is None:
        return {
            "available": False, "source": None, "requested": len(set(expected_keys)),
            "observed": 0, "verified": 0, "failed": 0, "missing": len(expected_ids),
            "trusted": False,
        }
    try:
        data = _load_json(path)
    except (OSError, ValueError, TypeError) as exc:
        return {
            "available": True, "source": str(path), "requested": len(expected_ids),
            "observed": 0, "verified": 0, "failed": 0, "missing": len(expected_ids),
            "trusted": False, "error": str(exc),
        }
    cases = [case for case in data.get("cases", []) if isinstance(case, Mapping)] \
        if isinstance(data, Mapping) else []
    case_ids = [str(case.get("id", "")) for case in cases if case.get("id") not in (None, "")]
    statuses = Counter(str(case.get("status", "missing")) for case in cases)
    summary = data.get("summary", {}) if isinstance(data, Mapping) else {}
    requested = summary.get("requested", len(expected_ids)) if isinstance(summary, Mapping) else len(expected_ids)
    expected_sha = manifest.get("library_sha256")
    actual_sha = data.get("library_sha256") if isinstance(data, Mapping) else None
    identity_ok = not (expected_sha not in (None, "") and actual_sha not in (None, "")
                       and str(expected_sha) != str(actual_sha))
    expected_mode = manifest.get("compile_mode")
    actual_mode = data.get("compile_mode") if isinstance(data, Mapping) else None
    identity_ok = identity_ok and not (expected_mode not in (None, "") and actual_mode not in (None, "")
                                       and str(expected_mode) != str(actual_mode))
    expected_gpu = manifest.get("gpu", {})
    expected_gpu_model = expected_gpu.get("model") if isinstance(expected_gpu, Mapping) else None
    actual_runtime = data.get("runtime", {}) if isinstance(data, Mapping) else {}
    actual_gpu_model = actual_runtime.get("device_name") if isinstance(actual_runtime, Mapping) else None
    identity_ok = identity_ok and not (expected_gpu_model not in (None, "")
                                       and actual_gpu_model not in (None, "")
                                       and str(expected_gpu_model) != str(actual_gpu_model))
    duplicate_ids = len(case_ids) - len(set(case_ids))
    ids_match = set(case_ids) == expected_ids and len(case_ids) == len(expected_ids)
    verified = sum(1 for case in cases if str(case.get("status", "")).lower() == "passed")
    failed = sum(1 for case in cases if str(case.get("status", "")).lower() in {"failed", "error"})
    return {
        "available": True, "source": str(path), "requested": int(requested or 0),
        "observed": len(cases), "verified": verified if identity_ok and not duplicate_ids and ids_match else 0,
        "failed": failed, "missing": max(len(expected_ids - set(case_ids)), 0),
        "unexpected": max(len(set(case_ids) - expected_ids), 0),
        "duplicate_ids": duplicate_ids, "trusted": bool(identity_ok and not duplicate_ids and ids_match),
        "identity": {
            "library_sha256": expected_sha in (None, "") or actual_sha in (None, "")
            or str(expected_sha) == str(actual_sha),
            "compile_mode": expected_mode in (None, "") or actual_mode in (None, expected_mode),
            "gpu_model": expected_gpu_model in (None, "") or actual_gpu_model in (None, expected_gpu_model),
        },
        "status_counts": dict(statuses),
    }


def _mapping_verified_rows(root: Path, manifest: Mapping[str, Any],
                           rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Join value checks to the exact timed mapping, never to a default plan."""
    result = [dict(row) for row in rows]
    paths = (root.parent / f"{root.name}_correctness.json", root / "correctness.json")
    path = next((path for path in paths if path.is_file()), None)
    if path is None:
        return result
    try:
        data = _load_json(path)
    except (OSError, ValueError, TypeError):
        return result
    if not isinstance(data, Mapping):
        return result
    # Legacy correctness reports can document semantics but cannot certify
    # which implementation produced a timing row.
    for field in ("library_sha256", "compile_mode"):
        if not manifest.get(field) or data.get(field) != manifest[field]:
            return result
    gpu = manifest.get("gpu", {})
    runtime = data.get("runtime", {})
    if gpu.get("model") and runtime.get("device_name") != gpu["model"]:
        return result
    checks: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in data.get("cases", []):
        if isinstance(case, Mapping) and case.get("id"):
            checks[str(case["id"])].append(case)

    def mapping(value: Any) -> Any:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                return None
        return value if isinstance(value, Mapping) and value else None

    for row in result:
        candidates = checks.get(str(row.get("id", "")), [])
        if len(candidates) != 1 or row.get("status") not in ("measured", "passed"):
            continue
        check = candidates[0]
        source = mapping(row.get("mapping_json"))
        if source is None or source != mapping(check.get("source_mapping_json")):
            continue
        if any(field not in check for field in
               ("operator", "precision", "shape", "batch", "inverse", "configuration")):
            continue
        if _row_key(row) != _row_key(check):
            continue
        if (check.get("status") == "passed" and _is_true(check.get("actual_result_checked"))
                and _is_true(check.get("mapping_replay_verified"))
                and source == mapping(check.get("mapping_json"))
                and not _is_false(row.get("correct"))):
            row.update(timing_status=row["status"], status="passed", correct=True,
                       mapping_replay_verified=True,
                       verification="independent value check of exact timed mapping")
        elif check.get("status") == "failed":
            row.update(timing_status=row["status"], status="failed", correct=False,
                       verification="exact timed mapping failed independent verification")
    return result


def _operator_report(
    manifest: Mapping[str, Any], rows: list[Mapping[str, Any]], coverage: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    completed = _completed_rows(rows)
    expected_by_operator: Counter[str] = Counter(
        str(record["operator"]) for record in coverage.get("_expected_records", [])
    )
    observed_by_operator: Counter[str] = Counter(
        str(row.get("operator", "")) for row in completed
    )
    aliases = manifest.get("semantic_aliases", {})
    if not isinstance(aliases, Mapping):
        aliases = {}
    names = [name for name, _ in _operator_specs(manifest, aliases)]
    names.extend(str(row.get("operator", "")) for row in rows)
    result: dict[str, dict[str, Any]] = {}
    for operator in _unique(names):
        op_rows = [row for row in completed if str(row.get("operator", "")) == operator]
        op_expected = [
            record for record in coverage.get("_expected_records", [])
            if record["operator"] == operator
        ]
        values: list[float] = []
        for row in op_rows:
            value = _safe_float(row.get("kernel_ms"))
            if value is not None:
                values.append(value)
        ranks = Counter(str(len(_normalise_shape(row.get("shape", [])))) for row in op_rows)
        directions = Counter(_direction_name(_normalise_direction(row.get("inverse", row.get("direction")))) for row in op_rows)
        operator_diag = {
            "expected": len(op_expected),
            "observed_rows": observed_by_operator[operator],
            "rows": len(op_rows),
            "missing": sum(record["operator"] == operator for record in coverage.get("missing", [])),
            "unexpected": sum(record["operator"] == operator for record in coverage.get("unexpected", [])),
            "duplicate": sum(item["key"]["operator"] == operator for item in coverage.get("duplicates", [])),
            "failed": sum(record["operator"] == operator for record in coverage.get("failed", [])),
            "unverified": sum(record["operator"] == operator for record in coverage.get("unverified", [])),
            "runtime_mismatch": sum(record["key"].get("operator") == operator
                                     for record in coverage.get("runtime_mismatch", [])),
            "precisions": sorted({str(row.get("precision", "")) for row in op_rows}),
            "ranks": dict(ranks),
            "batches": sorted({row.get("batch") for row in op_rows}, key=str),
            "forward": directions.get("forward", 0),
            "inverse": directions.get("inverse", 0),
            "algorithms": _row_counter(op_rows, "algorithm"),
            "median_kernel_ms": _median(values),
            "selected_rows": sum(
                _normalise_configuration(row.get("configuration", "")) == "selected"
                for row in op_rows
            ),
            "resident_t128_rows": sum(
                _normalise_configuration(row.get("configuration", "")) == "resident-t128"
                for row in op_rows
            ),
        }
        fft_values: list[float] = []
        for row in op_rows:
            value = _safe_float(row.get("speedup_vs_cufft"))
            if (
                operator == "fft"
                and _normalise_configuration(row.get("configuration", "")) == "selected"
                and value is not None
                and value > 0
            ):
                fft_values.append(value)
        operator_diag["fft_cufft_rows"] = len(fft_values)
        operator_diag["fft_cufft_geomean"] = (
            math.exp(sum(math.log(value) for value in fft_values) / len(fft_values))
            if fft_values else None
        )
        result[operator] = operator_diag
    return result


def _control_report(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    completed = _completed_rows(rows)
    selected = [
        row for row in completed
        if _normalise_configuration(row.get("configuration", "")) == "selected"
    ]
    resident = [
        row for row in completed
        if _normalise_configuration(row.get("configuration", "")) == "resident-t128"
    ]
    selected_by_key = _unique_rows_by_semantic_key(completed, "selected")
    resident_by_key = _unique_rows_by_semantic_key(completed, "resident-t128")
    paired: list[float] = []
    for key in sorted(selected_by_key.keys() & resident_by_key.keys(), key=_key_sort_value):
        selected_ms = _safe_float(selected_by_key[key].get("kernel_ms"))
        resident_ms = _safe_float(resident_by_key[key].get("kernel_ms"))
        if selected_ms is not None and resident_ms is not None and selected_ms > 0 and resident_ms > 0:
            paired.append(resident_ms / selected_ms)
    return {
        "selected": len(selected),
        "resident_t128": len(resident),
        "paired": len(paired),
        "selector_vs_resident_median_speedup": _median(paired),
        "selector_vs_resident_t128": {
            "paired_rows": len(paired),
            "median_speedup": _median(paired),
            "wins": sum(value > 1.0 for value in paired),
            "speedups": paired,
        },
        "selector_wins": sum(value > 1.0 for value in paired),
    }


def _population_boundary() -> dict[str, str]:
    return {
        "rank1": (
            "historical-only: rank-1 semantic matrices are not current xD "
            "acceptance evidence and do not count toward this report"
        ),
        "low_precision": (
            "historical-only: FP16/BF16 and other dated low-precision matrices "
            "are not current-build acceptance for this xD report"
        ),
        "external_baselines": (
            "FFT/cuFFT rows are in this runner; other external baselines are "
            "historical-only named populations and are not current acceptance"
        ),
    }


def summarize(root: Path | str) -> tuple[dict[str, Any], str]:
    root = Path(root)
    manifest = _load_json(root / "manifest.json")
    raw_rows = _load_json(root / "rows.json")
    rows = [row for row in raw_rows if isinstance(row, Mapping)]
    aliases = manifest.get("semantic_aliases", {})
    if not isinstance(aliases, Mapping):
        aliases = {}
    rows = _canonical_rows(rows, aliases)
    rows = _mapping_verified_rows(root, manifest, rows)
    configurations, configuration_contract = _requested_configurations(manifest, rows)
    expected_keys = _manifest_expected_keys(manifest, configurations)
    expected_records = [_key_record(key) for key in expected_keys]
    coverage = _diagnose_keys(expected_keys, rows)
    correctness = _correctness_evidence(root, manifest, rows, expected_keys)
    semantic_conflicts = _semantic_conflicts(rows)
    runtime_issues = _runtime_issues(manifest, rows)
    coverage["semantic_conflicts"] = semantic_conflicts
    coverage["semantic_conflict_count"] = len(semantic_conflicts)
    coverage["runtime_mismatch"] = runtime_issues
    coverage["runtime_mismatch_count"] = len(runtime_issues)
    if semantic_conflicts:
        coverage["population_complete"] = False
        coverage["execution_complete"] = False
        coverage["verified_complete"] = False
        coverage["complete"] = False
        coverage["status"] = "semantic-mismatch"
    if runtime_issues:
        coverage["population_complete"] = False
        coverage["execution_complete"] = False
        coverage["verified_complete"] = False
        coverage["complete"] = False
        coverage["status"] = "runtime-mismatch"
    bad_runtime_rows = {item["row"] for item in runtime_issues}
    # Determine duplicates before removing bad runtime rows. Otherwise one
    # invalid copy could disappear and make a duplicate cell seem unique.
    unique_performance = {id(row) for row in _performance_rows(rows)}
    performance_input = [row for index, row in enumerate(rows)
                         if index not in bad_runtime_rows and id(row) in unique_performance]
    eligible_keys = {_row_key(row) for row in performance_input} & set(expected_keys)
    coverage["verified"] = _sorted_key_records(eligible_keys)
    coverage["verified_count"] = len(eligible_keys)
    # Private data used while building the per-operator view; it is removed
    # before returning the public report.
    coverage_with_records = dict(coverage)
    coverage_with_records["_expected_records"] = expected_records

    completed = performance_input
    selected_completed = [
        row for row in completed
        if _normalise_configuration(row.get("configuration", "")) == "selected"
    ]
    selected_shapes = sorted(
        {_normalise_shape(row.get("shape", [])) for row in selected_completed},
        key=lambda shape: (len(shape), shape),
    )
    selected_rank_counts = Counter(
        len(_normalise_shape(row.get("shape", []))) for row in selected_completed
    )
    controls = _control_report(completed)
    report: dict[str, Any] = {
        "manifest": manifest,
        # `requested` now means the declared Cartesian population, not merely
        # the number of rows that happened to be emitted.
        "requested": len(expected_keys),
        "expected_keys": len(expected_keys),
        "observed_rows": len(rows),
        "row_count": len(rows),
        "status_counts": dict(Counter(str(row.get("status", "missing")) for row in rows)),
        "selected_rows": len(selected_completed),
        "selected_rank_counts": dict(selected_rank_counts),
        "selected_shapes": ["x".join(str(value) for value in shape) for shape in selected_shapes],
        "requested_configurations": configurations,
        "configuration_contract": configuration_contract,
        "configuration_source": configuration_contract["source"],
        "configuration_inferred": configuration_contract["inferred"],
        "coverage": coverage,
        "runtime_mismatch": runtime_issues,
        "semantic_conflicts": semantic_conflicts,
        "correctness": correctness,
        "mapping_bound_verified": sum(_is_true(row.get("mapping_replay_verified"))
                                      for row in performance_input),
        # Top-level aliases make the diagnostics easy to consume from shell
        # tools without requiring callers to know the nested report layout.
        "missing": coverage["missing"],
        "duplicate": coverage["duplicates"],
        "duplicates": coverage["duplicates"],
        "unexpected": coverage["unexpected"],
        "failed": coverage["failed"],
        "unverified": coverage["unverified"],
        "operators": _operator_report(manifest, completed, coverage_with_records),
        "controls": controls,
        "selector_vs_resident_t128": controls["selector_vs_resident_t128"],
        "selector_vs_resident_by_operator": {},
        "fft_forward_vs_cufft": {"rows": 0, "geomean_speedup": None, "wins": 0},
        "population_boundary": _population_boundary(),
    }

    # Preserve the historical per-operator control view while using true
    # statistics.median for even-sized populations.
    selected_by_key = _unique_rows_by_semantic_key(completed, "selected")
    resident_by_key = _unique_rows_by_semantic_key(completed, "resident-t128")
    for operator in report["operators"]:
        values: list[float] = []
        for key in selected_by_key.keys() & resident_by_key.keys():
            if key[0] != operator:
                continue
            selected_ms = _safe_float(selected_by_key[key].get("kernel_ms"))
            resident_ms = _safe_float(resident_by_key[key].get("kernel_ms"))
            if selected_ms is not None and resident_ms is not None and selected_ms > 0 and resident_ms > 0:
                values.append(resident_ms / selected_ms)
        report["selector_vs_resident_by_operator"][operator] = {
            "paired_rows": len(values),
            "median_speedup": _median(values),
            "wins": sum(value > 1.0 for value in values),
        }
    fft_values: list[float] = []
    for row in selected_completed:
        value = _safe_float(row.get("speedup_vs_cufft"))
        if str(row.get("operator", "")) == "fft" and value is not None and value > 0:
            fft_values.append(value)
    report["fft_forward_vs_cufft"] = {
        "rows": len(fft_values),
        "geomean_speedup": (
            math.exp(sum(math.log(value) for value in fft_values) / len(fft_values))
            if fft_values else None
        ),
        "wins": sum(value > 1.0 for value in fft_values),
    }
    return report, _markdown_report(report)


def _markdown_report(report: Mapping[str, Any]) -> str:
    manifest = report.get("manifest", {})
    gpu = manifest.get("gpu", {}) if isinstance(manifest, Mapping) else {}
    gpu_text = str(gpu.get("uuid", "unknown"))
    if gpu.get("model"):
        gpu_text += f" ({gpu['model']})"
    coverage = report["coverage"]
    contract = report["configuration_contract"]
    lines = [
        "# All-operator xD coverage report",
        "",
        f"- Profile: {manifest.get('profile', 'unknown')}",
        f"- GPU recorded by manifest: {gpu_text}",
        f"- Configurations: {', '.join(report['requested_configurations']) or '(none)'} "
        f"(source: {contract['source']}; inferred: {contract['inferred']})",
        f"- Configuration claim: {contract['claim']}",
        f"- Expected keys: **{report['expected_keys']}**; observed rows: **{report['observed_rows']}**",
        f"- Key population: **{coverage['status']}**; strict verified complete: **{coverage['verified_complete']}**",
        f"- Missing: **{coverage['missing_count']}**; duplicate keys: **{coverage['duplicate_count']}**; "
        f"unexpected: **{coverage['unexpected_count']}**; failed: **{coverage['failed_count']}**; "
        f"unverified: **{coverage['unverified_count']}**; semantic conflicts: **{coverage['semantic_conflict_count']}**; "
        f"runtime mismatches: **{coverage['runtime_mismatch_count']}**",
        f"- Separate correctness evidence: **{report['correctness']['verified']} / "
        f"{report['correctness']['requested']}** verified (trusted: **{report['correctness']['trusted']}**); "
        "this does not imply an eligible external timing reference.",
        f"- Timed mappings independently value-checked: **{report['mapping_bound_verified']}**",
        f"- Selected completed rows: **{report['selected_rows']}**",
        "",
        "## Per-operator coverage",
        "",
        "| operator | expected | observed | missing | duplicate | unexpected | failed | unverified | forward/inverse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for operator, info in report["operators"].items():
        lines.append(
            f"| {operator} | {info['expected']} | {info['observed_rows']} | "
            f"{info['missing']} | {info['duplicate']} | {info['unexpected']} | "
            f"{info['failed']} | {info['unverified']} | "
            f"{info['forward']}/{info['inverse']} |"
        )
    controls = report["controls"]
    lines.extend([
        "",
        "## Control comparison",
        "",
        f"Selected/control paired rows: **{controls['paired']}**; selector wins: "
        f"**{controls['selector_wins']}**; median control/selector speedup: "
        f"**{controls['selector_vs_resident_median_speedup']}x**",
        "",
        "## Population boundaries",
        "",
    ])
    for name, statement in report["population_boundary"].items():
        lines.append(f"- {name}: {statement}")
    lines.extend([
        "",
        "The report checks only the population declared or row-inferred for this run; it does not claim exhaustive external coverage.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="run directory containing manifest.json and rows.json")
    parser.add_argument("--output", type=Path, help="markdown report path (default: ROOT/coverage_report.md)")
    parser.add_argument("--require-verified-complete", action="store_true",
                        help="fail when declared cells are missing, duplicated, unverified or mismatched")
    args = parser.parse_args(argv)
    report, markdown = summarize(args.root)
    output = args.output or args.root / "coverage_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    (args.root / "coverage_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return int(args.require_verified_complete and
               (not report["coverage"]["verified_complete"] or report["configuration_inferred"]))


if __name__ == "__main__":
    raise SystemExit(main())
