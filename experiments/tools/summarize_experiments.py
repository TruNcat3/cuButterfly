#!/usr/bin/env python3
"""Summarize the finite paper-experiment acceptance snapshot.

This is a CPU-only audit tool.  It reads ``acceptance.json`` as the snapshot
authority and, when present, reads the small per-cell search journals for an
E9 trace.  It deliberately does not read ``summary.json`` or a module
manifest, and it never turns a confirmed-candidate ranking into a
whole-workload holdout claim.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


SCHEMA = "cubutterfly-paper-experiment-summary-v1"
ACCEPTANCE_SCHEMA = "cubutterfly-research-acceptance-v1"
SEARCH_SCHEMA = "cubutterfly-search-coverage-v1"
_RECOGNIZED_SEARCH_STATUSES = {"budget-omitted", "screened", "measured"}
_E9_BUDGETS = (4, 8, 16)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _finite_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _strict_true(value: Any) -> bool:
    """Accept the JSON/CSV truth spellings used by the acceptance harness."""
    return value is True or value == 1 or (isinstance(value, str) and value.strip().lower() in {"1", "true"})


def _p90(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.9
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _relative_stats(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "median": statistics.median(values) if values else None,
        "p90": _p90(values),
    }


def _read_json(path: Path) -> tuple[Any, dict[str, Any]]:
    """Read one small JSON artifact once and return its content hash."""
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    return value, {
        "path": path.name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _acceptance_snapshot(acceptance_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = acceptance_dir / "acceptance.json"
    if not path.is_file():
        raise ValueError(f"acceptance.json is missing: {path}")
    value, source = _read_json(path)
    if not isinstance(value, dict) or value.get("schema") != ACCEPTANCE_SCHEMA:
        raise ValueError(f"unexpected acceptance schema in {path}")
    if not isinstance(value.get("cells"), list):
        raise ValueError("acceptance snapshot has no cells list")
    return value, source


def _provided(value: Any) -> bool:
    return value is not None and value != ""


def _identity_value(identity: dict[str, Any], paths: tuple[tuple[str, ...], ...]) -> Any:
    for path in paths:
        value: Any = identity
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if _provided(value):
            return value
    return None


def _identity_values(identity: dict[str, Any], field: str) -> tuple[Any, ...]:
    aliases = {
        "uuid": (("device", "uuid"), ("device", "gpu_uuid"), ("uuid",), ("gpu_uuid",)),
        "gpu_uuid": (("device", "gpu_uuid"), ("device", "uuid"), ("gpu_uuid",), ("uuid",)),
        "global_memory_bytes": (("device", "global_memory_bytes"), ("device", "memory_bytes"),
                                 ("global_memory_bytes",), ("memory_bytes",)),
        "compile_mode": (("compile_mode",), ("device", "compile_mode")),
        "runtime_fingerprint": (("runtime_fingerprint",), ("device", "runtime_fingerprint"),
                                 ("build", "runtime_fingerprint")),
    }
    return tuple(
        value for path in aliases.get(field, ())
        for value in (_identity_value(identity, (path,)),)
        if _provided(value)
    )


def _identity_equal(field: str, observed: Any, expected: Any) -> bool:
    if field == "global_memory_bytes":
        try:
            return int(observed) == int(expected)
        except (TypeError, ValueError, OverflowError):
            pass
    if field in {"uuid", "gpu_uuid"}:
        return str(observed).strip().lower() == str(expected).strip().lower()
    return str(observed) == str(expected)


def _sample_identity_issues(sample: Any, identity: dict[str, Any], where: str) -> list[str]:
    """Check optional identity fields while tolerating adapters that omit them."""
    if not isinstance(sample, dict):
        return []
    issues: list[str] = []
    expected_device = _identity_value(identity, (("device", "device"), ("device", "name")))
    expected_cc = _identity_value(identity, (("device", "compute_capability"), ("compute_capability",)))
    if _provided(expected_device) and _provided(sample.get("device")):
        if str(sample.get("device")) != str(expected_device):
            issues.append(f"{where}.device={sample.get('device')!r} != {expected_device!r}")
    if _provided(expected_cc) and _provided(sample.get("compute_capability")):
        if str(sample.get("compute_capability")) != str(expected_cc):
            issues.append(f"{where}.compute_capability={sample.get('compute_capability')!r} != {expected_cc!r}")

    for field in ("uuid", "gpu_uuid", "global_memory_bytes", "compile_mode", "runtime_fingerprint"):
        observed = sample.get(field)
        expected_values = _identity_values(identity, field)
        if not _provided(observed) or not expected_values:
            continue
        if not any(_identity_equal(field, observed, expected) for expected in expected_values):
            issues.append(f"{where}.{field}={observed!r} differs from acceptance identity")
    observed_uuid = sample.get("uuid")
    observed_gpu_uuid = sample.get("gpu_uuid")
    if _provided(observed_uuid) and _provided(observed_gpu_uuid) and not _identity_equal("uuid", observed_uuid, observed_gpu_uuid):
        issues.append(f"{where}.uuid and gpu_uuid differ")
    return issues


def _cell_identity_issues(cell: dict[str, Any], identity: dict[str, Any]) -> list[str]:
    """Check optional sample identity fields without requiring external adapters to echo them."""
    issues: list[str] = []

    for row_index, row in enumerate(cell.get("measurements", [])):
        if isinstance(row, dict):
            issues.extend(_sample_identity_issues(row.get("sample"), identity, f"measurements[{row_index}].sample"))
    selected = cell.get("selected")
    if isinstance(selected, dict):
        for sample_index, sample in enumerate(selected.get("samples", [])):
            issues.extend(_sample_identity_issues(sample, identity, f"selected.samples[{sample_index}]"))
    replay = cell.get("selector_replay")
    replay_rows = replay if isinstance(replay, list) else ([replay] if isinstance(replay, dict) else [])
    for replay_index, row in enumerate(replay_rows):
        if isinstance(row, dict):
            issues.extend(_sample_identity_issues(row.get("sample"), identity, f"selector_replay[{replay_index}].sample"))
    return issues


def _complete_cells(document: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cells = document.get("cells", [])
    seen: set[str] = set()
    complete: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            invalid.append({"index": index, "reason": "cell-is-not-an-object"})
            continue
        cell_id = cell.get("id")
        if not isinstance(cell_id, str) or not cell_id:
            invalid.append({"index": index, "reason": "cell-id-missing"})
            continue
        if cell_id in seen:
            raise ValueError(f"duplicate semantic cell id: {cell_id}")
        seen.add(cell_id)
        if cell.get("status") == "complete":
            complete.append(cell)
    return complete, invalid


def _measurement_rows(cell: dict[str, Any]) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    """Return all-valid trial series, rejecting a named measurement with any bad trial."""
    by_name: dict[str, list[float]] = collections.defaultdict(list)
    invalid: list[dict[str, Any]] = []
    for index, row in enumerate(cell.get("measurements", [])):
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            invalid.append({"cell_id": cell.get("id"), "index": index, "reason": "measurement-row-invalid"})
            continue
        name = row["name"]
        sample = row.get("sample")
        if not _strict_true(row.get("correct")):
            invalid.append({"cell_id": cell.get("id"), "name": name, "index": index, "reason": "row-not-correct"})
            continue
        if not isinstance(sample, dict) or not _strict_true(sample.get("correct")):
            invalid.append({"cell_id": cell.get("id"), "name": name, "index": index, "reason": "sample-not-correct"})
            continue
        latency = _finite_positive(sample.get("kernel_ms"))
        if latency is None:
            invalid.append({"cell_id": cell.get("id"), "name": name, "index": index, "reason": "kernel-ms-not-positive-finite"})
            continue
        by_name[name].append(latency)

    bad_names = {item.get("name") for item in invalid if item.get("name")}
    for name in bad_names:
        by_name.pop(name, None)
    return dict(by_name), invalid


def _e1(document: dict[str, Any], complete: list[dict[str, Any]], identity_issues: dict[str, list[str]]) -> dict[str, Any]:
    expected = len(document.get("cells", []))
    by_operator: dict[str, dict[str, int]] = {}
    for cell in document.get("cells", []):
        if not isinstance(cell, dict):
            continue
        operator = str((cell.get("workload") or {}).get("operator", "unknown"))
        entry = by_operator.setdefault(operator, {"expected_cells": 0, "completed_cells": 0})
        entry["expected_cells"] += 1
        if cell.get("status") == "complete":
            entry["completed_cells"] += 1

    ratios: dict[str, list[float]] = collections.defaultdict(list)
    pair_rows: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    invalid_measurements: list[dict[str, Any]] = []
    for cell in complete:
        series, invalid = _measurement_rows(cell)
        invalid_measurements.extend(invalid)
        if identity_issues.get(cell.get("id")):
            gaps.append({"cell_id": cell.get("id"), "reason": "cohort-identity-mismatch", "details": identity_issues[cell.get("id")]})
            continue
        ours = series.get("cuButterfly")
        if not ours:
            gaps.append({"cell_id": cell.get("id"), "reason": "missing-valid-cuButterfly-trials"})
            continue
        ours_median = statistics.median(ours)
        for name, values in sorted(series.items()):
            if name == "cuButterfly":
                continue
            baseline_median = statistics.median(values)
            speedup = baseline_median / ours_median
            if not math.isfinite(speedup) or speedup <= 0:
                gaps.append({"cell_id": cell.get("id"), "baseline": name, "reason": "speedup-not-positive-finite"})
                continue
            ratios[name].append(speedup)
            pair_rows.append({
                "cell_id": cell.get("id"),
                "workload": copy.deepcopy(cell.get("workload")),
                "baseline": name,
                "cuButterfly_median_ms": ours_median,
                "baseline_median_ms": baseline_median,
                "speedup_baseline_over_cuButterfly": speedup,
                "cuButterfly_trials": len(ours),
                "baseline_trials": len(values),
            })
    baselines = {}
    for name, values in sorted(ratios.items()):
        baselines[name] = {
            "paired_cells": len(values),
            "geometric_mean_speedup": math.exp(statistics.mean(math.log(value) for value in values)),
            "minimum_speedup": min(values),
            "maximum_speedup": max(values),
            "faster_cells": sum(value >= 1.0 for value in values),
        }
    status = "not-ready" if expected == 0 else ("complete" if len(complete) == expected else "partial")
    return {
        "status": status,
        "acceptance_status": document.get("status"),
        "expected_cells": expected,
        "completed_cells": len(complete),
        "completion_fraction": (len(complete) / expected) if expected else None,
        "full_matrix_complete": bool(expected and len(complete) == expected),
        "full_migration_qualified": False,
        "cells_by_operator": dict(sorted(by_operator.items())),
        "baselines": baselines,
        "paired_rows": pair_rows,
        "gaps": gaps,
        "invalid_measurements": invalid_measurements,
    }


def _ranking_phase(
    complete: list[dict[str, Any]],
    identity_issues: dict[str, list[str]],
    field: str,
    label: str,
) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    prediction_errors: list[float] = []
    pairwise_correct = 0
    pairwise_count = 0
    top1 = 0
    top2 = 0
    regrets: list[float] = []
    for cell in complete:
        cell_id = cell.get("id")
        ranking = cell.get(field)
        if identity_issues.get(cell_id):
            gaps.append({"cell_id": cell_id, "reason": "cohort-identity-mismatch", "details": identity_issues[cell_id]})
            continue
        if not isinstance(ranking, dict) or ranking.get("status") != "evaluated":
            gaps.append({"cell_id": cell_id, "reason": "ranking-not-evaluated", "field": field})
            continue
        if not isinstance(ranking.get("top1"), bool) or not isinstance(ranking.get("top2"), bool):
            gaps.append({"cell_id": cell_id, "reason": "ranking-boolean-field-invalid", "field": field})
            continue
        correct = ranking.get("pairwise_correct")
        count = ranking.get("pairwise_count")
        regret = _finite_positive(ranking.get("latency_regret"))
        if not isinstance(correct, int) or isinstance(correct, bool) or not isinstance(count, int) or isinstance(count, bool) or count < 0 or correct < 0 or correct > count or regret is None:
            gaps.append({"cell_id": cell_id, "reason": "ranking-metric-invalid", "field": field})
            continue
        candidates = ranking.get("candidates", [])
        if not isinstance(candidates, list):
            gaps.append({"cell_id": cell_id, "reason": "ranking-candidates-invalid", "field": field})
            continue
        row_errors: list[float] = []
        invalid_candidates = 0
        for candidate in candidates:
            if not isinstance(candidate, dict):
                invalid_candidates += 1
                continue
            observed = _finite_positive(candidate.get("observed_ms"))
            predicted = _finite_positive(candidate.get("predicted_ms"))
            if observed is None or predicted is None:
                invalid_candidates += 1
                continue
            row_errors.append(abs(predicted - observed) / observed)
        if invalid_candidates:
            gaps.append({"cell_id": cell_id, "reason": "ranking-candidate-timing-invalid", "count": invalid_candidates, "field": field})
        prediction_errors.extend(row_errors)
        pairwise_correct += correct
        pairwise_count += count
        top1 += int(ranking["top1"])
        top2 += int(ranking["top2"])
        regrets.append(regret)
        evaluated.append({
            "cell_id": cell_id,
            "workload": copy.deepcopy(cell.get("workload")),
            "candidate_count": ranking.get("candidate_count", len(candidates)),
            "top1": ranking["top1"],
            "top2": ranking["top2"],
            "pairwise_correct": correct,
            "pairwise_count": count,
            "latency_regret": regret,
            "all_services_covered": ranking.get("all_services_covered"),
            "unestimated_count": len(ranking.get("unestimated", [])) if isinstance(ranking.get("unestimated"), list) else None,
            "service_gap_count": len(cell.get("service_gaps", [])) if isinstance(cell.get("service_gaps"), list) else None,
            "observed_order": copy.deepcopy(ranking.get("observed_order", [])),
            "predicted_order": copy.deepcopy(ranking.get("predicted_order", [])),
            "candidate_prediction_relative_error": _relative_stats(row_errors),
        })
        if isinstance(ranking.get("unestimated"), list) and ranking["unestimated"]:
            gaps.append({"cell_id": cell_id, "reason": "ranking-has-unestimated-candidates", "count": len(ranking["unestimated"]), "field": field})
        if isinstance(cell.get("service_gaps"), list) and cell["service_gaps"]:
            gaps.append({"cell_id": cell_id, "reason": "service-gaps", "count": len(cell["service_gaps"]), "field": field})

    n = len(evaluated)
    return {
        "label": label,
        "field": field,
        "status": "partial" if evaluated else "not-ready",
        "scope": "confirmed candidate population only; not exhaustive design-space regret",
        "evaluated_cells": n,
        "top1": {"count": top1, "rate": top1 / n if n else None},
        "top2": {"count": top2, "rate": top2 / n if n else None},
        "pairwise": {
            "correct": pairwise_correct,
            "count": pairwise_count,
            "accuracy": pairwise_correct / pairwise_count if pairwise_count else None,
        },
        "latency_regret": {
            "count": len(regrets),
            "median": statistics.median(regrets) if regrets else None,
            "p90": _p90(regrets),
            "maximum": max(regrets) if regrets else None,
        },
        "prediction_relative_error": _relative_stats(prediction_errors),
        "cells": evaluated,
        "gaps": gaps,
    }


def _e6(document: dict[str, Any], complete: list[dict[str, Any]], identity_issues: dict[str, list[str]]) -> dict[str, Any]:
    before = _ranking_phase(complete, identity_issues, "ranking_before_service_extension", "before-service-extension")
    after = _ranking_phase(complete, identity_issues, "ranking", "after-service-extension")
    return {
        "status": "partial" if before["evaluated_cells"] or after["evaluated_cells"] else "not-ready",
        "before_service_extension": before,
        "after_service_extension": after,
        "whole_workload_holdout": {
            "status": "not-ready",
            "complete": False,
            "reason": "acceptance rankings are confirmed-population rankings; no independent whole-workload train/validation assignment and held-out complete-plan timings are recorded",
        },
        "limitations": [
            "validation-point errors are not a whole-workload holdout",
            "confirmed-candidate ranking does not establish global design-space regret",
            "ranking phases are kept separate from the probe-type-fix snapshot",
        ],
    }


def _expected_search_identity(document: dict[str, Any]) -> dict[str, Any]:
    identity = document.get("identity", {})
    protocol = identity.get("protocol", {}) if isinstance(identity, dict) else {}
    binaries = identity.get("binaries", {}) if isinstance(identity, dict) else {}
    return {
        "compile_mode": identity.get("compile_mode"),
        "verify_batches": protocol.get("verify_batches"),
        "binary_sha256": {key: binaries[key] for key in ("cubutterfly_bench", "cuntt_bench") if key in binaries},
    }


def _search_identity_issues(search: dict[str, Any], document: dict[str, Any]) -> list[str]:
    expected = _expected_search_identity(document)
    issues: list[str] = []
    observed = search.get("measurement_identity")
    if not isinstance(observed, dict):
        return ["measurement_identity-missing"]
    for field in ("compile_mode", "verify_batches"):
        if expected.get(field) is not None and observed.get(field) != expected[field]:
            issues.append(f"measurement_identity.{field} differs")
    for name, value in expected["binary_sha256"].items():
        if (observed.get("binary_sha256") or {}).get(name) != value:
            issues.append(f"measurement_identity.binary_sha256.{name} differs")
    bench_hash = expected["binary_sha256"].get("cubutterfly_bench")
    if bench_hash is not None and search.get("binary_sha256") != bench_hash:
        issues.append("search binary_sha256 differs")
    if search.get("scope") != document.get("scope"):
        issues.append("search scope differs")
    return issues


def _load_cell_artifact(acceptance_dir: Path, cell_id: str, name: str) -> tuple[Any | None, dict[str, Any] | None]:
    # Cell IDs come from the snapshot; constrain them to one directory entry
    # and keep symlinked artifacts inside the acceptance cells root.
    if not isinstance(cell_id, str) or not cell_id or "/" in cell_id or "\\" in cell_id:
        return None, None
    if not isinstance(name, str) or not name or "/" in name or "\\" in name:
        return None, None
    cell_component = Path(cell_id)
    name_component = Path(name)
    if cell_component.is_absolute() or name_component.is_absolute() or len(cell_component.parts) != 1 or len(name_component.parts) != 1:
        return None, None
    if cell_id in {".", ".."} or name in {".", ".."}:
        return None, None
    cells_root = (acceptance_dir / "cells").resolve()
    path = acceptance_dir / "cells" / cell_id / name
    try:
        resolved = path.resolve()
        resolved.relative_to(acceptance_dir)
        resolved.relative_to(cells_root)
    except (OSError, RuntimeError, ValueError):
        return None, None
    if not resolved.is_file():
        return None, None
    return _read_json(resolved)


def _confirmed_rows(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        return {}
    rows = {}
    for row in value:
        if isinstance(row, dict) and isinstance(row.get("name"), str):
            rows[row["name"]] = row
    return rows


def _e9(document: dict[str, Any], acceptance_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[str]]]:
    traces: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    artifact_issues: dict[str, list[str]] = {}
    protocols: list[dict[str, Any]] = []
    for cell in document.get("cells", []):
        if not isinstance(cell, dict) or not isinstance(cell.get("id"), str):
            continue
        cell_id = cell["id"]
        search, source = _load_cell_artifact(acceptance_dir, cell_id, "search_coverage.json")
        if search is None:
            continue
        if source:
            source = {**source, "cell_id": cell_id, "kind": "search_coverage"}
            sources.append(source)
        if not isinstance(search, dict) or search.get("schema") != SEARCH_SCHEMA:
            artifact_issues[cell_id] = ["unexpected-search-coverage-schema"]
            continue
        issues = _search_identity_issues(search, document)
        if issues:
            artifact_issues[cell_id] = issues
        protocol = search.get("search_protocol") if isinstance(search.get("search_protocol"), dict) else {}
        protocols.append(protocol)
        workloads = search.get("workloads")
        if not isinstance(workloads, list) or not workloads:
            artifact_issues.setdefault(cell_id, []).append("search-workloads-missing")
            continue
        workload = next((item for item in workloads if isinstance(item, dict) and _canonical(item.get("workload")) == _canonical(cell.get("workload"))), None)
        if workload is None:
            artifact_issues.setdefault(cell_id, []).append("search-workload-does-not-match-cell")
            continue
        confirmed, confirmed_source = _load_cell_artifact(acceptance_dir, cell_id, "confirmed_records.json")
        if confirmed_source:
            sources.append({**confirmed_source, "cell_id": cell_id, "kind": "confirmed_records"})
        confirmed_by_id = _confirmed_rows(confirmed)
        candidates = workload.get("candidates") if isinstance(workload.get("candidates"), list) else []
        selected = sorted((item for item in candidates if isinstance(item, dict) and isinstance(item.get("selection_index"), int)), key=lambda item: item["selection_index"])
        trace_rows: list[dict[str, Any]] = []
        for item in selected:
            candidate_id = item.get("id")
            record = confirmed_by_id.get(candidate_id)
            observed = _finite_positive(record.get("median_kernel_ms")) if isinstance(record, dict) and _strict_true(record.get("correct")) else None
            sample_protocol = None
            if isinstance(record, dict) and isinstance(record.get("samples"), list) and record["samples"]:
                first = record["samples"][0]
                if isinstance(first, dict):
                    sample_protocol = {key: first.get(key) for key in ("warmup", "repeat", "trials") if key in first}
            trace_rows.append({
                "selection_index": item["selection_index"],
                "candidate_id": candidate_id,
                "selection_reason": item.get("selection_reason"),
                "coverage_status": item.get("status"),
                "record_status": record.get("status") if isinstance(record, dict) else None,
                "correct": _strict_true(record.get("correct")) if isinstance(record, dict) else None,
                "observed_screening_median_ms": observed,
                "predicted_screening_ms": _finite_positive(item.get("predicted_kernel_ms")),
                "timing_protocol": sample_protocol,
            })
        status_counts = collections.Counter(str(item.get("status")) for item in candidates if isinstance(item, dict))
        reason_counts = collections.Counter(str(item.get("selection_reason")) for item in selected if item.get("selection_reason") is not None)
        seed = workload.get("seed_coverage") if isinstance(workload.get("seed_coverage"), dict) else {}
        seed_fields = {key: seed.get(key) for key in ("available", "requested", "attempted", "confirmed", "budget_omitted") if key in seed}
        requested = workload.get("requested_screened_count")
        prefix_best = []
        for budget in _E9_BUDGETS:
            rows = [row for row in trace_rows if row["selection_index"] < budget and row["observed_screening_median_ms"] is not None]
            best = min(rows, key=lambda row: row["observed_screening_median_ms"]) if rows else None
            prefix_best.append({
                "budget": budget,
                "best_so_far_median_ms": best["observed_screening_median_ms"] if best else None,
                "best_candidate_id": best["candidate_id"] if best else None,
                "regret": None,
                "regret_status": "not-computable-without-independent-candidate-pool",
            })
        traces.append({
            "cell_id": cell_id,
            "workload": copy.deepcopy(cell.get("workload")),
            "source": f"cells/{cell_id}/search_coverage.json#workloads",
            "artifact_identity_status": "mismatch" if issues else "matched",
            "artifact_identity_gaps": issues,
            "complete": workload.get("complete") is True,
            "enumerated": workload.get("enumerated"),
            "screened": workload.get("screened"),
            "requested_screened_count": requested,
            "budget_omitted": workload.get("budget_omitted"),
            "generated_neighbors": workload.get("generated_neighbors", 0),
            "candidate_source": workload.get("candidate_source"),
            "stop_reason": workload.get("stop_reason"),
            "seed_coverage": seed_fields,
            "candidate_status_counts": dict(sorted(status_counts.items())),
            "failure_count": sum(count for name, count in status_counts.items() if name not in _RECOGNIZED_SEARCH_STATUSES),
            "selection_reason_counts": dict(sorted(reason_counts.items())),
            "selection_trace": trace_rows,
            "best_so_far_prefixes": prefix_best,
        })

    strategies = sorted({str(protocol.get("search_strategy")) for protocol in protocols if protocol.get("search_strategy")})
    independent_pool = False
    comparison_status = "not-ready"
    reasons = ["no independent candidate-pool holdout artifact", "only recorded search trace is insufficient for random/local/service-model comparison"]
    # A future explicit artifact may opt in, but enumerated/candidate counts are
    # intentionally not accepted as an independent pool.
    if any(isinstance(trace, dict) and trace.get("independent_candidate_pool") is True for trace in traces):
        independent_pool = True
    costs = {
        "online_search_elapsed_seconds": None,
        "compile_elapsed_seconds": None,
        "calibration_elapsed_seconds": None,
        "measurement_elapsed_seconds": None,
        "status": "not-recorded",
        "reason": "search_coverage records budget caps, not wall-clock costs",
    }
    return {
        "status": "not-ready",
        "trace_count": len(traces),
        "traces": traces,
        "budgets": list(_E9_BUDGETS),
        "budget_unit": "selected candidates including historical seeds",
        "configured_strategies": strategies,
        "independent_candidate_pool": {"status": "available" if independent_pool else "not-ready", "complete": independent_pool},
        "algorithm_comparison": {"status": comparison_status, "strategies": strategies, "reason": reasons[1]},
        "costs": costs,
        "reasons": reasons,
    }, sources, artifact_issues


def _replay_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return [value] if isinstance(value, dict) else []


def _e10(document: dict[str, Any], e9: dict[str, Any], artifact_issues: dict[str, list[str]]) -> dict[str, Any]:
    replay: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    identity = document.get("identity", {})
    for cell in document.get("cells", []):
        if not isinstance(cell, dict) or cell.get("status") != "complete":
            continue
        for index, row in enumerate(_replay_rows(cell.get("selector_replay"))):
            sample = row.get("sample") if isinstance(row.get("sample"), dict) else {}
            issues = []
            if artifact_issues.get(cell.get("id")):
                issues.extend(artifact_issues[cell.get("id")])
            if not _strict_true(row.get("correct")):
                issues.append("replay-not-correct")
            if not _strict_true(row.get("mapping_matches")):
                issues.append("replay-mapping-mismatch")
            issues.extend(_sample_identity_issues(sample, identity, "replay.sample"))
            item = {
                "cell_id": cell.get("id"),
                "source": f"acceptance.json#cells[{cell.get('id')}].selector_replay[{index}]",
                "name": row.get("name"),
                "correct": _strict_true(row.get("correct")),
                "mapping_matches": _strict_true(row.get("mapping_matches")),
                "status": "verified" if not issues else "rejected",
            }
            if issues:
                item["gaps"] = sorted(set(issues))
                rejected.append(item)
            else:
                replay.append(item)

    generated_inventory = []
    for trace in e9.get("traces", []):
        if "generated_neighbors" not in trace and "candidate_source" not in trace:
            continue
        generated_inventory.append({
            "cell_id": trace.get("cell_id"),
            "source": trace.get("source"),
            "candidate_source": trace.get("candidate_source"),
            "reported_generated_neighbors": trace.get("generated_neighbors", 0),
            "status": "reported-inventory-only",
            "correctness_inferred": False,
        })
    has_artifacts = bool(replay or rejected or generated_inventory)
    return {
        "status": "partial" if has_artifacts else "not-ready",
        "scope": "actual selector replays and explicitly reported generation inventory only",
        "replay": {
            "reported_records": len(replay) + len(rejected),
            "verified_records": len(replay),
            "rejected_records": len(rejected),
            "records": replay,
            "rejected": rejected,
        },
        "generated_inventory": {
            "reported_cells": len(generated_inventory),
            "reported_neighbor_total": sum(int(item.get("reported_generated_neighbors") or 0) for item in generated_inventory if isinstance(item.get("reported_generated_neighbors"), int)),
            "records": generated_inventory,
            "correctness_inferred": False,
        },
        "shared_code": {"status": "not-recorded"},
        "manual_schedule_modifications": {"status": "not-recorded"},
        "preparation_cost": {"status": "not-recorded"},
        "limitations": [
            "candidate inventory and compiled-unit counts are not correctness evidence",
            "no claim is made for a new operator or all possible generated shapes",
        ],
    }


def summarize_document(document: dict[str, Any], acceptance_source: dict[str, Any], acceptance_dir: Path) -> dict[str, Any]:
    if document.get("schema") != ACCEPTANCE_SCHEMA:
        raise ValueError("unexpected acceptance schema")
    identity = document.get("identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("device"), dict):
        raise ValueError("acceptance identity is missing")
    complete, invalid_cells = _complete_cells(document)
    identity_issues = {
        cell.get("id"): _cell_identity_issues(cell, identity)
        for cell in document.get("cells", [])
        if isinstance(cell, dict) and isinstance(cell.get("id"), str) and _cell_identity_issues(cell, identity)
    }
    e1 = _e1(document, complete, identity_issues)
    e6 = _e6(document, complete, identity_issues)
    e9, e9_sources, e9_artifact_issues = _e9(document, acceptance_dir)
    e10 = _e10(document, e9, e9_artifact_issues)
    files = [acceptance_source, *e9_sources]
    unique_files = []
    seen_files = set()
    for source in files:
        key = (source.get("path"), source.get("sha256"))
        if key not in seen_files:
            unique_files.append(source)
            seen_files.add(key)
    mismatches = []
    for cell_id, issues in sorted(e9_artifact_issues.items()):
        mismatches.append({"cell_id": cell_id, "source": "search_coverage.json", "issues": issues})
    for cell_id, issues in sorted(identity_issues.items()):
        mismatches.append({"cell_id": cell_id, "source": "acceptance.json", "issues": issues})
    return {
        "schema": SCHEMA,
        "snapshot": {
            "acceptance_path": acceptance_source["path"],
            "acceptance_status": document.get("status"),
            "scope": document.get("scope"),
            "expected_cells": len(document.get("cells", [])),
            "completed_cells": len(complete),
            "identity": copy.deepcopy(identity),
        },
        "provenance": {
            "files": unique_files,
            "identity": copy.deepcopy(identity),
            "identity_checks": {
                "acceptance_identity_present": True,
                "mixed_cohort_artifacts_rejected": bool(mismatches),
                "mismatches": mismatches,
            },
            "summary_does_not_read_summary_json": True,
            "module_manifest_read": False,
        },
        "e1": e1,
        "e6": e6,
        "e9": e9,
        "e10": e10,
        "invalid_cells": invalid_cells,
    }


def summarize_acceptance(acceptance_dir: Path) -> dict[str, Any]:
    acceptance_dir = Path(acceptance_dir).expanduser().resolve()
    document, source = _acceptance_snapshot(acceptance_dir)
    return summarize_document(document, source, acceptance_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    acceptance_dir = args.acceptance_dir.expanduser().resolve()
    output_input = args.output.expanduser()
    output_lexical = Path(os.path.abspath(str(output_input)))
    output = output_input.resolve()
    try:
        output_lexical.relative_to(acceptance_dir)
        inside_acceptance_dir = True
    except ValueError:
        inside_acceptance_dir = False
    try:
        output.relative_to(acceptance_dir)
        inside_resolved_acceptance_dir = True
    except ValueError:
        inside_resolved_acceptance_dir = False
    if inside_acceptance_dir or inside_resolved_acceptance_dir:
        parser.error("--output must be outside the acceptance directory")
    result = summarize_acceptance(acceptance_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
