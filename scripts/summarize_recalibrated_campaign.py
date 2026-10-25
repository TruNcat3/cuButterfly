#!/usr/bin/env python3
"""Report fresh campaign evidence without pooling unrelated populations.

Only files inside the requested campaign are read. Missing, partial and
unverified results remain visible; aliases and vendor fallbacks do not become
independent cuButterfly wins. Planning/compilation costs are not kernel times.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

from summarize_comprehensive_suite import summarize as summarize_rank1


EXTERNAL_POPULATION = "comparison/external_raw.csv"
EXTERNAL_ROLES = (("dao-fht", "butterfly"), ("gpuntt", "ntt"))
EXTERNAL_ROLES_BY_OPERATOR = {
    "fwht": (("dao-fht", "butterfly"),),
    "ntt": (("gpuntt", "ntt"),),
}
EXTERNAL_RUNNERS = frozenset(runner for role in EXTERNAL_ROLES for runner in role)
EXTERNAL_SEMANTIC_FIELDS = (
    "operator", "precision", "direction", "normalization", "placement", "output_order",
    "logN", "N", "batch", "modulus", "modulus_bits", "warmup", "repeat",
)
AGGREGATE_FIELDS = (
    "population", "operator", "precision", "direction", "reference", "reference_kind",
    "modulus", "modulus_bits", "normalization", "placement",
    "output_order", "warmup", "repeat",
)


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


_INVALID_QUEUE_STATUSES = frozenset({
    "pending", "running", "failed", "timeout", "launch-error",
    "gpu-not-exclusive", "gpu-conflict", "gpu-monitor-error",
    "registry-changed", "identity-changed", "identitychanged",
    "completed-unverified", "invalid",
})


def _path_matches(value, target, root):
    """Match an output declaration without relying on the caller's cwd."""
    if not isinstance(value, (str, Path)):
        return False
    text = str(value)
    target = Path(target).resolve()

    def same_or_child(candidate):
        candidate = Path(candidate).resolve()
        if candidate == target:
            return True
        try:
            target.relative_to(candidate)
            return True
        except ValueError:
            return False

    candidate = Path(text)
    if candidate.is_absolute():
        if same_or_child(candidate):
            return True
    else:
        bases = (root, root.parent.parent, Path.cwd())
        if any(same_or_child(base / candidate) for base in bases):
            return True
    # Strict manifests record output paths explicitly.  Do not search command
    # text: a summary step may read the raw file without producing it.
    relative = None
    try:
        relative = target.relative_to(root)
    except ValueError:
        pass
    return text == str(target) or (relative is not None and text == str(relative))


def _path_is_exact(value, target, root):
    if not isinstance(value, (str, Path)):
        return False
    candidate = Path(str(value))
    target = Path(target).resolve()
    if candidate.is_absolute():
        return candidate.resolve() == target
    return any((base / candidate).resolve() == target
               for base in (root, root.parent.parent, Path.cwd()))


def _string_values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _string_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _string_values(item)
    elif isinstance(value, (str, Path)):
        yield value


def _strict_output_gate(root, manifest, queue_state, target):
    """Return queue eligibility for one declared output artifact.

    Reviewed manifests deliberately bind outputs to their queue step.  A CSV
    left by an interrupted or identity-invalid step is therefore diagnostic
    evidence only, even when it still exists on disk.
    """
    if not isinstance(manifest, dict) or not manifest.get("measurement_contract"):
        return {"strict": False, "eligible": True, "reason": "legacy-manifest"}

    matches = []
    phases = manifest.get("phases", {})
    state_phases = queue_state.get("phases", {}) if isinstance(queue_state, dict) else {}
    if not isinstance(phases, dict):
        phases = {}
    if not isinstance(state_phases, dict):
        state_phases = {}
    for phase, steps in phases.items():
        if not isinstance(steps, list):
            continue
        records = state_phases.get(phase, {})
        records = records.get("steps", []) if isinstance(records, dict) else []
        if not isinstance(records, list):
            records = []
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            declared = list(_string_values(step.get("outputs", [])))
            if not any(_path_matches(value, target, root) for value in declared):
                continue
            record = records[index] if index < len(records) else None
            if not isinstance(record, dict) or record.get("id") != step.get("id"):
                record = next((item for item in records
                               if isinstance(item, dict) and item.get("id") == step.get("id")), None)
            matches.append((any(_path_is_exact(value, target, root) for value in declared),
                            phase, index, step, record))

    details = {"strict": True, "eligible": False, "output": str(Path(target))}
    if not matches:
        details["reason"] = "output-not-declared"
        return details
    _, selected = max(enumerate(matches), key=lambda item: (item[1][0], item[0]))
    _, phase, index, step, record = selected
    details.update({"phase": phase, "step_index": index, "step_id": step.get("id")})
    if not isinstance(record, dict):
        details["reason"] = "queue-step-missing"
        return details
    status = str(record.get("status", "missing")).strip().lower()
    details.update({
        "status": record.get("status", "missing"),
        "gate_passed": record.get("gate_passed"),
        "returncode": record.get("returncode"),
    })
    if status != "passed" or status in _INVALID_QUEUE_STATUSES:
        details["reason"] = "queue-status-" + status
        return details
    if record.get("gate_passed") is not True:
        details["reason"] = "gate-not-passed"
        return details
    if record.get("returncode") != 0:
        details["reason"] = "nonzero-returncode"
        return details
    details.update({"eligible": True, "reason": "queue-step-gated"})
    return details


def _filter_output_rows(root, manifest, queue_state, target, rows):
    gate = _strict_output_gate(root, manifest, queue_state, target)
    observed = list(rows or [])
    if gate["eligible"]:
        return observed, gate
    gate["observed_rows"] = len(observed)
    gate["eligible_rows"] = 0
    gate["blocked_rows"] = len(observed)
    return [], gate


def _relative_campaign_path(path, root):
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _declared_external_paths(root, manifest, external_manifest):
    """Find manifest-controlled raw external outputs, preferring a combined file."""
    values = []
    containers = [
        manifest.get("external_comparison", {}) if isinstance(manifest, dict) else {},
        manifest.get("fresh_outputs", {}) if isinstance(manifest, dict) else {},
        external_manifest if isinstance(external_manifest, dict) else {},
    ]
    for container in containers:
        if isinstance(container, dict):
            for field in ("fresh_artifacts", "required_artifacts", "outputs"):
                values.extend(_string_values(container.get(field, [])))
    phases = manifest.get("phases", {}) if isinstance(manifest, dict) else {}
    if isinstance(phases, dict):
        for steps in phases.values():
            if not isinstance(steps, list):
                continue
            for step in steps:
                if isinstance(step, dict):
                    values.extend(_string_values(step.get("outputs", [])))

    declared = []
    for value in values:
        text = str(value)
        if not (text.endswith("external_raw.csv")
                or text.endswith("external_dao_raw.csv")
                or text.endswith("external_gpuntt_raw.csv")):
            continue
        path = Path(text)
        if not path.is_absolute():
            path = root.parent.parent / path if text.startswith("results/") else root / path
        path = path.resolve()
        if path not in declared:
            declared.append(path)

    combined = (root / EXTERNAL_POPULATION).resolve()
    if declared:
        # A combined artifact is authoritative when declared and present.  If
        # it is absent, use the declared partitions that are available.
        if combined in declared and combined.is_file():
            selected = [combined]
        else:
            selected = [path for path in declared if path.is_file()]
    elif combined.is_file():
        # Historical manifests did not always carry fresh_artifacts metadata.
        selected = [combined]
    else:
        selected = []
    return declared, selected


def _read_external_sources(root, manifest, external_manifest, queue_state):
    declared, selected = _declared_external_paths(root, manifest, external_manifest)
    observed = []
    eligible = []
    gates = []
    source_files = []
    for path in selected:
        with path.open(newline="") as stream:
            source_rows = list(csv.DictReader(stream))
        source_files.append(_relative_campaign_path(path, root))
        source_eligible, gate = _filter_output_rows(root, manifest, queue_state, path, source_rows)
        gate["path"] = _relative_campaign_path(path, root)
        gates.append(gate)
        observed.extend(source_rows)
        eligible.extend(source_eligible)

    if not selected and declared:
        source_files = [_relative_campaign_path(path, root) for path in declared]
    strict = bool(manifest.get("measurement_contract")) if isinstance(manifest, dict) else False
    gate = {
        "strict": strict,
        "eligible": (any(item.get("eligible") for item in gates) if gates else not strict),
        "sources": gates,
    }
    if selected and all(not item.get("eligible") for item in gates):
        gate["eligible"] = False
        gate["reason"] = "all-sources-blocked"
    elif not selected and declared:
        gate["eligible"] = False
        gate["reason"] = "declared-sources-missing"
    return observed, eligible, source_files, gate


def positive(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _is_correct(value):
    return value is True or value == 1 or str(value).strip().lower() in ("1", "true")


def _inverse_flag(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "inverse", "inv")


def _external_semantic_key(row):
    """Return the complete external-runner semantic key, or None if incomplete."""
    if any(field not in row or row[field] is None for field in EXTERNAL_SEMANTIC_FIELDS):
        return None
    if row.get("group") is None:
        return None
    group = str(row.get("group", "")).strip()
    if not group:
        return None
    return group, tuple(str(row.get(field, "")).strip() for field in EXTERNAL_SEMANTIC_FIELDS)


def _external_timing(rows):
    if not rows or any(not _is_correct(row.get("correct")) for row in rows):
        return None
    trial_values = [str(row.get("trial", "")).strip() for row in rows]
    if any(not value for value in trial_values):
        # Multiple rows without trial identities cannot be distinguished from
        # a repeated/resumed trial and must not be pooled as independent data.
        if len(rows) > 1:
            return None
    elif len(set(trial_values)) != len(trial_values):
        return None
    for field in ("runtime_fingerprint", "build_fingerprint", "library_sha256"):
        identities = {str(row.get(field, "")).strip() for row in rows if row.get(field) not in (None, "")}
        if len(identities) > 1:
            return None
    values = [positive(row.get("kernel_ms")) for row in rows]
    if any(value is None for value in values):
        return None
    median = statistics.median(values)
    relative_range = (max(values) - min(values)) / median
    result = {
        "median": median,
        "relative_range": relative_range,
        "stability": "unstable" if relative_range > 0.03 else "stable",
        "trials": len(values),
    }
    return result


def _external_status(row):
    value = row.get("correct")
    if value is None or not str(value).strip():
        return "missing"
    return "correct" if _is_correct(value) else "incorrect"


def _external_pairs_and_pending(raw, population):
    """Build only complete current/external pairs and retain missing cases."""
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    pending = {}
    for index, row in enumerate(raw):
        runner = str(row.get("runner", "")).strip()
        if runner not in EXTERNAL_RUNNERS:
            continue
        key = _external_semantic_key(row)
        if key is None:
            pending[("invalid", index)] = {
                "case": str(row.get("group") or row.get("case_id") or index),
                "operator": str(row.get("operator", "")),
                "precision": str(row.get("precision", "")),
                "direction": str(row.get("direction", "")),
                "external_runner": runner,
                "reason": "incomplete semantic or timing-protocol fields",
            }
            continue
        # A missing implementation must stay one runner population. Falling
        # back to case_id would turn each trial into a separate implementation
        # and create a Cartesian product during pairing.
        implementation = str(row.get("implementation") or runner).strip()
        grouped[key][runner][implementation].append(row)

    pairs = []
    for key in sorted(grouped):
        group, semantic = key
        values = dict(zip(EXTERNAL_SEMANTIC_FIELDS, semantic))
        roles = EXTERNAL_ROLES_BY_OPERATOR.get(values["operator"], EXTERNAL_ROLES)
        for external_runner, ours_runner in roles:
            ours_entries = grouped[key].get(ours_runner, {})
            external_entries = grouped[key].get(external_runner, {})
            pending_key = (key, external_runner)
            if not ours_entries or not external_entries:
                missing = []
                if not ours_entries:
                    missing.append(ours_runner)
                if not external_entries:
                    missing.append(external_runner)
                pending[pending_key] = {
                    "case": group, "operator": values["operator"],
                    "precision": values["precision"], "direction": values["direction"],
                    "external_runner": external_runner,
                    "reason": "missing " + ", ".join(missing),
                }
                continue

            if len(ours_entries) != 1 or len(external_entries) != 1:
                pending[pending_key] = {
                    "case": group, "operator": values["operator"],
                    "precision": values["precision"], "direction": values["direction"],
                    "external_runner": external_runner,
                    "reason": "multiple implementations for one protocol key",
                }
                continue

            paired = 0
            for ours_implementation, ours_rows in sorted(ours_entries.items()):
                ours_timing = _external_timing(ours_rows)
                for external_implementation, external_rows in sorted(external_entries.items()):
                    external_timing = _external_timing(external_rows)
                    if ours_timing is None or external_timing is None:
                        continue
                    ours_ms = ours_timing["median"]
                    external_ms = external_timing["median"]
                    pairs.append({
                        "population": population, "case": group,
                        **values,
                        "shape": values["N"], "batch": values["batch"],
                        "implementation": ours_implementation, "reference": external_implementation,
                        "reference_kind": "external-library",
                        "external_runner": external_runner, "ours_runner": ours_runner,
                        "external_implementation": external_implementation,
                        "vendor_fallback": False,
                        "selector_status": "external-matching-protocol",
                        "ours_ms": ours_ms, "external_ms": external_ms,
                        "cuButterfly_ms": ours_ms, "reference_ms": external_ms,
                        "ratio_direction": "external_ms/cuButterfly_ms",
                        "speedup": external_ms / ours_ms,
                        "trials": min(ours_timing["trials"], external_timing["trials"]),
                        "verification": "all external-runner trials correct=1",
                        "stability": "unstable" if "unstable" in (
                            ours_timing["stability"], external_timing["stability"]
                        ) else "stable",
                    })
                    paired += 1
            if not paired:
                pending[pending_key] = {
                    "case": group, "operator": values["operator"],
                    "precision": values["precision"], "direction": values["direction"],
                    "external_runner": external_runner,
                    "reason": "incorrect or missing positive kernel_ms",
                }
    return pairs, list(pending.values())


def paired_external(raw, population=EXTERNAL_POPULATION):
    """Pair Dao-FHT/GPU-NTT against the matching current runner only."""
    return _external_pairs_and_pending(raw, population)[0]


def external_population(raw, population=EXTERNAL_POPULATION, expected=None,
                        observed_raw=None, measurement_gate=None, source_files=None):
    observed = list(raw if observed_raw is None else observed_raw)
    pairs, pending = _external_pairs_and_pending(raw, population)
    candidate_keys = {
        _external_semantic_key(row)
        for row in raw
        if str(row.get("runner", "")).strip() in EXTERNAL_RUNNERS
        and _external_semantic_key(row) is not None
    }
    requested = max(int(expected or 0), len(candidate_keys))
    pending_cases = max(len(pending), requested - len(pairs))
    statuses = Counter(
        _external_status(row)
        for row in observed
    )
    duplicate_trials = 0
    trial_groups = defaultdict(list)
    for row in observed:
        key = (_external_semantic_key(row), str(row.get("runner", "")).strip(),
               str(row.get("implementation") or row.get("runner", "")).strip())
        trial_groups[key].append(str(row.get("trial", "")).strip())
    for trials in trial_groups.values():
        if len(trials) > 1 and (any(not trial for trial in trials) or len(set(trials)) != len(trials)):
            duplicate_trials += 1
    if not raw:
        statuses["missing"] += 1
    result = {
        "population": population, "requested": requested,
        "measured_cases": len(pairs), "pending_cases": pending_cases,
        "trial_rows": len(observed), "observed_trial_rows": len(observed),
        "eligible_trial_rows": len(raw), "statuses": dict(statuses),
        "duplicate_trial_groups": duplicate_trials,
    }
    if source_files is not None:
        result["source_files"] = list(source_files)
    if measurement_gate is not None:
        result["measurement_gate"] = measurement_gate
    if pending:
        result["pending"] = pending
    return result


def paired_rank1(raw, population):
    """Require both measured sides and matching public semantics."""
    by_case = defaultdict(list)
    for row in raw:
        by_case[str(row.get("suite_case_id", ""))].append(row)

    semantic_fields = (
        "operator", "precision", "direction", "normalization", "placement", "N",
        "performance_batch", "element_stride", "batch_stride", "word_bits",
        "modulus", "modulus_bits", "inverse", "input_order", "output_order", "stage_matrices",
        "accumulation", "emulated_native", "warmup", "repeat",
    )
    identity_fields = (
        "runtime_fingerprint", "build_fingerprint", "library_version",
        "selection_confidence", "selection_status", "selection_source",
    )
    completed = []
    for case_id, samples in by_case.items():
        if not case_id:
            continue
        if any(not _is_correct(row.get("preflight_correct"))
               or not (_is_correct(row.get("correct")) or
                       (row.get("suite_runner") == "fft-library" and
                        row.get("implementation") in ("cuFFT", "VkFFT") and
                        str(row.get("correct")) == "-1"))
               or positive(row.get("kernel_ms")) is None for row in samples):
            continue
        trial_values = [str(row.get("trial", "")).strip() for row in samples]
        if len(samples) > 1 and (any(not value for value in trial_values)
                                 or len(set(trial_values)) != len(trial_values)):
            continue
        inconsistent = False
        for field in semantic_fields + identity_fields:
            values = {str(row.get(field, "")).strip() for row in samples}
            if len(values) > 1:
                inconsistent = True
                break
        if inconsistent:
            continue
        for row in samples:
            normalized = dict(row)
            normalized["preflight_correct"] = "1"
            normalized["correct"] = "1"
            completed.append(normalized)
    if not completed:
        return []
    summaries = summarize_rank1(completed)
    selector_states = {row["suite_case_id"]: row.get("selection_confidence") or "explicit-or-external"
                       for row in completed}
    timed_checks = defaultdict(list)
    for row in completed:
        timed_checks[row["suite_case_id"]].append(
            int(row.get("verified_batches") or 0) if row.get("correct") == "1" else 0)
    groups = defaultdict(list)
    for row in summaries:
        groups[row["group"]].append(row)
    result = []
    # A group name alone is insufficient if a manifest has a contract error.
    fields = ("operator", "precision", "direction", "normalization", "placement",
              "N", "performance_batch", "element_stride", "batch_stride",
              "word_bits", "modulus", "modulus_bits", "inverse", "output_order",
              "stage_matrices", "accumulation", "emulated_native", "warmup", "repeat")
    raw_case_metadata = {
        case_id: samples[0] for case_id, samples in by_case.items()
        if samples and case_id
    }
    raw_protocol_fields = ("warmup", "repeat")
    for group, rows in groups.items():
        for ours in rows:
            if not ours["implementation"].startswith(("cuButterfly", "cuNTT")):
                continue
            if ours.get("operator") == "xor-zeta":
                continue
            for other in rows:
                if other is ours or other["implementation"] == ours["implementation"]:
                    continue
                internal = other["implementation"].startswith(("cuButterfly", "cuNTT"))
                if internal and not other["reference"]:
                    continue
                if any(str(ours.get(field, "")) != str(other.get(field, "")) for field in fields):
                    continue
                ours_raw = raw_case_metadata.get(ours["case_id"], {})
                other_raw = raw_case_metadata.get(other["case_id"], {})
                if any(str(ours_raw.get(field, "")) != str(other_raw.get(field, ""))
                       for field in semantic_fields):
                    continue
                result.append({
                    "population": population, "case": group,
                    "operator": ours.get("operator"), "precision": ours.get("precision"),
                    "direction": ours.get("direction") or ("inverse" if ours.get("inverse") == "1" else "forward"),
                    "shape": str(ours.get("N", "")), "batch": ours.get("performance_batch"),
                    "modulus": ours.get("modulus", ""), "modulus_bits": ours.get("modulus_bits", ""),
                    **{key: ours_raw.get(key, "") for key in
                       ("modulus", "normalization", "placement", "input_order", "output_order",
                        "accumulation", "element_stride", "batch_stride", "stage_matrices", "warmup", "repeat")},
                    "implementation": ours["implementation"], "reference": other["implementation"],
                    "ours_case_id": ours["case_id"], "reference_case_id": other["case_id"],
                    "reference_kind": "internal-control" if internal else "external-library",
                    "vendor_fallback": ours.get("backend") == "cufft",
                    "selector_status": selector_states[ours["case_id"]],
                    "cuButterfly_ms": ours["median_kernel_ms"],
                    "reference_ms": other["median_kernel_ms"],
                    "speedup": other["median_kernel_ms"] / ours["median_kernel_ms"],
                    "trials": min(ours["trials"], other["trials"]),
                    "verification": (f"timed cuButterfly mapping: {min(timed_checks[ours['case_id']])} sampled transforms; reference preflight"
                                     if min(timed_checks[ours["case_id"]]) > 0 else
                                     "correctness-preflight; see raw verification batch"),
                    "stability": "unstable" if "unstable" in (ours["stability_class"], other["stability_class"]) else "stable",
                })
    return result


def paired_xd(rows, population):
    candidates = defaultdict(list)
    for row in rows:
        # Other operators' separate C-ABI checks are coverage evidence, not
        # verification of a potentially different measured selector mapping.
        if (row.get("operator") != "fft" or row.get("configuration") != "selected"
                or not _is_correct(row.get("correct")) or row.get("status") != "passed"):
            continue
        ours, reference = positive(row.get("kernel_ms")), positive(row.get("cufft_ms"))
        if ours is None or reference is None:
            continue
        shape = "x".join(map(str, row["shape"]))
        inverse = _inverse_flag(row.get("inverse"))
        key = (str(row.get("operator", "")), str(row.get("precision", "")), shape,
               str(row.get("batch", "")), inverse,
               str(row.get("configuration", "")))
        candidates[key].append({
            "population": population, "case": row.get("id", shape),
            "operator": "fft", "precision": row["precision"],
            "direction": "inverse" if inverse else "forward",
            "shape": shape, "batch": row["batch"],
            "modulus": row.get("modulus", ""), "modulus_bits": row.get("modulus_bits", ""),
            "implementation": row.get("algorithm", "selected"), "reference": "cuFFT",
            "reference_kind": "external-library", "vendor_fallback": False,
            "selector_status": "xD-complete-plan-measurement",
            "cuButterfly_ms": ours, "reference_ms": reference, "speedup": reference / ours,
            "trials": row.get("trials", 0), "verification": "timed plan verified",
            "stability": "not-reported",
        })
    # A resumed run can emit the same semantic cell more than once.  It is
    # coverage evidence, but neither copy is an independent comparison.
    return [values[0] for values in candidates.values() if len(values) == 1]


def aggregate(pairs):
    groups = defaultdict(list)
    for row in pairs:
        key = tuple(row.get(field, "") for field in AGGREGATE_FIELDS)
        groups[key].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        eligible = [row for row in rows if not row.get("vendor_fallback")]
        if not eligible:
            continue
        # Different sizes/batches form the declared population distribution.
        # Repeated aliases of one workload are observations, not extra votes.
        unique = defaultdict(list)
        for row in eligible:
            semantic = tuple(str(row.get(field, "")) for field in
                             ("shape", "batch", "input_order", "accumulation", "element_stride",
                              "batch_stride", "stage_matrices"))
            unique[semantic].append(row["speedup"])
        values = [statistics.median(observations) for observations in unique.values()]
        result.append(dict(zip(AGGREGATE_FIELDS, key),
                           cases=len(values), geomean=math.exp(statistics.mean(map(math.log, values))),
                           duplicate_observations=len(eligible) - len(values),
                           median=statistics.median(values), minimum=min(values), maximum=max(values),
                           faster=sum(v > 1.03 for v in values), parity=sum(0.97 <= v <= 1.03 for v in values),
                           slower=sum(v < 0.97 for v in values),
                           vendor_fallbacks=sum(bool(row["vendor_fallback"]) for row in rows),
                           unmeasured_selector_fallbacks=sum(str(row["selector_status"]).startswith("unmeasured") for row in eligible)))
    return result


def _rank1_population(raw, expected=None, expected_trials=1,
                      observed_raw=None, measurement_gate=None):
    observed = list(raw if observed_raw is None else observed_raw)
    current_case_ids = {
        str(row.get("suite_case_id", ""))
        for row in raw
        if str(row.get("suite_case_id", ""))
        and str(row.get("implementation", "")).startswith(("cuButterfly", "cuNTT"))
        and row.get("operator") != "xor-zeta"
    }
    observed_case_ids = {
        str(row.get("suite_case_id", ""))
        for row in observed
        if str(row.get("suite_case_id", ""))
        and str(row.get("implementation", "")).startswith(("cuButterfly", "cuNTT"))
        and row.get("operator") != "xor-zeta"
    }
    pairs = paired_rank1(raw, "comparison/rank1_raw.csv")
    paired_case_ids = {str(row.get("ours_case_id", "")) for row in pairs}
    by_case = defaultdict(list)
    for row in raw:
        if row.get("suite_case_id") in current_case_ids:
            by_case[row["suite_case_id"]].append(row)
    verified = set()
    for case, rows in by_case.items():
        trials = [str(row.get("trial", "")) for row in rows]
        if (len(rows) >= expected_trials and len(set(trials)) == len(trials)
                and all(_is_correct(row.get("preflight_correct")) and _is_correct(row.get("correct"))
                        and positive(row.get("kernel_ms")) is not None for row in rows)):
            verified.add(case)
    candidate_case_ids = current_case_ids | observed_case_ids
    requested = max(int(expected or 0), len(candidate_case_ids))
    pending_ids = sorted(candidate_case_ids - verified)
    pending_cases = max(len(pending_ids), requested - len(verified))
    statuses = Counter()
    selection_statuses = Counter()
    for row in observed:
        statuses["preflight_correct=" + str(row.get("preflight_correct", "missing"))] += 1
        statuses["correct=" + str(row.get("correct", "missing"))] += 1
        selection_statuses[str(row.get("selection_confidence") or row.get("selector_status") or "missing")] += 1
    result = {
        "population": "comparison/rank1_raw.csv",
        "requested": requested,
        "measured_cases": len(verified),
        "verified_cases": len(verified),
        "matched_pairs": len(pairs),
        "paired_cases": len(paired_case_ids),
        "verified_without_reference": len(verified - paired_case_ids),
        "pending_cases": pending_cases,
        "trial_rows": len(observed),
        "observed_trial_rows": len(observed),
        "eligible_trial_rows": len(raw),
        "statuses": dict(statuses),
        "selection_statuses": dict(selection_statuses),
        "pending_case_ids": pending_ids,
    }
    if measurement_gate is not None:
        result["measurement_gate"] = measurement_gate
    if observed is not raw:
        result["observed_cases"] = len({
            str(row.get("suite_case_id", "")) for row in observed
            if str(row.get("suite_case_id", ""))
        })
    return result


def _coverage_value(coverage, field, default=None):
    if not isinstance(coverage, dict):
        return default
    if field in coverage:
        return coverage[field]
    nested = coverage.get("coverage")
    if isinstance(nested, dict) and field in nested:
        return nested[field]
    return default


def _canonical_xd_operator(value, aliases):
    current = str(value)
    seen = set()
    while isinstance(aliases, dict) and current in aliases and current not in seen:
        seen.add(current)
        current = str(aliases[current])
    return current


def _xd_key(row, aliases):
    shape = row.get("shape", ())
    if isinstance(shape, str):
        shape = tuple(part for part in shape.replace(",", "x").split("x") if part)
    elif isinstance(shape, (list, tuple)):
        shape = tuple(str(part) for part in shape)
    else:
        shape = (str(shape),)
    return (
        _canonical_xd_operator(row.get("operator", ""), aliases),
        str(row.get("precision", "")), shape, str(row.get("batch", "")),
        _inverse_flag(row.get("inverse", row.get("direction"))),
        str(row.get("configuration", "")),
    )


def _fallback_xd_verified_count(rows, profile_manifest):
    aliases = profile_manifest.get("semantic_aliases", {}) if isinstance(profile_manifest, dict) else {}
    return len({
        _xd_key(row, aliases) for row in rows
        if row.get("status") == "passed" and _is_correct(row.get("correct"))
    })


def report(root):
    manifest = read_json(root / "manifest.json", {})
    if not isinstance(manifest, dict):
        manifest = {}
    queue_state = read_json(root / "campaign_queue_state.json", {})
    if not isinstance(queue_state, dict):
        queue_state = {}
    pairs, populations = [], []
    rank1_path = root / "comparison" / "rank1_raw.csv"
    rank1_observed_rows = []
    if rank1_path.is_file():
        with rank1_path.open(newline="") as stream:
            rank1_observed_rows = list(csv.DictReader(stream))
    rank1_rows, rank1_gate = _filter_output_rows(
        root, manifest, queue_state, rank1_path, rank1_observed_rows)
    comparison_meta = manifest.get("comparison_suite", {})
    if not isinstance(comparison_meta, dict):
        comparison_meta = {}
    cohorts = comparison_meta.get("cohorts", {})
    if not isinstance(cohorts, dict):
        cohorts = {}
    expected_rank1 = cohorts.get("standard_rank1", 0)
    suite = read_json(root / "comparison_suite.json", {})
    if suite.get("cases"):
        expected_rank1 = sum(case.get("implementation", "").startswith(("cuButterfly", "cuNTT"))
                             and case.get("operator") != "xor-zeta" for case in suite["cases"])
    expected_trials = int(suite.get("protocols", {}).get("full", {}).get("trials", 1))
    if rank1_rows and "suite_case_id" not in rank1_rows[0]:
        rank1_rows = []
    if rank1_observed_rows or rank1_rows or expected_rank1:
        rank1_population = _rank1_population(
            rank1_rows, expected_rank1, expected_trials,
            observed_raw=rank1_observed_rows, measurement_gate=rank1_gate)
        search_workloads = read_json(root / "rank1_search_workloads.json", {})
        if isinstance(search_workloads, dict):
            rank1_population["selection_calibration_status"] = search_workloads.get("calibration_status", {})
        pairs.extend(paired_rank1(rank1_rows, "comparison/rank1_raw.csv"))
        populations.append(rank1_population)
    external_manifest = read_json(root / "external_suite.json")
    if external_manifest is None:
        external_manifest = read_json(root / "comparison" / "external_suite.json", {})
    if not isinstance(external_manifest, dict):
        external_manifest = {}
    expected_external = sum(
        len(workload.get("batches", [1]))
        for workload in external_manifest.get("workloads", [])
        if isinstance(workload, dict)
    )
    external_observed_rows, external_rows, external_sources, external_gate = _read_external_sources(
        root, manifest, external_manifest, queue_state)
    pairs.extend(paired_external(external_rows, EXTERNAL_POPULATION))
    populations.append(external_population(
        external_rows, EXTERNAL_POPULATION, expected_external,
        observed_raw=external_observed_rows, measurement_gate=external_gate,
        source_files=external_sources))
    for name in ("xd_coverage", "xd_scale"):
        path = root / "profile" / name / "rows.json"
        observed_rows = read_json(path, [])
        if not isinstance(observed_rows, list):
            observed_rows = []
        rows, measurement_gate = _filter_output_rows(
            root, manifest, queue_state, path, observed_rows)
        profile_pairs = paired_xd(rows, "profile/" + name)
        pairs.extend(profile_pairs)
        coverage_path = root / "profile" / name / "coverage_summary.json"
        coverage = read_json(coverage_path, {})
        if not isinstance(coverage, dict):
            coverage = {}
        coverage_gate = _strict_output_gate(root, manifest, queue_state, coverage_path)
        correctness_path = root / "profile" / (name + "_correctness.json")
        correctness = read_json(correctness_path, {})
        correctness_summary = correctness.get("summary", {}) if isinstance(correctness, dict) else {}
        profile_manifest = read_json(root / "profile" / name / "manifest.json", {})
        correctness_requested = int(correctness_summary.get("requested", 0) or 0)
        correctness_verified = int(correctness_summary.get("passed", 0) or 0)
        correctness_gate = _strict_output_gate(root, manifest, queue_state, correctness_path)
        correctness_identity_ok = True
        if isinstance(correctness, dict) and isinstance(profile_manifest, dict):
            for field in ("library_sha256", "compile_mode"):
                expected_identity = profile_manifest.get(field)
                observed_identity = correctness.get(field)
                if expected_identity not in (None, "") and observed_identity not in (None, "") \
                        and str(expected_identity) != str(observed_identity):
                    correctness_identity_ok = False
            expected_gpu = profile_manifest.get("gpu", {})
            observed_runtime = correctness.get("runtime", {})
            expected_model = expected_gpu.get("model") if isinstance(expected_gpu, dict) else None
            observed_model = observed_runtime.get("device_name") if isinstance(observed_runtime, dict) else None
            if expected_model not in (None, "") and observed_model not in (None, "") \
                    and str(expected_model) != str(observed_model):
                correctness_identity_ok = False
        if correctness and not correctness_identity_ok:
            correctness_verified = 0
        if (manifest.get("measurement_contract") and not correctness_gate["eligible"]):
            correctness_verified = 0
        trusted_coverage = not manifest.get("measurement_contract") or coverage_gate["eligible"]
        expected = _coverage_value(coverage, "expected_keys") if trusted_coverage else None
        if expected is None:
            profiles = manifest.get("profiles", {})
            profile_definition = profiles.get(name.removeprefix("xd_"), {}) if isinstance(profiles, dict) else {}
            expected = profile_definition.get("arithmetic_rows") if isinstance(profile_definition, dict) else None
        verified = _coverage_value(coverage, "verified_count") if trusted_coverage else None
        if verified is None:
            verified = _fallback_xd_verified_count(rows, profile_manifest)
        if manifest.get("measurement_contract") and not measurement_gate["eligible"]:
            # A stale coverage_summary can still be useful as an observation,
            # but it cannot certify performance after its rows were quarantined.
            verified = 0
        correctness_status = correctness.get("status", "missing") if isinstance(correctness, dict) else "missing"
        if manifest.get("measurement_contract") and not correctness_gate["eligible"]:
            correctness_status = "pending-output-gate"
        populations.append(dict(
            population="profile/" + name,
            requested=expected,
            measured_cases=verified,
            verified_cases=verified,
            observed_cases=len(observed_rows),
            eligible_cases=len(rows),
            performance_pairs=len(profile_pairs),
            pending_cases=max(int(expected or 0) - int(verified or 0), 0),
            correctness_requested=correctness_requested,
            correctness_verified=correctness_verified,
            correctness_pending_cases=max(correctness_requested - correctness_verified, 0),
            correctness_status=correctness_status,
            correctness_identity_ok=correctness_identity_ok,
            matched_pairs=len(profile_pairs),
            eligible_reference_cases=len(profile_pairs),
            statuses=dict(Counter(row.get("status", "missing") for row in observed_rows)),
            coverage_status=(_coverage_value(coverage, "status", "unknown")
                             if trusted_coverage else "pending-output-gate"),
            duplicate_cases=(_coverage_value(coverage, "duplicate_count", 0)
                             if trusted_coverage else 0),
            measurement_gate=measurement_gate,
            coverage_gate=coverage_gate,
            correctness_gate=correctness_gate,
        ))
    return {"schema": "cubutterfly-campaign-report-v1", "campaign": str(root),
            "status": queue_state.get("status", "partial"),
            "populations": populations, "aggregates": aggregate(pairs), "pairs": pairs,
            "excluded_vendor_fallback_pairs": sum(bool(row.get("vendor_fallback")) for row in pairs),
            "limitations": ["No historic timing rows are imported.",
                "Populations, precisions, directions and references are reported separately.",
                "A prepared or partially measured matrix is not completed coverage.",
                "Aliases are excluded from independent win counts; vendor fallbacks are identified.",
                "Speedup is reference time / cuButterfly time; parity uses a descriptive +/-3% band, not a significance test.",
                "External raw pairs require matching semantics and correct=1 for every side; missing or incorrect sides remain pending.",
                "External speedup is external-library time / current implementation time; exact modulus is retained in case and aggregate keys.",
                "Separate semantic correctness checks do not certify a different selected mapping.",
                "Reviewed-manifest performance uses only queue outputs with returncode=0 and gate_passed; blocked output rows remain observed and pending.",
                "Independent xD three-way, real-FFT and large-memory anchor artifacts are not merged into the current geometric means."]}


def write_report(root):
    result = report(root)
    destination = root / "tables"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "performance.json").write_text(json.dumps(result, indent=2) + "\n")
    for filename, rows in (("performance_cases.csv", result["pairs"]), ("performance_summary.csv", result["aggregates"])):
        with (destination / filename).open("w", newline="") as stream:
            if rows:
                fieldnames = []
                for row in rows:
                    for field in row:
                        if field not in fieldnames:
                            fieldnames.append(field)
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    lines = ["# Current-build performance", "", "Status: " + result["status"], "",
             "Speedup = reference latency / cuButterfly latency. For external raw data, reference is the paired external library. Empty populations remain pending.", "",
             "| Population | Operator / precision / direction | Protocol | Reference | Cases | Geomean | Median | Min–max | Faster / parity / slower | Vendor / unmeasured selector fallback |",
             "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |"]
    for row in result["aggregates"]:
        protocol_fields = ("normalization", "placement", "output_order", "warmup", "repeat",
                           "modulus", "modulus_bits")
        protocol = "; ".join(
            f"{field}={row.get(field)}" for field in protocol_fields
            if row.get(field) not in (None, "")
        ) or "declared protocol"
        lines.append(f"| {row['population']} | {row['operator']} / {row['precision']} / {row['direction']} | {protocol} | "
                     f"{row['reference']} ({row['reference_kind']}) | {row['cases']} | {row['geomean']:.3f}× | "
                     f"{row['median']:.3f}× | {row['minimum']:.3f}–{row['maximum']:.3f}× | "
                     f"{row['faster']} / {row['parity']} / {row['slower']} | {row['vendor_fallbacks']} / {row['unmeasured_selector_fallbacks']} |")
    if not result["aggregates"]:
        lines.extend(["", "No verified current-build comparison pairs yet."])
    lines.extend(["", "## Coverage progress", "", "| Population | Observed | Performance verified | Pending | Correctness verified / requested | Matched pairs | Status counts |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in result["populations"]:
        lines.append(f"| {row['population']} | {row.get('observed_cases', row.get('trial_rows', row['measured_cases']))} | "
                     f"{row.get('verified_cases', row['measured_cases'])} | {row.get('pending_cases', 0)} | "
                     f"{row.get('correctness_verified', 'n/a')} / {row.get('correctness_requested', 'n/a')} | "
                     f"{row.get('matched_pairs', 'n/a')} | "
                     f"{json.dumps(row['statuses'], sort_keys=True)} |")
    lines.extend(["", "## Interpretation", "", *["- " + text for text in result["limitations"]], ""])
    (destination / "performance.md").write_text("\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    result = write_report(args.campaign.resolve())
    print(json.dumps({"status": result["status"], "verified_pairs": len(result["pairs"]),
                      "table": str(args.campaign / "tables/performance.md")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
