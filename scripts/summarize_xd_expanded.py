#!/usr/bin/env python3
"""Summarize the fixed 64-case xD FFT expansion.

The input directory is produced by ``benchmark_xd_expanded.py``.  A case is
paired only when the configuration row and the requested baseline row have a
passed, finite ``graph.median_ms`` value.  Missing and failed rows remain
visible in the output and are never converted to an artificial infinity.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BASELINES = ("cufft-native", "vkfft")
FIXED_POPULATION = "xd-v11-fixed64"
FIXED_CASE_COUNT = 64
SCHEMA = 1


class SummaryError(ValueError):
    """Raised when a report directory cannot be safely aggregated."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _identity_value(document: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in document and document[key] not in (None, ""):
            return document[key]
    return None


def _binary_sha256(document: Mapping[str, Any]) -> str | None:
    value = _identity_value(document, ("binary_sha256", "library_sha256"))
    return str(value) if value is not None else None


def _gpu_uuid(document: Mapping[str, Any]) -> str | None:
    value = _identity_value(document, ("gpu_uuid", "uuid"))
    if value is None:
        for key in ("gpu", "hardware", "identity"):
            nested = document.get(key)
            if isinstance(nested, Mapping):
                value = _identity_value(nested, ("gpu_uuid", "uuid"))
                if value is not None:
                    break
    if value is None:
        return None
    normalized = str(value).strip().lower()
    # CUDA/PyTorch report bare physical UUIDs; nvidia-smi prefixes them.
    if normalized.startswith("gpu-"):
        normalized = normalized[4:]
    return normalized or None


def _timing_protocol(document: Mapping[str, Any]) -> Any:
    return _identity_value(
        document,
        ("timing_protocol", "timingprotocol", "measurement_protocol", "protocol"),
    )


def _manifest_protocol(document: Mapping[str, Any]) -> Any:
    # benchmark_xd_expanded.py currently does not put a protocol in the
    # manifest.  Accept the explicit spellings used by guarded releases, and
    # compare a legacy ``protocol`` field when one is present.
    return _timing_protocol(document)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SummaryError(message)


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SummaryError(f"cannot read JSON {path}: {exc}") from exc
    _require(isinstance(value, Mapping), f"JSON document is not an object: {path}")
    return value


def load_manifest(report_dir: Path) -> Mapping[str, Any]:
    """Load and validate the fixed 64-case manifest."""
    manifest = _load_json(report_dir / "manifest.json")
    _require(manifest.get("schema") == SCHEMA,
             f"unsupported xD manifest schema: {manifest.get('schema')!r}")
    _require(manifest.get("population") == FIXED_POPULATION,
             f"unexpected xD population: {manifest.get('population')!r}")
    cases = manifest.get("cases")
    _require(isinstance(cases, list), "xD manifest cases must be an array")
    _require(len(cases) == FIXED_CASE_COUNT,
             f"fixed xD population requires {FIXED_CASE_COUNT} cases, got {len(cases)}")
    ids: set[str] = set()
    for case in cases:
        _require(isinstance(case, Mapping), "xD manifest case is not an object")
        case_id = case.get("id")
        _require(isinstance(case_id, str) and case_id,
                 "xD manifest case has no non-empty id")
        _require(case_id not in ids, f"duplicate xD manifest case id: {case_id}")
        ids.add(case_id)
        _require(isinstance(case.get("shape"), list) and case["shape"],
                 f"xD case {case_id} has no shape")
        _require(isinstance(case.get("batch"), int) and case["batch"] > 0,
                 f"xD case {case_id} has an invalid batch")
        _require(case.get("precision") in ("fp32", "fp64"),
                 f"xD case {case_id} has an invalid precision")
        _require(isinstance(case.get("inverse"), bool),
                 f"xD case {case_id} has an invalid direction")
    return manifest


def _case_matches(case: Mapping[str, Any], report: Mapping[str, Any]) -> bool:
    return (
        list(report.get("shape", ())) == list(case.get("shape", ()))
        and report.get("batch") == case.get("batch")
        and report.get("precision") == case.get("precision")
        and report.get("inverse") == case.get("inverse")
    )


def _validate_identity(manifest: Mapping[str, Any], reports: Iterable[tuple[str, Mapping[str, Any]]]) -> dict[str, Any]:
    expected_binary = _binary_sha256(manifest)
    _require(expected_binary is not None,
             "manifest has no binary_sha256/library_sha256 identity")
    expected_uuid = _gpu_uuid(manifest)
    expected_protocol = _manifest_protocol(manifest)
    first_uuid: str | None = None
    first_protocol: Any = None
    report_count = 0
    for case_id, report in reports:
        report_count += 1
        actual_binary = _binary_sha256(report)
        _require(actual_binary == expected_binary,
                 f"binary_sha256 mismatch for {case_id}: expected {expected_binary}, "
                 f"got {actual_binary}")
        actual_uuid = _gpu_uuid(report)
        _require(actual_uuid is not None,
                 f"missing gpu_uuid for {case_id}; refusing aggregation")
        if expected_uuid is not None and actual_uuid != expected_uuid:
            raise SummaryError(
                f"gpu_uuid mismatch for {case_id}: expected {expected_uuid}, got {actual_uuid}"
            )
        if first_uuid is None:
            first_uuid = actual_uuid
        elif actual_uuid != first_uuid:
            raise SummaryError(
                f"gpu_uuid mismatch for {case_id}: expected {first_uuid}, got {actual_uuid}"
            )

        actual_protocol = _timing_protocol(report)
        _require(actual_protocol is not None,
                 f"missing timing protocol for {case_id}; refusing aggregation")
        if expected_protocol is not None and _canonical(actual_protocol) != _canonical(expected_protocol):
            raise SummaryError(f"timing protocol mismatch for {case_id} against manifest")
        if first_protocol is None:
            first_protocol = actual_protocol
        elif _canonical(actual_protocol) != _canonical(first_protocol):
            raise SummaryError(f"timing protocol mismatch for {case_id}")

    return {
        "binary_sha256": expected_binary,
        "gpu_uuid": first_uuid,
        "timing_protocol": first_protocol,
        "reports_checked": report_count,
    }


def _validate_baseline_identity(
    manifest: Mapping[str, Any], reports: Iterable[tuple[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    """Reject mixed external baselines or mixed graph timing protocols."""
    expected_vkfft = _identity_value(manifest, ("vkfft_library_sha256",))
    expected_cufft = manifest.get("cufft_version")
    cufft_versions: set[str] = set()
    vkfft_hashes: set[str] = set()
    for case_id, report in reports:
        report_cufft = report.get("cufft_version")
        if report_cufft is not None:
            cufft_versions.add(str(report_cufft))
            if expected_cufft is not None and str(report_cufft) != str(expected_cufft):
                raise SummaryError(
                    f"cufft_version mismatch for {case_id}: expected {expected_cufft}, "
                    f"got {report_cufft}"
                )
        rows = report.get("rows")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            requested = str(row.get("requested", ""))
            if requested == "vkfft":
                metadata = row.get("baseline_metadata")
                actual_vkfft = (
                    metadata.get("library_sha256")
                    if isinstance(metadata, Mapping)
                    else None
                )
                if actual_vkfft is not None:
                    actual_vkfft = str(actual_vkfft)
                    vkfft_hashes.add(actual_vkfft)
                if expected_vkfft is not None:
                    if row.get("status") == "passed":
                        _require(
                            actual_vkfft == str(expected_vkfft),
                            f"vkfft baseline library_sha256 mismatch for {case_id}",
                        )
                    elif actual_vkfft is not None and actual_vkfft != str(expected_vkfft):
                        raise SummaryError(
                            f"vkfft baseline library_sha256 mismatch for {case_id}"
                        )
            graph = row.get("graph")
            if isinstance(graph, Mapping):
                for field, expected in (("graph_capture_calls", 20), ("graph_replays", 7)):
                    actual = graph.get(field)
                    if actual is None:
                        continue
                    try:
                        actual_number = int(actual)
                    except (TypeError, ValueError) as exc:
                        raise SummaryError(
                            f"invalid {field} in {case_id} {requested} timing"
                        ) from exc
                    if actual_number != expected:
                        raise SummaryError(
                            f"{field} mismatch for {case_id} {requested}: "
                            f"expected {expected}, got {actual}"
                        )
    if len(cufft_versions) > 1:
        raise SummaryError(f"cufft_version mismatch across reports: {sorted(cufft_versions)}")
    if len(vkfft_hashes) > 1:
        raise SummaryError(f"vkfft library_sha256 mismatch across reports: {sorted(vkfft_hashes)}")
    if expected_vkfft is not None and vkfft_hashes and vkfft_hashes != {str(expected_vkfft)}:
        raise SummaryError("vkfft library_sha256 does not match manifest")
    return {
        "cufft_version": next(iter(cufft_versions), expected_cufft),
        "vkfft_library_sha256": str(expected_vkfft) if expected_vkfft is not None else next(iter(vkfft_hashes), None),
    }


def _configuration_name(row: Mapping[str, Any]) -> str:
    requested = row.get("requested")
    requested_name = str(requested).strip() if requested not in (None, "") else ""
    if requested_name:
        return requested_name
    algorithm = row.get("algorithm")
    return str(algorithm).strip() if algorithm not in (None, "") else "unknown"


def _row_time(row: Any) -> tuple[str, float | None]:
    if not isinstance(row, Mapping) or row.get("status") != "passed":
        return "failed", None
    graph = row.get("graph")
    value = graph.get("median_ms") if isinstance(graph, Mapping) else row.get("median_ms")
    try:
        timing = float(value)
    except (TypeError, ValueError):
        return "failed", None
    if not math.isfinite(timing) or timing <= 0.0:
        return "failed", None
    return "passed", timing


def _row_maps(record: Mapping[str, Any]) -> tuple[dict[str, list[Mapping[str, Any]]], set[str]]:
    report = record.get("report")
    if not isinstance(report, Mapping):
        return {}, set()
    rows = report.get("rows")
    if not isinstance(rows, list):
        return {}, set()
    baselines: dict[str, list[Mapping[str, Any]]] = {name: [] for name in BASELINES}
    configurations: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        requested = str(row.get("requested", ""))
        if requested in BASELINES:
            baselines[requested].append(row)
        else:
            name = _configuration_name(row)
            configurations.setdefault(name, []).append(row)
    return {**baselines, **configurations}, set(configurations)


def _state(record: Mapping[str, Any], rows: Mapping[str, list[Mapping[str, Any]]], name: str) -> tuple[str, float | None]:
    if record.get("file_state") == "missing":
        return "missing", None
    if record.get("file_state") == "failed":
        return "failed", None
    candidates = rows.get(name, [])
    if not candidates:
        return "missing", None
    if len(candidates) != 1:
        return "failed", None
    return _row_time(candidates[0])


def _metric(ratios: Sequence[float], *, missing: int, failed: int,
            unpaired: int) -> dict[str, Any]:
    values = list(ratios)
    if values:
        geometric = math.exp(sum(math.log(value) for value in values) / len(values))
        median = float(statistics.median(values))
        minimum = min(values)
        maximum = max(values)
    else:
        geometric = median = minimum = maximum = None
    return {
        "gmean": geometric,
        "median": median,
        "wins_gt_1": sum(value > 1.0 for value in values),
        "wins_gt_1_03": sum(value > 1.03 for value in values),
        "min": minimum,
        "max": maximum,
        "n": len(values),
        "missing": int(missing),
        "failed": int(failed),
        "unpaired": int(unpaired),
    }


def paired_metric(records: Sequence[Mapping[str, Any]], configuration: str,
                  baseline: str) -> dict[str, Any]:
    """Return baseline/configuration speedup metrics for already loaded cases."""
    if baseline not in BASELINES and baseline != "min_baseline":
        raise ValueError(f"unknown xD comparison baseline: {baseline}")
    ratios: list[float] = []
    missing = failed = unpaired = 0
    for record in records:
        rows, _ = _row_maps(record)
        config_state, config_time = _state(record, rows, configuration)
        baseline_states = [_state(record, rows, name) for name in BASELINES]
        if baseline == "min_baseline":
            required = [*baseline_states, (config_state, config_time)]
            if all(state == "passed" for state, _ in required):
                assert config_time is not None
                baseline_time = min(value for _, value in baseline_states if value is not None)
                ratios.append(baseline_time / config_time)
            else:
                unpaired += 1
                if any(state == "missing" for state, _ in required):
                    missing += 1
                if any(state == "failed" for state, _ in required):
                    failed += 1
            continue
        baseline_state, baseline_time = _state(record, rows, baseline)
        required = ((config_state, config_time), (baseline_state, baseline_time))
        if all(state == "passed" for state, _ in required):
            assert config_time is not None and baseline_time is not None
            ratios.append(baseline_time / config_time)
        else:
            unpaired += 1
            if any(state == "missing" for state, _ in required):
                missing += 1
            if any(state == "failed" for state, _ in required):
                failed += 1
    return _metric(ratios, missing=missing, failed=failed, unpaired=unpaired)


def _fastest(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    # Fastest is a three-way complete comparison.  In particular, do not
    # omit a missing baseline and let the selected configuration win by
    # default; all three timings must be present and passed.
    competitors = ("selected", "cufft-native", "vkfft")
    winners: dict[str, int] = {name: 0 for name in competitors}
    excluded_missing_baseline = 0
    excluded_failed_baseline = 0
    excluded_missing_configuration = 0
    excluded_failed_configuration = 0
    n = 0
    for record in records:
        rows, _ = _row_maps(record)
        states = {name: _state(record, rows, name) for name in competitors}
        baseline_states = [states[name] for name in BASELINES]
        if not all(state == "passed" for state, _ in baseline_states):
            if any(state == "missing" for state, _ in baseline_states):
                excluded_missing_baseline += 1
            else:
                excluded_failed_baseline += 1
            continue
        selected_state, selected_timing = states["selected"]
        if selected_state != "passed" or selected_timing is None:
            if selected_state == "missing":
                excluded_missing_configuration += 1
            else:
                excluded_failed_configuration += 1
            continue
        candidates = [
            (states[name][1], name) for name in competitors
            if states[name][0] == "passed" and states[name][1] is not None
        ]
        # Both baseline states and the selected state passed above, so this
        # list is complete by construction.
        _, winner = min(candidates, key=lambda item: (item[0], item[1]))
        winners[winner] += 1
        n += 1
    return {
        "n": n,
        "wins": winners,
        "excluded_missing_baseline": excluded_missing_baseline,
        "excluded_failed_baseline": excluded_failed_baseline,
        "excluded_missing_configuration": excluded_missing_configuration,
        "excluded_failed_configuration": excluded_failed_configuration,
        "basis": "lowest median_ms among selected, cufft-native, vkfft; all three passed",
    }


def _dimension_groups(records: Sequence[Mapping[str, Any]]) -> dict[str, list[tuple[str, list[Mapping[str, Any]]]]]:
    dimensions: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        "original16": {}, "all64": {}, "precision": {}, "direction": {}, "rank": {}
    }
    for record in records:
        case = record["case"]
        if "original16" in case.get("groups", []):
            dimensions["original16"].setdefault("original16", []).append(record)
        dimensions["all64"].setdefault("all64", []).append(record)
        dimensions["precision"].setdefault(str(case["precision"]), []).append(record)
        direction = "inverse" if case["inverse"] else "forward"
        dimensions["direction"].setdefault(direction, []).append(record)
        rank = str(len(case["shape"]))
        dimensions["rank"].setdefault(rank, []).append(record)
    result: dict[str, list[tuple[str, list[Mapping[str, Any]]]]] = {}
    for dimension, groups in dimensions.items():
        if dimension in ("original16", "all64"):
            order = [name for name in ("original16", "all64") if name in groups]
        elif dimension == "rank":
            order = sorted(groups, key=int)
        else:
            order = sorted(groups)
        result[dimension] = [(name, groups[name]) for name in order]
    return result


def _configuration_metadata(records: Sequence[Mapping[str, Any]], configurations: Sequence[str]) -> list[dict[str, Any]]:
    metadata: dict[str, dict[str, set[str]]] = {
        name: {"requested": set(), "algorithm": set()} for name in configurations
    }
    for record in records:
        report = record.get("report")
        if not isinstance(report, Mapping) or not isinstance(report.get("rows"), list):
            continue
        for row in report["rows"]:
            if not isinstance(row, Mapping) or str(row.get("requested", "")) in BASELINES:
                continue
            name = _configuration_name(row)
            if name not in metadata:
                continue
            if row.get("requested") not in (None, ""):
                metadata[name]["requested"].add(str(row["requested"]))
            if row.get("algorithm") not in (None, ""):
                metadata[name]["algorithm"].add(str(row["algorithm"]))
    return [
        {
            "name": name,
            "requested": sorted(metadata[name]["requested"]),
            "algorithm": sorted(metadata[name]["algorithm"]),
        }
        for name in configurations
    ]


def aggregate_records(records: Sequence[Mapping[str, Any]], configurations: Sequence[str]) -> dict[str, Any]:
    """Aggregate loaded case records without reading the filesystem."""
    configurations = sorted(set(configurations))
    output: dict[str, list[dict[str, Any]]] = {}
    for dimension, groups in _dimension_groups(records).items():
        dimension_output: list[dict[str, Any]] = []
        for value, group_records in groups:
            config_output: dict[str, Any] = {}
            for configuration in configurations:
                config_output[configuration] = {
                    baseline: paired_metric(group_records, configuration, baseline)
                    for baseline in (*BASELINES, "min_baseline")
                }
            dimension_output.append({
                "value": int(value) if dimension == "rank" else value,
                "case_count": len(group_records),
                "configurations": config_output,
                "fastest": _fastest(group_records),
            })
        output[dimension] = dimension_output
    return output


def _load_records(report_dir: Path, manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str], list[str], list[str]]:
    records: list[dict[str, Any]] = []
    missing_files: list[str] = []
    failed_files: list[str] = []
    identity_reports: list[tuple[str, Mapping[str, Any]]] = []
    for case in manifest["cases"]:
        case_id = str(case["id"])
        path = report_dir / f"{case_id}.json"
        if not path.exists():
            missing_files.append(case_id)
            records.append({"case": case, "report": None, "file_state": "missing"})
            continue
        try:
            report = _load_json(path)
        except SummaryError as exc:
            failed_files.append(case_id)
            records.append({"case": case, "report": None, "file_state": "failed", "error": str(exc)})
            continue
        _require(_case_matches(case, report), f"case metadata mismatch for {case_id}")
        identity_reports.append((case_id, report))
        if report.get("status") != "passed":
            failed_files.append(case_id)
        records.append({"case": case, "report": report, "file_state": "loaded"})
    return records, missing_files, failed_files, [case_id for case_id, _ in identity_reports]


def summarize_report_dir(report_dir: Path) -> dict[str, Any]:
    """Validate identity and return an auditable summary for ``report_dir``."""
    report_dir = Path(report_dir)
    manifest = load_manifest(report_dir)
    records, missing_files, failed_files, _ = _load_records(report_dir, manifest)
    valid_reports = [
        (str(record["case"]["id"]), record["report"])
        for record in records
        if isinstance(record.get("report"), Mapping)
    ]
    identity = _validate_identity(manifest, valid_reports)
    identity.update(_validate_baseline_identity(manifest, valid_reports))
    configurations: set[str] = set()
    for record in records:
        _, names = _row_maps(record)
        configurations.update(names)
    groups = aggregate_records(records, sorted(configurations))
    loaded_cases = sum(1 for record in records if isinstance(record.get("report"), Mapping))
    failed_rows = 0
    for record in records:
        report = record.get("report")
        if isinstance(report, Mapping):
            rows = report.get("rows")
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, Mapping) or row.get("status") != "passed":
                        failed_rows += 1
                    elif _row_time(row)[0] != "passed":
                        failed_rows += 1
    return {
        "schema": SCHEMA,
        "status": "complete" if not missing_files and not failed_files and not failed_rows else "partial",
        "aggregated": True,
        "population": manifest["population"],
        "identity": identity,
        "coverage": {
            "expected_cases": len(manifest["cases"]),
            "loaded_cases": loaded_cases,
            "missing_files": len(missing_files),
            "missing_file_ids": missing_files,
            "failed_files": len(failed_files),
            "failed_file_ids": failed_files,
            "failed_rows": failed_rows,
        },
        "configuration_metadata": _configuration_metadata(records, sorted(configurations)),
        "groups": groups,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        summary = summarize_report_dir(args.report_dir)
    except SummaryError as exc:
        print(f"refusing xD aggregation: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
