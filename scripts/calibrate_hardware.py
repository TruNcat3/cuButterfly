#!/usr/bin/env python3
"""Single entry point for cuButterfly hardware migration calibration.

This module owns orchestration only. Candidate generation, measurement,
registry promotion and cost-model fitting remain in calibrate_local_hardware.py
so the installation path and the standalone recovery path share one engine.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any

from hardware_registry import canonical_semantics, default_path
from local_selector_data import select_points


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if not (ROOT / "CMakeLists.txt").exists():
    ROOT = ROOT / "share/cuButterfly"
CALIBRATOR = SCRIPT_DIR / "calibrate_local_hardware.py"
SELECTOR_VERIFIER = SCRIPT_DIR / "verify_local_selector.py"
DEFAULT_SOURCE_WORKLOADS = ROOT / "config/install_search_workloads.json"
DEFAULT_INSTALLED_WORKLOADS = ROOT / "config/install_search_workloads.json"
DEFAULT_PROTOCOL = {
    "trials": 5,
    "operator_trials": 3,
    "operator_warmup": 100,
    "operator_repeat": 100,
    "verify_batches": 0,
    "search_budget": 64,
    "search_finalists": 3,
    "search_seconds": 300.0,
    "compile_seconds": 600.0,
    # New installations revalidate a small set of portable historical
    # mappings.  _apply_resume_manifest restores zero for pre-seed manifests
    # so their old search protocol remains unchanged.
    "seed_budget": 8,
    "cost_model": "staged",
    "search_strategy": "model",
    "stage_calibration": "full",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect the current GPU and run or preview the complete cuButterfly migration calibration."
    )
    parser.add_argument("--build-dir", type=pathlib.Path,
                        help="configured cuButterfly build containing the benchmark binaries")
    parser.add_argument("--resume-from", type=pathlib.Path,
                        help="resume the exact paths and protocol recorded by migration_manifest.json")
    parser.add_argument("--profile-dir", type=pathlib.Path,
                        help="exact output directory; defaults below --profile-root")
    parser.add_argument("--profile-root", type=pathlib.Path,
                        help="parent for an automatically named profile directory")
    parser.add_argument("--output-dir", type=pathlib.Path,
                        help="calibration output directory; defaults to the profile directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mode", choices=("dry-run", "execute"), default="execute",
                      help="preview the resolved plan or execute it (default: execute)")
    mode.add_argument("--dry-run", dest="mode", action="store_const", const="dry-run",
                      help="alias for --mode dry-run")
    mode.add_argument("--execute", dest="mode", action="store_const", const="execute",
                      help="alias for --mode execute")
    parser.add_argument("--search-workloads", "--workloads", dest="search_workloads", type=pathlib.Path,
                        help="explicit workload configuration (default: config/install_search_workloads.json)")
    parser.add_argument("--trials", type=int,
                        help="hardware capability probe trials")
    parser.add_argument("--operator-trials", type=int)
    parser.add_argument("--operator-warmup", type=int)
    parser.add_argument("--operator-repeat", type=int)
    parser.add_argument("--verify-batches", type=int,
                        help="verify this many transforms in each timed batch; 0 verifies all")
    parser.add_argument("--search-budget", type=int,
                        help="screened mappings per workload; 0 exhausts the exported inventory")
    parser.add_argument("--search-finalists", type=int)
    parser.add_argument("--cost-model", choices=("staged", "legacy"))
    parser.add_argument("--stage-calibration", choices=("full", "bounded", "skip"),
                        help="independent stage calibration policy (default: full)")
    parser.add_argument("--stage-import-checkpoint", type=pathlib.Path, action="append",
                        help="reuse strictly compatible stage checkpoints; may be repeated")
    parser.add_argument("--search-strategy", choices=("model", "evolutionary"))
    parser.add_argument("--search-seconds", type=float,
                        help="total search wall-time budget; 0 is unrestricted")
    parser.add_argument("--compile-seconds", type=float,
                        help="specialization compilation budget; 0 is unrestricted")
    parser.add_argument("--mapping-seeds", type=pathlib.Path, action="append",
                        help="mapping seed file; may be repeated (mapping-only, no historical timing)")
    parser.add_argument("--seed-budget", type=int,
                        help="historical mapping seeds to revalidate per workload; 0 disables seed replay")
    parser.add_argument("--resume-search", action="store_true",
                        help="resume compatible checkpoints instead of mixing identities")
    parser.add_argument("--search-only", action="store_true",
                        help="calibrate only the explicitly configured workload cells")
    parser.add_argument("--skip-operator-calibration", action="store_true")
    parser.add_argument("--embed-legacy-selector", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing capability profile")
    return parser.parse_args(argv)


def _visible_gpu_index() -> str | None:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or visible in ("-1", "NoDevFiles"):
        return "__hidden__"
    return visible.split(",", 1)[0].strip()


def _nvidia_smi_identity() -> dict[str, Any] | None:
    """Read a device fingerprint without requiring a built benchmark binary."""
    query = ["nvidia-smi"]
    index = _visible_gpu_index()
    if index == "__hidden__":
        return None
    if index:
        query += ["-i", index]
    query += ["--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(query, cwd=ROOT, text=True, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    row = next(csv.reader([line for line in result.stdout.splitlines() if line.strip()]), None)
    if not row or len(row) < 3:
        return None
    name, capability, memory_mib = (value.strip() for value in row[:3])
    if not re.fullmatch(r"\d+\.\d+", capability):
        return None
    memory_match = re.search(r"\d+(?:\.\d+)?", memory_mib)
    if not memory_match:
        return None
    return {
        "device": name,
        "compute_capability": capability,
        "global_memory_bytes": int(float(memory_match.group()) * 1024 * 1024),
        "source": "nvidia-smi-estimate",
    }


def _benchmark_identity(build_dir: pathlib.Path) -> dict[str, Any] | None:
    if _visible_gpu_index() == "__hidden__":
        return None
    binary = build_dir / "cubutterfly_bench"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        return None
    try:
        result = subprocess.run([str(binary), "--device-identity"], cwd=ROOT,
                                text=True, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        row = next(csv.DictReader(line for line in result.stdout.splitlines() if line.strip()))
    except (StopIteration, csv.Error):
        return None
    required = ("device", "compute_capability", "global_memory_bytes")
    if any(not row.get(key) for key in required):
        return None
    try:
        memory = int(row["global_memory_bytes"])
    except (TypeError, ValueError):
        return None
    return {
        "device": row["device"],
        "compute_capability": row["compute_capability"],
        "global_memory_bytes": memory,
        "source": "cubutterfly_bench",
    }


def detect_identity(build_dir: pathlib.Path, *, required: bool = False) -> dict[str, Any]:
    if _visible_gpu_index() == "__hidden__":
        identity = None
    elif required:
        # The CUDA benchmark is the only source with the exact runtime memory
        # capacity used for profile-directory identity. nvidia-smi is suitable
        # for dry-run display only.
        identity = _benchmark_identity(build_dir)
    else:
        identity = _benchmark_identity(build_dir) or _nvidia_smi_identity()
    if identity:
        return identity
    result = {
        "device": "unknown",
        "compute_capability": "unknown",
        "global_memory_bytes": 0,
        "source": "unavailable",
    }
    if required:
        raise RuntimeError("unable to detect GPU identity exactly; execute mode requires cubutterfly_bench --device-identity")
    return result


def resolve_workloads(requested: pathlib.Path | None) -> pathlib.Path:
    if requested:
        path = requested.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"workload configuration was not found: {path}")
        return path
    for candidate in (DEFAULT_SOURCE_WORKLOADS, DEFAULT_INSTALLED_WORKLOADS):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("config/install_search_workloads.json was not found; pass --search-workloads FILE")


def load_workload_summary(path: pathlib.Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("schema") != "cubutterfly-install-search-v1":
        raise ValueError("unsupported workload configuration schema")
    workloads = document.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("workload configuration must contain a non-empty workloads list")
    semantic_workloads = [_normalize_workload(row) for row in workloads]
    operators = sorted({str(row["operator"]) for row in semantic_workloads})
    return {
        "schema": document["schema"],
        "scope": document.get("scope", "unspecified"),
        "workload_count": len(workloads),
        "operators": operators,
        "semantic_workloads": semantic_workloads,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "path": str(path),
    }


def _normalize_workload(workload: Any) -> dict[str, Any]:
    """Return the explicit semantic cell used by both planning and replay.

    The lower-level calibrator applies the same FFT default while building
    search coverage. Keeping it here makes a missing operator explicit instead
    of silently comparing an operator family with an entire workload matrix.
    """
    if not isinstance(workload, dict):
        raise ValueError("each workload must be an object")
    return {"operator": "fft", **workload}


def _semantic_key(workload: Any) -> str:
    return json.dumps(_workload_semantics(workload), sort_keys=True, separators=(",", ":"))


def _workload_semantics(workload: dict[str, Any]) -> dict[str, str]:
    values = _normalize_workload(workload)
    values.setdefault("placement", "natural" if values["operator"] == "ntt" else "out-of-place")
    if "stage_matrix" in values:
        values["stage_matrices"] = _canonical_stage_matrix(values.pop("stage_matrix"))
    if "stage_matrices" in values:
        values["stage_matrices"] = _canonical_stage_matrix(values["stage_matrices"])
    return canonical_semantics(values)


def _read_json(path: pathlib.Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _artifact_path(value: Any, output_dir: pathlib.Path) -> pathlib.Path | None:
    if not value or not isinstance(value, str):
        return None
    path = pathlib.Path(value).expanduser()
    if not path.is_absolute():
        path = output_dir / path
    return path


def _is_true(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _is_cufft_candidate(candidate: dict[str, Any], measured: dict[str, Any]) -> bool:
    configurations = [candidate.get("configuration"), measured.get("configuration")]
    configurations.extend(sample for sample in measured.get("samples", []) if isinstance(sample, dict))
    for configuration in configurations:
        if isinstance(configuration, dict) and str(configuration.get("backend", "")).lower() == "cufft":
            return True
    command = measured.get("command", candidate.get("command", []))
    if isinstance(command, list):
        return any(str(command[index]).lower() == "cufft" and index > 0 and
                   str(command[index - 1]) in {"--backend", "-backend"}
                   for index in range(len(command)))
    return False


def _canonical_stage_matrix(value: Any) -> str:
    if isinstance(value, list):
        value = ":".join(str(item) for item in value)
    value = str(value).replace(",", ":").replace(" ", "")
    if not value:
        return ""
    return "x".join(":".join(format(float(number), ".17g") for number in matrix.split(":"))
                    for matrix in value.split("x"))


def _candidate_matches_workload(workload: dict[str, Any], candidate: dict[str, Any],
                                measured: dict[str, Any]) -> bool:
    # Requested configuration alone cannot prove what the binary executed.
    samples = measured.get("samples", [])
    expected = _workload_semantics(workload)
    return bool(samples) and all(isinstance(sample, dict) and all(
        _workload_semantics(sample).get(key) == value for key, value in expected.items())
        for sample in samples)


def _confirmed_internal_candidate(candidate: dict[str, Any], measured: dict[str, Any] | None,
                                  min_samples: int = 2,
                                  workload: dict[str, Any] | None = None) -> bool:
    """Require a finalist and repeated correct measurements from the search.

    A one-sample screened row is useful evidence for diagnostics, but it is not
    enough to promote or replay a mapping. The coverage gate intentionally
    ignores cuFFT baselines even if they happen to be marked measured.
    """
    if not measured or candidate.get("status") != "measured":
        return False
    samples = measured.get("samples")
    if measured.get("status") != "measured" or not _is_true(measured.get("correct")):
        return False
    if not isinstance(samples, list) or len(samples) < max(2, min_samples):
        return False
    if not all(isinstance(sample, dict) and _is_true(sample.get("correct")) for sample in samples):
        return False
    if workload is not None and not _candidate_matches_workload(workload, candidate, measured):
        return False
    return not _is_cufft_candidate(candidate, measured)


def _int_field(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _mapping_seed_files(paths: list[pathlib.Path] | None) -> list[dict[str, str]]:
    """Resolve and fingerprint explicit seed files for the entry manifest.

    The automatic registry is deliberately not included here: it is a
    mutable cumulative input and the lower-level calibrator freezes it in its
    own mapping_seed_snapshot.json.  Explicit files, by contrast, are part of
    the caller's reproducibility contract and must remain byte-identical on
    resume.
    """
    records = []
    for value in paths or []:
        path = value.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"mapping seed file was not found: {path}")
        records.append({
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return records


def _mapping_seed_summary(paths: list[pathlib.Path] | None,
                          *, registry_path: pathlib.Path | str | None = None) -> dict[str, Any]:
    files = _mapping_seed_files(paths)
    automatic_registry = pathlib.Path(registry_path) if registry_path else default_path()
    return {
        "schema": "cubutterfly-mapping-seeds-v1",
        "files": files,
        "paths": [record["path"] for record in files],
        "automatic_registry_path": str(automatic_registry.expanduser().resolve()),
    }


def _resume_mapping_seed_files(manifest: dict[str, Any]) -> list[dict[str, str]] | None:
    """Read the current explicit seed-file records from a migration manifest."""
    record = manifest.get("mapping_seeds")
    if isinstance(record, dict):
        files = record.get("files")
        if isinstance(files, list):
            return [item for item in files if isinstance(item, dict) and item.get("path")]
    return None


def _validate_resume_mapping_seeds(args: argparse.Namespace,
                                   manifest: dict[str, Any]) -> None:
    expected = _resume_mapping_seed_files(manifest)
    if expected is None:
        # A pre-seed manifest has no seed input.  An explicitly supplied seed
        # would change the checkpoint protocol, so do not silently mix it.
        if args.mapping_seeds:
            raise ValueError("resume manifest has no mapping seed inputs; use a new output directory")
        return
    actual = _mapping_seed_files(args.mapping_seeds)
    expected_paths = [str(pathlib.Path(item["path"]).expanduser().resolve()) for item in expected]
    actual_paths = [item["path"] for item in actual]
    if actual_paths != expected_paths:
        raise ValueError("mapping seed inputs changed since the resume manifest was written")
    for old, current in zip(expected, actual):
        old_hash = old.get("sha256")
        if old_hash and old_hash != current["sha256"]:
            raise ValueError("mapping seed file changed since the resume manifest was written")


def _workload_coverage(output_dir: pathlib.Path,
                       expected_workloads: list[dict[str, Any]]) -> dict[str, Any]:
    coverage_path = output_dir / "search_coverage.json"
    measurements_path = output_dir / "search_measurements.json"
    coverage = _read_json(coverage_path)
    measurements = _read_json(measurements_path)
    expected = [_normalize_workload(row) for row in expected_workloads]
    # The explicit workload file is a semantic set. Stable de-duplication also
    # keeps a duplicated config row from changing completion counts on resume.
    unique_expected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for workload in expected:
        key = _semantic_key(workload)
        if key not in seen:
            seen.add(key)
            unique_expected.append(workload)

    base = {
        "status": "missing-artifact",
        "expected_count": len(unique_expected),
        "covered_count": 0,
        "missing_count": len(unique_expected),
        "covered": [],
        "missing": unique_expected,
        "cells": [],
        "confirmed_candidate_count": 0,
        "attempted_candidate_count": 0,
        "valid_candidate_count": 0,
        "required_confirmation_trials": 2,
        "budget_omitted_count": 0,
        "failed_candidate_count": 0,
        "search_protocol": {
            "requested_budget_per_workload": 0,
            "seed_budget": 0,
            "complete": False,
            "cells_complete": 0,
            "cell_count": 0,
        },
        "observed_search_protocol": {},
        "search_protocol_complete": False,
        "search_coverage": str(coverage_path),
        "search_measurements": str(measurements_path),
        "artifacts_present": coverage is not None and measurements is not None,
        "complete": False,
    }
    if not isinstance(coverage, dict) or not isinstance(coverage.get("workloads"), list):
        return base
    if not isinstance(measurements, list):
        base["status"] = "missing-measurements"
        return base

    search_protocol = coverage.get("search_protocol") if isinstance(coverage.get("search_protocol"), dict) else {}
    requested_budget = coverage.get("search_budget_per_workload", search_protocol.get("search_budget", 0))
    requested_budget = _int_field(requested_budget)
    seed_budget = _int_field(search_protocol.get("seed_budget", 0))
    confirmation_trials = max(2, _int_field(search_protocol.get("operator_trials", 2)))

    measured_by_id = {
        str(row.get("name")): row for row in measurements
        if isinstance(row, dict) and row.get("name")
    }
    incumbents = _read_json(output_dir / "operator_calibration.json")
    incumbents = incumbents if isinstance(incumbents, list) else []
    cells_by_key: dict[str, list[dict[str, Any]]] = {}
    for raw_cell in coverage["workloads"]:
        if not isinstance(raw_cell, dict) or not isinstance(raw_cell.get("workload"), dict):
            continue
        try:
            key = _semantic_key(raw_cell["workload"])
        except ValueError:
            continue
        cells_by_key.setdefault(key, []).append(raw_cell)

    covered: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    all_cells: list[dict[str, Any]] = []
    for workload in unique_expected:
        key = _semantic_key(workload)
        matching = cells_by_key.get(key, [])
        confirmed: list[str] = []
        omissions = 0
        failures = 0
        screened = 0
        valid = 0
        enumerated = 0
        protocol_stops: list[str] = []
        seed_coverage_values: list[dict[str, int]] = []
        requested_screened_values: list[int] = []
        for cell in matching:
            omissions += _int_field(cell.get("budget_omitted"))
            enumerated += _int_field(cell.get("enumerated"))
            if cell.get("stop_reason"):
                protocol_stops.append(str(cell["stop_reason"]))
            raw_seed_coverage = cell.get("seed_coverage")
            if isinstance(raw_seed_coverage, dict):
                seed_coverage_values.append({
                    field: _int_field(raw_seed_coverage.get(field, 0))
                    for field in ("available", "requested", "attempted", "confirmed", "budget_omitted")
                })
            if "requested_screened_count" in cell:
                requested_screened_values.append(_int_field(cell.get("requested_screened_count")))
            candidates = cell.get("candidates", [])
            if not isinstance(candidates, list):
                candidates = []
            screened += sum(isinstance(c, dict) and str(c.get("status", "")).lower() not in
                            ("", "budget-omitted", "seed-budget-omitted", "omitted", "seed-omitted")
                            and not str(c.get("status", "")).lower().endswith("-omitted")
                            for c in candidates)
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                status = str(candidate.get("status", ""))
                if status not in {"screened", "measured", "budget-omitted", ""}:
                    failures += 1
                candidate_id = str(candidate.get("id", ""))
                measured = measured_by_id.get(candidate_id)
                if measured and status in {"screened", "measured"} and _is_true(measured.get("correct")) and \
                        not _is_cufft_candidate(candidate, measured) and \
                        _candidate_matches_workload(workload, candidate, measured):
                    valid += 1
                if candidate_id and _confirmed_internal_candidate(candidate, measured, confirmation_trials, workload):
                    confirmed.append(candidate_id)
        # A candidate is counted once even if an interrupted resume produced
        # duplicate coverage cells.
        search_confirmed = sorted(set(confirmed))
        incumbent_confirmed = [str(row["name"]) for row in incumbents if isinstance(row, dict) and
                               row.get("name") and
                               _confirmed_internal_candidate(row, row, confirmation_trials, workload)]
        confirmed = sorted(set(search_confirmed + incumbent_confirmed))
        if seed_coverage_values:
            # Resume journals may contain duplicate records for a cell.  The
            # available pool is a snapshot property; work counters describe
            # the actual attempts represented by each matching cell.
            seed_coverage = {
                "available": max(row["available"] for row in seed_coverage_values),
                "requested": sum(row["requested"] for row in seed_coverage_values),
                "attempted": sum(row["attempted"] for row in seed_coverage_values),
                "confirmed": sum(row["confirmed"] for row in seed_coverage_values),
                "budget_omitted": sum(row["budget_omitted"] for row in seed_coverage_values),
            }
        else:
            seed_coverage = {
                "available": 0, "requested": 0, "attempted": 0,
                "confirmed": 0, "budget_omitted": 0,
            }
        if requested_screened_values:
            requested_screened_count = sum(requested_screened_values)
        else:
            requested_screened_count = sum(
                min(requested_budget, _int_field(cell.get("enumerated", 0))) if requested_budget
                else _int_field(cell.get("enumerated", 0))
                for cell in matching
            )
        cell_record = {
            "workload": workload,
            "status": "covered" if confirmed else ("missing-confirmation" if matching else "missing-cell"),
            "matching_cell_count": len(matching),
            "confirmed_candidate_ids": confirmed,
            "confirmed_candidate_count": len(confirmed),
            "search_confirmed_candidate_ids": search_confirmed,
            "incumbent_confirmed_candidate_ids": sorted(set(incumbent_confirmed)),
            "screened_count": screened,
            "attempted_count": screened,
            "valid_candidate_count": valid,
            "budget_omitted_count": omissions,
            "failed_candidate_count": failures,
            "cell_complete": bool(matching) and all(bool(cell.get("complete")) for cell in matching),
            "enumerated_count": enumerated,
            "requested_screened_count": requested_screened_count,
            "seed_coverage": seed_coverage,
            "search_protocol_stop_reasons": sorted(set(protocol_stops)),
        }
        requested = cell_record["requested_screened_count"]
        required_finalists = min(_int_field(search_protocol.get("search_finalists", 1)), valid)
        cell_record["required_finalists"] = required_finalists
        seed_protocol_complete = not seed_coverage_values or (
            seed_coverage["attempted"] >= seed_coverage["requested"])
        cell_record["search_protocol_complete"] = cell_record["cell_complete"] and requested > 0 and \
            screened >= requested and len(search_confirmed) >= required_finalists and \
            seed_protocol_complete and not protocol_stops
        all_cells.append(cell_record)
        if confirmed:
            covered.append(workload)
        else:
            missing.append(workload)

    base.update({
        "status": "complete" if not missing else ("partial" if covered else "missing-confirmation"),
        "covered_count": len(covered),
        "missing_count": len(missing),
        "covered": covered,
        "missing": missing,
        "cells": all_cells,
        "confirmed_candidate_count": sum(cell["confirmed_candidate_count"] for cell in all_cells),
        "attempted_candidate_count": sum(cell["attempted_count"] for cell in all_cells),
        "valid_candidate_count": sum(cell["valid_candidate_count"] for cell in all_cells),
        "required_confirmation_trials": confirmation_trials,
        "budget_omitted_count": sum(cell["budget_omitted_count"] for cell in all_cells),
        "failed_candidate_count": sum(cell["failed_candidate_count"] for cell in all_cells),
        "search_protocol": {
            "requested_budget_per_workload": requested_budget,
            "seed_budget": seed_budget,
            "complete": bool(all_cells) and all(cell["search_protocol_complete"] for cell in all_cells),
            "cells_complete": sum(bool(cell["search_protocol_complete"]) for cell in all_cells),
            "cell_count": len(all_cells),
        },
        "observed_search_protocol": search_protocol,
        "search_protocol_complete": bool(all_cells) and all(cell["search_protocol_complete"] for cell in all_cells),
        "complete": not missing,
    })
    return base


def _tag(identity: dict[str, Any]) -> str:
    device = re.sub(r"[^a-zA-Z0-9.-]+", "-", str(identity["device"])).strip("-") or "unknown"
    capability = str(identity["compute_capability"]).replace(".", "")
    memory = int(identity.get("global_memory_bytes", 0))
    return f"{device}-sm{capability}-{memory}B"


def resolve_directories(args: argparse.Namespace, identity: dict[str, Any]) -> tuple[pathlib.Path, pathlib.Path]:
    if args.profile_dir:
        profile_dir = args.profile_dir.expanduser().resolve()
    else:
        root = args.profile_root
        if root is None:
            root = pathlib.Path(os.environ.get("CUBUTTERFLY_PROFILE_ROOT", "~/.local/share/cuButterfly/hardware"))
        profile_dir = root.expanduser().resolve() / _tag(identity)
    output_dir = (args.output_dir or profile_dir).expanduser().resolve()
    return profile_dir, output_dir


def calibration_command(args: argparse.Namespace, profile_dir: pathlib.Path,
                        output_dir: pathlib.Path, workloads: pathlib.Path) -> list[str]:
    python = sys.executable
    command = [python, str(CALIBRATOR), "--build-dir", str(args.build_dir.resolve()),
               "--profile-dir", str(profile_dir), "--output-dir", str(output_dir),
               "--search-workloads", str(workloads), "--trials", str(args.trials),
               "--operator-trials", str(args.operator_trials), "--operator-warmup", str(args.operator_warmup),
               "--operator-repeat", str(args.operator_repeat), "--verify-batches", str(args.verify_batches),
               "--search-budget", str(args.search_budget), "--search-finalists", str(args.search_finalists),
               "--search-seconds", str(args.search_seconds), "--compile-seconds", str(args.compile_seconds),
               "--seed-budget", str(args.seed_budget),
               "--cost-model", str(getattr(args, "cost_model", "legacy")),
               "--stage-calibration", str(getattr(args, "stage_calibration", "skip")),
               "--search-strategy", str(getattr(args, "search_strategy", "model"))]
    for seed_path in args.mapping_seeds or []:
        command.extend(("--mapping-seeds", str(seed_path.resolve())))
    for checkpoint_path in getattr(args, "stage_import_checkpoint", None) or []:
        command.extend(("--stage-import-checkpoint", str(checkpoint_path.resolve())))
    for enabled, flag in ((args.resume_search, "--resume-search"), (args.search_only, "--search-only"),
                          (args.skip_operator_calibration, "--skip-operator-calibration"),
                          (args.embed_legacy_selector, "--embed-legacy-selector"), (args.force, "--force")):
        if enabled:
            command.append(flag)
    return command


def _compile_mode_from_artifacts(output_dir: pathlib.Path) -> str | None:
    identity = _read_json(output_dir / "operator_measurement_identity.json")
    if isinstance(identity, dict) and identity.get("compile_mode"):
        return str(identity["compile_mode"])
    return None


def _resume_manifest(path: pathlib.Path) -> dict[str, Any]:
    manifest = _read_json(path.expanduser().resolve())
    if not isinstance(manifest, dict):
        raise ValueError(f"resume manifest is missing or invalid: {path}")
    if manifest.get("schema") != "cubutterfly-hardware-migration-v1":
        raise ValueError(f"unsupported resume manifest schema: {manifest.get('schema')!r}")
    return manifest


def _apply_resume_manifest(args: argparse.Namespace) -> str | None:
    """Restore exact paths/protocol from a prior entry-point manifest."""
    if not args.resume_from:
        return os.environ.get("CUBUTTERFLY_COMPILE_MODE", "auto") or "auto"
    manifest_path = args.resume_from.expanduser().resolve()
    manifest = _resume_manifest(manifest_path)
    command = manifest.get("calibration_command") if isinstance(manifest.get("calibration_command"), list) else []
    if args.stage_import_checkpoint is None:
        args.stage_import_checkpoint = [pathlib.Path(command[i + 1])
                                       for i, value in enumerate(command[:-1])
                                       if value == "--stage-import-checkpoint"]
    for attribute, flag in (("search_only", "--search-only"),
                            ("skip_operator_calibration", "--skip-operator-calibration"),
                            ("embed_legacy_selector", "--embed-legacy-selector")):
        if flag in command:
            setattr(args, attribute, True)
    paths = {
        "build_dir": manifest.get("build_dir"),
        "profile_dir": manifest.get("profile_dir"),
        "output_dir": manifest.get("output_dir"),
    }
    for name, value in paths.items():
        if getattr(args, name) is None and value:
            setattr(args, name, pathlib.Path(str(value)))
    workload_record = manifest.get("workloads")
    if args.search_workloads is None and isinstance(workload_record, dict) and workload_record.get("path"):
        args.search_workloads = pathlib.Path(str(workload_record["path"]))
    if isinstance(workload_record, dict) and workload_record.get("sha256"):
        args._resume_workload_sha256 = str(workload_record["sha256"])
    seed_files = _resume_mapping_seed_files(manifest)
    if seed_files is not None:
        if args.mapping_seeds is None:
            args.mapping_seeds = [pathlib.Path(str(item["path"])) for item in seed_files]
        args._resume_mapping_seed_files = seed_files
    protocol = manifest.get("protocol") if isinstance(manifest.get("protocol"), dict) else {}
    protocol_keys = {
        "trials": "capability_trials",
        "operator_trials": "operator_trials",
        "operator_warmup": "operator_warmup",
        "operator_repeat": "operator_repeat",
        "verify_batches": "verify_batches",
        "search_budget": "search_budget",
        "search_finalists": "search_finalists",
        "search_seconds": "search_seconds",
        "compile_seconds": "compile_seconds",
        "seed_budget": "seed_budget",
        "cost_model": "cost_model",
        "search_strategy": "search_strategy",
        "stage_calibration": "stage_calibration",
    }
    for argument, manifest_key in protocol_keys.items():
        if getattr(args, argument) is None and manifest_key in protocol:
            setattr(args, argument, protocol[manifest_key])
    if args.seed_budget is None and "seed_budget" not in protocol:
        # A migration manifest written before seed replay must resume with its
        # original search-only behavior, even though fresh runs default to 8.
        args.seed_budget = 0
    for argument, legacy in (("cost_model", "legacy"), ("search_strategy", "model"), ("stage_calibration", "skip")):
        if getattr(args, argument) is None and argument not in protocol:
            setattr(args, argument, legacy)
    # The manifest records whether the checkpoint traversal was resumed; a
    # resume-from invocation must keep using that checkpoint identity.
    args.resume_search = True
    compile_mode = manifest.get("compile_mode")
    if not compile_mode and isinstance(protocol, dict):
        compile_mode = protocol.get("compile_mode")
    if not compile_mode:
        output = pathlib.Path(str(args.output_dir)) if args.output_dir else pathlib.Path(str(manifest.get("output_dir", "")))
        compile_mode = _compile_mode_from_artifacts(output)
    registry_path = manifest.get("registry_path")
    if not registry_path:
        output = pathlib.Path(str(args.output_dir)) if args.output_dir else pathlib.Path(str(manifest.get("output_dir", "")))
        old_calibration = _read_json(output / "calibration_manifest.json")
        if isinstance(old_calibration, dict):
            registry_path = old_calibration.get("registry_path")
    if registry_path:
        args._resume_registry_path = str(registry_path)
    _validate_resume_mapping_seeds(args, manifest)
    return str(compile_mode or os.environ.get("CUBUTTERFLY_COMPILE_MODE", "auto") or "auto")


def _fill_protocol_defaults(args: argparse.Namespace) -> None:
    if args.search_seconds is None and (getattr(args, "stage_calibration", None) or "full") == "full":
        args.search_seconds = 0
    if getattr(args, "compile_seconds", None) is None and (getattr(args, "stage_calibration", None) or "full") == "full":
        args.compile_seconds = 0
    for name, value in DEFAULT_PROTOCOL.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)


def _selector_replay(args: argparse.Namespace, output_dir: pathlib.Path,
                     environment: dict[str, str]) -> dict[str, Any]:
    """Replay promoted mappings through the common selector after calibration."""
    status_path = output_dir / "selector_replay_status.json"
    calibration = output_dir / "operator_calibration.json"
    result_path = output_dir / "selector_replay.json"
    if not calibration.is_file():
        result = {"status": "missing-artifact", "reason": "operator_calibration.json is absent",
                  "output": str(result_path), "result_count": 0}
        write_manifest(status_path, result)
        return result
    pending_path = output_dir / "selector_replay.pending.json"
    if pending_path.exists():
        pending_path.unlink()
    command = [sys.executable, str(SELECTOR_VERIFIER), "--build-dir", str(args.build_dir),
               "--calibration", str(calibration), "--output", str(pending_path)]
    env = os.environ.copy()
    env.update(environment)
    try:
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=env)
    except OSError as error:
        result = {"status": "failed", "returncode": None, "error": str(error),
                  "command": command, "output": str(result_path), "result_count": 0}
        write_manifest(status_path, result)
        return result
    if completed.returncode:
        result = {
            "status": "failed", "returncode": completed.returncode,
            "error": completed.stderr.strip() or completed.stdout.strip(),
            "command": command, "output": str(result_path), "result_count": 0,
        }
        write_manifest(status_path, result)
        return result
    results = _read_json(pending_path)
    if not _valid_selector_results(results, _read_json(calibration)):
        result = {"status": "failed", "returncode": 0,
                  "error": "selector verifier did not confirm every selected mapping",
                  "command": command, "output": str(result_path), "result_count": 0}
        write_manifest(status_path, result)
        return result
    pending_path.replace(result_path)
    result = {"status": "passed", "returncode": 0, "command": command,
              "output": str(result_path), "result_count": len(results),
              "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest()}
    write_manifest(status_path, result)
    return result


def write_manifest(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _valid_selector_results(results: Any, calibration: Any) -> bool:
    if isinstance(calibration, list):
        calibration = {"candidates": calibration}
    if not isinstance(results, list) or not results or not isinstance(calibration, dict):
        return False
    try:
        expected = {point["name"] for point in select_points(calibration)}
    except (KeyError, TypeError, ValueError):
        return False
    return bool(expected) and len(results) == len(expected) and all(
        isinstance(row, dict) and _is_true(row.get("correct")) and
        _is_true(row.get("mapping_matches")) and isinstance(row.get("sample"), dict) and
        _is_true(row["sample"].get("correct")) and
        row["sample"].get("selected_implementation") == row.get("name")
        for row in results) and {row.get("name") for row in results} == expected


def _selector_replay_summary(output_dir: pathlib.Path) -> dict[str, Any]:
    status_path = output_dir / "selector_replay_status.json"
    result_path = output_dir / "selector_replay.json"
    status = _read_json(status_path)
    if isinstance(status, dict) and status.get("status") != "passed":
        return status
    results = _read_json(result_path)
    calibration_path = output_dir / "operator_calibration.json"
    if _valid_selector_results(results, _read_json(calibration_path)):
        if isinstance(status, dict) and status.get("calibration_sha256") and \
                status["calibration_sha256"] != hashlib.sha256(calibration_path.read_bytes()).hexdigest():
            return {**status, "status": "stale", "reason": "calibration changed since selector replay"}
        return {
            **(status if isinstance(status, dict) else {}),
            "status": "passed",
            "result_count": len(results),
            "output": str(result_path),
        }
    return {"status": "failed" if status is not None or results is not None else "not-run",
            "result_count": 0, "output": str(result_path)}


def _execution_summary(output_dir: pathlib.Path,
                       expected_workloads: list[dict[str, Any]] | list[str] | None = None,
                       *, expected_operators: list[str] | None = None,
                       expected_protocol: dict[str, Any] | None = None,
                       expected_compile_mode: str | None = None) -> dict[str, Any]:
    """Summarize independent calibration evidence without overclaiming.

    Workload coverage, cost-model validation and selector replay are separate
    contracts. In particular, an operator family in the low-level manifest is
    not evidence that every explicit workload cell has a measured mapping.
    """
    if expected_workloads is None:
        expected_workloads = [{"operator": operator} for operator in (expected_operators or [])]
    elif expected_workloads and isinstance(expected_workloads[0], str):
        expected_workloads = [{"operator": operator} for operator in expected_workloads]  # type: ignore[list-item]
    expected_workloads = [_normalize_workload(row) for row in expected_workloads]  # type: ignore[arg-type]

    calibration_path = output_dir / "calibration_manifest.json"
    manifest = _read_json(calibration_path)
    coverage = _workload_coverage(output_dir, expected_workloads)
    selector = _selector_replay_summary(output_dir)
    observed_protocol = coverage.get("observed_search_protocol", {})
    protocol_fields = ("search_budget", "search_finalists", "search_seconds", "compile_seconds",
                       "operator_trials", "operator_warmup", "operator_repeat", "seed_budget", "cost_model", "search_strategy", "stage_calibration")
    protocol_mismatches: list[str] = []
    if expected_protocol:
        for field in protocol_fields:
            expected_value = expected_protocol.get(field)
            if expected_value is None:
                continue
            observed_value = observed_protocol.get(field)
            if observed_value is None:
                # Old low-level artifacts predate seed replay.  Unified
                # manifests default to seed_budget=0, which is semantically
                # identical to the absent field in those artifacts.
                if field == "seed_budget" and _int_field(expected_value) == 0:
                    continue
                if (field, expected_value) in (("cost_model", "legacy"), ("search_strategy", "model"), ("stage_calibration", "skip")):
                    continue
                protocol_mismatches.append(field)
                continue
            try:
                if float(observed_value) != float(expected_value):
                    protocol_mismatches.append(field)
            except (TypeError, ValueError):
                if str(observed_value) != str(expected_value):
                    protocol_mismatches.append(field)
    observed_identity = _read_json(output_dir / "search_coverage.json")
    observed_compile_mode = None
    if isinstance(observed_identity, dict) and isinstance(observed_identity.get("measurement_identity"), dict):
        observed_compile_mode = observed_identity["measurement_identity"].get("compile_mode")
    if expected_compile_mode and observed_compile_mode != expected_compile_mode:
        protocol_mismatches.append("compile_mode")
    protocol_identity = {
        "match": not protocol_mismatches,
        "mismatches": sorted(set(protocol_mismatches)),
        "expected": expected_protocol or {},
        "expected_compile_mode": expected_compile_mode,
        "observed": observed_protocol,
        "observed_compile_mode": observed_compile_mode,
    }
    coverage["protocol_identity"] = protocol_identity
    partial_names = ("search_coverage.json", "search_measurements.json",
                     "operator_calibration.json", "calibration_device.json",
                     "cost_model.json", "selector_replay.json",
                     "selector_replay_status.json")
    partial = {name: str(output_dir / name) for name in partial_names
               if (output_dir / name).is_file()}
    if not isinstance(manifest, dict):
        expected_operators_set = sorted({str(row["operator"]) for row in expected_workloads})
        return {
            "status": "missing-calibration-manifest" if not calibration_path.exists() else "invalid-calibration-manifest",
            "calibration_status": "missing" if not calibration_path.exists() else "invalid",
            "partial_artifacts": partial,
            "partial_workload_count": len(coverage.get("cells", [])),
            "partial_screened_count": sum(int(cell.get("screened_count", 0)) for cell in coverage.get("cells", [])),
            "operator_families": [],
            "manifest_operator_families": [],
            "expected_operator_families": expected_operators_set,
            "missing_operator_families": expected_operators_set,
            "registry_promoted_records": 0,
            "registry_path": None,
            "workload_coverage": coverage,
            "workload_coverage_complete": False,
            "mapping_coverage_complete": False,
            "search_protocol_complete": False,
            "protocol_identity": protocol_identity,
            "cost_model": {"status": "missing", "validated": False},
            "cost_model_status": "missing",
            "cost_model_validated": False,
            "cost_model_training_rows": 0,
            "cost_model_holdout_top1_accuracy": None,
            "selector_replay": selector,
            "calibration_manifest": str(calibration_path),
            "complete": False,
        }

    cost_model_path = _artifact_path(manifest.get("cost_model"), output_dir) or (output_dir / "cost_model.json")
    cost_model_raw = _read_json(cost_model_path)
    cost_model = cost_model_raw if isinstance(cost_model_raw, dict) else {}
    cost_status = str(cost_model.get("status", "missing"))
    validation = cost_model.get("validation") if isinstance(cost_model.get("validation"), dict) else {}
    stage_service = _read_json(output_dir / "stage_service_profile.json") or {}
    stage_required = manifest.get("stage_calibration_mode", "skip") != "skip"
    stage_validated = bool(stage_service.get("validated", False)) and stage_service.get("calibration_status") == "complete"
    composition_required = bool(stage_service.get("coverage", {}).get("composition_request_count"))
    composition_validation = cost_model.get("composition_validation", validation.get("composition_validation", {}))
    composition_validated = composition_validation.get("status") == "passed"
    global_ranking_validated = bool(validation.get("global_validation", {}).get("passed"))
    cost_validated = (cost_status == "calibrated-local" or
                      (cost_status == "calibrated-local-serial" and stage_validated and global_ranking_validated)) and \
        (not composition_required or composition_validated)
    manifest_operators = sorted({str(operator) for operator in manifest.get("operator_families", [])})
    confirmed_operators = sorted({str(row["operator"]) for row in coverage.get("covered", [])
                                  if isinstance(row, dict) and row.get("operator")})
    expected = sorted({str(row["operator"]) for row in expected_workloads})
    missing_operators = sorted(set(expected) - set(confirmed_operators))
    registry_path = manifest.get("registry_path")
    if registry_path:
        registry_path = str(registry_path)
    mapping_coverage_complete = bool(coverage.get("complete"))
    search_protocol_complete = bool(coverage.get("search_protocol_complete"))
    complete = mapping_coverage_complete and search_protocol_complete and protocol_identity["match"] and \
        cost_validated and selector.get("status") == "passed" and (not stage_required or stage_validated)
    return {
        "status": str(manifest.get("status", "unknown")),
        "calibration_status": str(manifest.get("status", "unknown")),
        "operator_families": confirmed_operators,
        "manifest_operator_families": manifest_operators,
        "expected_operator_families": expected,
        "missing_operator_families": missing_operators,
        "registry_promoted_records": _int_field(manifest.get("registry_promoted_records", 0)),
        "registry_path": registry_path,
        "workload_coverage": coverage,
        "workload_coverage_complete": mapping_coverage_complete,
        "mapping_coverage_complete": mapping_coverage_complete,
        "search_protocol_complete": search_protocol_complete,
        "protocol_identity": protocol_identity,
        "cost_model": {
            "status": cost_status,
            "validated": cost_validated,
            "training_rows": cost_model.get("training_rows", 0),
            "holdout_top1_accuracy": validation.get("holdout_top1_accuracy"),
            "holdout_mean_latency_regret": validation.get("holdout_mean_latency_regret"),
            "artifact": str(cost_model_path),
        },
        "cost_model_status": cost_status,
        "cost_model_validated": cost_validated,
        "stage_service_validated": stage_validated,
        "stage_service_coverage": stage_service.get("coverage", {}),
        "composition_required": composition_required,
        "composition_validated": composition_validated,
        "composition_validation": composition_validation,
        "cost_model_training_rows": cost_model.get("training_rows", 0),
        "cost_model_holdout_top1_accuracy": validation.get("holdout_top1_accuracy"),
        "selector_replay": selector,
        "calibration_manifest": str(calibration_path),
        "complete": complete,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    compile_mode = _apply_resume_manifest(args)
    _fill_protocol_defaults(args)
    if args.build_dir is None:
        raise SystemExit("--build-dir is required unless --resume-from supplies it")
    if args.trials <= 0 or args.operator_trials <= 0 or args.operator_warmup < 0 or args.operator_repeat <= 0 or args.verify_batches < 0:
        raise SystemExit("trials and repeat values must be positive")
    if args.search_budget < 0 or args.search_finalists < 1 or args.search_seconds < 0 or args.compile_seconds < 0 or args.seed_budget < 0:
        raise SystemExit("search budget must be nonnegative and finalists positive")
    args.build_dir = args.build_dir.expanduser().resolve()
    workloads = resolve_workloads(args.search_workloads)
    workload_summary = load_workload_summary(workloads)
    expected_workload_sha256 = getattr(args, "_resume_workload_sha256", None)
    if expected_workload_sha256 and workload_summary["sha256"] != expected_workload_sha256:
        raise ValueError("workload configuration changed since the resume manifest was written")
    mapping_seed_summary = _mapping_seed_summary(
        args.mapping_seeds, registry_path=getattr(args, "_resume_registry_path", None))
    identity = detect_identity(args.build_dir, required=args.mode == "execute")
    profile_dir, output_dir = resolve_directories(args, identity)
    command = calibration_command(args, profile_dir, output_dir, workloads)
    manifest_path = output_dir / "migration_manifest.json"
    plan: dict[str, Any] = {
        "schema": "cubutterfly-hardware-migration-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": args.mode,
        "device": identity,
        "build_dir": str(args.build_dir),
        "profile_dir": str(profile_dir),
        "output_dir": str(output_dir),
        "workloads": workload_summary,
        "mapping_seeds": mapping_seed_summary,
        "compile_mode": compile_mode,
        "calibration_command": command,
        "protocol": {
            "capability_trials": args.trials,
            "operator_trials": args.operator_trials,
            "operator_warmup": args.operator_warmup,
            "operator_repeat": args.operator_repeat,
            "verify_batches": args.verify_batches,
            "search_budget": args.search_budget,
            "search_finalists": args.search_finalists,
            "search_seconds": args.search_seconds,
            "compile_seconds": args.compile_seconds,
            "seed_budget": args.seed_budget,
            "cost_model": args.cost_model,
            "search_strategy": args.search_strategy,
            "stage_calibration": args.stage_calibration,
            "resume_search": args.resume_search,
            "compile_mode": compile_mode,
        },
        "registry_path": getattr(args, "_resume_registry_path", None) or str(default_path()),
        "status": "planned" if args.mode == "dry-run" else "running",
    }
    if args.mode == "dry-run":
        if manifest_path.exists():
            manifest_path = output_dir / "migration_plan.json"
        plan["warnings"] = ([] if identity["source"] != "unavailable" else
                            ["GPU identity unavailable; execute mode will fail until a GPU is visible"])
        write_manifest(manifest_path, plan)
        print(f"hardware migration: dry-run")
        print(f"device: {identity['device']} (source={identity['source']})")
        print(f"workloads: {workload_summary['workload_count']} cells ({', '.join(workload_summary['operators'])})")
        print(f"profile: {profile_dir}")
        print(f"manifest: {manifest_path}")
        print("command: " + " ".join(command))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest_path, plan)
    environment = {"CUBUTTERFLY_COMPILE_MODE": str(compile_mode)}
    resume_registry = getattr(args, "_resume_registry_path", None)
    if resume_registry:
        environment["CUBUTTERFLY_REGISTRY"] = str(resume_registry)
    expected_protocol = {
        "search_budget": args.search_budget,
        "search_finalists": args.search_finalists,
        "search_seconds": args.search_seconds,
        "compile_seconds": args.compile_seconds,
        "operator_trials": args.operator_trials,
        "operator_warmup": args.operator_warmup,
        "operator_repeat": args.operator_repeat,
        "seed_budget": args.seed_budget,
        "cost_model": args.cost_model,
        "search_strategy": args.search_strategy,
        "stage_calibration": args.stage_calibration,
    }
    try:
        completed = subprocess.run(command, cwd=ROOT, text=True, env={**os.environ, **environment})
    except OSError as error:
        plan.update(status="failed", error=str(error))
        write_manifest(manifest_path, plan)
        raise
    if completed.returncode:
        plan.update(status="failed", returncode=completed.returncode,
                    execution=_execution_summary(output_dir, workload_summary["semantic_workloads"],
                                                 expected_protocol=expected_protocol,
                                                 expected_compile_mode=compile_mode))
        write_manifest(manifest_path, plan)
        return completed.returncode
    calibration_metadata = _read_json(output_dir / "calibration_manifest.json")
    if isinstance(calibration_metadata, dict) and calibration_metadata.get("registry_path"):
        environment["CUBUTTERFLY_REGISTRY"] = str(calibration_metadata["registry_path"])
    if args.skip_operator_calibration:
        selector_status = {"status": "skipped", "reason": "operator calibration was skipped",
                           "output": str(output_dir / "selector_replay.json"), "result_count": 0}
        write_manifest(output_dir / "selector_replay_status.json", selector_status)
    else:
        _selector_replay(args, output_dir, environment)
    execution = _execution_summary(output_dir, workload_summary["semantic_workloads"],
                                   expected_protocol=expected_protocol,
                                   expected_compile_mode=compile_mode)
    complete = bool(execution.get("complete"))
    plan.update(status="complete" if complete else "incomplete", execution=execution,
                registry_path=execution.get("registry_path") or plan["registry_path"])
    write_manifest(manifest_path, plan)
    print(f"hardware migration: {plan['status']}")
    print(f"device: {identity['device']} (sm_{str(identity['compute_capability']).replace('.', '')}, {identity['global_memory_bytes']} bytes)")
    print(f"workloads: {workload_summary['workload_count']} cells ({', '.join(workload_summary['operators'])})")
    coverage = execution["workload_coverage"]
    print(f"confirmed mappings: {coverage['covered_count']}/{coverage['expected_count']} workloads; "
          f"search protocol: {coverage['search_protocol']['cells_complete']}/{coverage['expected_count']}")
    print(f"cost model: {execution['cost_model_status']}; selector replay: {execution['selector_replay']['status']}")
    print(f"migration manifest: {manifest_path}")
    return int((args.stage_calibration == "full" and not args.skip_operator_calibration and not complete) or
               execution["calibration_status"] in {"missing", "invalid"} or
               execution["selector_replay"].get("status") in {"failed", "missing-artifact", "stale"} or
               (not args.skip_operator_calibration and
                (execution["cost_model_status"] == "missing" or not coverage["artifacts_present"])))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
