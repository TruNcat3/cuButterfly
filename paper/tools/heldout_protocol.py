#!/usr/bin/env python3
"""Freeze and evaluate the bounded E6 whole-workload holdout protocol.

Preparation is CPU-only.  It freezes a disjoint workload list and a small
fixed-core candidate set, audits the existing stage checkpoint, and emits a
GPU command for the later full-plan measurements.  The frozen stage
checkpoint is a physical-stage service source; its composite probe timings
are never used as whole-workload holdout observations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


SCHEMA = "cubutterfly-paper-heldout-protocol-v2"
POINTS_SCHEMA = "cubutterfly-paper-heldout-points-v1"
RESULT_SCHEMA = "cubutterfly-paper-heldout-validation-v1"
DESCRIPTORS_SCHEMA = "cubutterfly-paper-holdout-descriptors-v1"
PREDICTIONS_SCHEMA = "cubutterfly-paper-holdout-predictions-v1"
VALIDATION_SCHEMA = "cubutterfly-stage-plan-validation-v1"
TIMING_FIELDS = frozenset({
    "trial_kernel_ms", "plan_trial_kernel_ms", "kernel_ms", "median_kernel_ms",
    "public_plan_ms", "whole_workload_ms", "full_plan_ms", "full_workload_ms",
    "relative_error", "stage_sum_ms",
})
IDENTITY_FIELDS = ("uuid", "device", "compute_capability", "global_memory_bytes",
                   "compile_policy")
# Only these mapping knobs identify a frozen candidate.  The probe may add
# resource/runtime defaults; including those in a join would reject a valid
# resolved alias.  This is deliberately a projection, not a timing-based key.
JOIN_MAPPING_FIELDS = frozenset({
    "schema_version", "kind", "backend", "dispatch", "compute_unit",
    "stage_partition", "tile_threads", "threads_per_block", "fft_core", "core",
    "local_exchange", "shared_layout", "dataflow_layout", "exchange",
})
DEFAULT_E6 = (
    ("fft", 18, 16),
    ("fwht", 12, 1),
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def point_signature(point: dict[str, Any]) -> str:
    """Normalize a runner point to its frozen workload/mapping contract.

    The stage probe adds resolved aliases and bulk-policy defaults to its
    output point.  Those fields are useful in a measurement row but must not
    prevent joining it back to the frozen request.
    """
    if not isinstance(point, dict):
        raise ValueError("point must be an object")
    mapping = point.get("mapping_json", {})
    if isinstance(mapping, str):
        try:
            mapping = json.loads(mapping)
        except (TypeError, ValueError):
            raise ValueError("point mapping_json must encode an object")
    if mapping in (None, ""):
        mapping = {}
    elif not isinstance(mapping, dict):
        raise ValueError("point mapping_json must encode an object")
    else:
        mapping = dict(mapping)
    mapping = {field: mapping[field] for field in JOIN_MAPPING_FIELDS
               if field in mapping}
    operator = point.get("operator", mapping.get("operator", "fft"))
    precision = point.get("precision", mapping.get("precision", "fp32"))
    if operator == "ntt" and precision in (None, ""):
        bits = point.get("word_bits", mapping.get("word_bits"))
        precision = f"word{bits}" if bits not in (None, "") else "word64"
    log_n = point.get("logN", point.get("log_n", mapping.get("logN")))
    if log_n in (None, "") and point.get("N") not in (None, ""):
        log_n = int(math.log2(int(point["N"])))
    if log_n not in (None, ""):
        log_n = int(log_n)
    batch = point.get("batch", 1)
    if batch in (None, ""):
        batch = 1
    if batch not in (None, ""):
        batch = int(batch)
    result = {
        "operator": operator,
        "precision": precision,
        "logN": log_n,
        "batch": batch,
        "placement": point.get("placement", "natural" if operator == "ntt" else "out-of-place"),
        "direction": point.get("direction", "forward"),
        "normalization": point.get("normalization", "none"),
        "mapping_json": mapping,
    }
    for field in ("modulus", "input_order", "output_order", "stage_matrix"):
        value = point.get(field, mapping.get(field))
        if value not in (None, ""):
            # Benchmark CSV encodes the integer modulus as text. Preserve
            # all word64 bits when joining it to the JSON request.
            if field == "modulus" and isinstance(value, str):
                value = int(value)
            result[field] = value
    return canonical(result)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _identity_value(identity: dict[str, Any] | None, field: str) -> Any:
    """Read the UUID/compile aliases used by checkpoints and profiles."""
    if not isinstance(identity, dict):
        return None
    if field == "uuid":
        return identity.get("uuid") or identity.get("gpu_uuid")
    if field == "compile_policy":
        return identity.get("compile_policy") or identity.get("compile_mode")
    if field == "global_memory_bytes":
        return identity.get("global_memory_bytes") or identity.get("memory_bytes")
    return identity.get(field)


def _require_identity_match(actual: dict[str, Any] | None,
                            expected: dict[str, Any] | None,
                            label: str, *, require: bool = True) -> None:
    """Validate all available target identity fields without platform defaults."""
    actual = actual if isinstance(actual, dict) else {}
    expected = expected if isinstance(expected, dict) else {}
    for field in IDENTITY_FIELDS:
        expected_value = _identity_value(expected, field)
        if expected_value in (None, ""):
            continue
        actual_value = _identity_value(actual, field)
        if actual_value in (None, ""):
            if require:
                raise ValueError(f"{label} is missing identity field: {field}")
            continue
        if not _identity_equal(actual_value, expected_value, field):
            raise ValueError(f"{label} identity mismatch: {field}")


def _path_ref(path: Path) -> dict[str, str]:
    """Record a path and hash when the input already exists.

    Preparation is intentionally CPU-only and may emit commands before a
    target build is mounted.  Later phases require the hash when it was
    available at freeze time.
    """
    resolved = path.resolve()
    result = {"path": str(resolved)}
    if resolved.is_file():
        result["sha256"] = sha256_file(resolved)
    return result


def read_json(path: Path) -> tuple[Any, dict[str, str]]:
    raw = path.read_bytes()
    return json.loads(raw), {
        "path": str(path.resolve()),
        "sha256": sha256_bytes(raw),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def workload_id(workload: dict[str, Any]) -> str:
    """Stable ID for a complete public workload contract."""
    payload = canonical(workload).encode()
    return "workload-" + sha256_bytes(payload)[:16]


def _same_workload(workload: dict[str, Any], operator: str, log_n: int,
                   batch: int) -> bool:
    return (workload.get("operator") == operator
            and int(workload.get("logN", -1)) == log_n
            and int(workload.get("batch", -1)) == batch
            and workload.get("placement") == "out-of-place"
            and workload.get("direction", "forward") == "forward"
            and workload.get("normalization", "none") == "none")


def select_workloads(domain: list[dict[str, Any]], requested: list[str] | None = None,
                     defaults: tuple[tuple[str, int, int], ...] = DEFAULT_E6
                     ) -> list[dict[str, Any]]:
    """Select fixed representatives without looking at any timing artifact."""
    by_id = {workload_id(item): item for item in domain}
    if requested:
        selected = []
        for identifier in requested:
            if identifier not in by_id:
                raise ValueError(f"requested workload is absent from domain: {identifier}")
            selected.append(dict(by_id[identifier]))
    else:
        selected = []
        for operator, log_n, batch in defaults:
            matches = [item for item in domain if _same_workload(item, operator, log_n, batch)]
            if not matches:
                raise ValueError(f"representative workload is absent from domain: {operator}/{log_n}/{batch}")
            selected.append(dict(sorted(matches, key=canonical)[0]))
    identifiers = [workload_id(item) for item in selected]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("workload representatives must be unique")
    return selected


def balanced_partition(log_n: int, groups: int) -> list[int]:
    if groups < 1 or groups > log_n:
        raise ValueError("partition group count must be within logN")
    base, remainder = divmod(log_n, groups)
    return [base + (1 if index < remainder else 0) for index in range(groups)]


def _mapping_for(workload: dict[str, Any], partition: list[int], threads: int) -> dict[str, Any]:
    operator = workload["operator"]
    mapping = {
        "schema_version": 1,
        "kind": "ntt" if operator == "ntt" else "butterfly",
        "backend": "shared-iterative",
        "compute_unit": "radix2",
        "stage_partition": partition,
    }
    if operator == "ntt":
        mapping.update(threads_per_block=threads, dataflow_layout="hermes-xor")
    else:
        mapping.update(tile_threads=threads, fft_core="scalar",
                       local_exchange="shared", shared_layout="writer-aligned")
    return mapping


def make_point(workload: dict[str, Any], partition: list[int], threads: int) -> dict[str, Any]:
    point = dict(workload)
    point["mapping_json"] = canonical(_mapping_for(workload, partition, threads))
    return point


def validate_point(point: dict[str, Any]) -> dict[str, Any]:
    mapping = json.loads(point["mapping_json"])
    partition = mapping.get("stage_partition")
    log_n = int(point["logN"])
    if (mapping.get("backend") != "shared-iterative"
            or mapping.get("compute_unit") != "radix2"):
        raise ValueError("E6 candidates must use shared-iterative radix2")
    if not isinstance(partition, list) or not partition or any(int(x) <= 0 for x in partition):
        raise ValueError("candidate stage_partition must be positive")
    if sum(int(x) for x in partition) != log_n:
        raise ValueError("candidate stage_partition must cover logN")
    if point["operator"] == "ntt":
        threads = mapping.get("threads_per_block")
        if threads not in (128, 256) or mapping.get("dataflow_layout") != "hermes-xor":
            raise ValueError("unsupported NTT fixed-core candidate")
    else:
        threads = mapping.get("tile_threads")
        if threads not in (128, 256) or mapping.get("fft_core") != "scalar":
            raise ValueError("unsupported butterfly fixed-core candidate")
        if mapping.get("local_exchange") != "shared":
            raise ValueError("E6 requires shared local exchange")
    return mapping


def _candidate_id(point: dict[str, Any]) -> str:
    return "heldout-" + sha256_bytes(canonical(point).encode())[:20]


def _candidate_entry(point: dict[str, Any]) -> dict[str, Any]:
    mapping = validate_point(point)
    threads = mapping.get("tile_threads", mapping.get("threads_per_block"))
    fallback_score = (float(len(mapping["stage_partition"])) * float(1 << int(point["logN"]))
                      * float(point["batch"]) / float(threads))
    return {
            "candidate_id": _candidate_id(point),
            "core": "scalar",
            "point": point,
            "mapping": mapping,
            "fallback_score": fallback_score,
            "legality": {
                "structural": True,
                "compile_request": "shared-iterative-research",
                "runtime_lowering_verified": False,
            },
        }


def candidate_set(workload: dict[str, Any], count: int = 4) -> list[dict[str, Any]]:
    """Return the original deterministic scalar-core representative set."""
    if count != 4:
        raise ValueError("E6's default representative set supports exactly four candidates")
    return [_candidate_entry(make_point(workload, balanced_partition(int(workload["logN"]), groups), threads))
            for groups, threads in ((2, 128), (2, 256), (3, 128), (3, 256))]


def explicit_candidate_sets(path: Path, workloads: list[dict[str, Any]]):
    """Freeze a declared pool without consulting whole-plan measurements."""
    value, source = read_json(path)
    points = value.get("points") if isinstance(value, dict) else value
    if not isinstance(points, list) or not points:
        raise ValueError("candidate-points must contain a nonempty points list")
    by_semantics = {_semantic_key(w): workload_id(w) for w in workloads}
    result = {workload_id(w): [] for w in workloads}
    signatures = set()
    for point in points:
        if not isinstance(point, dict) or _semantic_key(point) not in by_semantics:
            raise ValueError("candidate point is outside the selected workload semantics")
        entry = _candidate_entry(point)
        signature = point_signature(point)
        if signature in signatures:
            raise ValueError("candidate-points repeats a mapping")
        signatures.add(signature)
        result[by_semantics[_semantic_key(point)]].append(entry)
    if any(len(entries) < 2 for entries in result.values()):
        raise ValueError("each held-out workload needs at least two declared candidates")
    return result, source


def _role_counts(checkpoint: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    records = checkpoint.get("records", [])
    record_roles: dict[str, int] = {}
    group_roles: dict[str, int] = {}
    for record in records if isinstance(records, list) else []:
        role = str(record.get("role", "train")).strip().lower()
        role = "validation" if role == "holdout" else role
        record_roles[role] = record_roles.get(role, 0) + 1
        groups = record.get("groups", [])
        for group in groups if isinstance(groups, list) else []:
            group_role = str(group.get("role", role)).strip().lower()
            group_role = "validation" if group_role == "holdout" else group_role
            group_roles[group_role] = group_roles.get(group_role, 0) + 1
    return record_roles, group_roles


def audit_stage_checkpoint(checkpoint: dict[str, Any], checkpoint_source: dict[str, str]
                           ) -> dict[str, Any]:
    """Audit roles and identity while excluding stage timing values from output."""
    records = checkpoint.get("records", [])
    records = records if isinstance(records, list) else []
    record_roles, group_roles = _role_counts(checkpoint)
    candidates = checkpoint.get("candidates", [])
    candidate_semantics = []
    for candidate in candidates if isinstance(candidates, list) else []:
        point = candidate.get("point", {})
        candidate_semantics.append({
            "operator": point.get("operator"),
            "precision": point.get("precision"),
            "logN": point.get("logN"),
            "placement": point.get("placement"),
            "mapping_json": point.get("mapping_json"),
        })
    probe_plan_timing_rows = sum("plan_trial_kernel_ms" in row for row in records)
    whole_workload_fields = ("whole_workload_ms", "public_plan_ms", "full_plan_ms", "full_workload_ms")
    whole_workload_field_counts = {
        field: sum(field in row or (isinstance(row.get("sample"), dict)
                                   and field in row["sample"])
                   for row in records) for field in whole_workload_fields
    }
    inventory = checkpoint.get("acquisition_inventory_ref") or {}
    identity = checkpoint.get("identity", {})
    return {
        "source": checkpoint_source,
        "schema": checkpoint.get("schema"),
        "status": checkpoint.get("status"),
        "calibration_status": checkpoint.get("calibration_status"),
        "identity": {
            "uuid": _identity_value(identity, "uuid"),
            "device": _identity_value(identity, "device"),
            "compute_capability": _identity_value(identity, "compute_capability"),
            "global_memory_bytes": _identity_value(identity, "global_memory_bytes"),
            "compile_policy": _identity_value(identity, "compile_policy"),
        },
        "record_count": len(records),
        "record_role_counts": record_roles,
        "group_role_counts": group_roles,
        "candidate_count": len(candidates) if isinstance(candidates, list) else 0,
        "model_curve_count": (checkpoint.get("model") or {}).get("curve_count"),
        "model_training_groups": (checkpoint.get("model") or {}).get("training_groups"),
        "model_validation_groups": (checkpoint.get("model") or {}).get("validation_groups"),
        "probe_plan_timing_rows": probe_plan_timing_rows,
        "whole_workload_field_counts": whole_workload_field_counts,
        "whole_workload_training_eligible": False,
        "training_scope": "physical-stage-groups-only",
        "whole_workload_ids": [],
        "reason": (
            "stage_calibration records are independent physical stage probes; even their "
            "plan_trial_kernel_ms fields are probe-plan diagnostics, not public whole-workload "
            "timings, so no whole workload is imported as E6 training"
        ),
        "acquisition_inventory_ref": {
            "path": inventory.get("path"),
            "sha256": inventory.get("sha256"),
            "count": inventory.get("count"),
        },
        "candidate_semantics": candidate_semantics,
    }


def _hash_basis(document: dict[str, Any]) -> dict[str, Any]:
    basis = json.loads(json.dumps(document))
    basis.pop("freeze", None)
    return basis


def protocol_hash(document: dict[str, Any]) -> str:
    return sha256_bytes(canonical(_hash_basis(document)).encode())


def _source_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _gpu_command(*, python: str, checkpoint: Path, profile: Path, build_dir: Path,
                 points: Path, output: Path, target_uuid: str,
                 compile_policy: str = "research") -> list[str]:
    root = Path(__file__).resolve().parents[2]
    runner = root / "scripts" / "validate_stage_service_profile.py"
    return [
        "env", f"CUDA_VISIBLE_DEVICES={target_uuid}",
        f"CUBUTTERFLY_COMPILE_MODE={compile_policy}", python, str(runner),
        "--checkpoint", str(checkpoint.resolve()), "--profile", str(profile.resolve()),
        "--build-dir", str(build_dir.resolve()), "--points", str(points.resolve()),
        "--output", str(output.resolve()),
    ]


def _describe_command(*, build_dir: Path, point: dict[str, Any]) -> list[str]:
    probe = build_dir.resolve() / "cubutterfly_stage_microbench"
    return [str(probe), "--describe-only", "--point-json", canonical(point)]


def _source_matches(path: Path, expected: dict[str, Any], label: str) -> dict[str, str]:
    actual = _source_record(path)
    if str(expected.get("path", "")) != actual["path"]:
        raise ValueError(f"{label} path differs from frozen protocol")
    if expected.get("sha256") != actual["sha256"]:
        raise ValueError(f"{label} sha256 differs from frozen protocol")
    return actual


def _validate_profile_source(path: Path, expected: dict[str, Any] | None,
                             label: str = "capability profile") -> tuple[dict[str, Any], dict[str, str]]:
    profile, actual = read_json(path)
    if not isinstance(profile, dict):
        raise ValueError(f"{label} must be an object")
    if isinstance(expected, dict) and expected.get("path"):
        if str(path.resolve()) != str(Path(expected["path"]).resolve()):
            raise ValueError(f"{label} path differs from frozen protocol")
        if expected.get("sha256") and expected["sha256"] != actual["sha256"]:
            raise ValueError(f"{label} sha256 differs from frozen protocol")
    return profile, actual


def _descriptor_binaries(descriptor: dict[str, Any]) -> dict[str, dict[str, Any]]:
    build = descriptor.get("build", {})
    binaries = build.get("benchmark_binaries")
    if not isinstance(binaries, dict):
        binaries = descriptor.get("target", {}).get("benchmark_binaries", {})
    return binaries if isinstance(binaries, dict) else {}


def _validate_descriptor_build(descriptor: dict[str, Any], build_dir: Path,
                               *, label: str = "build") -> dict[str, dict[str, Any]]:
    """Bind CPU phases to the exact binaries used by describe-only."""
    build_dir = build_dir.resolve()
    recorded = descriptor.get("build", {})
    recorded_dir = recorded.get("directory")
    if recorded_dir and Path(recorded_dir).resolve() != build_dir:
        raise ValueError(f"{label} directory differs from describe-only artifact")
    probe_path = recorded.get("probe_path")
    probe_sha = recorded.get("probe_sha256")
    if probe_path and Path(probe_path).resolve().parent != build_dir:
        raise ValueError(f"{label} stage probe path differs from describe-only artifact")
    if probe_sha:
        probe = build_dir / (Path(probe_path).name if probe_path else "cubutterfly_stage_microbench")
        if not probe.is_file() or sha256_file(probe) != probe_sha:
            raise ValueError(f"{label} stage probe changed since describe-only")
    binaries = _descriptor_binaries(descriptor)
    if not binaries:
        raise ValueError(f"{label} has no describe-only benchmark binary identities")
    for name, expected in binaries.items():
        if not isinstance(expected, dict) or not expected.get("sha256"):
            raise ValueError(f"describe-only artifact has no hash for {name}")
        binary = build_dir / name
        recorded_path = expected.get("path")
        if recorded_path and Path(recorded_path).resolve() != binary.resolve():
            raise ValueError(f"{label} binary path differs from describe-only artifact: {name}")
        if not binary.is_file() or sha256_file(binary) != expected["sha256"]:
            raise ValueError(f"{label} binary changed since describe-only: {name}")
    return binaries


def _validate_runtime_inputs(protocol: dict[str, Any], descriptor: dict[str, Any],
                             profile_path: Path | None = None,
                             build_dir: Path | None = None,
                             predictions: dict[str, Any] | None = None
                             ) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    """Validate optional strict evaluate/replay profile and build bindings."""
    profile = profile_source = None
    if profile_path is not None:
        expected_profile = ((predictions or {}).get("profile")
                            or protocol.get("sources", {}).get("capability_profile"))
        profile, profile_source = _validate_profile_source(profile_path, expected_profile)
        expected = protocol.get("target", {})
        expected_hardware = {
            field: _identity_value(expected, field)
            for field in ("uuid", "device", "compute_capability", "global_memory_bytes")
            if _identity_value(expected, field) not in (None, "")
        }
        _require_identity_match(_profile_identity(profile), expected_hardware,
                                "capability profile")
    if build_dir is not None:
        expected_build = protocol.get("sources", {}).get("build_dir", {})
        expected_build_path = expected_build.get("path") if isinstance(expected_build, dict) else None
        if expected_build_path and Path(expected_build_path).resolve() != Path(build_dir).resolve():
            raise ValueError("build directory differs from frozen protocol")
        _validate_descriptor_build(descriptor, build_dir)
    return profile, profile_source


def _load_protocol(path: Path, schema: str) -> tuple[dict[str, Any], dict[str, str]]:
    protocol, source = read_json(path)
    if not isinstance(protocol, dict) or protocol.get("schema") != schema:
        raise ValueError(f"unsupported protocol schema: expected {schema}")
    expected_hash = (protocol.get("freeze") or {}).get("protocol_sha256")
    if expected_hash != protocol_hash(protocol):
        raise ValueError("protocol hash mismatch")
    return protocol, source


def _protocol_stage_checkpoint(protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    source = protocol.get("sources", {}).get("stage_checkpoint")
    if not isinstance(source, dict) or not source.get("path"):
        raise ValueError("protocol has no frozen stage checkpoint source")
    checkpoint_path = Path(source["path"])
    checkpoint, actual = read_json(checkpoint_path)
    _source_matches(checkpoint_path, source, "stage checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != "cubutterfly-stage-calibration-v1":
        raise ValueError("frozen stage checkpoint has an unsupported schema")
    _require_identity_match(checkpoint.get("identity"), protocol.get("target", {}),
                            "stage checkpoint")
    return checkpoint, actual


def _protocol_groups(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    groups = protocol.get("heldout_workloads", protocol.get("workloads", []))
    if not isinstance(groups, list) or not groups:
        raise ValueError("protocol contains no workload groups")
    return groups


def _protocol_candidates(protocol: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    result = []
    for group in _protocol_groups(protocol):
        workload = group.get("workload_id")
        for candidate in group.get("candidates", []):
            if not isinstance(candidate, dict) or not candidate.get("candidate_id"):
                raise ValueError("protocol candidate is missing candidate_id")
            point = candidate.get("point")
            if not isinstance(point, dict):
                raise ValueError("protocol candidate is missing point")
            result.append((group, candidate, point))
    if not result:
        raise ValueError("protocol contains no candidates")
    return result


def _timing_fields(value: Any, path: str = "") -> list[str]:
    """Find timing observations, including JSON-encoded nested descriptors."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in TIMING_FIELDS:
                found.append(child_path)
            found.extend(_timing_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_timing_fields(child, f"{path}[{index}]"))
    elif isinstance(value, str) and value[:1] in "[{":
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            decoded = None
        if decoded is not None:
            found.extend(_timing_fields(decoded, path))
    return found


def _number_equal(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=0.5)
    except (TypeError, ValueError):
        return str(left) == str(right)


def _identity_equal(left: Any, right: Any, field: str) -> bool:
    if field == "global_memory_bytes":
        return _number_equal(left, right)
    return str(left).strip() == str(right).strip()


def _gpu_environment(uuid: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = uuid
    return environment


def _query_gpu_uuid(requested: str) -> str:
    result = subprocess.run(
        ["nvidia-smi", "-i", requested, "--query-gpu=uuid", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True, env=_gpu_environment(requested),
    )
    actual = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    if not actual or not _identity_equal(actual, requested, "uuid"):
        raise ValueError(f"describe-only GPU UUID mismatch: expected {requested}, got {actual or '<none>'}")
    return actual


def _query_benchmark_identity(binary: Path, environment: dict[str, str] | None = None) -> dict[str, Any]:
    result = subprocess.run([str(binary), "--device-identity"],
                            text=True, capture_output=True, check=True, env=environment)
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    if len(rows) != 1:
        raise ValueError(f"{binary.name} --device-identity did not return one row")
    identity = dict(rows[0])
    for key in ("device", "compute_capability", "global_memory_bytes"):
        if identity.get(key) in (None, ""):
            raise ValueError(f"{binary.name} device identity is missing {key}")
    return identity


def _descriptor_hash(document: dict[str, Any]) -> str:
    basis = json.loads(json.dumps(document))
    basis.pop("freeze", None)
    return sha256_bytes(canonical(basis).encode())


def _prediction_hash(document: dict[str, Any]) -> str:
    return _descriptor_hash(document)


def _descriptor_path(args: argparse.Namespace) -> Path:
    return Path(args.output).resolve()


def _descriptor_identity(protocol: dict[str, Any], checkpoint: dict[str, Any],
                         build_dir: Path, target_uuid: str) -> dict[str, Any]:
    target = protocol.get("target", {})
    checkpoint_identity = checkpoint.get("identity", {})
    expected_identity = dict(target)
    for field in ("uuid", "device", "compute_capability", "global_memory_bytes", "compile_policy"):
        if _identity_value(expected_identity, field) in (None, ""):
            value = _identity_value(checkpoint_identity, field)
            if value not in (None, ""):
                expected_identity[field] = value
    expected_identity["uuid"] = target_uuid
    _require_identity_match(checkpoint_identity, expected_identity, "stage checkpoint")
    environment = _gpu_environment(target_uuid)
    actual_uuid = _query_gpu_uuid(target_uuid)
    cubutterfly_binary = build_dir / "cubutterfly_bench"
    if not cubutterfly_binary.exists():
        raise ValueError("describe-only requires cubutterfly_bench in --build-dir")
    identity = _query_benchmark_identity(cubutterfly_binary, environment)
    bench_binaries = {
        "cubutterfly_bench": {
            "path": str(cubutterfly_binary.resolve()),
            "sha256": sha256_file(cubutterfly_binary),
            "identity": identity,
        }
    }
    cuntt_binary = build_dir / "cuntt_bench"
    if cuntt_binary.exists():
        # cuntt_bench does not implement --device-identity; its path and hash
        # are still bound by the descriptor and checked by later phases.
        bench_binaries["cuntt_bench"] = {
            "path": str(cuntt_binary.resolve()),
            "sha256": sha256_file(cuntt_binary),
        }
    _require_identity_match({**identity, "uuid": actual_uuid,
                             "compile_policy": (target.get("compile_policy")
                                                 or checkpoint_identity.get("compile_policy"))},
                            expected_identity, "describe-only target (cubutterfly_bench)")
    return {
        "uuid": actual_uuid,
        "device": identity["device"],
        "compute_capability": identity["compute_capability"],
        "global_memory_bytes": identity["global_memory_bytes"],
        "benchmark_binaries": bench_binaries,
        "compile_policy": target.get("compile_policy") or checkpoint_identity.get("compile_policy"),
    }


def describe(args: argparse.Namespace, schema: str = SCHEMA) -> dict[str, Any]:
    """Run only stage-probe descriptor calls on the requested GPU.

    The output is an identity-bound, timing-free input to the CPU prediction
    phase.  A failed descriptor is retained as a gap so the prediction phase
    can distinguish an unresolved service from a missing candidate request.
    """
    protocol, protocol_source = _load_protocol(args.protocol, schema)
    checkpoint, checkpoint_source = _protocol_stage_checkpoint(protocol)
    build_dir = Path(args.build_dir).resolve()
    probe = build_dir / "cubutterfly_stage_microbench"
    if not probe.exists():
        raise ValueError(f"describe-only probe is missing: {probe}")
    expected_probe_sha = checkpoint.get("identity", {}).get("binary_sha256")
    actual_probe_sha = sha256_file(probe)
    if expected_probe_sha and actual_probe_sha != expected_probe_sha:
        raise ValueError("stage probe binary changed since frozen calibration")
    target_uuid = args.target_uuid or protocol.get("target", {}).get("uuid")
    if not target_uuid:
        raise ValueError("describe-only requires protocol target UUID or --target-uuid")
    if protocol.get("target", {}).get("uuid") and target_uuid != protocol["target"]["uuid"]:
        raise ValueError("describe-only target UUID differs from protocol")
    target = _descriptor_identity(protocol, checkpoint, build_dir, target_uuid)
    needed_binaries = {
        "cuntt_bench" if point.get("operator") == "ntt" else "cubutterfly_bench"
        for _, _, point in _protocol_candidates(protocol)
    }
    missing_binaries = sorted(needed_binaries - set(target["benchmark_binaries"]))
    if missing_binaries:
        raise ValueError(f"describe-only build is missing benchmark binaries: {missing_binaries}")
    descriptors = []
    for group, candidate, point in _protocol_candidates(protocol):
        command = _describe_command(build_dir=build_dir, point=point)
        try:
            result = subprocess.run(command, text=True, capture_output=True, check=False,
                                    env=_gpu_environment(target_uuid))
            payload = json.loads(result.stdout) if result.stdout.strip() else {}
        except (OSError, json.JSONDecodeError) as error:
            # Preserve a gap row rather than silently converting an unresolved
            # resource description into a synthetic all-missing prior.
            descriptors.append({
                "workload_id": group["workload_id"], "candidate_id": candidate["candidate_id"],
                "point": point, "status": "unavailable", "reason": str(error),
                "command": command, "sample": None, "groups": [],
                "hardware": target, "runtime_fingerprint": None,
            })
            continue
        if result.returncode and payload.get("status") != "unavailable":
            raise ValueError(result.stderr.strip() or f"probe exited {result.returncode}")
        if payload.get("schema") != "cubutterfly-stage-probe-v1":
            raise ValueError("stage probe returned an unsupported schema")
        if _timing_fields(payload):
            raise ValueError("describe-only response contains timing observations")
        status = str(payload.get("status", "unavailable"))
        sample = payload.get("sample") if isinstance(payload.get("sample"), dict) else None
        if status == "resolved" and sample is None:
            raise ValueError("resolved descriptor has no sample")
        if sample is not None and point_signature(sample) != point_signature(point):
            raise ValueError("stage probe resolved a different candidate mapping")
        hardware = payload.get("hardware", {})
        for field in ("device", "compute_capability", "global_memory_bytes"):
            if field in hardware and not _identity_equal(hardware[field], target[field], field):
                raise ValueError(f"stage probe identity mismatch: {field}")
        descriptors.append({
            "workload_id": group["workload_id"],
            "candidate_id": candidate["candidate_id"],
            "point": point,
            "status": status,
            "reason": payload.get("reason"),
            "command": command,
            "sample": sample,
            "groups": payload.get("groups", []),
            "hardware": hardware,
            "runtime_fingerprint": (sample or {}).get("runtime_fingerprint") or payload.get("runtime_fingerprint"),
        })
    document = {
        "schema": DESCRIPTORS_SCHEMA,
        "status": "complete-describe",
        "experiment": protocol.get("experiment"),
        "protocol": protocol_source,
        "protocol_sha256": protocol["freeze"]["protocol_sha256"],
        "stage_checkpoint": checkpoint_source,
        "target": target,
        "build": {
            "directory": str(build_dir),
            "probe_path": str(probe.resolve()),
            "probe_sha256": actual_probe_sha,
            "benchmark_binaries": target["benchmark_binaries"],
        },
        "timing_free": True,
        "descriptor_count": len(descriptors),
        "resolved_count": sum(item["status"] == "resolved" for item in descriptors),
        "unavailable_count": sum(item["status"] != "resolved" for item in descriptors),
        "descriptors": descriptors,
        "rules_checked": {
            "no_timing_observations": True,
            "uuid_preflight": True,
            "stage_checkpoint_binary": True,
            "protocol_hash": True,
        },
    }
    document["freeze"] = {
        "algorithm": "sha256",
        "canonicalization": "json-sort-keys-separators-utf8; freeze field excluded",
        "descriptor_sha256": _descriptor_hash(document),
    }
    write_json(_descriptor_path(args), document)
    return document


def _semantic_key(point: dict[str, Any]) -> str:
    """Return workload semantics with the probe's implicit defaults applied.

    ``records[*].sample`` is a resolved physical sample and therefore carries
    fields such as ``N``, strides, and NTT word size that the public workload
    domain may omit.  Compare those fields after applying the same defaults so
    a shape is not accidentally treated as disjoint merely because one side
    came from the runner and the other from a concise request.
    """
    point = point if isinstance(point, dict) else {}
    operator = str(point.get("operator", "fft")).strip().lower()
    precision = point.get("precision")
    if operator == "ntt" and precision in (None, ""):
        bits = point.get("word_bits")
        precision = f"word{bits}" if bits not in (None, "") else "word64"
    log_n = point.get("logN", point.get("log_n"))
    n = point.get("N")
    if log_n in (None, "") and n not in (None, ""):
        log_n = int(math.log2(int(n)))
    if log_n not in (None, ""):
        log_n = int(log_n)
    if n in (None, "") and log_n not in (None, ""):
        n = 1 << log_n
    batch = point.get("batch", 1)
    if batch not in (None, ""):
        batch = int(batch)
    placement = point.get("placement", "natural" if operator == "ntt" else "out-of-place")
    direction = point.get("direction", "forward")
    normalization = point.get("normalization", "none")
    element_stride = point.get("element_stride", 1)
    if element_stride in (None, ""):
        element_stride = 1
    value = {
        "operator": operator,
        "precision": precision,
        "logN": log_n,
        "N": n,
        "batch": batch,
        "placement": placement,
        "direction": direction,
        "normalization": normalization,
        "accumulation": point.get("accumulation", "native"),
        "element_stride": element_stride,
    }
    # The benchmark's default batch stride is the contiguous extent.  Keep an
    # explicit non-contiguous stride in the semantic key because it changes
    # address geometry even when the transform shape is identical.
    extent = None
    try:
        extent = (int(n) - 1) * int(value["element_stride"]) + 1
    except (TypeError, ValueError):
        pass
    batch_stride = point.get("batch_stride", extent)
    if batch_stride in (None, ""):
        batch_stride = extent
    value["batch_stride"] = batch_stride
    if operator == "ntt":
        word_bits = point.get("word_bits")
        if word_bits in (None, "") and isinstance(precision, str) and precision.lower().startswith("word"):
            word_bits = precision[4:]
        if isinstance(word_bits, str) and word_bits.strip().isdigit():
            word_bits = int(word_bits)
        value.update({
            "word_bits": word_bits,
            "modulus": point.get("modulus"),
            "input_order": point.get("input_order", "natural"),
            "output_order": point.get("output_order", placement),
        })
    for field in ("value_type", "complex_values", "address_layout", "dimension_order", "stage_matrix"):
        if point.get(field) not in (None, ""):
            value[field] = point[field]
    return canonical(value)


def _semantic_inventory(checkpoint: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_role: dict[str, set[str]] = {}
    for record in checkpoint.get("records", []) if isinstance(checkpoint.get("records", []), list) else []:
        sample = record.get("sample")
        if not isinstance(sample, dict):
            continue
        record_role = str(record.get("role", "train")).strip().lower()
        record_role = "validation" if record_role == "holdout" else record_role
        group_roles = {
            str(group.get("role", record_role)).strip().lower()
            for group in record.get("groups", [])
            if isinstance(group, dict)
        }
        roles = group_roles or {record_role}
        for role in roles:
            role = "validation" if role == "holdout" else role
            by_role.setdefault(role, set()).add(_semantic_key(sample))
    result = {}
    for role, keys in sorted(by_role.items()):
        result[role] = [json.loads(key) for key in sorted(keys)]
    return result


def _audit_holdout_semantics(audit: dict[str, Any], checkpoint: dict[str, Any],
                             groups: list[dict[str, Any]]) -> dict[str, Any]:
    inventory = _semantic_inventory(checkpoint)
    stage_keys = {
        role: {canonical(item) for item in values}
        for role, values in inventory.items()
    }
    heldout = []
    overlap = []
    for group in groups:
        workload = group.get("workload", {})
        semantic = json.loads(_semantic_key(workload))
        item = {"workload_id": group.get("workload_id"), **semantic}
        heldout.append(item)
        roles = [role for role, keys in stage_keys.items() if canonical(semantic) in keys]
        if roles:
            overlap.append({**item, "stage_record_roles": sorted(roles)})
    audit = dict(audit)
    training = inventory.get("train", [])
    validation = inventory.get("validation", [])
    audit.update({
        "stage_record_workloads_by_role": inventory,
        "stage_training_workloads": training,
        "stage_validation_workloads": validation,
        "whole_workload_training_workloads": [],
        "whole_workload_validation_workloads": [],
        "heldout_workloads": heldout,
        "heldout_stage_semantic_overlap": overlap,
        "whole_workload_holdout_eligible": not overlap,
        "semantic_comparison_fields": [
            "operator", "precision", "logN", "N", "batch", "placement", "direction",
            "normalization", "accumulation", "element_stride", "batch_stride", "word_bits",
            "modulus", "input_order", "output_order", "value_type", "complex_values",
            "address_layout", "dimension_order", "stage_matrix",
        ],
    })
    return audit


def _load_descriptor(path: Path, protocol: dict[str, Any], schema: str = SCHEMA
                     ) -> tuple[dict[str, Any], dict[str, str]]:
    descriptor, source = read_json(path)
    if not isinstance(descriptor, dict) or descriptor.get("schema") != DESCRIPTORS_SCHEMA:
        raise ValueError("unsupported describe-only artifact schema")
    expected_hash = (descriptor.get("freeze") or {}).get("descriptor_sha256")
    if expected_hash != _descriptor_hash(descriptor):
        raise ValueError("describe-only artifact hash mismatch")
    if descriptor.get("protocol_sha256") != protocol.get("freeze", {}).get("protocol_sha256"):
        raise ValueError("describe-only artifact is bound to a different protocol")
    if descriptor.get("experiment") != protocol.get("experiment"):
        raise ValueError("describe-only artifact experiment differs from protocol")
    expected_checkpoint = protocol.get("sources", {}).get("stage_checkpoint", {})
    descriptor_checkpoint = descriptor.get("stage_checkpoint", {})
    if descriptor_checkpoint.get("path") != expected_checkpoint.get("path"):
        raise ValueError("describe-only artifact uses a different stage checkpoint")
    if descriptor_checkpoint.get("sha256") != expected_checkpoint.get("sha256"):
        raise ValueError("describe-only artifact uses a changed stage checkpoint")
    if descriptor.get("target", {}).get("uuid") != protocol.get("target", {}).get("uuid"):
        raise ValueError("describe-only artifact UUID differs from protocol")
    _require_identity_match(descriptor.get("target"), protocol.get("target", {}),
                            "describe-only artifact target")
    if descriptor.get("timing_free") is not True or _timing_fields(descriptor):
        raise ValueError("describe-only artifact contains timing observations")
    expected = {(group["workload_id"], candidate["candidate_id"])
                for group, candidate, _ in _protocol_candidates(protocol)}
    descriptor_items = descriptor.get("descriptors", [])
    if not isinstance(descriptor_items, list):
        raise ValueError("describe-only artifact has no descriptors list")
    expected_points = {(group["workload_id"], candidate["candidate_id"]): point
                       for group, candidate, point in _protocol_candidates(protocol)}
    actual_items = []
    for item in descriptor_items:
        if not isinstance(item, dict):
            raise ValueError("describe-only artifact contains a non-object descriptor")
        key = (item.get("workload_id"), item.get("candidate_id"))
        actual_items.append(key)
        if key not in expected_points:
            continue
        item_point = item.get("point")
        if not isinstance(item_point, dict) or point_signature(item_point) != point_signature(expected_points[key]):
            raise ValueError(f"describe-only artifact mapping differs for candidate {key[1]}")
    actual = set(actual_items)
    if len(actual_items) != len(actual):
        raise ValueError("describe-only artifact repeats a candidate")
    if actual != expected:
        raise ValueError("describe-only artifact candidate set differs from protocol")
    return descriptor, source


def _profile_identity(profile: dict[str, Any]) -> dict[str, Any]:
    identity = dict(profile.get("identity", {})) if isinstance(profile.get("identity"), dict) else {}
    for field in ("device", "compute_capability", "global_memory_bytes", "memory_bytes",
                  "gpu_uuid", "uuid", "compile_policy", "compile_mode"):
        if profile.get(field) not in (None, ""):
            identity.setdefault(field, profile[field])
    return identity


def _build_prediction_model(checkpoint: dict[str, Any], profile: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    import stage_cost_model
    import stage_service_model

    service = stage_service_model.build_profile(checkpoint.get("records", []), checkpoint.get("identity"))
    if not service.get("curves"):
        raise ValueError("frozen stage checkpoint has no independent service curves")
    model_profile = dict(profile)
    # The service table is rebuilt from the frozen checkpoint so a cached
    # profile cannot smuggle in a post-hoc timing fit.
    model_profile["stage_service"] = service
    return stage_cost_model.fit([], model_profile), service


def predict(args: argparse.Namespace, schema: str = SCHEMA) -> dict[str, Any]:
    """Build frozen priors from descriptors and independent stage curves only."""
    protocol, protocol_source = _load_protocol(args.protocol, schema)
    checkpoint, checkpoint_source = _protocol_stage_checkpoint(protocol)
    explicit_checkpoint = getattr(args, "stage_checkpoint", None)
    if explicit_checkpoint is not None:
        _source_matches(Path(explicit_checkpoint).resolve(), protocol["sources"]["stage_checkpoint"],
                        "explicit stage checkpoint")
    descriptor, descriptor_source = _load_descriptor(args.descriptors, protocol, schema)
    profile_ref = protocol.get("sources", {}).get("capability_profile")
    profile, profile_source = _validate_profile_source(Path(args.profile), profile_ref)
    expected_identity = protocol.get("target", {})
    checkpoint_identity = checkpoint.get("identity", {})
    profile_identity = _profile_identity(profile)
    expected_profile_identity = {}
    for field in ("uuid", "device", "compute_capability", "global_memory_bytes"):
        expected = _identity_value(expected_identity, field) or _identity_value(checkpoint_identity, field)
        if expected not in (None, ""):
            expected_profile_identity[field] = expected
    _require_identity_match(profile_identity, expected_profile_identity,
                            "capability profile")
    if _timing_fields(profile):
        raise ValueError("capability profile unexpectedly contains whole-workload timing fields")
    build_dir = getattr(args, "build_dir", None)
    if build_dir is not None:
        _validate_descriptor_build(descriptor, Path(build_dir), label="prediction build")
    model, service = _build_prediction_model(checkpoint, profile)
    expected_by_id = {
        (group["workload_id"], candidate["candidate_id"]): (candidate, point)
        for group, candidate, point in _protocol_candidates(protocol)
    }
    descriptor_by_id = {
        (item.get("workload_id"), item.get("candidate_id")): item
        for item in descriptor.get("descriptors", [])
    }
    predictions = []
    for key, (candidate, point) in expected_by_id.items():
        item = descriptor_by_id.get(key)
        if item is None:
            raise ValueError(f"describe-only artifact is missing candidate {key[1]}")
        sample = item.get("sample")
        prediction = None
        status = str(item.get("status", "unavailable"))
        missing_groups = []
        missing_primitives = []
        extrapolated = []
        if status == "resolved" and isinstance(sample, dict):
            if _timing_fields(sample) or _timing_fields(item.get("groups", [])):
                raise ValueError(f"descriptor for {key[1]} contains timing observations")
            if point_signature(sample) != point_signature(point):
                raise ValueError(f"descriptor candidate mapping mismatch: {key[1]}")
            try:
                prediction = __import__("stage_cost_model").predict(sample, model, explain=True)
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
                status = "prediction-unavailable"
                item = dict(item, reason=str(error))
            else:
                missing_groups = list(prediction.get("unmeasured_groups", []))
                missing_primitives = list(prediction.get("missing_primitives", []))
                extrapolated = list(prediction.get("extrapolated_primitives", []))
                fallback_value = prediction.get("kernel_ms")
                covered = bool(
                    fallback_value is not None
                    and not prediction.get("whole_plan_fallback_used")
                    and not missing_groups
                    and not missing_primitives
                    and not extrapolated
                )
                status = "covered" if covered else (
                    "service-gap" if missing_groups or missing_primitives or extrapolated
                    else "fallback-prior")
        model_value = prediction.get("kernel_ms") if isinstance(prediction, dict) else None
        try:
            model_value = float(model_value)
            if not math.isfinite(model_value) or model_value <= 0:
                model_value = None
        except (TypeError, ValueError):
            model_value = None
        # Keep the model's diagnostic value separate from the usable prior.
        # Extrapolation and legacy whole-plan fallback are evidence gaps, not
        # latency predictions that a cold-start policy may consume.
        prediction_usable = bool(status == "covered" and model_value is not None)
        prediction_source = (
            "stage-model" if prediction_usable else
            "stage-model-gap" if model_value is not None else None
        )
        predictions.append({
            "workload_id": key[0],
            "candidate_id": key[1],
            "predicted_ms": model_value if prediction_usable else None,
            "stage_model_predicted_ms": model_value,
            "fallback_predicted_ms": None,
            "fallback_score": candidate.get(
                "fallback_score",
                (candidate.get("structural") or {}).get("logical_traffic_score")),
            "prediction_usable": prediction_usable,
            "prediction_source": prediction_source,
            "status": status,
            "source": "frozen-stage-service-profile",
            "uses_whole_workload_observation": False,
            "missing_service_groups": missing_groups,
            "missing_primitives": missing_primitives,
            "extrapolated_primitives": extrapolated,
            "runtime_fingerprint": item.get("runtime_fingerprint"),
            "reason": item.get("reason"),
            "explanation": prediction,
        })
    document = {
        "schema": PREDICTIONS_SCHEMA,
        "status": "frozen-prior-predictions",
        "prediction_state": "frozen-before-full-plan-measurement",
        "experiment": protocol.get("experiment"),
        "protocol": protocol_source,
        "protocol_sha256": protocol["freeze"]["protocol_sha256"],
        "descriptors": descriptor_source,
        "stage_checkpoint": checkpoint_source,
        "profile": profile_source,
        "build": descriptor.get("build", {}),
        "model_source": {
            "stage_service": "frozen checkpoint records, role=train curves only",
            "capability_profile": "explicit --profile",
            "whole_workload_observations_read": False,
            "training_scope": "physical-stage-groups-only",
            "stage_service_schema": service.get("schema"),
            "stage_service_hash": sha256_bytes(canonical(service).encode()),
        },
        "prior_contract": {
            "published_before_full_plan_measurement": True,
            "uses_whole_workload_observation": False,
            "fallback_kind": "structural-order-only; never a latency estimate",
        },
        "prediction_count": len(predictions),
        "covered_count": sum(item["status"] == "covered" for item in predictions),
        "service_gap_count": sum(item["status"] != "covered" for item in predictions),
        "model_prediction_count": sum(item["stage_model_predicted_ms"] is not None
                                      for item in predictions),
        "stage_model_gap_count": sum(item["prediction_source"] == "stage-model-gap"
                                      for item in predictions),
        "fallback_count": sum(not item["prediction_usable"] for item in predictions),
        "prior_predictions": predictions,
        "rules_checked": {
            "no_whole_workload_observations": True,
            "descriptor_is_timing_free": descriptor.get("timing_free") is True,
            "protocol_hash": True,
            "stage_checkpoint_hash": True,
            "profile_hash": True,
            "predictions_frozen_before_measurement": True,
        },
    }
    document["freeze"] = {
        "algorithm": "sha256",
        "canonicalization": "json-sort-keys-separators-utf8; freeze field excluded",
        "prediction_sha256": _prediction_hash(document),
    }
    write_json(args.output, document)
    if getattr(args, "require_covered", False) and document["fallback_count"]:
        raise ValueError("declared covered-prior experiment has service gaps; predictions saved, do not measure")
    return document


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    domain, domain_source = read_json(args.workloads)
    if isinstance(domain, dict):
        domain = domain.get("workloads", domain.get("points", []))
    if not isinstance(domain, list) or not all(isinstance(item, dict) for item in domain):
        raise ValueError("workloads input must contain a list")
    checkpoint, checkpoint_source = read_json(args.stage_checkpoint)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != "cubutterfly-stage-calibration-v1":
        raise ValueError("stage checkpoint has an unsupported schema")
    selected = select_workloads(domain, args.workload_id)
    selected_ids = {workload_id(item) for item in selected}
    declared_candidates = candidate_source = None
    if getattr(args, "candidate_points", None) is not None:
        declared_candidates, candidate_source = explicit_candidate_sets(args.candidate_points, selected)
    audit = audit_stage_checkpoint(checkpoint, checkpoint_source)
    target = checkpoint.get("identity", {})
    target_uuid = args.target_uuid or _identity_value(target, "uuid")
    if not target_uuid:
        raise ValueError("stage checkpoint has no target UUID; pass --target-uuid")
    checkpoint_uuid = _identity_value(target, "uuid")
    if checkpoint_uuid and target_uuid != checkpoint_uuid:
        raise ValueError("target UUID differs from frozen stage checkpoint")
    groups = []
    points = []
    prior_predictions = []
    for workload in selected:
        identifier = workload_id(workload)
        candidates = (declared_candidates[identifier] if declared_candidates is not None
                      else candidate_set(workload))
        entries = []
        for candidate in candidates:
            point = candidate["point"]
            entries.append({
                "candidate_id": candidate["candidate_id"],
                "core": candidate["core"],
                "mapping": candidate["mapping"],
                "point": point,
                "legality": candidate["legality"],
            })
            points.append(point)
            prior_predictions.append({
                "workload_id": identifier,
                "candidate_id": candidate["candidate_id"],
                "predicted_ms": None,
                "stage_model_predicted_ms": None,
                "fallback_predicted_ms": None,
                "prediction_usable": False,
                "prediction_source": None,
                "fallback_score": candidate["fallback_score"],
                "source": "awaiting-describe-only-stage-profile",
                "uses_whole_workload_observation": False,
                "status": "pending-describe",
                "reason": (
                    "descriptor and frozen stage-service prediction are generated before any "
                    "whole-workload measurement; no timing-derived prior is substituted"
                ),
            })
        groups.append({
            "workload_id": identifier,
            "workload": workload,
            "core": "scalar",
            "candidate_ids": [entry["candidate_id"] for entry in entries],
            "candidates": entries,
        })
    audit = _audit_holdout_semantics(audit, checkpoint, groups)
    if audit["heldout_stage_semantic_overlap"] and not getattr(args, "composition_holdout", False):
        overlap = ", ".join(item["workload_id"] for item in audit["heldout_stage_semantic_overlap"])
        raise ValueError(f"held-out workloads overlap checkpoint.records.sample: {overlap}")
    output_dir = args.output_dir.resolve()
    protocol_path = output_dir / "heldout_protocol.json"
    points_path = output_dir / "heldout_points.json"
    measurements_path = output_dir / "heldout_measurements.json"
    output_path = output_dir / "heldout_validation.json"
    descriptors_path = output_dir / "heldout_descriptors.json"
    predictions_path = output_dir / "heldout_predictions.json"
    write_json(points_path, {"schema": POINTS_SCHEMA, "points": points})
    profile = args.profile.resolve()
    build_dir = args.build_dir.resolve()
    gpu_trials = int((checkpoint.get("protocol") or {}).get("trials", 3))
    document = {
        "schema": SCHEMA,
        "status": "frozen-awaiting-gpu-whole-workload-timing",
        "experiment": "E6",
        "metric_population": ("whole-workload-heldout" if audit["whole_workload_holdout_eligible"]
                               else "whole-workload-composition-holdout"),
        "target": {
            "uuid": target_uuid,
            "device": target.get("device"),
            "compute_capability": target.get("compute_capability"),
            "global_memory_bytes": _identity_value(target, "global_memory_bytes"),
            "compile_policy": target.get("compile_policy"),
        },
        "sources": {
            "workload_domain": domain_source,
            **({"candidate_points": candidate_source} if candidate_source else {}),
            "stage_checkpoint": checkpoint_source,
            "capability_profile": _path_ref(Path(args.profile)),
            "build_dir": {"path": str(Path(args.build_dir).resolve())},
            "generator": _source_record(Path(__file__)),
        },
        "training_inventory_audit": audit,
        "training_workloads": [],
        "validation_workloads": [],
        "stage_training_workloads": audit["stage_record_workloads_by_role"],
        "stage_training_workload_semantics": audit["stage_training_workloads"],
        "stage_validation_workloads": audit["stage_validation_workloads"],
        "stage_record_workloads_by_role": audit["stage_record_workloads_by_role"],
        "training_scope": "no-whole-workload-training; stage-service-only prior",
        "composition_holdout_declared": bool(getattr(args, "composition_holdout", False)),
        "heldout_workloads": groups,
        "workload_ids": sorted(selected_ids),
        "prior_predictions": prior_predictions,
        "prior_contract": {
            "published_before_gpu_measurement": True,
            "predictions_frozen": False,
            "prediction_artifact_required_before_measurement": True,
            "uses_whole_workload_observation": False,
            "prediction_phase": "describe-only then CPU stage-service prediction",
            "service_gap_count": None,
            "fallback_allowed": True,
            "fallback_kind": "structural-order-only; never a latency estimate",
        },
        "gpu_measurement": {
            "points_path": str(points_path.resolve()),
            "output_path": str(measurements_path.resolve()),
            "full_workload_point_count": len(points),
            "trials_per_point": gpu_trials,
            "expected_full_workload_trials": len(points) * gpu_trials,
            "expected_describe_only_calls": len(points),
            "requires_gpu_supplement": True,
            "reason": "E6 whole-workload observations do not exist in the frozen checkpoint.",
            "describe_command": [
                "env", f"CUDA_VISIBLE_DEVICES={target_uuid}", args.python,
                str(Path(__file__).resolve()), "--mode", "describe",
                "--protocol", str(protocol_path.resolve()), "--build-dir", str(build_dir),
                "--output", str(descriptors_path.resolve()), "--target-uuid", target_uuid,
            ],
            "predict_command": [
                "env", "CUDA_VISIBLE_DEVICES=", args.python,
                str(Path(__file__).resolve()), "--mode", "predict",
                "--protocol", str(protocol_path.resolve()), "--descriptors", str(descriptors_path.resolve()),
                "--profile", str(profile), "--stage-checkpoint", str(args.stage_checkpoint.resolve()),
                "--build-dir", str(build_dir),
                "--output", str(predictions_path.resolve()),
            ],
            "command": _gpu_command(
                python=args.python, checkpoint=args.stage_checkpoint,
                profile=profile, build_dir=build_dir, points=points_path,
                output=measurements_path, target_uuid=target_uuid,
                compile_policy=_identity_value(target, "compile_policy") or "research"),
            "evaluate_command": [
                "env", "CUDA_VISIBLE_DEVICES=", args.python,
                str(Path(__file__).resolve()), "--mode", "evaluate",
                "--protocol", str(protocol_path.resolve()), "--descriptors", str(descriptors_path.resolve()),
                "--predictions", str(predictions_path.resolve()), "--measurements", str(measurements_path.resolve()),
                "--profile", str(profile), "--build-dir", str(build_dir),
                "--output", str((output_dir / "heldout_validation.json").resolve()),
            ],
        },
        "rules": [
            ("Known independent stage semantics may overlap; public whole-plan timing is held out."
             if audit["heldout_stage_semantic_overlap"] else
             "Whole-workload IDs are disjoint and frozen before target-driven timing."),
            "The stage checkpoint is audited as physical-stage training only; no record is imported as a whole-workload training point.",
            "Prior predictions are published before reading heldout measurements and never fit a heldout plan time.",
            "Targeted stage extension, known-stage composition and structural cold start are separate populations.",
        ],
    }
    document["freeze"] = {
        "algorithm": "sha256",
        "canonicalization": "json-sort-keys-separators-utf8; freeze field excluded",
        "protocol_sha256": protocol_hash(document),
    }
    write_json(protocol_path, document)
    return document


def _measurement_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        for key in ("rows", "measurements", "records"):
            if isinstance(value.get(key), list):
                return value[key]
    if isinstance(value, list):
        return value
    raise ValueError("measurement input must contain rows")


def _observed_ms(row: dict[str, Any]) -> float | None:
    for key in ("public_plan_ms", "whole_workload_ms", "kernel_ms", "median_kernel_ms"):
        value = row.get(key)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed > 0:
            return parsed
    samples = row.get("samples")
    if isinstance(samples, list):
        values = []
        for sample in samples:
            try:
                parsed = float(sample.get("kernel_ms"))
            except (AttributeError, TypeError, ValueError):
                continue
            if math.isfinite(parsed) and parsed > 0:
                values.append(parsed)
        if values:
            return statistics.median(values)
    return None


def _load_predictions(path: Path, protocol: dict[str, Any], descriptor: dict[str, Any] | None = None,
                      descriptor_source: dict[str, str] | None = None
                      ) -> tuple[dict[str, Any], dict[str, str]]:
    predictions, source = read_json(path)
    if not isinstance(predictions, dict) or predictions.get("schema") != PREDICTIONS_SCHEMA:
        raise ValueError("unsupported frozen prediction artifact schema")
    expected_hash = (predictions.get("freeze") or {}).get("prediction_sha256")
    if expected_hash != _prediction_hash(predictions):
        raise ValueError("frozen prediction artifact hash mismatch")
    if predictions.get("protocol_sha256") != protocol.get("freeze", {}).get("protocol_sha256"):
        raise ValueError("frozen predictions are bound to a different protocol")
    if descriptor is not None:
        expected_descriptor = predictions.get("descriptors", {})
        if descriptor_source is not None:
            if expected_descriptor.get("path") != descriptor_source.get("path"):
                raise ValueError("frozen predictions use a different descriptor artifact")
            if expected_descriptor.get("sha256") != descriptor_source.get("sha256"):
                raise ValueError("frozen predictions use a changed descriptor artifact")
        elif expected_descriptor.get("path") and descriptor.get("protocol", {}).get("path"):
            # Older artifacts lacked their own source record.  Retain the
            # protocol-path check for those files but never mistake it for a
            # descriptor hash check.
            if expected_descriptor.get("path") == descriptor.get("protocol", {}).get("path"):
                raise ValueError("frozen predictions use a different descriptor artifact")
    expected = {(group["workload_id"], candidate["candidate_id"])
                for group, candidate, _ in _protocol_candidates(protocol)}
    actual_items = predictions.get("prior_predictions")
    if not isinstance(actual_items, list):
        raise ValueError("frozen prediction artifact has no prior_predictions list")
    if not all(isinstance(item, dict) for item in actual_items):
        raise ValueError("frozen prediction artifact contains a non-object prediction")
    actual_keys = [(item.get("workload_id"), item.get("candidate_id")) for item in actual_items]
    actual = set(actual_keys)
    if len(actual_keys) != len(actual):
        raise ValueError("frozen prediction artifact repeats a candidate")
    if actual != expected:
        raise ValueError("frozen prediction artifact candidate set differs from protocol")
    if any(item.get("uses_whole_workload_observation") is not False for item in actual_items):
        raise ValueError("prediction artifact does not prove whole-workload independence")
    return predictions, source


def _load_measurement_report(path: Path, protocol: dict[str, Any], descriptor: dict[str, Any],
                             *, strict: bool = True) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    measurements, source = read_json(path)
    if not isinstance(measurements, dict):
        raise ValueError("measurement artifact must be an object")
    if measurements.get("schema") != VALIDATION_SCHEMA:
        if strict:
            raise ValueError("measurement artifact is not validate_stage_service_profile output")
        rows = _measurement_rows(measurements)
        return measurements, {}, {"strict_schema": False, "source": source}
    if measurements.get("status") != "complete-diagnostic":
        raise ValueError("stage-plan validation report is not complete")
    checkpoint_source = protocol.get("sources", {}).get("stage_checkpoint", {})
    report_checkpoint = measurements.get("checkpoint")
    if report_checkpoint != checkpoint_source.get("path"):
        raise ValueError("measurement report checkpoint path differs from protocol")
    _source_matches(Path(report_checkpoint), checkpoint_source, "measurement checkpoint")
    expected_mode = protocol.get("target", {}).get("compile_policy")
    if expected_mode not in (None, "") and str(measurements.get("compile_mode")) != str(expected_mode):
        raise ValueError("measurement report compile mode differs from protocol")
    target = descriptor.get("target", {})
    hardware = measurements.get("hardware", {})
    if not isinstance(hardware, dict):
        raise ValueError("measurement report has no hardware identity")
    for field in ("device", "compute_capability", "global_memory_bytes"):
        if field not in hardware or not _identity_equal(hardware[field], target.get(field), field):
            raise ValueError(f"measurement report hardware differs from describe-only identity: {field}")
    report_uuid = _identity_value(hardware, "uuid") or _identity_value(measurements, "uuid")
    if report_uuid not in (None, "") and not _identity_equal(report_uuid, target.get("uuid"), "uuid"):
        raise ValueError("measurement report UUID differs from describe-only identity")
    descriptor_by_id = {(item["workload_id"], item["candidate_id"]): item
                        for item in descriptor.get("descriptors", [])}
    candidate_by_signature = {point_signature(point): (group["workload_id"], candidate["candidate_id"])
                              for group, candidate, point in _protocol_candidates(protocol)}
    rows = _measurement_rows(measurements)
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    checkpoint, _ = _protocol_stage_checkpoint(protocol)
    trial_protocol = checkpoint.get("protocol", {})
    expected_trials = int(trial_protocol.get("trials", 0))
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("measurement report contains a non-object row")
        point = row.get("point")
        identity = candidate_by_signature.get(point_signature(point)) if isinstance(point, dict) else None
        if identity is None:
            raise ValueError("measurement report contains a point outside the frozen candidate set")
        if row.get("candidate_id") not in (None, "", identity[1]):
            raise ValueError(f"measurement row candidate_id differs for {identity[1]}")
        if identity in indexed:
            raise ValueError(f"measurement report repeats candidate {identity[1]}")
        descriptor_item = descriptor_by_id.get(identity)
        if descriptor_item is None:
            raise ValueError(f"missing descriptor for candidate {identity[1]}")
        status = str(row.get("status", ""))
        if status == "measured":
            operator = descriptor_item["point"].get("operator", "fft")
            binary_name = "cuntt_bench" if operator == "ntt" else "cubutterfly_bench"
            binary = _descriptor_binaries(descriptor).get(binary_name, {})
            if not binary or row.get("binary_sha256") != binary.get("sha256"):
                raise ValueError(f"measurement row build hash mismatch for candidate {identity[1]}")
            samples = row.get("samples")
            if not isinstance(samples, list) or len(samples) != expected_trials:
                raise ValueError(f"measurement row trial count mismatch for candidate {identity[1]}")
            expected_runtime = descriptor_item.get("runtime_fingerprint")
            if descriptor_item.get("status") == "resolved" and expected_runtime in (None, ""):
                raise ValueError(f"resolved descriptor has no runtime fingerprint for candidate {identity[1]}")
            expected_mapping = point_signature(descriptor_item.get("sample") or descriptor_item["point"])
            for sample in samples:
                if not isinstance(sample, dict):
                    raise ValueError(f"measurement row contains a non-object trial for candidate {identity[1]}")
                if sample.get("correct") not in ("1", 1, True):
                    raise ValueError(f"measurement row has failed correctness for candidate {identity[1]}")
                if expected_runtime not in (None, "") and sample.get("runtime_fingerprint") != expected_runtime:
                    raise ValueError(f"measurement row runtime fingerprint mismatch for candidate {identity[1]}")
                sample_point = {**descriptor_item["point"], **sample}
                if point_signature(sample_point) != expected_mapping:
                    raise ValueError(f"measurement row resolved mapping mismatch for candidate {identity[1]}")
            command = row.get("command")
            command = command if isinstance(command, list) else str(command or "").split()
            for flag in ("--warmup", "--repeat", "--verify", "--verify-batches", "--csv"):
                if flag not in command:
                    raise ValueError(f"measurement row command lacks {flag} for candidate {identity[1]}")
            for flag, key in (("--warmup", "warmup"), ("--repeat", "repeat")):
                index = command.index(flag)
                if index + 1 >= len(command) or str(command[index + 1]) != str(trial_protocol.get(key)):
                    raise ValueError(f"measurement row {flag} differs from frozen protocol")
            if _observed_ms(row) is None:
                raise ValueError(f"measured row has no positive whole-workload observation for {identity[1]}")
        elif status not in {"failed", "unavailable", "incorrect", "missing"}:
            raise ValueError(f"unsupported measurement row status: {status}")
        indexed[identity] = {
            "observed_ms": _observed_ms(row) if status == "measured" else None,
            "status": status,
            "row": row,
        }
    expected = {(group["workload_id"], candidate["candidate_id"])
                for group, candidate, _ in _protocol_candidates(protocol)}
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        raise ValueError(f"measurement report is missing frozen candidates: {missing[:3]}")
    metadata = {
        "strict_schema": True,
        "source": source,
        "uuid": target.get("uuid"),
        "uuid_source": ("measurement-report" if report_uuid not in (None, "")
                        else "describe-only-preflight-only"),
        "uuid_verified": bool(report_uuid not in (None, "") and target.get("uuid")),
        "uuid_preflight_verified": bool(target.get("uuid")),
        "build_verified": True,
        "protocol_verified": True,
        "checkpoint_verified": True,
    }
    return measurements, indexed, metadata


def _rank_metrics(group: dict[str, Any], predictions: dict[str, dict[str, Any]],
                  observed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate_ids = list(group["candidate_ids"])
    usable = [cid for cid in candidate_ids if observed.get(cid, {}).get("observed_ms") is not None]
    measured_order = sorted(usable, key=lambda cid: (observed[cid]["observed_ms"], cid))
    predicted_ids = [cid for cid in candidate_ids
                     if predictions.get(cid, {}).get("predicted_ms") is not None
                     and predictions.get(cid, {}).get("prediction_usable",
                                                       predictions.get(cid, {}).get("predicted_ms") is not None)]
    predicted_order = sorted(predicted_ids,
                             key=lambda cid: (predictions[cid]["predicted_ms"], cid))
    best_observed = measured_order[0] if measured_order else None
    predicted_best = predicted_order[0] if predicted_order else None
    pairwise_correct = pairwise_total = 0
    errors = []
    for left_index, left in enumerate(predicted_order):
        for right in predicted_order[left_index + 1:]:
            if left not in observed or right not in observed:
                continue
            if observed[left].get("observed_ms") is None or observed[right].get("observed_ms") is None:
                continue
            pairwise_total += 1
            pairwise_correct += int((predictions[left]["predicted_ms"] <= predictions[right]["predicted_ms"])
                                    == (observed[left]["observed_ms"] <= observed[right]["observed_ms"]))
        if left in observed and observed[left].get("observed_ms"):
            errors.append(abs(predictions[left]["predicted_ms"] / observed[left]["observed_ms"] - 1.0))
    regret_ratio = None
    regret = None
    predicted_best_observed = (predicted_best is not None
                               and observed.get(predicted_best, {}).get("observed_ms") is not None)
    if predicted_best_observed and best_observed:
        regret_ratio = (observed[predicted_best]["observed_ms"]
                        / observed[best_observed]["observed_ms"])
        regret = regret_ratio - 1.0
    return {
        "workload_id": group["workload_id"],
        "candidate_count": len(candidate_ids),
        "measured_count": len(usable),
        "failed_or_missing_count": len(candidate_ids) - len(usable),
        "observed_order": measured_order,
        "predicted_order": predicted_order,
        "top1": (predicted_best == best_observed) if predicted_best_observed and best_observed else None,
        "top2": (best_observed in predicted_order[:2]) if best_observed and predicted_order else None,
        "pairwise_correct": pairwise_correct,
        "pairwise_count": pairwise_total,
        "regret_ratio": regret_ratio,
        "regret_ratio_definition": "observed latency of predicted best / observed latency of measured best",
        "relative_regret": regret,
        "relative_regret_definition": "regret_ratio - 1.0",
        "regret": regret,
        "relative_errors": errors,
        "service_gaps": [cid for cid in candidate_ids if cid not in predicted_ids],
        "observed_candidates": observed,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    protocol, protocol_source = _load_protocol(args.protocol, SCHEMA)
    expected_hash = protocol["freeze"]["protocol_sha256"]
    descriptor_path = getattr(args, "descriptors", None)
    prediction_path = getattr(args, "predictions", None)
    strict = descriptor_path is not None or prediction_path is not None
    descriptor = predictions_document = None
    measurement_identity = {"strict_schema": False, "uuid_verified": False}
    if strict and (descriptor_path is None or prediction_path is None):
        raise ValueError("evaluate requires both --descriptors and --predictions")
    if strict:
        descriptor, descriptor_source = _load_descriptor(descriptor_path, protocol)
        predictions_document, predictions_source = _load_predictions(
            prediction_path, protocol, descriptor, descriptor_source)
        _validate_runtime_inputs(protocol, descriptor,
                                 profile_path=getattr(args, "profile", None),
                                 build_dir=getattr(args, "build_dir", None),
                                 predictions=predictions_document)
        measurements, measurement_index, measurement_identity = _load_measurement_report(
            args.measurements, protocol, descriptor, strict=True)
        measurement_source = measurement_identity["source"]
        observed = measurement_index
    else:
        measurements, measurement_source = read_json(args.measurements)
        rows = _measurement_rows(measurements)
        observed = {}
    by_point = {}
    for group in protocol.get("heldout_workloads", []):
        for candidate in group.get("candidates", []):
            by_point[point_signature(candidate["point"])] = (group["workload_id"], candidate["candidate_id"])
    if not strict:
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("measurement input contains a non-object row")
            point = row.get("point")
            if not isinstance(point, dict):
                continue
            identity = by_point.get(point_signature(point))
            if identity is None:
                continue
            workload, candidate = identity
            value = _observed_ms(row)
            status = str(row.get("status", "measured"))
            if row.get("correct") is False or row.get("correct") in ("0", 0, "false"):
                value = None
                status = "incorrect"
            if status not in {"measured", "failed", "unavailable", "incorrect", "missing"}:
                raise ValueError(f"unsupported measurement row status: {status}")
            if (workload, candidate) in observed:
                raise ValueError(f"measurement report repeats candidate {candidate}")
            observed[workload, candidate] = {
                "observed_ms": value if status == "measured" else None,
                "status": status,
            }
    predictions = {}
    prediction_items = (predictions_document or {}).get("prior_predictions")
    if prediction_items is None:
        prediction_items = protocol.get("prior_predictions", [])
    for item in prediction_items:
        predictions[item["workload_id"], item["candidate_id"]] = item
    reports = []
    for group in protocol.get("heldout_workloads", []):
        group_predictions = {cid: predictions.get((group["workload_id"], cid), {})
                             for cid in group["candidate_ids"]}
        group_observed = {cid: observed.get((group["workload_id"], cid), {"observed_ms": None, "status": "missing"})
                          for cid in group["candidate_ids"]}
        reports.append(_rank_metrics(group, group_predictions, group_observed))
    top1_values = [item["top1"] for item in reports if item["top1"] is not None]
    top2_values = [item["top2"] for item in reports if item["top2"] is not None]
    pairwise_correct = sum(item["pairwise_correct"] for item in reports)
    pairwise_count = sum(item["pairwise_count"] for item in reports)
    regrets = [item["regret"] for item in reports if item["regret"] is not None]
    regret_ratios = [item["regret_ratio"] for item in reports if item["regret_ratio"] is not None]
    errors = [error for item in reports for error in item["relative_errors"]]
    gaps = sum(len(item["service_gaps"]) for item in reports)
    result = {
        "schema": RESULT_SCHEMA,
        "status": "evaluated" if not gaps and all(item["measured_count"] == item["candidate_count"] for item in reports)
        else "evaluated-with-gaps",
        "protocol": protocol_source,
        "measurement_source": measurement_source,
        "protocol_sha256": expected_hash,
        "metric_population": protocol.get("metric_population", "whole-workload-heldout"),
        "used_whole_workload_observations_for_predictions": False,
        "prediction_source": (predictions_source if strict else None),
        "measurement_identity": measurement_identity,
        "training_scope": protocol.get("training_scope"),
        "workloads": reports,
        "metrics": {
            "top1": statistics.mean(top1_values) if top1_values else None,
            "top1_count": sum(top1_values),
            "top1_eligible": len(top1_values),
            "top2": statistics.mean(top2_values) if top2_values else None,
            "top2_count": sum(top2_values),
            "top2_eligible": len(top2_values),
            "pairwise_correct": pairwise_correct,
            "pairwise_count": pairwise_count,
            "regret_median": statistics.median(regrets) if regrets else None,
            "regret_p90": _percentile(regrets, .90),
            "regret_ratio_median": statistics.median(regret_ratios) if regret_ratios else None,
            "regret_ratio_p90": _percentile(regret_ratios, .90),
            "regret_ratio_definition": "observed latency of predicted best / observed latency of measured best",
            "relative_regret_definition": "regret_ratio - 1.0",
            "relative_error_median": statistics.median(errors) if errors else None,
            "relative_error_p90": _percentile(errors, .90),
            "service_gap_count": gaps,
            "missing_or_failed_measurements": sum(item["failed_or_missing_count"] for item in reports),
        },
        "rules_checked": {
            "protocol_hash": True,
            "measurement_schema": measurement_identity.get("strict_schema", False),
            "measurement_uuid": measurement_identity.get("uuid_preflight_verified",
                                                           measurement_identity.get("uuid_verified", False)),
            "measurement_build": measurement_identity.get("build_verified", False),
            "measurement_protocol": measurement_identity.get("protocol_verified", False),
            "predictions_precede_measurements": True,
            "whole_workload_train_imported": False,
            "candidate_only_ranking_claimed": False,
        },
    }
    write_json(args.output, result)
    return result


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "describe", "predict", "evaluate"), default="prepare")
    parser.add_argument("--workloads", type=Path)
    parser.add_argument("--stage-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workload-id", action="append")
    parser.add_argument("--candidate-points", type=Path,
                        help="explicit finite E6 pool; defaults to the original four representatives")
    parser.add_argument("--composition-holdout", action="store_true",
                        help="allow calibrated stage semantics; label as composition holdout, never unseen workload")
    parser.add_argument("--require-covered", action="store_true",
                        help="predict: save the prior and fail before measurement if any service is missing")
    parser.add_argument("--target-uuid")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--profile", type=Path,
                        help="target capability profile; no platform default is assumed")
    parser.add_argument("--build-dir", type=Path,
                        help="target build directory; no platform default is assumed")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--descriptors", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        if (not args.workloads or not args.stage_checkpoint or not args.output_dir
                or not args.profile or not args.build_dir):
            parser.error("prepare requires --workloads, --stage-checkpoint, --output-dir, --profile and --build-dir")
    elif args.mode == "describe":
        if not args.protocol or not args.build_dir or not args.output:
            parser.error("describe requires --protocol, --build-dir and --output")
    elif args.mode == "predict":
        if (not args.protocol or not args.descriptors or not args.profile
                or not args.build_dir or not args.output):
            parser.error("predict requires --protocol, --descriptors, --profile, --build-dir and --output")
    elif (not args.protocol or not args.descriptors or not args.predictions
          or not args.measurements or not args.profile or not args.build_dir or not args.output):
        parser.error("evaluate requires --protocol, --descriptors, --predictions, --measurements, --profile, --build-dir and --output")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "prepare":
        result = freeze(args)
    elif args.mode == "describe":
        result = describe(args)
    elif args.mode == "predict":
        result = predict(args)
    else:
        result = evaluate(args)
    print(json.dumps({"status": result["status"], "schema": result["schema"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
