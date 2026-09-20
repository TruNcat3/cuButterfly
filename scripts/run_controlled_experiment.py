#!/usr/bin/env python3
"""Run explicit, paired cuButterfly research points.

The study file owns the semantic workload and every requested mapping.  This
runner does not select or save a configuration.  It only serializes the
requested mapping into the public benchmark command, records the mapping that
the benchmark resolved, and accepts a timing sample after all contracts pass.

``--mode prepare`` is deliberately CPU-only: it validates the study shape and
prints/stores commands without asking the driver about a device.  ``run`` is
hardware-specific and writes an atomic journal after every preflight and
timing attempt so an interrupted measurement can be resumed without mixing
targets or protocols.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Iterable

from calibration_space import command_for
from hardware_registry import canonical_semantics
from run_comprehensive_suite import parse_csv_record, require_exclusive_gpu, visible_gpu


SCHEMA = "cubutterfly-controlled-experiment-v1"
PROTOCOL_FIELDS = ("trials", "warmup", "repeat", "verify_batches")
SUPPORTED_OPERATORS = {
    "fft", "fwht", "subset-zeta", "superset-zeta", "xor-zeta", "structured-2x2", "ntt",
}
SUPPORTED_PRECISIONS = {"fp32", "fp64", "bf16", "uint32", "word32", "word64"}
SEMANTIC_FIELDS = {
    "operator", "precision", "placement", "logN", "batch", "modulus", "accumulation",
    "direction", "normalization", "element_stride", "batch_stride", "stage_matrices",
    "input_order", "output_order", "stage_matrix",
}

# These fields are emitted by a resolved plan but are resources/implementation
# details, rather than independent controls.  They may change when a control
# changes without turning a declared one-factor comparison into a two-factor
# comparison.
DERIVED_MAPPING_FIELDS = {
    "execution_group_mappings", "segment_mappings", "subgraph_mappings",
    "prefix_units_per_cta", "suffix_units_per_cta", "execution_stage_partition",
    "logical_subgraphs", "materialized_boundaries", "decomposition_count",
    "stages_per_decomposition", "execution_groups", "execution_groups_json",
    "workspace_bytes", "segment_cores", "group_cores", "segment_threads",
    "segment_ept", "group_threads", "group_ept", "runtime_fingerprint",
}
MAPPING_METADATA_FIELDS = {"schema_version", "kind"}

BENCHMARK_BY_OPERATOR = {
    "ntt": "cuntt_bench",
}
DEFAULT_BENCHMARK = "cubutterfly_bench"


class StudyError(ValueError):
    """A malformed controlled-study contract."""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity() -> dict[str, str]:
    """Hash code which contributes to command construction/validation."""
    root = Path(__file__).resolve().parent
    names = (
        "run_controlled_experiment.py", "calibration_space.py", "hardware_registry.py",
        "run_comprehensive_suite.py",
    )
    return {name: _sha256(root / name) for name in names}


def _atomic_write(path: Path, document: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(document, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _protocol(protocol: Any) -> dict[str, int]:
    if not isinstance(protocol, dict):
        raise StudyError("study.protocol must be an object")
    missing = [field for field in PROTOCOL_FIELDS if field not in protocol]
    if missing:
        raise StudyError("study.protocol is missing " + ", ".join(missing))
    unknown = set(protocol) - set(PROTOCOL_FIELDS)
    if unknown:
        raise StudyError("study.protocol has unsupported fields: " + ", ".join(sorted(unknown)))
    values = {field: protocol[field] for field in PROTOCOL_FIELDS}
    if not _is_int(values["trials"]) or values["trials"] < 1:
        raise StudyError("protocol.trials must be a positive integer")
    if not _is_int(values["warmup"]) or values["warmup"] < 0:
        raise StudyError("protocol.warmup must be a non-negative integer")
    if not _is_int(values["repeat"]) or values["repeat"] < 1:
        raise StudyError("protocol.repeat must be a positive integer")
    if not _is_int(values["verify_batches"]) or values["verify_batches"] < 0:
        raise StudyError("protocol.verify_batches must be a non-negative integer")
    return values


def _validate_workload(workload: Any, location: str) -> dict[str, Any]:
    if not isinstance(workload, dict):
        raise StudyError(f"{location} must be an object")
    required = ("operator", "precision", "logN", "batch")
    missing = [field for field in required if field not in workload]
    if missing:
        raise StudyError(f"{location} is missing " + ", ".join(missing))
    operator = workload["operator"]
    precision = workload["precision"]
    if operator not in SUPPORTED_OPERATORS:
        raise StudyError(f"{location}.operator is unsupported: {operator}")
    if precision not in SUPPORTED_PRECISIONS:
        raise StudyError(f"{location}.precision is unsupported: {precision}")
    if not _is_int(workload["logN"]) or not 1 <= workload["logN"] <= 30:
        raise StudyError(f"{location}.logN must be an integer in [1, 30]")
    if not _is_int(workload["batch"]) or workload["batch"] < 1:
        raise StudyError(f"{location}.batch must be a positive integer")
    if "mapping_json" in workload or "backend" in workload or "fft_core" in workload:
        raise StudyError(f"{location} must keep execution mapping under each variant.mapping")
    if operator == "ntt" and precision not in {"word32", "word64"}:
        raise StudyError(f"{location}: NTT precision must be word32 or word64")
    if operator != "ntt" and precision in {"word32", "word64"}:
        raise StudyError(f"{location}: word precision is only valid for NTT")
    if "stage_matrix" in workload:
        if not isinstance(workload["stage_matrix"], str):
            raise StudyError(f"{location}.stage_matrix must be a comma-separated string")
        if len(workload["stage_matrix"].split(",")) != 4:
            raise StudyError(f"{location}.stage_matrix requires four coefficients")
    for field in ("element_stride", "batch_stride"):
        if field in workload and (not _is_int(workload[field]) or workload[field] <= 0):
            raise StudyError(f"{location}.{field} must be a positive integer")
    return dict(workload)


def _validate_mapping(mapping: Any, workload: dict[str, Any], location: str) -> dict[str, Any]:
    if not isinstance(mapping, dict) or not mapping:
        raise StudyError(f"{location} must be a non-empty explicit object")
    semantic_overlap = set(mapping) & SEMANTIC_FIELDS
    if semantic_overlap:
        raise StudyError(
            f"{location} cannot override workload semantics: {', '.join(sorted(semantic_overlap))}")
    if "mapping_json" in mapping:
        raise StudyError(f"{location} must contain mapping fields, not nested mapping_json")
    if "schema_version" in mapping and mapping["schema_version"] != 1:
        raise StudyError(f"{location}.schema_version must be 1")
    expected_kind = "ntt" if workload["operator"] == "ntt" else "butterfly"
    if "kind" in mapping and mapping["kind"] != expected_kind:
        raise StudyError(f"{location}.kind must be {expected_kind}")
    for field in ("stage_partition", "factor_partition"):
        if field in mapping:
            value = mapping[field]
            if not isinstance(value, list) or not value or any(not _is_int(x) or x <= 0 for x in value):
                raise StudyError(f"{location}.{field} must be a non-empty positive integer list")
    for field in ("batch_tile_count", "factor_slices", "factor_ept", "factor_columns", "data_tiles_per_cta", "prefetch_depth"):
        if field in mapping and (not _is_int(mapping[field]) or mapping[field] < 0):
            raise StudyError(f"{location}.{field} must be a non-negative integer")
    for field in ("stage_overlap", "factor_overlap"):
        if field in mapping and not isinstance(mapping[field], (bool, int)):
            raise StudyError(f"{location}.{field} must be boolean-like")
    return dict(mapping)


def _axis_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StudyError("comparison.changed_axes entries must be non-empty strings")
    return value.strip()


def _mapping_diff(left: dict[str, Any], right: dict[str, Any], *, include_derived: bool = True) -> set[str]:
    keys = set(left) | set(right)
    if not include_derived:
        keys -= DERIVED_MAPPING_FIELDS
    keys -= MAPPING_METADATA_FIELDS
    return {key for key in keys if left.get(key) != right.get(key)}


def _validate_study(study: Any) -> tuple[dict[str, Any], dict[str, int]]:
    if not isinstance(study, dict) or study.get("schema") != SCHEMA:
        raise StudyError(f"study schema must be {SCHEMA}")
    protocol = _protocol(study.get("protocol"))
    cases = study.get("cases")
    if not isinstance(cases, list) or not cases:
        raise StudyError("study.cases must be a non-empty list")
    seen_cases: set[str] = set()
    normalized_cases = []
    for case_index, case in enumerate(cases):
        location = f"cases[{case_index}]"
        if not isinstance(case, dict):
            raise StudyError(f"{location} must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise StudyError(f"{location}.id must be a non-empty string")
        if case_id in seen_cases:
            raise StudyError(f"case ids must be unique: {case_id}")
        seen_cases.add(case_id)
        experiment = case.get("experiment")
        if not isinstance(experiment, str) or not experiment:
            raise StudyError(f"{location}.experiment must be a non-empty string")
        workload = _validate_workload(case.get("workload"), f"{location}.workload")
        variants = case.get("variants")
        if not isinstance(variants, list) or not variants:
            raise StudyError(f"{location}.variants must be a non-empty list")
        seen_variants: set[str] = set()
        normalized_variants = []
        for variant_index, variant in enumerate(variants):
            variant_location = f"{location}.variants[{variant_index}]"
            if not isinstance(variant, dict):
                raise StudyError(f"{variant_location} must be an object")
            variant_id = variant.get("id")
            if not isinstance(variant_id, str) or not variant_id:
                raise StudyError(f"{variant_location}.id must be a non-empty string")
            if variant_id in seen_variants:
                raise StudyError(f"variant ids must be unique in {case_id}: {variant_id}")
            seen_variants.add(variant_id)
            mapping = _validate_mapping(variant.get("mapping"), workload, f"{variant_location}.mapping")
            normalized_variants.append({"id": variant_id, "mapping": mapping})
        variant_ids = {variant["id"] for variant in normalized_variants}
        comparisons = case.get("comparisons")
        if not isinstance(comparisons, list) or not comparisons:
            raise StudyError(f"{location}.comparisons must be a non-empty list")
        normalized_comparisons = []
        for comparison_index, comparison in enumerate(comparisons):
            comparison_location = f"{location}.comparisons[{comparison_index}]"
            if not isinstance(comparison, dict):
                raise StudyError(f"{comparison_location} must be an object")
            baseline = comparison.get("baseline")
            treatment = comparison.get("treatment")
            if baseline not in variant_ids or treatment not in variant_ids:
                raise StudyError(f"{comparison_location} references an unknown variant")
            if baseline == treatment:
                raise StudyError(f"{comparison_location} baseline and treatment must differ")
            changed_axes = comparison.get("changed_axes")
            if not isinstance(changed_axes, list) or not changed_axes:
                raise StudyError(f"{comparison_location}.changed_axes must be a non-empty list")
            axes = [_axis_name(axis) for axis in changed_axes]
            if len(set(axes)) != len(axes):
                raise StudyError(f"{comparison_location}.changed_axes must be unique")
            baseline_mapping = next(v["mapping"] for v in normalized_variants if v["id"] == baseline)
            treatment_mapping = next(v["mapping"] for v in normalized_variants if v["id"] == treatment)
            requested_diff = _mapping_diff(baseline_mapping, treatment_mapping, include_derived=False)
            if not requested_diff:
                raise StudyError(f"{comparison_location} treatment does not change the requested mapping")
            undeclared = requested_diff - set(axes)
            if undeclared:
                raise StudyError(
                    f"{comparison_location} is not single-factor; undeclared mapping axes: "
                    + ", ".join(sorted(undeclared)))
            normalized_comparisons.append({
                "baseline": baseline, "treatment": treatment, "changed_axes": axes,
            })
        normalized_cases.append({
            "id": case_id, "experiment": experiment, "workload": workload,
            "variants": normalized_variants, "comparisons": normalized_comparisons,
        })
    normalized = dict(study)
    normalized["protocol"] = protocol
    normalized["cases"] = normalized_cases
    return normalized, protocol


def _binary_for(build_dir: Path, workload: dict[str, Any]) -> Path:
    name = BENCHMARK_BY_OPERATOR.get(workload["operator"], DEFAULT_BENCHMARK)
    return (build_dir / name).resolve()


def _mapping_payload(mapping: dict[str, Any], workload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(mapping)
    payload.setdefault("schema_version", 1)
    payload.setdefault("kind", "ntt" if workload["operator"] == "ntt" else "butterfly")
    return payload


def _base_command(binary: Path, workload: dict[str, Any]) -> list[str]:
    # command_for is the repository's semantic lowering.  Mapping axes stay in
    # --mapping-json because that is the only lossless representation for
    # arrays, nested physical mappings, and booleans.
    return [str(value) for value in command_for(binary, workload)]


def _commands(binary: Path, workload: dict[str, Any], mapping: dict[str, Any], protocol: dict[str, int]) -> dict[str, list[str]]:
    base = _base_command(binary, workload)
    mapping_text = _json(_mapping_payload(mapping, workload))
    explicit = ["--mapping-json", mapping_text]
    preflight = base + explicit + [
        "--warmup", "0", "--repeat", "1", "--verify", "--verify-batches", "0", "--csv",
    ]
    timed = base + explicit + [
        "--warmup", str(protocol["warmup"]), "--repeat", str(protocol["repeat"]), "--csv",
    ]
    return {"preflight": preflight, "timed": timed}


def _command_text(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(value)) for value in command)


def _prepared_identity(study: dict[str, Any], study_path: Path, build_dir: Path) -> dict[str, Any]:
    binaries = {}
    for case in study["cases"]:
        binary = _binary_for(build_dir, case["workload"])
        binaries[binary.name] = {"path": str(binary), "sha256": None}
    # Keep the optional descriptor binary in the prepared identity by path so
    # a run that records its SHA can still be resumed against the same build.
    probe = (build_dir / "cubutterfly_stage_microbench").resolve()
    if probe.is_file():
        binaries[probe.name] = {"path": str(probe), "sha256": None}
    return {
        "schema": SCHEMA,
        "study_path": str(study_path.resolve()),
        "build_dir": str(build_dir.resolve()),
        "study_sha256": _sha256(study_path),
        "protocol": study["protocol"],
        "source_sha256": _source_identity(),
        "binaries": binaries,
        "gpu_queried": False,
        "executable_claim": False,
    }


def prepare_document(study: dict[str, Any], study_path: Path, build_dir: Path) -> dict[str, Any]:
    """Validate and lower a study without touching the CUDA driver."""
    normalized, protocol = _validate_study(study)
    cases = []
    for case in normalized["cases"]:
        binary = _binary_for(build_dir, case["workload"])
        variants = []
        for variant in case["variants"]:
            commands = _commands(binary, case["workload"], variant["mapping"], protocol)
            variants.append({
                "id": variant["id"], "mapping": variant["mapping"],
                "command": commands["timed"], "preflight_command": commands["preflight"],
                "timed_command": commands["timed"], "status": "pending",
            })
        cases.append({
            "id": case["id"], "experiment": case["experiment"], "workload": case["workload"],
            "binary": str(binary), "variants": variants, "comparisons": case["comparisons"],
            "status": "pending", "trial_orders": [],
        })
    return {
        "schema": SCHEMA, "status": "prepared", "identity": _prepared_identity(normalized, study_path, build_dir),
        "study": normalized, "protocol": protocol, "cases": cases,
        "limitations": [
            "prepare performs structural and single-factor checks only; it does not query a GPU or claim executability",
            "runtime legality, resolved mappings, full-batch correctness, and resources require run mode",
        ],
    }


def _parse_csv_row(output: str) -> dict[str, str]:
    """Parse device-identity CSV, which has no kernel_ms column."""
    lines = [line for line in output.splitlines() if line.strip()]
    for index, line in enumerate(lines[:-1]):
        if "," not in line or "device" not in line.lower():
            continue
        try:
            rows = list(csv.DictReader(io.StringIO(f"{line}\n{lines[index + 1]}\n")))
        except csv.Error:
            continue
        if len(rows) == 1:
            return dict(rows[0])
    raise RuntimeError(f"benchmark did not produce a CSV identity row:\n{output}")


def _parse_json_field(sample: dict[str, Any], field: str) -> Any:
    value = sample.get(field)
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _normal_stage_matrices(workload: dict[str, Any]) -> str | None:
    if "stage_matrices" in workload:
        return str(workload["stage_matrices"])
    if "stage_matrix" not in workload:
        return None
    values = workload["stage_matrix"].split(",")
    return ":".join(format(float(value), ".17g") for value in values)


def _expected_semantics(workload: dict[str, Any]) -> dict[str, str]:
    point = dict(workload)
    matrices = _normal_stage_matrices(workload)
    if matrices is not None:
        point["stage_matrices"] = matrices
    return canonical_semantics(point)


def _sample_semantics(sample: dict[str, Any]) -> dict[str, str]:
    return canonical_semantics(sample)


def _sample_correct(sample: dict[str, Any]) -> bool:
    return str(sample.get("correct", "")).strip().lower() in {"1", "true"}


def _finite_kernel(sample: dict[str, Any]) -> bool:
    try:
        value = float(sample["kernel_ms"])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0


def _resource_description(sample: dict[str, Any], resolved_mapping: dict[str, Any]) -> dict[str, Any]:
    resources: dict[str, Any] = {"resolved_mapping": resolved_mapping}
    for field in (
        "workspace_bytes", "execution_group_count", "segment_cores", "group_cores",
        "segment_threads", "segment_ept", "group_threads", "group_ept",
        "verified_batches",
    ):
        if field in sample:
            resources[field] = sample[field]
    groups = _parse_json_field(sample, "execution_groups_json")
    if groups is not None:
        resources["execution_groups"] = groups
        if "execution_group_count" not in resources:
            resources["execution_group_count"] = len(groups) if isinstance(groups, list) else None
    return resources


def _validate_sample(
    sample: dict[str, Any], workload: dict[str, Any], expected: dict[str, str], device: dict[str, Any],
    *, require_correctness: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(sample, dict):
        raise RuntimeError("benchmark CSV sample is not an object")
    observed = _sample_semantics(sample)
    mismatches = {
        field: {"expected": value, "observed": observed.get(field)}
        for field, value in expected.items() if observed.get(field) != value
    }
    if mismatches:
        raise RuntimeError("semantic contract mismatch: " + _json(mismatches))
    for field in ("device", "compute_capability"):
        expected_hardware = device.get("name") if field == "device" else device.get(field)
        if sample.get(field) and expected_hardware and str(sample[field]) != str(expected_hardware):
            raise RuntimeError(f"benchmark hardware mismatch for {field}: {sample[field]} != {expected_hardware}")
    mapping = _parse_json_field(sample, "mapping_json")
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError("benchmark did not report a resolved mapping_json")
    if not sample.get("runtime_fingerprint"):
        raise RuntimeError("benchmark did not report a runtime_fingerprint")
    if require_correctness:
        if not _sample_correct(sample):
            raise RuntimeError("full-batch correctness verification failed")
        try:
            verified = int(str(sample["verified_batches"]).strip())
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("full-batch verification count is missing") from None
        if verified != int(workload["batch"]):
            raise RuntimeError(
                f"full-batch verification incomplete: verified {verified}, expected {workload['batch']}")
    resources = _resource_description(sample, mapping)
    return mapping, resources


def _query_hardware(binary: Path, requested_uuid: str) -> dict[str, Any]:
    if not requested_uuid:
        raise RuntimeError("--gpu-uuid is required in run mode for explicit hardware binding")
    gpu = visible_gpu()
    queried = subprocess.run([
        "nvidia-smi", "-i", gpu,
        "--query-gpu=name,compute_cap,memory.total,uuid,driver_version",
        "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True)
    rows = list(csv.reader(io.StringIO(queried.stdout), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) < 5:
        raise RuntimeError("nvidia-smi did not return one target GPU identity")
    name, compute_capability, memory_mib, actual_uuid, driver = [value.strip() for value in rows[0][:5]]
    if actual_uuid.lower() != requested_uuid.strip().lower():
        raise RuntimeError(f"requested GPU UUID {requested_uuid} does not match target {actual_uuid}")
    identity = subprocess.run([str(binary), "--device-identity"], check=True, capture_output=True, text=True)
    device_row = _parse_csv_row(identity.stdout)
    if device_row.get("device") and device_row["device"].strip('"') != name:
        raise RuntimeError("benchmark and nvidia-smi disagree about the GPU name")
    if device_row.get("compute_capability") and device_row["compute_capability"] != compute_capability:
        raise RuntimeError("benchmark and nvidia-smi disagree about compute capability")
    try:
        cuda_memory_bytes = int(device_row["global_memory_bytes"]) if device_row.get("global_memory_bytes") else None
    except ValueError:
        cuda_memory_bytes = None
    try:
        nvml_memory_total_mib: Any = float(memory_mib)
        if nvml_memory_total_mib.is_integer():
            nvml_memory_total_mib = int(nvml_memory_total_mib)
    except ValueError:
        nvml_memory_total_mib = memory_mib
    return {
        "name": name, "compute_capability": compute_capability,
        # CUDA and NVML report related but not contractually identical
        # capacities. Preserve both observations without inventing an
        # equivalence threshold for the target identity.
        "global_memory_bytes": cuda_memory_bytes,
        "cuda_global_memory_bytes": cuda_memory_bytes,
        "memory_total_mib": nvml_memory_total_mib,
        "nvml_memory_total_mib": nvml_memory_total_mib,
        "uuid": actual_uuid, "driver": driver,
        "visible_gpu": gpu,
    }


def _runtime_identity(
    study: dict[str, Any], study_path: Path, build_dir: Path, device: dict[str, Any], gpu_uuid: str,
) -> dict[str, Any]:
    binaries: dict[str, Any] = {}
    for case in study["cases"]:
        binary = _binary_for(build_dir, case["workload"])
        if not binary.is_file():
            raise FileNotFoundError(f"benchmark binary not found: {binary}")
        binaries[binary.name] = {"path": str(binary), "sha256": _sha256(binary)}
    probe = (build_dir / "cubutterfly_stage_microbench").resolve()
    if probe.is_file():
        binaries[probe.name] = {"path": str(probe), "sha256": _sha256(probe)}
    return {
        "schema": SCHEMA, "study_path": str(study_path.resolve()), "build_dir": str(build_dir.resolve()),
        "study_sha256": _sha256(study_path), "protocol": study["protocol"],
        "source_sha256": _source_identity(), "binaries": binaries,
        "compile_mode": os.environ.get("CUBUTTERFLY_COMPILE_MODE", "auto"),
        "gpu_uuid": gpu_uuid.lower(), "device": device,
        "gpu_queried": True, "executable_claim": True,
    }


def _prepared_compatible(existing: dict[str, Any], prepared: dict[str, Any]) -> None:
    if existing.get("schema") != SCHEMA:
        raise RuntimeError("checkpoint schema does not match controlled study")
    old = existing.get("identity", {})
    new = prepared.get("identity", {})
    for field in ("study_path", "build_dir", "study_sha256", "protocol", "source_sha256", "binaries"):
        if field == "binaries":
            old_paths = {name: value.get("path") for name, value in old.get(field, {}).items()}
            new_paths = {name: value.get("path") for name, value in new.get(field, {}).items()}
            if old_paths != new_paths:
                raise RuntimeError("checkpoint identity changed: binaries")
            continue
        if old.get(field) != new.get(field):
            raise RuntimeError(f"checkpoint identity changed: {field}")
    if existing.get("study") != prepared.get("study") or existing.get("cases") is None:
        raise RuntimeError("checkpoint study or generated commands changed")
    def plan_signature(value: dict[str, Any]) -> list[dict[str, Any]]:
        signature = []
        for case in value.get("cases", []):
            signature.append({
                "id": case.get("id"), "experiment": case.get("experiment"),
                "workload": case.get("workload"), "binary": case.get("binary"),
                "comparisons": [
                    {field: comparison.get(field) for field in ("baseline", "treatment", "changed_axes")}
                    for comparison in case.get("comparisons", [])
                ],
                "variants": [
                    {field: variant.get(field) for field in (
                        "id", "mapping", "command", "preflight_command", "timed_command")}
                    for variant in case.get("variants", [])
                ],
            })
        return signature
    if plan_signature(existing) != plan_signature(prepared):
        raise RuntimeError("checkpoint study or generated commands changed")


def _invoke(command: list[str]) -> tuple[dict[str, str] | None, str | None]:
    require_exclusive_gpu()
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    finally:
        require_exclusive_gpu()
    if completed.returncode:
        return None, (
            f"command failed with exit code {completed.returncode}: {_command_text(command)}\n"
            f"stdout:\n{getattr(completed, 'stdout', '')}\nstderr:\n{getattr(completed, 'stderr', '')}"
        )
    try:
        return parse_csv_record(completed.stdout), None
    except Exception as error:  # keep malformed output in the journal, fail closed
        return None, f"could not parse benchmark CSV: {error}"


def _probe_resources(
    build_dir: Path, workload: dict[str, Any], mapping: dict[str, Any],
) -> dict[str, Any] | None:
    probe = (build_dir / "cubutterfly_stage_microbench").resolve()
    if not probe.is_file():
        return None
    point = dict(workload)
    point["mapping_json"] = _json(mapping)
    command = [str(probe), "--describe-only", "--point-json", _json(point)]
    try:
        require_exclusive_gpu()
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        finally:
            require_exclusive_gpu()
    except Exception as error:
        return {"status": "unavailable", "command": command, "reason": str(error)}
    try:
        value = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        value = {"status": "unavailable", "reason": getattr(completed, "stderr", "") or "invalid probe JSON"}
    value.setdefault("command", command)
    if completed.returncode and value.get("status") == "resolved":
        value["status"] = "unavailable"
    return value


def _variant_by_id(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {variant["id"]: variant for variant in case["variants"]}


def _ensure_variant_fields(variant: dict[str, Any]) -> None:
    variant.setdefault("status", "pending")
    variant.setdefault("trials", [])


def _preflight_variant(
    case: dict[str, Any], variant: dict[str, Any], expected: dict[str, str], device: dict[str, Any],
    build_dir: Path,
) -> None:
    _ensure_variant_fields(variant)
    prior = variant.get("preflight")
    if isinstance(prior, dict) and prior.get("status") in {"accepted", "rejected", "error"}:
        return
    sample, error = _invoke(list(variant["preflight_command"]))
    if error:
        variant["preflight"] = {"status": "error", "performance_accepted": False, "reason": error}
        variant["status"] = "error"
        return
    try:
        mapping, resources = _validate_sample(
            sample, case["workload"], expected, device, require_correctness=True)
        probe = _probe_resources(build_dir, case["workload"], mapping)
        if probe is not None:
            resources["stage_probe"] = probe
        variant["preflight"] = {
            "status": "accepted", "performance_accepted": False,
            "sample": sample, "resolved_mapping": mapping,
            "runtime_fingerprint": sample.get("runtime_fingerprint"), "resources": resources,
        }
        variant["status"] = "verified"
    except Exception as validation_error:
        variant["preflight"] = {
            "status": "rejected", "performance_accepted": False,
            "sample": sample, "reason": str(validation_error),
        }
        variant["status"] = "rejected"


def _timed_command(case: dict[str, Any], variant: dict[str, Any], protocol: dict[str, int]) -> list[str]:
    preflight = variant.get("preflight", {})
    mapping = preflight.get("resolved_mapping")
    if not isinstance(mapping, dict):
        return list(variant["timed_command"])
    binary = Path(case["binary"])
    commands = _commands(binary, case["workload"], mapping, protocol)
    return commands["timed"]


def _trial_result(variant: dict[str, Any], trial: int) -> dict[str, Any] | None:
    for result in variant.get("trials", []):
        if int(result.get("trial", -1)) == trial:
            return result
    return None


def _measure_trial(
    case: dict[str, Any], variant: dict[str, Any], trial: int, expected: dict[str, str], device: dict[str, Any],
    protocol: dict[str, int], order: list[str],
) -> None:
    if variant.get("preflight", {}).get("status") != "accepted":
        return
    if _trial_result(variant, trial) is not None:
        return
    sample, error = _invoke(_timed_command(case, variant, protocol))
    result: dict[str, Any] = {"trial": trial, "order": list(order), "performance_accepted": False}
    if error:
        result.update(status="error", reason=error)
        variant.setdefault("trials", []).append(result)
        return
    try:
        mapping, resources = _validate_sample(
            sample, case["workload"], expected, device, require_correctness=False)
        preflight = variant["preflight"]
        if _json(mapping) != _json(preflight["resolved_mapping"]):
            raise RuntimeError("resolved mapping changed after correctness preflight")
        if sample.get("runtime_fingerprint") != preflight.get("runtime_fingerprint"):
            raise RuntimeError("runtime fingerprint changed after correctness preflight")
        if sample.get("correct") not in (None, "", "-1", -1) and not _sample_correct(sample):
            raise RuntimeError("timed sample reports incorrect output")
        if not _finite_kernel(sample):
            raise RuntimeError("timed sample has no positive finite kernel_ms")
        result.update(
            status="measured", performance_accepted=True, sample=sample,
            kernel_ms=float(sample["kernel_ms"]), resolved_mapping=mapping, resources=resources,
        )
    except Exception as validation_error:
        result.update(status="rejected", reason=str(validation_error), sample=sample)
    variant.setdefault("trials", []).append(result)


def _control_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    control = {}
    for key, value in mapping.items():
        if key in DERIVED_MAPPING_FIELDS | MAPPING_METADATA_FIELDS:
            continue
        if key in {"boundaries", "boundary_mappings"}:
            # A stage partition derives the number of boundary records.  Keep
            # a uniform policy comparable while retaining heterogeneous policy
            # lists so an undeclared boundary change cannot be hidden.
            if isinstance(value, list) and value:
                first = value[0]
                if all(entry == first for entry in value[1:]):
                    value = ("uniform-boundary-policy", _json(first))
        control[key] = value
    return control


def _comparison_result(case: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    variants = _variant_by_id(case)
    baseline = variants[comparison["baseline"]]
    treatment = variants[comparison["treatment"]]
    # Recompute from the immutable comparison contract.  Journals may contain
    # a previous blocked/measured result; copying those derived fields would
    # leave stale reasons or alias flags after a successful re-analysis.
    result = {field: comparison[field] for field in ("baseline", "treatment", "changed_axes")}
    result.update(status="blocked", performance_accepted=False)
    bp = baseline.get("preflight", {})
    tp = treatment.get("preflight", {})
    if bp.get("status") != "accepted" or tp.get("status") != "accepted":
        result["reason"] = "correctness or semantic preflight was not accepted"
        return result
    baseline_mapping = bp.get("resolved_mapping")
    treatment_mapping = tp.get("resolved_mapping")
    if not isinstance(baseline_mapping, dict) or not isinstance(treatment_mapping, dict):
        result["reason"] = "resolved mapping evidence is missing"
        return result
    control_diff = _mapping_diff(_control_mapping(baseline_mapping), _control_mapping(treatment_mapping))
    if not control_diff:
        result["reason"] = "runtime mapping alias: treatment resolved to the baseline mapping"
        result["resolved_mapping_alias"] = True
        return result
    declared = set(comparison["changed_axes"])
    undeclared = control_diff - declared
    if undeclared:
        result["reason"] = "runtime mapping changed undeclared axes: " + ", ".join(sorted(undeclared))
        result["actual_changed_axes"] = sorted(control_diff)
        return result
    baseline_trials = {int(row["trial"]): row for row in baseline.get("trials", [])}
    treatment_trials = {int(row["trial"]): row for row in treatment.get("trials", [])}
    trial_ids = list(range(1, 1 + max((int(row) for row in baseline_trials), default=-1)))
    # The protocol's zero-based trial id is used by this runner.  Keep this
    # fallback for journals written by an early interrupted invocation.
    trial_ids = sorted(set(baseline_trials) | set(treatment_trials))
    pairs = []
    for trial in trial_ids:
        base = baseline_trials.get(trial)
        treat = treatment_trials.get(trial)
        if not base or not treat or not base.get("performance_accepted") or not treat.get("performance_accepted"):
            result["reason"] = f"trial {trial} is not performance-accepted"
            return result
        base_ms = float(base["kernel_ms"])
        treatment_ms = float(treat["kernel_ms"])
        pairs.append({"trial": trial, "baseline_kernel_ms": base_ms, "treatment_kernel_ms": treatment_ms,
                      "treatment_over_baseline": base_ms / treatment_ms})
    if not pairs:
        result["reason"] = "no paired timing trials"
        return result
    result.update(
        status="measured", performance_accepted=True,
        actual_changed_axes=sorted(control_diff),
        baseline_resolved_mapping=baseline_mapping, treatment_resolved_mapping=treatment_mapping,
        pairs=pairs,
        baseline_median_kernel_ms=statistics.median(row["baseline_kernel_ms"] for row in pairs),
        treatment_median_kernel_ms=statistics.median(row["treatment_kernel_ms"] for row in pairs),
    )
    result["median_treatment_over_baseline"] = (
        result["baseline_median_kernel_ms"] / result["treatment_median_kernel_ms"])
    return result


def _evaluate_case(case: dict[str, Any]) -> None:
    comparisons = []
    for comparison in case["comparisons"]:
        comparisons.append(_comparison_result(case, comparison))
    case["comparisons"] = comparisons
    valid_variants = all(
        variant.get("preflight", {}).get("status") == "accepted"
        and all(row.get("performance_accepted") for row in variant.get("trials", []))
        for variant in case["variants"]
        if variant.get("preflight", {}).get("status") == "accepted"
    )
    case["status"] = "complete" if valid_variants and all(
        comparison.get("status") == "measured" for comparison in comparisons
    ) else "complete-with-gaps"


def run_document(
    document: dict[str, Any], study: dict[str, Any], study_path: Path, build_dir: Path,
    output: Path, gpu_uuid: str, *, resume: bool,
) -> dict[str, Any]:
    normalized, protocol = _validate_study(study)
    prepared = prepare_document(normalized, study_path, build_dir)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        _prepared_compatible(existing, prepared)
        if existing.get("status") not in {"prepared", "running", "interrupted", "complete-with-gaps", "complete"}:
            raise RuntimeError("unsupported checkpoint status")
        if existing.get("status") != "prepared" and not resume:
            raise RuntimeError("controlled journal exists; use --resume")
    else:
        existing = prepared
    # Hardware and binary identities are collected before the first benchmark,
    # and are part of the immutable resume key.
    first_binary = _binary_for(build_dir, normalized["cases"][0]["workload"])
    device = _query_hardware(first_binary, gpu_uuid)
    identity = _runtime_identity(normalized, study_path, build_dir, device, gpu_uuid)
    if existing.get("status") not in {"prepared"} and existing.get("identity") != identity:
        raise RuntimeError("checkpoint target/build/hardware/protocol/source identity changed")
    if existing.get("status") == "prepared":
        document = existing
        document["identity"] = identity
    else:
        document = existing
    document["status"] = "running"
    document.setdefault("protocol", protocol)
    _atomic_write(output, document)
    try:
        for case in document["cases"]:
            expected = _expected_semantics(case["workload"])
            for variant in case["variants"]:
                _preflight_variant(case, variant, expected, device, build_dir)
                _atomic_write(output, document)
            variant_ids = [variant["id"] for variant in case["variants"]]
            existing_orders = {int(row["trial"]): row.get("order", []) for row in case.get("trial_orders", [])}
            for trial in range(protocol["trials"]):
                order = variant_ids if trial % 2 == 0 else list(reversed(variant_ids))
                if trial not in existing_orders:
                    case.setdefault("trial_orders", []).append({"trial": trial, "order": order})
                    existing_orders[trial] = order
                for variant_id in order:
                    variant = _variant_by_id(case)[variant_id]
                    _measure_trial(case, variant, trial, expected, device, protocol, order)
                    _atomic_write(output, document)
            _evaluate_case(case)
            _atomic_write(output, document)
        document["status"] = "complete" if all(case.get("status") == "complete" for case in document["cases"]) else "complete-with-gaps"
        document["completed"] = document["status"] == "complete"
        _atomic_write(output, document)
        return document
    except BaseException as error:
        document["status"] = "interrupted"
        document["last_error"] = str(error)
        _atomic_write(output, document)
        raise


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StudyError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise StudyError(f"JSON root must be an object: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True, help="controlled-study JSON")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="atomic JSON journal")
    parser.add_argument("--mode", choices=("prepare", "run"), default="prepare")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu-uuid", help="explicit target UUID; required in run mode")
    args = parser.parse_args(argv)
    study_path = args.study.expanduser().resolve()
    build_dir = args.build_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output == study_path:
        raise StudyError("--output must not overwrite --study")
    study = _load_json(study_path)
    if args.mode == "prepare":
        document = prepare_document(study, study_path, build_dir)
        if output.exists():
            if not args.resume:
                raise StudyError("prepared journal exists; use --resume to regenerate it")
            existing = _load_json(output)
            if existing.get("status") != "prepared":
                raise StudyError("refusing to overwrite a non-prepared journal in prepare mode")
            _prepared_compatible(existing, document)
        _atomic_write(output, document)
        print(json.dumps({"status": "prepared", "output": str(output), "gpu_queried": False}, sort_keys=True))
        return 0
    normalized, _ = _validate_study(study)
    document = run_document(document={}, study=normalized, study_path=study_path, build_dir=build_dir,
                            output=output, gpu_uuid=args.gpu_uuid or "", resume=args.resume)
    print(json.dumps({"status": document["status"], "output": str(output), "completed": document.get("completed", False)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (StudyError, RuntimeError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
