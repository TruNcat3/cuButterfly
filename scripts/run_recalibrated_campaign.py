#!/usr/bin/env python3
"""Run the prepared recalibration campaign as a resumable serial queue.

The preparation script deliberately writes shell-shaped command strings so the
plan stays readable.  This runner accepts only the small command grammar used
by that local manifest: environment assignments, argv tokens, and stdout
redirection.  It never invokes a shell and it records a durable state entry
after every command.  A dry run performs no calibration, GPU, or subprocess
work; it is useful for checking the full queue on a machine without CUDA.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
PHASES = ("prerequisites", "correctness", "large_memory", "profile_gate", "comparison")
SUCCESS_STATUSES = {"complete", "ready-with-measured-mappings"}
ALLOWED_ASSIGNMENTS = {"CUDA_VISIBLE_DEVICES", "CUBUTTERFLY_COMPILE_MODE"}
SHELL_OPERATORS = {";", "&&", "||", "|", "||", "&", "<", "<<", ">>", "2>", "1>"}
ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=(.*)$")
NON_PASS_CLASSIFICATIONS = {"unsupported", "missing-external-reference", "memory-limit"}


try:
    # Keep the existing repository GPU guard as the single source of truth.
    from run_comprehensive_suite import require_exclusive_gpu
except ImportError:  # pragma: no cover - only relevant to unusual import paths.
    try:
        from scripts.run_comprehensive_suite import require_exclusive_gpu
    except ImportError:
        # Tests import this file by path, where the scripts directory is not
        # automatically on sys.path.  Load the existing helper by file path.
        import importlib.util

        _suite_spec = importlib.util.spec_from_file_location(
            "cubutterfly_run_comprehensive_suite", ROOT / "scripts" / "run_comprehensive_suite.py"
        )
        if _suite_spec is None or _suite_spec.loader is None:  # pragma: no cover
            raise
        _suite_module = importlib.util.module_from_spec(_suite_spec)
        _suite_spec.loader.exec_module(_suite_module)
        require_exclusive_gpu = _suite_module.require_exclusive_gpu


try:
    # The isolated runner owns the process-tree safety primitives.  The queue
    # uses them for monitoring only; it never signals a foreign PID.
    from run_isolated_gpu import descendant, gpu_pids, stop_owned
except ImportError:  # pragma: no cover - exercised when imported by file path.
    try:
        from scripts.run_isolated_gpu import descendant, gpu_pids, stop_owned
    except ImportError:
        import importlib.util

        _isolated_spec = importlib.util.spec_from_file_location(
            "cubutterfly_run_isolated_gpu", ROOT / "scripts" / "run_isolated_gpu.py"
        )
        if _isolated_spec is None or _isolated_spec.loader is None:  # pragma: no cover
            raise
        _isolated_module = importlib.util.module_from_spec(_isolated_spec)
        _isolated_spec.loader.exec_module(_isolated_module)
        descendant = _isolated_module.descendant
        gpu_pids = _isolated_module.gpu_pids
        stop_owned = _isolated_module.stop_owned


class CampaignError(RuntimeError):
    """A plan, calibration, identity, or queue-state error."""


def require_campaign_active(manifest_path: Path) -> None:
    marker = manifest_path.resolve().parent / "cancellation.json"
    if marker.exists():
        document = _load_json(marker)
        raise CampaignError("campaign cancelled: {} ({})".format(
            document.get("reason", "execution disabled"), marker))


def _now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(document: Dict[str, Any]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _resolve_repo_path(value: str, base: Path = ROOT) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open() as stream:
            document = json.load(stream)
    except (OSError, ValueError) as exc:
        raise CampaignError("could not read JSON {}: {}".format(path, exc)) from exc
    if not isinstance(document, dict):
        raise CampaignError("JSON document is not an object: {}".format(path))
    return document


def load_manifest(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise CampaignError("campaign manifest does not exist: {}".format(path))
    document = _load_json(path)
    expected = document.get("manifest_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise CampaignError("campaign manifest has no valid manifest_sha256: {}".format(path))
    body = dict(document)
    body.pop("manifest_sha256", None)
    actual = _sha256_bytes(_canonical(body))
    if actual != expected:
        raise CampaignError(
            "campaign manifest hash mismatch: expected {}, got {}".format(expected, actual)
        )
    if document.get("schema") not in (None, "cubutterfly-recalibrated-campaign-v1"):
        raise CampaignError("unsupported campaign manifest schema: {}".format(document.get("schema")))
    for phase in PHASES:
        commands = document.get("phases", {}).get(phase, [])
        if not isinstance(commands, list):
            raise CampaignError("manifest phase is not a list: {}".format(phase))
        seen = set()
        for item in commands:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not isinstance(item.get("run"), str):
                raise CampaignError("invalid {} command entry".format(phase))
            if item["id"] in seen:
                raise CampaignError("duplicate command id in {}: {}".format(phase, item["id"]))
            seen.add(item["id"])
    return document


def validate_plan_inputs(manifest: Dict[str, Any], manifest_path: Path) -> None:
    """Validate generated input artifacts; timing rows are never imported."""
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        raise CampaignError("manifest files section is not an object")
    for name, record in files.items():
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
            raise CampaignError("invalid input fingerprint for {}".format(name))
        path = _resolve_repo_path(str(name), manifest_path.parent)
        if not path.is_file():
            raise CampaignError("manifest input is missing: {}".format(path))
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise CampaignError("manifest input hash mismatch: {}".format(path))

    freeze = manifest.get("source_freeze")
    if not isinstance(freeze, dict):
        return
    inputs = freeze.get("inputs", [])
    if not isinstance(inputs, list):
        raise CampaignError("manifest source_freeze inputs are not a list")
    digest = hashlib.sha256()
    for record in inputs:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not isinstance(record.get("sha256"), str):
            raise CampaignError("invalid source_freeze input record")
        path = _resolve_repo_path(record["path"])
        if not path.is_file():
            raise CampaignError("source input is missing: {}".format(path))
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise CampaignError("source input hash mismatch: {}".format(path))
        digest.update(record["path"].encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    expected_source = freeze.get("source_sha256")
    if expected_source is not None and digest.hexdigest() != expected_source:
        raise CampaignError("source_freeze aggregate hash mismatch")


def _artifact_records(manifest: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    builds = manifest.get("builds", {})
    if not isinstance(builds, dict):
        raise CampaignError("manifest builds section is not an object")
    for build_name, build in builds.items():
        if not isinstance(build, dict):
            continue
        artifacts = build.get("artifacts", {})
        if not isinstance(artifacts, dict):
            continue
        for artifact_name, record in artifacts.items():
            if isinstance(record, dict) and isinstance(record.get("path"), str):
                yield "{}.{}".format(build_name, artifact_name), record


def fingerprint_builds(manifest: Dict[str, Any], *, allow_unresolved: bool = False) -> Dict[str, str]:
    """Return all required artifact hashes, optionally tolerating prepared gaps."""
    fingerprints: Dict[str, str] = {}
    for name, record in _artifact_records(manifest):
        if not record.get("required", True):
            continue
        path = _resolve_repo_path(record["path"])
        expected = record.get("sha256")
        if not path.is_file():
            if allow_unresolved and expected is None:
                continue
            raise CampaignError("required build artifact is missing: {} ({})".format(path, name))
        actual = sha256_file(path)
        if expected is not None and actual != expected:
            raise CampaignError(
                "build artifact hash mismatch for {}: expected {}, got {}".format(name, expected, actual)
            )
        fingerprints[name] = actual
    return fingerprints


def validate_resolved_builds(manifest: Dict[str, Any], resolved: Dict[str, str]) -> None:
    current = fingerprint_builds(manifest, allow_unresolved=False)
    if set(current) != set(resolved):
        missing = sorted(set(resolved) - set(current))
        extra = sorted(set(current) - set(resolved))
        raise CampaignError("resolved build artifact set changed (missing={}, extra={})".format(missing, extra))
    for name, expected in resolved.items():
        if current[name] != expected:
            raise CampaignError("resolved build hash changed for {}".format(name))


def _substitute_gpu(command: str, gpu_uuid: str) -> str:
    command = command.replace("${GPU_UUID}", gpu_uuid).replace("$GPU_UUID", gpu_uuid)
    if "${" in command or "$(" in command or "`" in command:
        raise CampaignError("unsupported shell expansion in manifest command")
    return command


def parse_command(command: str, gpu_uuid: str, log_root: Path) -> Tuple[List[str], Dict[str, str], Optional[Path]]:
    """Parse the local manifest grammar without invoking a shell."""
    tokens = shlex.split(_substitute_gpu(command, gpu_uuid), posix=True)
    if not tokens:
        raise CampaignError("empty manifest command")
    assignments: Dict[str, str] = {}
    while tokens:
        match = ASSIGNMENT.match(tokens[0])
        if not match:
            break
        key, value = tokens.pop(0).split("=", 1)
        if key not in ALLOWED_ASSIGNMENTS:
            raise CampaignError("manifest command sets unapproved environment variable: {}".format(key))
        assignments[key] = value
    argv: List[str] = []
    redirect: Optional[Path] = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in SHELL_OPERATORS:
            raise CampaignError("unsupported shell operator in manifest command: {}".format(token))
        if token == ">":
            if redirect is not None or index + 1 >= len(tokens):
                raise CampaignError("invalid stdout redirection in manifest command")
            redirect = _resolve_repo_path(tokens[index + 1])
            try:
                redirect.relative_to(ROOT)
            except ValueError as exc:
                raise CampaignError("manifest redirection escapes repository: {}".format(redirect)) from exc
            index += 2
            continue
        if token == "2>&1":
            # The generated anchors use `> file 2>&1`; stderr is already joined
            # by subprocess.run, so this token is safe to consume.
            index += 1
            continue
        if token in ("2>", "1>", "&>"):
            raise CampaignError("unsupported stderr redirection in manifest command")
        argv.append(token)
        index += 1
    if not argv:
        raise CampaignError("manifest command has no executable")
    if Path(argv[0]).name in ("python3", "python"):
        argv[0] = sys.executable
    return argv, assignments, redirect


def is_cpu_summary(step: Dict[str, Any], phase: str) -> bool:
    if phase == "prerequisites" or bool(step.get("cpu_only")):
        return True
    command_id = str(step.get("id", ""))
    return command_id.startswith("summarize_") or "summarize_" in Path(str(step.get("run", ""))).name


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:100] or "command"


def _atomic_write(path: Path, document: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    status = Path("/proc/{}/status".format(pid))
    if status.is_file():
        for line in status.read_text(errors="replace").splitlines():
            if line.startswith("State:"):
                return "zombie" not in line.lower()
    return True


def calibration_is_successful(path: Path, accepted_statuses: Sequence[str]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    if not path.is_file():
        return False, None
    try:
        document = _load_json(path)
    except CampaignError:
        return False, None
    status = document.get("status")
    calibration_status = document.get("calibration_status")
    successful = status in accepted_statuses or calibration_status in accepted_statuses
    return bool(successful), document


def _registry_from_calibration(document: Dict[str, Any]) -> Tuple[Path, str]:
    value = document.get("registry_path")
    if not isinstance(value, str) or not value:
        raise CampaignError("calibration_manifest has no registry_path")
    path = Path(value)
    if not path.is_absolute():
        raise CampaignError("calibration registry_path must be absolute: {}".format(value))
    path = path.resolve()
    if not path.is_file():
        raise CampaignError("calibration registry_path is missing: {}".format(path))
    return path, sha256_file(path)


def validate_calibration_binding(manifest: Dict[str, Any], calibration: Dict[str, Any],
                                 gpu_uuid: str) -> None:
    """Bind confirmed measurements to this campaign, not merely a success label."""
    contract = manifest.get("measurement_contract")
    if not contract:
        return  # Historical manifests retain their recorded protocol.
    if gpu_uuid != contract["gpu_uuid"]:
        raise CampaignError("campaign target GPU UUID differs from requested GPU")
    if os.environ.get("LD_PRELOAD"):
        raise CampaignError("reviewed campaign requires LD_PRELOAD unset for reproducible library loading")
    if contract.get("execution_environment", {}) != execution_environment():
        raise CampaignError("execution environment changed since campaign preparation")
    identity = calibration.get("measurement_identity", {})
    if identity.get("compile_mode") != contract["compile_mode"]:
        raise CampaignError("calibration compile mode differs from campaign")
    rules = os.environ.get("CUBUTTERFLY_PLATFORM_TUNING")
    rules_hash = sha256_file(Path(rules)) if rules else None
    if identity.get("platform_tuning_sha256") != rules_hash:
        raise CampaignError("calibration platform tuning rules differ from campaign")
    for key, value in contract["hardware"].items():
        if str(calibration.get(key)) != str(value):
            raise CampaignError("calibration hardware differs from campaign: " + key)
    artifacts = manifest["builds"]["current"]["artifacts"]
    library = artifacts["current_library"]
    if identity.get("library_sha256") != sha256_file(_resolve_repo_path(library["path"])):
        raise CampaignError("calibration shared library differs from campaign")
    for artifact_name in ("current_butterfly", "current_ntt"):
        path = _resolve_repo_path(artifacts[artifact_name]["path"])
        if identity.get("binary_sha256", {}).get(path.name) != sha256_file(path):
            raise CampaignError("calibration benchmark differs from campaign: " + path.name)
    if calibration.get("selector_replay", {}).get("status") != "passed":
        raise CampaignError("calibration selector replay has not passed")


def validate_runtime_hardware(manifest: Dict[str, Any], gpu_uuid: str) -> Dict[str, Any]:
    contract = manifest.get("measurement_contract")
    if not contract:
        return {}
    _ensure_gpu_exclusive(gpu_uuid)
    path = _resolve_repo_path(manifest["builds"]["current"]["artifacts"]["current_butterfly"]["path"])
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid,
                       CUBUTTERFLY_COMPILE_MODE=contract["compile_mode"])
    completed = subprocess.run([str(path), "--device-identity"], env=environment,
                               capture_output=True, text=True, check=True)
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    if len(rows) != 1:
        raise CampaignError("device identity did not return one hardware record")
    for key, value in contract["hardware"].items():
        if str(rows[0].get(key)) != str(value):
            raise CampaignError("runtime hardware differs from campaign: " + key)
    return dict(rows[0], uuid=gpu_uuid)


def execution_environment() -> Dict[str, str]:
    excluded = {"CUBUTTERFLY_COMPILE_MODE", "CUBUTTERFLY_REGISTRY", "CUBUTTERFLY_GPU_UUID"}
    return {key: value for key, value in os.environ.items()
            if (key.startswith("CUBUTTERFLY_") and key not in excluded)
            or key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "CUDA_MODULE_LOADING", "CUDA_LAUNCH_BLOCKING")}


def quarantine_retry_outputs(step: Dict[str, Any], state: Dict[str, Any],
                             state_path: Path, phase: str, previous: Dict[str, Any]) -> None:
    """Preserve a failed attempt, but never feed its partial timings to resume."""
    root = state_path.parent.resolve()
    destination = root / "quarantine" / (str(time.time_ns()) + "_" + _safe_name(step["id"]))
    moved = []
    for value in step.get("outputs", []):
        path = _resolve_repo_path(value)
        if root not in path.parents:
            raise CampaignError("retry output must be strictly inside the campaign: " + str(path))
        if path.exists():
            target = destination / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)
            moved.append({"source": str(path), "preserved": str(target)})
    state.setdefault("attempt_history", []).append({"phase": phase, "record": dict(previous),
                                                   "quarantined_outputs": moved})
    _atomic_write(state_path, state)


def _validate_state_registry(state: Dict[str, Any]) -> Path:
    rules_path = os.environ.get("CUBUTTERFLY_PLATFORM_TUNING")
    if state.get("platform_tuning_sha256") != (sha256_file(Path(rules_path)) if rules_path else None):
        raise CampaignError("platform tuning rules changed during campaign")
    calibration = state.get("calibration", {})
    path_value = calibration.get("registry_path")
    expected = calibration.get("registry_sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise CampaignError("queue state has no calibrated registry fingerprint")
    path = Path(path_value)
    if not path.is_absolute():
        raise CampaignError("queue registry_path is not absolute: {}".format(path))
    path = path.resolve()
    if not path.is_file():
        raise CampaignError("queue registry_path is missing: {}".format(path))
    actual = sha256_file(path)
    if actual != expected:
        raise CampaignError("calibration registry hash changed: {}".format(path))
    return path


def wait_for_calibration(
    calibration_pid: Optional[int],
    manifest_path: Path,
    accepted_statuses: Sequence[str],
    poll_seconds: float,
    timeout_seconds: Optional[float],
) -> Dict[str, Any]:
    started = time.monotonic()
    while calibration_pid is not None and _process_alive(calibration_pid):
        if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
            raise CampaignError("calibration process did not finish before timeout: {}".format(calibration_pid))
        time.sleep(max(0.0, poll_seconds))
    successful, document = calibration_is_successful(manifest_path, accepted_statuses)
    if not successful or document is None:
        raise CampaignError(
            "calibration_manifest is missing or not successful: {} (accepted statuses: {})".format(
                manifest_path, ",".join(accepted_statuses)
            )
        )
    return document


def _command_fingerprint(command: str, gpu_uuid: str) -> str:
    return _sha256_bytes(_substitute_gpu(command, gpu_uuid).encode("utf-8"))


def _command_steps(manifest: Dict[str, Any]) -> Iterable[Tuple[str, int, Dict[str, Any]]]:
    for phase in PHASES:
        for index, step in enumerate(manifest.get("phases", {}).get(phase, [])):
            yield phase, index, step


def dry_run(manifest: Dict[str, Any], manifest_path: Path, gpu_uuid: str) -> int:
    count = 0
    print("campaign_manifest={}".format(manifest_path.resolve()))
    print("gpu_uuid={} (dry-run; no GPU query or subprocess)".format(gpu_uuid or "<unset>"))
    for phase, index, step in _command_steps(manifest):
        argv, assignments, redirect = parse_command(step["run"], gpu_uuid or "GPU-dry-run", ROOT)
        kind = "cpu" if is_cpu_summary(step, phase) else "gpu"
        prefix = " ".join("{}={}".format(key, value) for key, value in assignments.items())
        rendered = " ".join(shlex.quote(x) for x in argv)
        if prefix:
            rendered = prefix + " " + rendered
        suffix = " > {}".format(redirect) if redirect is not None else ""
        print("{:02d}. {}[{}] {} {}{}".format(count + 1, phase, index, kind, rendered, suffix))
        count += 1
    print("TOTAL_COMMANDS={}".format(count))
    return 0


def _phase_step_records(state: Dict[str, Any], phase: str) -> List[Dict[str, Any]]:
    return state.setdefault("phases", {}).setdefault(phase, {}).setdefault("steps", [])


def _existing_success(
    state: Dict[str, Any], phase: str, index: int, step: Dict[str, Any], gpu_uuid: str
) -> Optional[Dict[str, Any]]:
    records = _phase_step_records(state, phase)
    if index >= len(records):
        return None
    record = records[index]
    if record.get("id") != step.get("id") or record.get("command_sha256") != _command_fingerprint(step["run"], gpu_uuid):
        raise CampaignError("resume command identity changed: {}".format(step.get("id")))
    if record.get("gate_passed") and record.get("returncode") == 0:
        return record
    return None


def _classification(step: Dict[str, Any]) -> Optional[str]:
    for key in ("expected_status", "comparison_reference", "status_policy"):
        value = step.get(key)
        if not isinstance(value, str):
            continue
        value = value.strip().lower().replace("_", "-")
        if value in NON_PASS_CLASSIFICATIONS:
            return value
    return None


def _record_failed(record: Dict[str, Any]) -> bool:
    if "gate_passed" in record:
        return not bool(record["gate_passed"])
    return record.get("status") not in ("passed", "completed-unverified")


def _ensure_gpu_exclusive(gpu_uuid: str) -> None:
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    try:
        require_exclusive_gpu()
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def _foreign_gpu_pids(gpu_uuid: str, owner_pid: int) -> List[int]:
    """Return live GPU PIDs outside the command's newly-created process tree."""
    pids = gpu_pids(gpu_uuid)
    return sorted({
        pid for pid in pids
        if Path("/proc/{}".format(pid)).exists() and not descendant(pid, owner_pid)
    })


def _run_gpu_process(
    argv: Sequence[str],
    environment: Dict[str, str],
    log_path: Path,
    gpu_uuid: str,
    poll_seconds: float,
    timeout_seconds: Optional[float],
) -> Dict[str, Any]:
    """Run one GPU command and quarantine it if a foreign GPU PID appears."""
    result: Dict[str, Any] = {}
    with log_path.open("w") as stream:
        child = subprocess.Popen(
            list(argv),
            cwd=str(ROOT),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        conflict: List[int] = []
        monitor_error: Optional[str] = None
        timed_out = False
        started = time.monotonic()
        try:
            while child.poll() is None:
                if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
                    timed_out = True
                    stop_owned(child)
                    break
                try:
                    conflict = _foreign_gpu_pids(gpu_uuid, child.pid)
                except Exception as exc:  # noqa: BLE001 - preserve a safe stop.
                    monitor_error = "GPU process monitor failed: {}".format(exc)
                    stop_owned(child)
                    break
                if conflict:
                    stop_owned(child)
                    break
                time.sleep(max(0.0, poll_seconds))
            if monitor_error is None and not conflict:
                try:
                    conflict = _foreign_gpu_pids(gpu_uuid, child.pid)
                except Exception as exc:  # noqa: BLE001 - preserve a safe stop.
                    monitor_error = "GPU process monitor failed at exit: {}".format(exc)
        except BaseException:
            stop_owned(child)
            raise
        if child.poll() is None:
            stop_owned(child)
        result["returncode"] = child.returncode
        if conflict:
            result.update({
                "status": "gpu-conflict",
                "passed": False,
                "gate_passed": False,
                "contaminated_by_foreign_pids": sorted(set(conflict)),
            })
        elif timed_out:
            result.update({
                "status": "timeout",
                "passed": False,
                "gate_passed": False,
                "error": "GPU command exceeded timeout",
            })
        elif monitor_error is not None:
            result.update({
                "status": "gpu-monitor-error",
                "passed": False,
                "gate_passed": False,
                "error": monitor_error,
            })
    return result


def _run_step(
    phase: str,
    index: int,
    step: Dict[str, Any],
    state: Dict[str, Any],
    state_path: Path,
    gpu_uuid: str,
    step_timeout: Optional[float],
    gpu_poll_seconds: float,
) -> Dict[str, Any]:
    command = step["run"]
    require_campaign_active(Path(state["manifest"]))
    strict = state.get("strict_measurement_contract", False)
    if strict:
        manifest_path = Path(state["manifest"])
        manifest = load_manifest(manifest_path)
        validate_plan_inputs(manifest, manifest_path)
        validate_resolved_builds(manifest, state["resolved_build_hashes"])
        if manifest["measurement_contract"].get("execution_environment", {}) != execution_environment():
            raise CampaignError("execution environment changed during campaign")
    cpu_only = is_cpu_summary(step, phase)
    argv, assignments, declared_log = parse_command(command, gpu_uuid, ROOT)
    registry_path = _validate_state_registry(state)
    registry_hash = state["calibration"]["registry_sha256"]
    log_path = declared_log or state_path.parent / "queue_logs" / "{}_{:03d}_{}.log".format(
        _safe_name(phase), index, _safe_name(step["id"])
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "id": step["id"],
        "command_sha256": _command_fingerprint(command, gpu_uuid),
        "argv": argv,
        "cpu_only": cpu_only,
        "classification": _classification(step),
        "registry_path": str(registry_path),
        "registry_sha256": registry_hash,
        "log": str(log_path),
        "started_at": _now(),
        "status": "running",
    }
    records = _phase_step_records(state, phase)
    while len(records) <= index:
        records.append({})
    records[index] = record
    _atomic_write(state_path, state)
    if not cpu_only:
        try:
            _ensure_gpu_exclusive(gpu_uuid)
        except Exception as exc:  # noqa: BLE001 - persist and continue independent steps.
            record.update({"status": "gpu-not-exclusive", "passed": False, "gate_passed": False, "error": str(exc), "finished_at": _now()})
            _atomic_write(state_path, state)
            return record
    environment = os.environ.copy()
    environment.update(assignments)
    environment["CUBUTTERFLY_REGISTRY"] = str(registry_path)
    if strict:
        environment["CUBUTTERFLY_COMPILE_MODE"] = state["compile_mode"]
    if not cpu_only:
        environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    try:
        if cpu_only:
            with log_path.open("w") as stream:
                completed = subprocess.run(
                    argv,
                    cwd=str(ROOT),
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=step_timeout,
                )
            record["returncode"] = completed.returncode
        else:
            record.update(_run_gpu_process(argv, environment, log_path, gpu_uuid, gpu_poll_seconds, step_timeout))
    except subprocess.TimeoutExpired as exc:
        record.update({"returncode": None, "status": "timeout", "error": str(exc)})
    except OSError as exc:
        record.update({"returncode": None, "status": "launch-error", "error": str(exc)})
    if "status" not in record or record["status"] == "running":
        if record.get("returncode") == 0:
            if record.get("classification"):
                record["status"] = record["classification"]
                record["passed"] = False
            else:
                record["status"] = "passed"
                record["passed"] = True
            record["gate_passed"] = True
        else:
            record["status"] = "failed"
            record["passed"] = False
            record["gate_passed"] = False
    if not cpu_only and record.get("returncode") is not None and record.get("status") not in ("gpu-conflict", "gpu-monitor-error"):
        try:
            _ensure_gpu_exclusive(gpu_uuid)
        except Exception as exc:  # noqa: BLE001 - a post-run conflict invalidates this row.
            record.update({"status": "gpu-conflict", "passed": False, "gate_passed": False, "error": str(exc)})
    record.setdefault("passed", record.get("status") == "passed")
    record.setdefault(
        "gate_passed",
        record.get("returncode") == 0
        and (record.get("status") == "passed" or record.get("classification") in NON_PASS_CLASSIFICATIONS),
    )
    try:
        _validate_state_registry(state)
        if strict:
            validate_plan_inputs(manifest, manifest_path)
            validate_resolved_builds(manifest, state["resolved_build_hashes"])
    except CampaignError as exc:
        record.update({"status": "identity-changed" if strict else "registry-changed", "passed": False, "gate_passed": False, "error": str(exc)})
    record["finished_at"] = _now()
    _atomic_write(state_path, state)
    return record


def run_campaign(
    manifest_path: Path,
    gpu_uuid: str,
    calibration_manifest: Path,
    calibration_pid: Optional[int] = None,
    state_path: Optional[Path] = None,
    resume: bool = False,
    poll_seconds: float = 15.0,
    calibration_timeout: Optional[float] = None,
    step_timeout: Optional[float] = None,
    gpu_poll_seconds: float = 1.0,
    accepted_calibration_statuses: Sequence[str] = ("complete", "ready-with-measured-mappings"),
) -> int:
    manifest_path = manifest_path.resolve()
    require_campaign_active(manifest_path)
    manifest = load_manifest(manifest_path)
    validate_plan_inputs(manifest, manifest_path)
    if not gpu_uuid:
        raise CampaignError("--gpu-uuid is required for a non-dry run")
    calibration_manifest = calibration_manifest.resolve()
    state_path = (state_path or manifest_path.parent / "campaign_queue_state.json").resolve()
    if state_path.exists() and not resume:
        raise CampaignError("queue state exists; pass --resume to reuse it: {}".format(state_path))
    if not resume:
        for relative in ("correctness", "profile", "comparison", "queue_logs"):
            if (manifest_path.parent / relative).exists():
                raise CampaignError("campaign output directory exists; use --resume or a fresh manifest: {}".format(relative))
    accepted = tuple(status.strip() for status in accepted_calibration_statuses if status.strip())
    if not accepted:
        raise CampaignError("at least one calibration success status is required")
    manifest_digest = manifest["manifest_sha256"]
    if resume:
        state = _load_json(state_path)
        if state.get("manifest_sha256") != manifest_digest:
            raise CampaignError("resume manifest hash does not match queue state")
        if state.get("gpu_uuid") != gpu_uuid:
            raise CampaignError("resume GPU UUID does not match queue state")
        resolved = state.get("resolved_build_hashes")
        if isinstance(resolved, dict):
            validate_resolved_builds(manifest, resolved)
        elif state.get("status") not in ("waiting-for-calibration", "calibration-not-ready"):
            raise CampaignError("resume queue has no resolved build hashes")
        saved_calibration = state.get("calibration", {})
        if saved_calibration.get("path") != str(calibration_manifest):
            raise CampaignError("resume calibration manifest path changed")
        if saved_calibration.get("sha256") and calibration_manifest.is_file() and saved_calibration.get("sha256") != sha256_file(calibration_manifest):
            raise CampaignError("calibration manifest changed since queue start")
        if saved_calibration.get("registry_path"):
            _validate_state_registry(state)
        rules_path = os.environ.get("CUBUTTERFLY_PLATFORM_TUNING")
        rules_hash = sha256_file(Path(rules_path)) if rules_path else None
        if state.get("platform_tuning_sha256") != rules_hash:
            raise CampaignError("platform tuning rules changed since queue start")
    else:
        # Validate known build/library identities before the queue can launch.
        fingerprint_builds(manifest, allow_unresolved=True)
        state = {
            "schema": "cubutterfly-recalibrated-campaign-queue-v1",
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_digest,
            "gpu_uuid": gpu_uuid,
            "platform_tuning_sha256": sha256_file(Path(os.environ["CUBUTTERFLY_PLATFORM_TUNING"]))
                if os.environ.get("CUBUTTERFLY_PLATFORM_TUNING") else None,
            "created_at": _now(),
            "status": "waiting-for-calibration",
            "calibration": {"path": str(calibration_manifest), "pid": calibration_pid},
            "phases": {phase: {"status": "pending", "steps": []} for phase in PHASES},
        }
        _atomic_write(state_path, state)
    try:
        calibration_document = wait_for_calibration(
            calibration_pid,
            calibration_manifest,
            accepted,
            poll_seconds,
            calibration_timeout,
        )
    except CampaignError as exc:
        state["status"] = "calibration-not-ready"
        state["calibration"]["error"] = str(exc)
        _atomic_write(state_path, state)
        raise
    require_campaign_active(manifest_path)
    calibration_hash = sha256_file(calibration_manifest)
    validate_calibration_binding(manifest, calibration_document, gpu_uuid)
    registry_path, registry_hash = _registry_from_calibration(calibration_document)
    state["calibration"].update({
        "sha256": calibration_hash,
        "status": calibration_document.get("status", calibration_document.get("calibration_status")),
        "registry_path": str(registry_path),
        "registry_sha256": registry_hash,
        "finished_at": _now(),
    })
    state["status"] = "running"
    if manifest.get("measurement_contract"):
        state["strict_measurement_contract"] = True
        state["compile_mode"] = manifest["measurement_contract"]["compile_mode"]
        state["resolved_build_hashes"] = fingerprint_builds(manifest, allow_unresolved=False)
        state["runtime_hardware"] = validate_runtime_hardware(manifest, gpu_uuid)
    _atomic_write(state_path, state)
    if resume and isinstance(state.get("resolved_build_hashes"), dict):
        validate_resolved_builds(manifest, state["resolved_build_hashes"])

    for phase in PHASES:
        phase_state = state["phases"][phase]
        phase_state["status"] = "running"
        _atomic_write(state_path, state)
        if phase == "prerequisites" or not state.get("resolved_build_hashes"):
            # The prepared manifest can intentionally contain build artifacts
            # absent when it was prepared.  Resolve them after the build queue.
            pass
        for index, step in enumerate(manifest.get("phases", {}).get(phase, [])):
            prior = _existing_success(state, phase, index, step, gpu_uuid) if resume else None
            if prior is not None:
                continue
            if resume and state.get("strict_measurement_contract"):
                records = _phase_step_records(state, phase)
                if index < len(records) and records[index]:
                    quarantine_retry_outputs(step, state, state_path, phase, records[index])
            try:
                record = _run_step(phase, index, step, state, state_path, gpu_uuid, step_timeout, gpu_poll_seconds)
            except CampaignError as exc:
                phase_state["status"] = "invalid"
                phase_state["error"] = str(exc)
                state["status"] = "invalid"
                state["error"] = str(exc)
                _atomic_write(state_path, state)
                raise
            if record.get("status") in ("gpu-not-exclusive", "gpu-conflict", "gpu-monitor-error", "registry-changed", "identity-changed"):
                phase_state["status"] = record["status"]
                state["status"] = record["status"]
                state["error"] = record.get("error", "foreign GPU process detected")
                _atomic_write(state_path, state)
                return 1
            if phase != "prerequisites" and not state.get("correctness_gate", {}).get("passed", True):
                # A later command may still be useful for diagnostics, but its
                # timing/result row is never a passed campaign result.
                if record.get("status") == "passed":
                    record["status"] = "completed-unverified"
                    record["passed"] = False
                    _atomic_write(state_path, state)
        records = _phase_step_records(state, phase)
        failures = [record for record in records if _record_failed(record)]
        if phase == "prerequisites":
            if failures:
                phase_state["status"] = "failed"
                state["status"] = "failed-prerequisites"
                _atomic_write(state_path, state)
                return 1
            try:
                state["resolved_build_hashes"] = fingerprint_builds(manifest, allow_unresolved=False)
            except CampaignError as exc:
                phase_state["status"] = "failed"
                phase_state["error"] = str(exc)
                state["status"] = "failed-build-identity"
                _atomic_write(state_path, state)
                return 1
            phase_state["status"] = "passed"
        elif phase == "correctness":
            state["correctness_gate"] = {"passed": not failures, "finished_at": _now()}
            phase_state["status"] = "passed" if not failures else "failed"
        else:
            correctness_passed = bool(state.get("correctness_gate", {}).get("passed"))
            for record in records:
                if record.get("status") == "passed" and not correctness_passed:
                    record["status"] = "completed-unverified"
                    record["passed"] = False
            failures = [record for record in records if _record_failed(record)]
            phase_state["status"] = "passed" if not failures and correctness_passed else ("completed-unverified" if not failures else "failed")
        _atomic_write(state_path, state)
        if (manifest.get("measurement_contract") and failures
                and phase in ("correctness", "profile_gate")):
            state["status"] = "blocked-" + phase
            state["finished_at"] = _now()
            _atomic_write(state_path, state)
            return 1
    all_phases_passed = all(
        state.get("phases", {}).get(phase, {}).get("status") == "passed"
        for phase in PHASES
    )
    state["status"] = "completed" if all_phases_passed and state.get("correctness_gate", {}).get("passed") else "completed-with-failures"
    state["finished_at"] = _now()
    _atomic_write(state_path, state)
    # The normal summary step observes a still-running queue. Refresh only
    # the report after publishing the terminal state; no measurements repeat.
    comparison_steps = manifest.get("phases", {}).get("comparison", [])
    if comparison_steps and comparison_steps[-1]["id"] == "summarize_recalibrated_campaign":
        final_report = _run_step("comparison", len(comparison_steps) - 1,
                                 comparison_steps[-1], state, state_path, gpu_uuid,
                                 step_timeout, gpu_poll_seconds)
        if not final_report.get("passed"):
            state["status"] = "completed-with-failures"
            state["phases"]["comparison"]["status"] = "failed"
            _atomic_write(state_path, state)
    return 0 if state["status"] == "completed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu-uuid", default=os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0])
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--calibration-pid", type=int)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--calibration-timeout-seconds", type=float)
    parser.add_argument("--step-timeout-seconds", type=float)
    parser.add_argument("--gpu-poll-seconds", type=float, default=1.0)
    parser.add_argument("--accepted-calibration-status", default=",".join(sorted(SUCCESS_STATUSES)))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest_path = args.manifest.resolve()
        manifest = load_manifest(manifest_path)
        if args.dry_run:
            return dry_run(manifest, manifest_path, args.gpu_uuid or "GPU-dry-run")
        calibration_manifest = args.calibration_manifest
        if calibration_manifest is None:
            raise CampaignError("--calibration-manifest is required unless --dry-run is used")
        return run_campaign(
            manifest_path,
            args.gpu_uuid,
            calibration_manifest,
            calibration_pid=args.calibration_pid,
            state_path=args.state,
            resume=args.resume,
            poll_seconds=args.poll_seconds,
            calibration_timeout=args.calibration_timeout_seconds,
            step_timeout=args.step_timeout_seconds,
            gpu_poll_seconds=args.gpu_poll_seconds,
            accepted_calibration_statuses=tuple(args.accepted_calibration_status.split(",")),
        )
    except CampaignError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
