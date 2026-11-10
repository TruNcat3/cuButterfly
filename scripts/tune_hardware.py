#!/usr/bin/env python3
"""Bounded, resumable rounds of local hardware search.

The tuner is deliberately a driver.  Measurement, checkpoint reuse, model
fitting, registry promotion and selector replay remain owned by
``calibrate_hardware.py``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if not (ROOT / "CMakeLists.txt").exists():
    ROOT = ROOT / "share/cuButterfly"
CALIBRATOR = SCRIPT_DIR / "calibrate_hardware.py"
STATE_SCHEMA = "cubutterfly-hardware-tuner-v1"
DEFAULT_TIME_BUDGET = 1800.0
DEFAULT_ROUND_BUDGET = 16
DEFAULT_MAX_ROUNDS = 4
DEFAULT_POLL_SECONDS = 15
MAX_POLL_SECONDS = 60
IDLE_CONTEXT_MEMORY_MB = 64
INTERRUPT_GRACE_SECONDS = 5.0
TERMINATE_GRACE_SECONDS = 2.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded evolutionary hardware-search rounds from a migration manifest."
    )
    parser.add_argument("--resume-from", required=True, type=pathlib.Path,
                        help="migration_manifest.json produced by calibrate_hardware.py")
    parser.add_argument("--gpu", help="GPU UUID to expose through CUDA_VISIBLE_DEVICES")
    parser.add_argument("--time-budget", type=float, default=DEFAULT_TIME_BUDGET,
                        help="total tuner wall-time budget in seconds")
    parser.add_argument("--round-budget", type=int, default=DEFAULT_ROUND_BUDGET,
                        help="additional search candidates per workload each round")
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS,
                        help="maximum number of successful search rounds")
    parser.add_argument("--watch", action="store_true",
                        help="wait for a busy GPU until the remaining wall budget expires")
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS,
                        help="GPU-busy polling interval, capped at 60 seconds")
    parser.add_argument("--dry-run", action="store_true",
                        help="write the next command to tuner_state.json without executing it")
    args = parser.parse_args(argv)
    if args.time_budget <= 0.0:
        parser.error("--time-budget must be positive")
    if args.round_budget <= 0:
        parser.error("--round-budget must be positive")
    if args.max_rounds <= 0:
        parser.error("--max-rounds must be positive")
    if args.poll_seconds <= 0.0 or args.poll_seconds > MAX_POLL_SECONDS:
        parser.error(f"--poll-seconds must be in (0, {MAX_POLL_SECONDS:g}]")
    return args


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"manifest not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON manifest: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"manifest must contain an object: {path}")
    return value


def _write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _workload_hash(manifest: dict[str, Any]) -> str:
    workloads = manifest.get("workloads")
    if isinstance(workloads, dict):
        recorded = workloads.get("sha256")
        if recorded:
            return str(recorded)
    return _hash_json(workloads if workloads is not None else [])


def _identity_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    # Search budgets and round status intentionally do not belong here: the
    # tuner changes the former and the calibrator rewrites the latter.
    return {
        "device": manifest.get("device", {}),
        "build_dir": manifest.get("build_dir", ""),
        "profile_dir": manifest.get("profile_dir", ""),
        "output_dir": manifest.get("output_dir", ""),
        "compile_mode": manifest.get("compile_mode", ""),
    }


def _identity_hash(manifest: dict[str, Any]) -> str:
    return _hash_json(_identity_payload(manifest))


def _manifest_gpu_target(manifest: dict[str, Any]) -> str | None:
    device = manifest.get("device")
    if not isinstance(device, dict):
        return None
    for field in ("uuid", "gpu_uuid", "device_uuid", "id"):
        value = device.get(field)
        if value:
            return str(value)
    return None


def _environment_gpu_target() -> str | None:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return value.split(",", 1)[0].strip() or None


def _target_conflicts(left: str | None, right: str | None) -> bool:
    if not left or not right or left == right:
        return False
    # An index and a UUID may select the same device, but cannot be resolved
    # without querying the host.  Leave that comparison to the calibrator.
    index_like = lambda value: value.isdigit()
    if index_like(left) != index_like(right):
        return False
    return True


def _validate_gpu_target(manifest: dict[str, Any], state: dict[str, Any] | None,
                         requested: str | None) -> str | None:
    manifest_target = _manifest_gpu_target(manifest)
    state_target = str(state["gpu"]) if state and state.get("gpu") else None
    environment_target = _environment_gpu_target()
    for label, previous in (("manifest", manifest_target), ("tuner state", state_target)):
        if _target_conflicts(requested, previous):
            raise ValueError(f"GPU target conflicts with {label}: {requested} != {previous}")
    if _target_conflicts(requested, environment_target):
        raise ValueError(
            f"--gpu conflicts with CUDA_VISIBLE_DEVICES: {requested} != {environment_target}"
        )
    if _target_conflicts(state_target, environment_target):
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES conflicts with tuner state: {environment_target} != {state_target}"
        )
    if requested:
        return requested
    if _target_conflicts(manifest_target, environment_target):
        raise ValueError("CUDA_VISIBLE_DEVICES conflicts with migration manifest GPU")
    return state_target or environment_target or manifest_target


def _base_search_budget(manifest: dict[str, Any], state: dict[str, Any] | None) -> int:
    if state and state.get("base_search_budget") is not None:
        return max(0, int(state["base_search_budget"]))
    protocol = manifest.get("protocol")
    if isinstance(protocol, dict):
        try:
            return max(0, int(protocol.get("search_budget", 0)))
        except (TypeError, ValueError):
            pass
    return 0


def _round_budget(state: dict[str, Any], round_number: int) -> int:
    return max(0, int(state["base_search_budget"])) + round_number * int(state["round_budget"])


def calibration_command(manifest_path: pathlib.Path, search_budget: int) -> list[str]:
    return [
        sys.executable,
        str(CALIBRATOR),
        "--resume-from",
        str(manifest_path),
        "--search-strategy",
        "evolutionary",
        "--cost-model",
        "staged",
        "--search-budget",
        str(search_budget),
    ]


def _new_state(manifest_path: pathlib.Path, manifest: dict[str, Any], args: argparse.Namespace,
               gpu: str | None) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "manifest": str(manifest_path),
        "identity_hash": _identity_hash(manifest),
        "workload_sha256": _workload_hash(manifest),
        "gpu": gpu,
        "base_search_budget": _base_search_budget(manifest, None),
        "round_budget": int(args.round_budget),
        "max_rounds": int(args.max_rounds),
        "time_budget": float(args.time_budget),
        "elapsed_seconds": 0.0,
        "completed_rounds": 0,
        "next_round": 1,
        "status": "planned",
        "rounds": [],
        "commands": [],
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def _load_or_create_state(state_path: pathlib.Path, manifest_path: pathlib.Path,
                          manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    existing = _read_json(state_path) if state_path.exists() else None
    gpu = _validate_gpu_target(manifest, existing, args.gpu)
    if existing is None:
        state = _new_state(manifest_path, manifest, args, gpu)
        _write_json(state_path, state)
        return state
    if existing.get("schema") != STATE_SCHEMA:
        raise ValueError(f"unsupported tuner state schema: {existing.get('schema')!r}")
    if pathlib.Path(str(existing.get("manifest", ""))).resolve() != manifest_path:
        raise ValueError("tuner state belongs to a different migration manifest")
    if existing.get("identity_hash") != _identity_hash(manifest):
        raise ValueError("migration manifest hardware/build identity changed since tuning began")
    if existing.get("workload_sha256") != _workload_hash(manifest):
        raise ValueError("workload configuration changed since tuning began")
    for field, requested in (("round_budget", args.round_budget),
                             ("max_rounds", args.max_rounds),
                             ("time_budget", args.time_budget)):
        if field in existing and float(existing[field]) != float(requested):
            raise ValueError(f"{field} changed since tuning began; use the original tuner protocol")
    if existing.get("gpu") and gpu and _target_conflicts(str(existing["gpu"]), gpu):
        raise ValueError("GPU target changed since tuning began")
    existing["gpu"] = existing.get("gpu") or gpu
    existing.setdefault("rounds", [])
    existing.setdefault("commands", [])
    existing.setdefault("elapsed_seconds", 0.0)
    existing.setdefault("completed_rounds", 0)
    existing["next_round"] = max(
        int(existing.get("next_round", 1)), int(existing["completed_rounds"]) + 1
    )
    existing["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_json(state_path, existing)
    return existing


def gpu_is_busy(target: str | None = None) -> bool | None:
    """Return True when nvidia-smi reports active work, None when unknown."""
    command = ["nvidia-smi"]
    selected = target or _environment_gpu_target()
    if selected:
        command += ["-i", selected]
    command += ["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        utilization = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        busy = int((utilization.stdout.strip().splitlines() or ["0"])[0].strip()) > 0
    except ValueError:
        return None
    if busy:
        return True
    apps_command = ["nvidia-smi"]
    if selected:
        apps_command += ["-i", selected]
    apps_command += ["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    try:
        apps = subprocess.run(apps_command, cwd=ROOT, text=True, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in apps.stdout.splitlines():
        if not line.strip():
            continue
        try:
            memory_mb = int(line.rsplit(",", 1)[-1].strip())
        except ValueError:
            memory_mb = IDLE_CONTEXT_MEMORY_MB + 1
        if memory_mb > IDLE_CONTEXT_MEMORY_MB:
            return True
    return False


def _remaining(deadline: float) -> float:
    return deadline - time.monotonic()


def _wait_for_gpu(target: str | None, watch: bool, poll_seconds: float,
                  deadline: float) -> tuple[bool, str | None]:
    while True:
        busy = gpu_is_busy(target)
        if busy is not True:
            return True, None
        if not watch:
            return False, "gpu-busy"
        remaining = _remaining(deadline)
        if remaining <= 0.0:
            return False, "gpu-busy-time-budget"
        time.sleep(min(poll_seconds, remaining))


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGINT)
    except (AttributeError, OSError, ProcessLookupError):
        try:
            process.send_signal(signal.SIGINT)
        except (AttributeError, OSError, ProcessLookupError):
            return
    try:
        process.wait(timeout=INTERRUPT_GRACE_SECONDS)
        return
    except (subprocess.TimeoutExpired, OSError):
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (AttributeError, OSError, ProcessLookupError):
        try:
            process.terminate()
        except (AttributeError, OSError, ProcessLookupError):
            pass
    try:
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
        return
    except (subprocess.TimeoutExpired, OSError):
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (AttributeError, OSError, ProcessLookupError):
        try:
            process.kill()
        except (AttributeError, OSError, ProcessLookupError):
            pass


def run_calibration(command: list[str], env: dict[str, str], deadline: float) -> dict[str, Any]:
    """Run one calibrator process, interrupting only its own process group."""
    try:
        process = subprocess.Popen(command, cwd=ROOT, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
    except OSError as error:
        return {"returncode": None, "stdout": "", "stderr": str(error), "failed": True,
                "timed_out": False}
    while True:
        returncode = process.poll()
        if returncode is not None:
            stdout, stderr = process.communicate()
            return {"returncode": returncode, "stdout": _text(stdout), "stderr": _text(stderr),
                    "failed": False, "timed_out": False}
        if _remaining(deadline) <= 0.0:
            _terminate_process_group(process)
            stdout, stderr = process.communicate()
            return {"returncode": process.returncode, "stdout": _text(stdout), "stderr": _text(stderr),
                    "failed": False, "timed_out": True}
        # Drain both pipes while the child runs. Polling without reading can
        # deadlock a verbose calibration when its pipe buffer fills.
        try:
            stdout, stderr = process.communicate(timeout=min(1.0, max(0.001, _remaining(deadline))))
            return {"returncode": process.returncode, "stdout": _text(stdout), "stderr": _text(stderr),
                    "failed": False, "timed_out": False}
        except subprocess.TimeoutExpired:
            continue


def _busy_output(result: dict[str, Any]) -> bool:
    text = (_text(result.get("stdout")) + "\n" + _text(result.get("stderr"))).lower()
    return any(marker in text for marker in (
        "not exclusive", "active compute processes", "gpu became busy", "gpu is busy",
        "target gpu" + " is occupied", "target gpu" + " is not exclusive",
    ))


def _manifest_status(manifest_path: pathlib.Path) -> tuple[str | None, str | None]:
    if not manifest_path.exists():
        return None, "migration manifest disappeared"
    try:
        manifest = _read_json(manifest_path)
    except ValueError as error:
        return None, str(error)
    status = manifest.get("status")
    return str(status) if status else None, None


def _set_elapsed(state: dict[str, Any], initial_elapsed: float, invocation_start: float) -> None:
    state["elapsed_seconds"] = initial_elapsed + max(0.0, time.monotonic() - invocation_start)
    state["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()


def _save_round(state_path: pathlib.Path, state: dict[str, Any], record: dict[str, Any],
                initial_elapsed: float, invocation_start: float) -> None:
    _set_elapsed(state, initial_elapsed, invocation_start)
    state["rounds"][-1] = record
    _write_json(state_path, state)


def _print_output(result: dict[str, Any]) -> None:
    stdout = _text(result.get("stdout"))
    stderr = _text(result.get("stderr"))
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")


def _report(state_path: pathlib.Path, state: dict[str, Any], message: str) -> None:
    print(f"hardware tuner: {message}")
    print(f"rounds: {state.get('completed_rounds', 0)}/{state.get('max_rounds', 0)}")
    print(f"tuner state: {state_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.resume_from.expanduser().resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != "cubutterfly-hardware-migration-v1":
        raise ValueError(f"unsupported migration manifest schema: {manifest.get('schema')!r}")
    state_path = manifest_path.parent / "tuner_state.json"
    state = _load_or_create_state(state_path, manifest_path, manifest, args)
    if state.get("status") in {"complete", "complete-with-warning"}:
        _report(state_path, state, "already complete")
        return 0
    if int(state.get("completed_rounds", 0)) >= int(state["max_rounds"]):
        state["status"] = "complete-with-warning"
        _write_json(state_path, state)
        _report(state_path, state, "maximum rounds already reached")
        return 0

    next_round = max(1, int(state.get("next_round", int(state.get("completed_rounds", 0)) + 1)))
    budget = _round_budget(state, next_round)
    command = calibration_command(manifest_path, budget)
    if args.dry_run:
        state["status"] = "planned"
        state["next_round"] = next_round
        state["planned_command"] = command
        state["planned_search_budget"] = budget
        _write_json(state_path, state)
        print("hardware tuner: dry-run")
        print("command: " + " ".join(command))
        print(f"tuner state: {state_path}")
        return 0

    invocation_start = time.monotonic()
    initial_elapsed = float(state.get("elapsed_seconds", 0.0))
    deadline = invocation_start + max(0.0, float(state["time_budget"]) - initial_elapsed)
    target = state.get("gpu")
    child_env = os.environ.copy()
    if target:
        child_env["CUDA_VISIBLE_DEVICES"] = str(target)

    while int(state.get("completed_rounds", 0)) < int(state["max_rounds"]):
        next_round = max(1, int(state.get("next_round", int(state.get("completed_rounds", 0)) + 1)))
        budget = _round_budget(state, next_round)
        command = calibration_command(manifest_path, budget)
        if _remaining(deadline) <= 0.0:
            state["status"] = "paused"
            state["pause_reason"] = "time-budget-expired"
            _set_elapsed(state, initial_elapsed, invocation_start)
            _write_json(state_path, state)
            _report(state_path, state, "paused: time budget expired")
            return 0

        available, reason = _wait_for_gpu(target, args.watch, args.poll_seconds, deadline)
        if not available:
            state["status"] = "paused"
            state["pause_reason"] = reason
            state["next_round"] = next_round
            state["planned_command"] = command
            _set_elapsed(state, initial_elapsed, invocation_start)
            _write_json(state_path, state)
            _report(state_path, state, f"paused: {reason}")
            return 0

        record = {
            "round": next_round,
            "search_budget": budget,
            "command": command,
            "status": "running",
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        state["status"] = "running"
        state["next_round"] = next_round
        state.setdefault("rounds", []).append(record)
        state.setdefault("commands", []).append(command)
        _save_round(state_path, state, record, initial_elapsed, invocation_start)

        result = run_calibration(command, child_env, deadline)
        _print_output(result)
        _set_elapsed(state, initial_elapsed, invocation_start)
        record["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        record["returncode"] = result.get("returncode")
        if result.get("timed_out"):
            record["status"] = "paused"
            record["reason"] = "time-budget-expired"
            state["status"] = "paused"
            state["pause_reason"] = "time-budget-expired"
            _save_round(state_path, state, record, initial_elapsed, invocation_start)
            _report(state_path, state, "paused: calibrator interrupted at deadline")
            return 0
        if result.get("failed") or result.get("returncode") is None:
            record["status"] = "failed"
            record["error"] = _text(result.get("stderr"))
            state["status"] = "failed"
            state["error"] = record["error"]
            _save_round(state_path, state, record, initial_elapsed, invocation_start)
            _report(state_path, state, "failed: calibrator could not start")
            return 1
        if int(result.get("returncode", 1)) != 0:
            if _busy_output(result):
                record["status"] = "gpu-busy"
                record["reason"] = "calibrator-reported-gpu-busy"
                _save_round(state_path, state, record, initial_elapsed, invocation_start)
                if args.watch and _remaining(deadline) > 0.0:
                    continue
                state["status"] = "paused"
                state["pause_reason"] = "gpu-busy"
                _write_json(state_path, state)
                _report(state_path, state, "paused: calibrator reported a busy GPU")
                return 0
            record["status"] = "failed"
            record["error"] = _text(result.get("stderr")) or _text(result.get("stdout"))
            state["status"] = "failed"
            state["error"] = record["error"]
            _save_round(state_path, state, record, initial_elapsed, invocation_start)
            _report(state_path, state, f"failed: calibrator exit {result.get('returncode')}")
            return 1

        calibrator_status, status_error = _manifest_status(manifest_path)
        if status_error:
            record["status"] = "failed"
            record["error"] = status_error
            state["status"] = "failed"
            state["error"] = status_error
            _save_round(state_path, state, record, initial_elapsed, invocation_start)
            _report(state_path, state, "failed: calibrator did not leave a valid manifest")
            return 1
        if calibrator_status == "failed":
            record["status"] = "failed"
            record["error"] = "calibrator manifest reports failed"
            state["status"] = "failed"
            state["error"] = record["error"]
            _save_round(state_path, state, record, initial_elapsed, invocation_start)
            _report(state_path, state, "failed: calibrator manifest reports failed")
            return 1

        warning = calibrator_status != "complete"
        record["status"] = "warning" if warning else "complete"
        record["calibrator_status"] = calibrator_status or "unknown"
        if warning:
            record["warning"] = "calibration completed without a complete migration status"
        state["completed_rounds"] = int(state.get("completed_rounds", 0)) + 1
        state["next_round"] = int(state["completed_rounds"]) + 1
        state["status"] = "complete" if not warning else "running"
        _save_round(state_path, state, record, initial_elapsed, invocation_start)
        if not warning:
            _report(state_path, state, "complete")
            return 0
        _report(state_path, state, f"round {next_round} finished with warning; continuing")
        if int(state["completed_rounds"]) >= int(state["max_rounds"]):
            state["status"] = "complete-with-warning"
            _write_json(state_path, state)
            _report(state_path, state, "finished with model/search warning")
            return 0
        # A warning/incomplete round is valid evidence; the next round gets a
        # larger budget and lets the unified calibrator reuse its checkpoints.

    state["status"] = "complete-with-warning"
    _write_json(state_path, state)
    _report(state_path, state, "finished with model/search warning")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
