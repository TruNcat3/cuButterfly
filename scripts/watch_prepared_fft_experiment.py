#!/usr/bin/env python3
"""Wait for the fixed FFT GPU, then resume the finite prepared experiment.

Only explicit GPU busy/foreign-work interruptions retry. The original summary
is never rewritten by this adapter; the runner owns its measurement checkpoint.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import watch_idle_campaign as idle

ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = os.environ.get("CUBUTTERFLY_GPU_UUID", "")
DEFAULT_SUMMARY = ROOT / "results/fft_intra_stage_20261024/measurement/summary.json"
COMPLETE = "measured_and_errors_reported"


class ForeignGPUInterruption(RuntimeError):
    pass


class MonitorStopped(RuntimeError):
    pass


class GPUObservationFailure(RuntimeError):
    pass


def read_summary(path):
    return json.loads(path.read_text())


def validate_scope(report):
    if (report.get("preparation_complete") is not True or
            report.get("identity", {}).get("gpu") != GPU_UUID or
            len(report.get("prepared_cases", [])) != 4 or len(report.get("modules", [])) != 20):
        raise ValueError("expected the fully prepared four-request, twenty-module FFT experiment on the fixed UUID")


def campaign_state(report):
    validate_scope(report)
    names = {case["name"] for case in report["prepared_cases"]}
    completed = {case["name"] for case in report.get("cases", []) if case.get("status") == COMPLETE}
    remaining = len(names - completed)
    state = dict(remaining=remaining, measured_rows=len(completed), retryable=False,
                 updated_unix=time.time(), source_status=report.get("status"), error=report.get("error"))
    if any(case.get("failures") for case in report.get("cases", [])):
        return dict(state, status="failed", error="recorded validation failures require inspection")
    if report.get("status") == COMPLETE and completed == names:
        return dict(state, status="complete")
    error = report.get("error", "")
    if report.get("status") == "failed":
        if report.get("error_type") == "RuntimeError" and error.startswith("GPU is busy; no experiment started:"):
            return dict(state, status="waiting-for-gpu", retryable=True)
        if (report.get("error_type") == "ForeignGPUInterruption" or
                (report.get("error_type") == "RuntimeError" and error.startswith("contaminated GPU measurement at "))):
            return dict(state, status="interrupted-by-foreign-gpu-work", retryable=True)
        return dict(state, status="failed")
    if report.get("status") == "cpu_and_jit_prepared" and remaining == 4:
        return dict(state, status="waiting-for-gpu", retryable=True)
    return dict(state, status="failed", error=report.get("error") or "unconfirmed or unknown runner state requires inspection")


def record(control, state, event):
    state = dict(state, updated_unix=time.time(), driver_pid=os.getpid())
    temporary = control / "status.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(control / "status.json")
    with (control / "driver_events.jsonl").open("a") as events:
        events.write(json.dumps(dict(state, event=event)) + "\n")


def resume_command(report, summary):
    value = report["identity"]
    command = [value["python"], str(ROOT / "scripts/run_fft_intra_stage_experiment.py"),
               "--library", value["library"], "--gpu", GPU_UUID, "--output", str(summary.parent),
               "--nvcc", value["nvcc"], "--mathdx", value["mathdx"], "--jit-cache", value["jit_cache"]]
    for key in ("chain", "repeats", "trials", "warmup", "sm"):
        command.extend(["--" + key, str(value[key])])
    return command + ["--resume"]


def validate_cpu(report, summary):
    import run_fft_intra_stage_experiment as runner
    args = runner.make_parser().parse_args(resume_command(report, summary)[2:])
    args.compile_policy = os.environ.get("CUBUTTERFLY_COMPILE_MODE", report["identity"]["compile_policy"])
    runner.validate_prepared(args, report)


def descendants(pid):
    result, pending = set(), [pid]
    while pending:
        parent = pending.pop()
        try:
            children = Path(f"/proc/{parent}/task/{parent}/children").read_text().split()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        for raw in children:
            child = int(raw)
            if child not in result:
                result.add(child)
                pending.append(child)
    return result


def foreign_pids(snapshot):
    return sorted(set(snapshot["pids"]) - {os.getpid()} - descendants(os.getpid()))


def cleanup_children():
    for pid in descendants(os.getpid()):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def fingerprint(path):
    if not path.exists():
        return None
    return (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())


def drive(summary, control, poll_seconds=10):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("the driver and runner signal handlers require the main thread")
    with (control / "driver.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 73
        report = read_summary(summary)
        state = campaign_state(report)
        if state["status"] in ("complete", "failed"):
            record(control, state, "preflight")
            return 0 if state["status"] == "complete" else 2
        (control / f"attempt_{time.time_ns()}.before.json").write_bytes(summary.read_bytes())
        resume_failure = summary.parent / "resume_failure.json"
        before_failure, before_summary = fingerprint(resume_failure), fingerprint(summary)
        stop = threading.Event()
        interruption = {}

        def receive(number, frame):
            if number == signal.SIGUSR1:
                raise ForeignGPUInterruption("foreign GPU process observed: " + str(interruption.get("pids")))
            if number == signal.SIGUSR2:
                raise GPUObservationFailure(str(interruption.get("error")))
            interruption.update(manual_signal=number)
            raise MonitorStopped("monitor received signal " + str(number))

        handlers = {sig: signal.signal(sig, receive) for sig in
                    (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1, signal.SIGUSR2)}

        def guard():
            while not stop.wait(poll_seconds):
                try:
                    snapshot = idle.gpu_snapshot(GPU_UUID)
                    others = foreign_pids(snapshot)
                except Exception as error:
                    if stop.is_set():
                        return
                    interruption["error"] = repr(error)
                    os.kill(os.getpid(), signal.SIGUSR2)
                    return
                if others:
                    if stop.is_set():
                        return
                    interruption.update(pids=others, snapshot=snapshot)
                    os.kill(os.getpid(), signal.SIGUSR1)
                    return

        observer = threading.Thread(target=guard, daemon=True)
        code = 2
        try:
            import run_fft_intra_stage_experiment as runner
            os.environ.setdefault("CUBUTTERFLY_COMPILE_MODE", report["identity"]["compile_policy"])
            record(control, dict(state, status="running", retryable=False), "starting-runner")
            observer.start()
            code = runner.main(resume_command(report, summary)[2:])
        except (ForeignGPUInterruption, MonitorStopped, GPUObservationFailure) as error:
            # Signals before the runner enters its exception handler still have
            # explicit provenance here, without altering its raw checkpoint.
            interruption["exception"] = str(error)
        except Exception as error:
            interruption["error"] = repr(error)
        finally:
            stop.set()
            if observer.is_alive():
                observer.join(timeout=20)
            cleanup_children()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        updated = read_summary(summary)
        state = campaign_state(updated)
        if interruption.get("manual_signal"):
            state.update(status="stopped-manually", retryable=False, interruption=interruption)
            code = 128 + interruption["manual_signal"]
        elif interruption.get("error"):
            state.update(status="failed", retryable=False, interruption=interruption)
            code = 2
        elif fingerprint(resume_failure) != before_failure:
            state.update(status="failed", retryable=False, error="runner rejected resume; see resume_failure.json")
            code = 2
        elif interruption.get("pids"):
            if state["status"] == "failed" and fingerprint(summary) != before_summary:
                state.update(retryable=False, interruption=interruption)
                code = 2
            else:
                state.update(status="interrupted-by-foreign-gpu-work", retryable=True, interruption=interruption)
                code = 1
        elif code != 0 and fingerprint(summary) == before_summary:
            state.update(status="failed", retryable=False, error="runner exited without a fresh checkpoint")
        elif code == 0 and state["status"] != "complete":
            state.update(status="failed", retryable=False, error="runner exited before finite completion")
            code = 2
        record(control, state, "runner-exited")
        return code


def monitor(summary, control):
    control.mkdir(parents=True, exist_ok=True)
    # Protect bootstrap status publication as well as the reused watcher's lock.
    with (control / "adapter.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if idle.driver_locked(control):
            raise RuntimeError("an existing driver owns this control directory")
        report = read_summary(summary)
        validate_scope(report)
        initial = control / "initial_summary.json"
        if not initial.exists():
            initial.write_bytes(summary.read_bytes())
        try:
            validate_cpu(report, summary)
            state = campaign_state(report)
        except Exception as error:
            record(control, dict(status="failed", retryable=False, error=repr(error)), "cpu-validation-failed")
            return 2
        record(control, state, "bootstrap")
        command = [report["identity"]["python"], str(Path(__file__).resolve()), "--driver",
                   "--summary", str(summary), "--control-dir", str(control), "--gpu-uuid", GPU_UUID]
        return idle.watch(control, GPU_UUID, command, poll_seconds=10, idle_seconds=30,
                          max_idle_memory_mib=64, idle_graphics_names=())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--control-dir", type=Path,
                        default=ROOT / "results/fft_intra_stage_20261024/monitor")
    parser.add_argument("--driver", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu-uuid", default=GPU_UUID, required=not GPU_UUID)
    args = parser.parse_args(argv)
    globals()["GPU_UUID"] = args.gpu_uuid
    summary, control = args.summary.resolve(strict=True), args.control_dir.resolve()
    if control == summary.parent:
        parser.error("control directory must be separate from the measurement directory")
    return drive(summary, control) if args.driver else monitor(summary, control)


if __name__ == "__main__":
    raise SystemExit(main())
