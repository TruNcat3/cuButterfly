#!/usr/bin/env python3
"""Run one experiment command with read-only NVML clock observations.

Samples cover the child process lifetime, including setup and checking; they
are not attributed to CUDA-event timing intervals. No GPU clocks are changed.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from run_comprehensive_suite import require_exclusive_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--stop-on-thermal-throttle", action="store_true",
                        help="stop the owned child process group if NVML reports thermal slowdown")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a child command is required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise ValueError("CUDA_VISIBLE_DEVICES must equal the requested UUID")
    paths = {key: Path(str(args.output_prefix) + suffix) for key, suffix in
             (("stdout", ".stdout.log"), ("stderr", ".stderr.log"), ("metadata", ".telemetry.json"))}
    if any(p.exists() for p in paths.values()):
        raise ValueError("telemetry output already exists; use a new prefix for a continuation")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    require_exclusive_gpu()
    nvml = ctypes.CDLL("libnvidia-ml.so.1")
    if nvml.nvmlInit_v2() != 0:
        raise RuntimeError("NVML initialization failed")
    handle = ctypes.c_void_p()
    if nvml.nvmlDeviceGetHandleByUUID(args.gpu_uuid.encode(), ctypes.byref(handle)) != 0:
        raise RuntimeError("NVML cannot resolve the target UUID")
    record = dict(schema="cubutterfly-command-telemetry-v1", gpu_uuid=args.gpu_uuid,
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), command=command,
        compile_mode=os.environ.get("CUBUTTERFLY_COMPILE_MODE"),
        files={k: str(v.resolve()) for k, v in paths.items()}, status="running",
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sampling_scope="child lifetime; not exact CUDA-event interval", clock_samples=[],
        thermal_samples=[], stop_on_thermal_throttle=args.stop_on_thermal_throttle)
    for arg in command:
        # Command arguments can include serialized mappings larger than the
        # filesystem NAME_MAX; they are data, not candidate source paths.
        try:
            p = Path(arg)
            is_file = p.is_file()
        except OSError:
            continue
        if is_file and (p.suffix == ".py" or p == Path(command[0])):
            record.setdefault("command_file_hashes", {})[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()

    def save():
        temporary = Path(str(paths["metadata"]) + ".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(paths["metadata"])

    started = time.monotonic()
    process = None
    try:
        with paths["stdout"].open("w") as stdout, paths["stderr"].open("w") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            record["pid"] = process.pid
            save()
            saved = started
            while process.poll() is None:
                mhz = ctypes.c_uint()
                if nvml.nvmlDeviceGetClockInfo(handle, 1, ctypes.byref(mhz)) == 0:
                    record["clock_samples"].append([round(time.monotonic() - started, 3), mhz.value])
                temperature = ctypes.c_uint()
                reasons = ctypes.c_ulonglong()
                temperature_ok = nvml.nvmlDeviceGetTemperature(handle, 0, ctypes.byref(temperature)) == 0
                reasons_ok = nvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle, ctypes.byref(reasons)) == 0
                record["thermal_samples"].append([round(time.monotonic() - started, 3),
                    temperature.value if temperature_ok else None, reasons.value if reasons_ok else None])
                if args.stop_on_thermal_throttle and reasons_ok and reasons.value & (0x20 | 0x40):
                    raise RuntimeError("NVML reports thermal slowdown; this command is diagnostic, not accepted performance")
                if time.monotonic() - saved >= 5:
                    save()
                    saved = time.monotonic()
                time.sleep(0.1)
        record["returncode"] = process.returncode
        record["status"] = "completed" if process.returncode == 0 else "child-failed"
        require_exclusive_gpu()
    except BaseException as error:
        record["status"] = "interrupted"
        record["error"] = str(error)
        if process is not None and process.poll() is None:
            # Only this wrapper's new child session is signalled. Give a
            # resumable Python runner time to save its interrupted checkpoint.
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        record["ended_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
        nvml.nvmlShutdown()
    print(json.dumps({k: record[k] for k in ("status", "returncode", "elapsed_seconds")}))
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
