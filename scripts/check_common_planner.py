#!/usr/bin/env python3
"""Bounded rank-one planner checks using the existing output-checking benches.

Defaults to CPU-only command generation. Execution requires --run. Existing
benchmark tolerances and reference implementations remain authoritative. Raw
outputs and every CSV metric are retained; floating-point benches currently
report max_error rather than an error distribution.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

from watch_idle_campaign import gpu_snapshot, is_idle
from watch_prepared_fft_experiment import descendants


PROFILE_FILES = ("hardware_model.json", "hardware_profile.json", "calibration_device.json", "cost_model.json")
PLANNER_SCRIPTS = ("check_common_planner.py", "rank_static_candidates.py", "stage_cost_model.py",
                   "static_hardware_profile.py", "stage_schedule_native.py", "stage_schedule_native.cpp",
                   "stage_service_model.py", "schedule_cost_model.py", "hardware_registry.py",
                   "resident_mapping.py")


@dataclass(frozen=True)
class Case:
    operator: str
    precision: str
    log_n: int
    batch: int
    mode: str
    phase: str = "correctness"
    mapping: dict | None = None

    @property
    def name(self) -> str:
        return f"{self.phase}-{self.operator}-{self.precision}-n{self.log_n}-b{self.batch}-{self.mode}"


def cases(phase: str) -> list[Case]:
    if phase == "buffering":
        return [Case("fft", "fp64", log_n, 4, f"factor-c{columns}-d{depth}", "buffering",
                     dict(backend="factor-streamed", fft_core="cufftdx-block",
                          factor_partition=[factor, factor], factor_columns=columns,
                          prefetch_depth=depth, factor_ept=8, data_tiles_per_cta=4,
                          cross_twiddle="recurrence", shared_layout="writer-aligned"))
                for log_n, factor, columns, depth in ((20, 10, 2, 2), (20, 10, 4, 0), (18, 9, 4, 2))]
    types = [("fft", "fp32"), ("fft", "fp64"), ("fwht", "fp32"),
             ("subset-zeta", "uint32"), ("superset-zeta", "uint32"),
             ("xor-zeta", "uint32"), ("structured-2x2", "fp32"),
             ("ntt", "word32"), ("ntt", "word64")]
    correctness = [Case(op, precision, n, batch, mode)
                   for op, precision in types
                   for n, batch, mode in ((10, 1, "shared-jit"),
                                          (16, 4, "shared-jit"), (16, 16, "auto"))]
    performance = [Case("fft", precision, n, 16, "auto", "performance")
                   for precision in ("fp32", "fp64") for n in (16, 20)]
    performance += [Case(op, precision, 16, 16, "auto", "performance")
                    for op, precision in (("fwht", "fp32"), ("ntt", "word32"), ("ntt", "word64"))]
    return correctness if phase == "correctness" else performance if phase == "performance" else correctness + performance


def command(case: Case, build: Path) -> list[str]:
    ntt = case.operator == "ntt"
    result = [str(build / ("cuntt_bench" if ntt else "cubutterfly_bench")),
              "--logN", str(case.log_n), "--batch", str(case.batch),
              "--verify", "--verify-batches", "0", "--csv",
              "--warmup", "5" if case.phase in ("performance", "buffering") else "1",
              "--repeat", "20" if case.phase in ("performance", "buffering") else "1"]
    if ntt:
        result += ["--word-bits", case.precision.removeprefix("word")]
        if case.precision == "word32":
            result += ["--modulus", "998244353"]
    else:
        result += ["--operator", case.operator, "--precision", case.precision]
        if case.operator == "structured-2x2":
            result += ["--stage-matrix", "1,0.25,-0.5,1"]
    if case.mapping is not None:
        result += ["--mapping-json", json.dumps(case.mapping, sort_keys=True, separators=(",", ":"))]
    elif case.mode == "auto":
        result += ["--auto-select"]
    else:
        result += ["--backend", "shared-iterative", "--stage-partition",
                   "5,5" if case.log_n == 10 else "8,8"]
        result += ["--threads-per-block" if ntt else "--tile-threads", "128"]
    return result


def parse_row(stdout: str) -> dict[str, str]:
    # CUDA or compile logs may precede CSV, but the benchmark owns its header.
    lines = stdout.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("device,compute_capability,"))
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    if len(rows) != 1 or None in rows[0]:
        raise ValueError("expected exactly one complete benchmark CSV row")
    return rows[0]


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    return dict(state="present", sha256=file_digest(path)) if path.is_file() else dict(state="missing")


def profile_identities(directory):
    return {name: file_identity(directory / name) for name in PROFILE_FILES}


def planner_input_identities(env, script_directory=None):
    directory = script_directory or Path(__file__).resolve().parent
    result = {"scripts": {name: file_identity(directory / name) for name in PLANNER_SCRIPTS}}
    for key in ("CUBUTTERFLY_HARDWARE_MODEL", "CUBUTTERFLY_COST_MODEL", "CUBUTTERFLY_STATIC_RANKER"):
        if env.get(key):
            path = Path(env[key]).resolve()
            result[key] = dict(path=str(path), identity=file_identity(path))
            if key != "CUBUTTERFLY_STATIC_RANKER":
                result[key]["profile_siblings"] = profile_identities(path.parent)
    return result


def consumed_profile_identity(provenance):
    directories = set()
    reported = provenance.get("profile_directory")
    if isinstance(reported, str) and reported:
        directories.add(str(Path(reported).resolve()))
    for source in provenance.get("hardware_resource_sources", []):
        if source.startswith("static-hardware-model:"):
            directories.add(str(Path(source.removeprefix("static-hardware-model:")).resolve().parent))
    return dict(confirmed=bool(directories), directories={directory: profile_identities(Path(directory))
                                                       for directory in sorted(directories)})


def profile_resume_matches(previous):
    saved = previous.get("consumed_profile_identity", {})
    return saved.get("confirmed") is True and saved == consumed_profile_identity(
        previous.get("selection_provenance", {}))


def design_point(row):
    mapping = json.loads(row.get("mapping_json", "{}"))
    groups = json.loads(row.get("execution_groups_json", "[]"))
    partition = mapping.get("stage_partition")
    group_fields = ("first_stage", "stage_count", "grid_ctas", "threads", "core", "codelet", "io_policy",
                    "shared_layout", "local_stage_partition", "exchange_chunk", "data_tiles_per_cta",
                    "prefetch_depth", "stage_space", "stage_time", "data_space", "data_time",
                    "batch_space", "batch_time", "elements_per_thread", "launch_count",
                    "live_shared_bytes", "compiler_resources_known")
    tile_fields = ("local_stages", "flow_tile_log_n", "factor_partition", "local_stage_partitions",
                   "tile_threads", "factor_ept", "factor_columns", "data_tiles_per_cta", "prefetch_depth")
    overlap_fields = ("stage_overlap", "batch_tile_count", "factor_overlap", "factor_slices",
                      "pipeline_buffers", "stage_space", "pipeline_warps", "stage_handoff")
    def resolved_value(key):
        if key in mapping:
            return mapping[key]
        value = row.get(key)
        if value in (None, ""):
            return None
        if key in ("stage_overlap", "factor_overlap") and value in ("0", "1"):
            return value == "1"
        if value.isdecimal():
            return int(value)
        return value
    return dict(backend=mapping.get("backend", row.get("backend")), stage_partition=partition,
                logical_segment_count=len(partition) if isinstance(partition, list) else None,
                physical_group_count=len(groups),
                physical_stage_partition=[group.get("stage_count") for group in groups],
                tile={key: resolved_value(key) for key in tile_fields},
                overlap={key: resolved_value(key) for key in overlap_fields},
                groups=[{key: group.get(key) for key in group_fields} for group in groups],
                measured_residency=None)


def selection_provenance(row, stderr):
    confidence = row.get("selection_confidence")
    reason = row.get("selection_reason")
    mode_match = re.search(r"(?:^|;\s*)mode=([^;]*)", reason or "")
    profile_match = re.search(r'(?:^|;\s*)profile=(null|"(?:[^"\\]|\\.)*")', reason or "")
    mode = mode_match.group(1) if mode_match else None
    classifications = {"validated-static-model": "static-model-estimate",
                       "unmeasured-static-model": "static-model-estimate",
                       "hardware-rate-prior": "rate-prior",
                       "unmeasured-feasible": "unmeasured-fallback"}
    # Candidate confidence identifies the selected source even in mixed rankings.
    source = classifications.get(confidence, classifications.get(mode, "unknown"))
    if row.get("auto_select") == "0":
        source = "explicit-mapping"
    return dict(source=source, confidence=confidence, mode=mode, reason=reason,
                profile_directory=json.loads(profile_match.group(1)) if profile_match else None,
                selected_implementation=row.get("selected_implementation"),
                selection_target=row.get("selection_target"),
                predicted_kernel_ms=row.get("predicted_kernel_ms"),
                exact_workload_measurement_used=False if source in classifications.values() else None,
                hardware_resource_sources=re.findall(r"^cuButterfly hardware resources: (.+)$", stderr, re.MULTILINE))


def annotate_result(record, stderr):
    metrics = record.get("metrics")
    if metrics is not None:
        record["design_point"] = design_point(metrics)
        record["selection_provenance"] = selection_provenance(metrics, stderr)
        record["consumed_profile_identity"] = consumed_profile_identity(record["selection_provenance"])


def check_row(case: Case, row: dict[str, str]) -> None:
    if row.get("correct") != "1" or int(row.get("verified_batches", "0")) != case.batch:
        raise ValueError("benchmark did not verify every requested batch successfully")
    if not row.get("mapping_json") or not row.get("runtime_fingerprint"):
        raise ValueError("mapping or build provenance is missing")
    groups = json.loads(row["execution_groups_json"])
    if not groups:
        raise ValueError("physical execution groups are missing")
    if case.mapping is not None:
        resolved_mapping = json.loads(row["mapping_json"])
        if any(resolved_mapping.get(key) != value for key, value in case.mapping.items()):
            raise ValueError("explicit mapping parameters were changed")
        if not all(group.get("compiler_resources_known") for group in groups):
            raise ValueError("explicit factor mapping is missing compiled group resources")
    elif case.mode == "shared-jit":
        if row.get("backend") != "shared-iterative":
            raise ValueError("explicit shared mapping was changed")
        if not all(group.get("compiler_resources_known") for group in groups):
            raise ValueError("shared JIT did not attach compiled resources to every group")
    elif row.get("auto_select") != "1" or not row.get("selection_reason"):
        raise ValueError("AUTO selection provenance is missing")
    if case.operator != "ntt" and "max_error" not in row:
        raise ValueError("floating/integer benchmark error metric is missing")


class GPUInterruption(RuntimeError):
    def __init__(self, status, snapshot):
        super().__init__(status)
        self.status, self.snapshot = status, snapshot


class DriverStopped(RuntimeError):
    pass


def stop_owned_group(child):
    # The command owns this new process group, including any JIT subprocesses.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=5)


def guarded_run(command, env, directory, gpu_uuid, timeout, poll_seconds, idle_memory, idle_graphics_names=()):
    snapshot = gpu_snapshot(gpu_uuid)
    if not is_idle(snapshot, idle_memory, idle_graphics_names):
        raise GPUInterruption("waiting-for-gpu", snapshot)
    child = None
    started = time.monotonic()
    with (directory / "stdout.txt").open("w") as stdout, (directory / "stderr.txt").open("w") as stderr:
        try:
            child = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr,
                                     text=True, start_new_session=True)
            while True:
                observed = gpu_snapshot(gpu_uuid)
                allowed = {child.pid} | descendants(child.pid)
                allowed.update(process["pid"] for process in observed.get("processes", [])
                               if process["type"] == "G" and process["name"] in idle_graphics_names)
                foreign = sorted(set(observed["pids"]) - allowed)
                if foreign:
                    raise GPUInterruption("interrupted-by-foreign-gpu-work", dict(observed, foreign_pids=foreign))
                if child.poll() is not None:
                    break
                if time.monotonic() - started >= timeout:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    child.wait(timeout=min(poll_seconds, max(0.01, timeout - (time.monotonic() - started))))
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            if child is not None:
                stop_owned_group(child)
            raise
    result = subprocess.CompletedProcess(command, child.returncode,
                                         (directory / "stdout.txt").read_text(),
                                         (directory / "stderr.txt").read_text())
    result.gpu_observations = dict(before=snapshot, after=observed)
    return result


def write_status(output, selected, results, status, gpu_uuid, retryable=False, **extra):
    passed = sum(result["status"] == "passed" for result in results)
    value = dict(status=status, remaining=len(selected) - passed, measured_rows=passed,
                 updated_unix=time.time(), retryable=retryable, gpu_uuid=gpu_uuid,
                 driver_pid=os.getpid(), **extra)
    temporary = output / "status.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(output / "status.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--suite", "--phase", dest="phase", choices=("correctness", "performance", "all", "buffering"), default="correctness",
                        help="buffering is a separate three-point factor experiment; all retains the original 34 cases")
    parser.add_argument("--run", action="store_true", help="execute GPU commands; default only prints commands")
    parser.add_argument("--output", type=Path, default=Path("results/common_planner_gate"))
    parser.add_argument("--case", action="append", default=[], help="select exact case ID; repeatable")
    parser.add_argument("--case-limit", type=int, help="run at most this many selected cases")
    parser.add_argument("--resume", action="store_true", help="reuse matching successful case records")
    parser.add_argument("--unsupported-case", action="append", default=[],
                        help="reviewed case ID whose backend rejection should be recorded as unsupported")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--gpu-uuid", help="explicit GPU UUID; --run may also use a single UUID in CUDA_VISIBLE_DEVICES")
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--max-idle-memory-mib", type=float, default=64)
    parser.add_argument("--idle-graphics-name", action="append", default=[],
                        help="explicit exact name of a permitted idle G-only display process; repeatable")
    args = parser.parse_args()
    gpu_uuid = args.gpu_uuid or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    uuid_pattern = r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    if (args.run or args.gpu_uuid) and not re.fullmatch(uuid_pattern, gpu_uuid):
        parser.error("execution requires --gpu-uuid or CUDA_VISIBLE_DEVICES containing exactly one full GPU UUID")
    if not 0 < args.timeout < float("inf") or not 0 < args.poll_seconds <= 10 or not 0 <= args.max_idle_memory_mib < float("inf"):
        parser.error("timeout must be finite/positive, poll-seconds in (0,10], and idle memory finite/nonnegative")
    selected = cases(args.phase)
    valid_names = {case.name for case in selected}
    if set(args.case + args.unsupported_case) - valid_names:
        parser.error("unknown case ID for the selected phase")
    selected = [case for case in selected if not args.case or case.name in args.case]
    if args.case_limit is not None:
        if args.case_limit < 1:
            parser.error("case-limit must be positive")
        selected = selected[:args.case_limit]
    build = args.build.resolve()
    env = dict(os.environ, CUBUTTERFLY_COMPILE_MODE="research")
    if re.fullmatch(uuid_pattern, gpu_uuid):
        env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    if not args.run:
        for case in selected:
            print(f"# {case.name}")
            prefix = "CUBUTTERFLY_COMPILE_MODE=research "
            if re.fullmatch(uuid_pattern, gpu_uuid):
                prefix += "CUDA_VISIBLE_DEVICES=" + gpu_uuid + " "
            print(prefix + shlex.join(command(case, build)))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    driver_lock = (args.output / "driver.lock").open("a")
    try:
        fcntl.flock(driver_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 73
    identities = {}
    for name in ("cubutterfly_bench", "cuntt_bench", "libcuntt.so", "libcuntt.a"):
        binary = build / name
        if binary.is_file():
            identities[name] = file_digest(binary)
    environment = {key: value for key, value in env.items()
                   if key.startswith("CUBUTTERFLY_") or key in
                   ("CUDA_VISIBLE_DEVICES", "XDG_CACHE_HOME", "LD_LIBRARY_PATH", "PYTHONPATH", "PATH")}
    for key in ("CUBUTTERFLY_HARDWARE_MODEL", "CUBUTTERFLY_COST_MODEL"):
        model = Path(env.get(key, ""))
        if model.is_file():
            identities[key] = file_digest(model)
    planner_inputs = planner_input_identities(env)
    idle_graphics_names = tuple(sorted(set(args.idle_graphics_name)))
    planner_inputs["idle_graphics_names"] = list(idle_graphics_names)
    results = []
    def stopped(number, frame):
        raise DriverStopped(f"driver received signal {number}")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stopped)
    write_status(args.output, selected, results, "running", gpu_uuid)
    for case in selected:
        case_directory = args.output / case.name
        case_directory.mkdir(exist_ok=True)
        result_path = case_directory / "result.json"
        record = {"case": case.name, "phase": case.phase, "command": command(case, build),
                  "compile_mode": "research", "status": "error", "build_identities": identities,
                  "environment": environment, "planner_input_identities": planner_inputs,
                  "idle_graphics_names": list(idle_graphics_names)}
        if args.resume and result_path.is_file():
            previous = json.loads(result_path.read_text())
            if previous.get("status") == "passed" and profile_resume_matches(previous) and all(
                previous.get(key) == record[key] for key in
                ("command", "build_identities", "environment", "planner_input_identities")
            ):
                stderr_path = case_directory / "stderr.txt"
                annotate_result(previous, stderr_path.read_text() if stderr_path.is_file() else "")
                result_path.write_text(json.dumps(previous, indent=2) + "\n")
                results.append(previous)
                print(f"reused: {case.name}", flush=True)
                continue
        started = time.monotonic()
        interruption = None
        try:
            completed = guarded_run(record["command"], env, case_directory, gpu_uuid,
                                    args.timeout, args.poll_seconds, args.max_idle_memory_mib, idle_graphics_names)
            record["returncode"] = completed.returncode
            record["gpu_observations"] = getattr(completed, "gpu_observations", None)
            if completed.returncode:
                if case.name in args.unsupported_case and "error:" in completed.stderr and "cuda" not in completed.stderr.lower() and "verification mismatch" not in completed.stderr:
                    record["status"] = "unsupported"
                else:
                    raise ValueError("benchmark failed; inspect original stderr/stdout")
            else:
                record["metrics"] = parse_row(completed.stdout)
                annotate_result(record, completed.stderr)
                check_row(case, record["metrics"])
                record["status"] = "passed"
        except GPUInterruption as error:
            interruption = error
            record.update(status=error.status, snapshot=error.snapshot, retryable=True)
        except (OSError, ValueError, KeyError, StopIteration, RuntimeError, subprocess.SubprocessError) as error:
            record["error"] = str(error)
            if isinstance(error, DriverStopped):
                record["status"] = "stopped-manually"
        record["process_wall_seconds"] = time.monotonic() - started
        results.append(record)
        result_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"{record['status']}: {case.name}", flush=True)
        (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        if interruption is not None or record["status"] == "stopped-manually":
            write_status(args.output, selected, results, record["status"], gpu_uuid,
                         retryable=interruption is not None, case=case.name)
            return 1
        write_status(args.output, selected, results, "running", gpu_uuid, case=case.name)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    # Unsupported cells are missing coverage, never counted as passes.
    success = all(result["status"] == "passed" for result in results)
    write_status(args.output, selected, results, "complete" if success else "failed", gpu_uuid)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
