#!/usr/bin/env python3
"""Build and run native/NTT-adapted CKKS complete calls using pinned archives.

Example: python experiments/workloads/run_heongpu_ckks_e2e.py --sizes 4096,16384
  --batches 1,8 --warmup 3 --repetitions 10 --output results/ckks_e2e.csv
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "results/e13_assets"
CONDA_CMAKE = Path("cmake")
GPU_MONITOR_INTERVAL = 1.0


def bootstrap_profile(profile_name: str = "primary") -> dict:
    """Return one explicit bootstrap contract, shared with the generated header and reports.

    ``primary`` is the pinned N=65536 example.  ``bootstrap_v2_mid`` keeps the
    same Q/P chain at N=32768 for an intermediate capacity probe, while
    ``bootstrap_v2_small`` instantiates the independent reduced-N smoke at
    N=16384.  ``bootstrap_v1_small`` is the separate N=4096 v1 example with
    its own bit-size chain and legacy bootstrap configuration.  The profiles
    stay separate so a reduced workload can never be mistaken for the primary
    application result.
    """
    profiles = json.loads((ROOT / "experiments/workloads/ckks_tfhe_workloads.json").read_text())["ckks"]
    if profile_name not in ("primary", "bootstrap_v2_mid", "bootstrap_v2_small", "bootstrap_v1_small"):
        raise ValueError(f"unknown bootstrap profile: {profile_name}")
    definition = profiles[profile_name]
    profile = dict(profiles[definition["inherits"]]) if "inherits" in definition else {}
    profile.update(definition)
    for prefix in ("Q", "P"):
        hex_key = f"{prefix}_moduli_hex"
        if hex_key in profile:
            values = [int(value, 0) for value in profile[hex_key]]
            if len(values) != profile[f"{prefix.lower()}_primes"] or any(
                    value <= 2 or value >= 2**62 or (value - 1) % (2 * profile["N"])
                    for value in values) or len(set(values)) != len(values):
                raise ValueError(f"invalid bootstrap {prefix} modulus contract")
            profile[f"{prefix}_bits"] = ":".join(str(value.bit_length()) for value in values)
        else:
            bits = profile.get(f"{prefix}_bits")
            if not isinstance(bits, list) or len(bits) != profile[f"{prefix.lower()}_primes"]:
                raise ValueError(f"invalid bootstrap {prefix} bit-size contract")
            values = []
            profile[f"{prefix}_bits"] = ":".join(str(int(value)) for value in bits)
        profile[f"{prefix}_values"] = values
    return profile


def bootstrap_header(profile_name: str = "primary") -> str:
    profile = bootstrap_profile(profile_name)
    lines = ["#pragma once", "#include <vector>", "namespace ckks_bootstrap_profile {"]
    for prefix in ("Q", "P"):
        values = ",".join(f"{value}ULL" for value in profile[f"{prefix}_values"])
        lines += [f"inline const std::vector<Data64> {prefix.lower()}_moduli{{{values}}};",
                  f'inline constexpr const char* {prefix.lower()}_bits = "{profile[f"{prefix}_bits"]}";']
    for key in ("N", "secret_weight", "ephemeral_secret_weight", "scale_bits",
                "evalmod_log_message_ratio", "slots_to_coeffs_level", "coeffs_to_slots_level"):
        lines.append(f"inline constexpr int {key} = {int(profile[key])};")
    config = profile.get("bootstrap_config", {})
    for key, default in (("ctos_piece", 0), ("stoc_piece", 0), ("taylor_number", 0)):
        lines.append(f"inline constexpr int bootstrap_{key} = {int(config.get(key, default))};")
    lines.append(f"inline constexpr bool bootstrap_less_key_mode = {str(bool(config.get('less_key_mode', False))).lower()};")
    return "\n".join(lines + ["}", ""])


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_identity(path: Path) -> dict:
    identity = {"path": str(path), "sha256": None}
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        identity.update(sha256=digest.hexdigest(), size_bytes=path.stat().st_size)
    except OSError as error:
        identity["error"] = str(error)
    return identity


def read_only_probe(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, timeout=15, check=False)
        return {"argv": command, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": command, "returncode": None, "error": str(error)}


def target_gpu_uuid(env: dict[str, str]) -> str:
    value = env.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value):
        raise ValueError("performance runs require CUDA_VISIBLE_DEVICES to name exactly one full GPU UUID; GPU indices are not CUDA device identities")
    return value


def parse_gpu_processes(stdout: str, gpu_uuid: str, own_pid: int | None = None) -> list[dict]:
    processes = []
    for row in csv.reader(stdout.splitlines()):
        if not row or not any(field.strip() for field in row):
            continue
        if len(row) != 4:
            raise ValueError("malformed compute-process row")
        uuid, pid_text, name, memory = (field.strip() for field in row)
        if uuid.lower() != gpu_uuid.lower():
            continue
        pid = int(pid_text)
        if pid <= 0:
            raise ValueError("invalid compute-process PID")
        if pid != own_pid:
            processes.append({"gpu_uuid": uuid, "pid": pid,
                              "process_name": name, "used_memory": memory})
    return processes


def gpu_observation(gpu_uuid: str, own_pid: int | None = None) -> dict:
    probe = read_only_probe([
        "nvidia-smi", f"--id={gpu_uuid}",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits"])
    observation = {"observed_utc": utc_now(), "gpu_uuid": gpu_uuid,
                   "excluded_pid": own_pid, "probe": probe,
                   "query_ok": False, "foreign_processes": []}
    if probe.get("returncode") != 0:
        observation["error"] = "target GPU compute-process query failed"
        return observation
    try:
        observation["foreign_processes"] = parse_gpu_processes(probe["stdout"], gpu_uuid, own_pid)
    except (AttributeError, KeyError, TypeError, ValueError, csv.Error) as error:
        observation["error"] = str(error)
        return observation
    observation["query_ok"] = True
    return observation


def stop_own_process(process) -> dict:
    action = {"pid": process.pid, "terminate_requested": False, "kill_requested": False}
    if process.poll() is None:
        try:
            process.terminate()
            action["terminate_requested"] = True
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            action["kill_requested"] = True
            process.wait()
    return action


def run_with_evidence(command: list[str], env: dict[str, str], output: Path,
                      source: Path, cuntt: Path, archive: Path,
                      original_archive: Path, skip_build: bool) -> int:
    log_path = output.with_suffix(".log")
    metadata_path = output.with_suffix(".metadata.json")
    if output in (log_path, metadata_path):
        raise ValueError("output path collides with its evidence sidecar")
    selection_path = Path(str(output) + ".selection.csv")
    samples_path = Path(str(output) + ".samples.csv")
    if any(path.exists() for path in (output, log_path, metadata_path, selection_path, samples_path)):
        raise FileExistsError("refusing to overwrite existing benchmark output or evidence sidecars")
    source_files = [ROOT / "adapters/heongpu" / name for name in
                    ("ntt_adapter.hpp", "ntt_adapter.cu", "ckks_operator_overlay.cu",
                     "graph_memory.hpp", "graph_memory.cpp", "ntt_capture.hpp",
                     "rns_stage_batch.hpp", "rns_stage_batch.cu",
                     "ntt_graph_batch.hpp", "ntt_graph_batch.cpp", "graph_schedule.hpp",
                     "rns_preparation.hpp")]
    source_files += [Path(__file__).resolve(),
                     ROOT / "experiments/workloads/ckks_tfhe_workloads.json",
                     Path(command[0]).parent / "wrapper/ckks_bootstrap_parameters.hpp",
                     ROOT / "include/cuntt/rns_ntt.hpp",
                     ROOT / "include/cuntt/rns_tuning.hpp", ROOT / "src/rns_tuning.cpp",
                     ROOT / "src/rns_ntt.cu", ROOT / "src/ntt_hybrid2d.cuh",
                     ROOT / "src/rns_fused_ctgs.cuh",
                     ROOT / "experiments/workloads/heongpu_ckks_e2e_bench.cu",
                     ROOT / "experiments/workloads/gpu_thermal_guard.hpp",
                     source / "src/lib/host/ckks/operator.cu"]
    if "--stream-ordering-fix" in command:
        staged = Path(command[0]).parent / "wrapper/heongpu-stream-fixed"
        source_files += [ROOT / "adapters/heongpu/patches/ckks_v2_stream_ordering.patch",
                         staged / "stream_ordering.json", staged / "src/lib/host/ckks/operator.cu",
                         staged / "src/include/heongpu/host/ckks/operator.cuh"]
    git = read_only_probe(["git", "-C", str(source), "rev-parse", "HEAD"])
    gpuntt_source = source / "thirdparty/GPU-NTT"
    gpuntt_git = read_only_probe(["git", "-C", str(gpuntt_source), "rev-parse", "HEAD"])
    metadata = {
        "schema_version": 2,
        "wrapper_argv": [sys.executable, *sys.argv],
        "benchmark_argv": command,
        "cwd": str(Path.cwd()),
        "started_utc": utc_now(),
        "completed_utc": None,
        "returncode": None,
        "status": "running",
        "performance_eligible": False,
        "gpu_observations": [],
        "monitor_scope": {
            "target_gpu_uuid": None,
            "interval_seconds": GPU_MONITOR_INTERVAL,
            "own_process_exclusion": "benchmark Popen.pid only",
            "coverage": "Preflight, sampled monitoring and final target compute-process checks; sampling cannot detect every transient workload."},
        "output": str(output),
        "log": str(log_path),
        "stderr_merged_into_stdout": True,
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
        "CUBUTTERFLY_COMPILE_MODE": env["CUBUTTERFLY_COMPILE_MODE"],
        "rns_preparation": {key: env.get(key) for key in (
            "CUBUTTERFLY_RNS_CACHE", "CUBUTTERFLY_RNS_CACHE_POLICY",
            "CUBUTTERFLY_RNS_TUNE_REPETITIONS", "CUBUTTERFLY_RNS_REQUEST_EXPORT")},
        "heongpu_source": str(source),
        "heongpu_pinned_git_sha": git.get("stdout", "").strip() if git.get("returncode") == 0 else None,
        "heongpu_git_probe": git,
        "gpuntt_git_probe": gpuntt_git,
        "gpuntt_source_git_sha": gpuntt_git.get("stdout", "").strip() if gpuntt_git.get("returncode") == 0 else None,
        "source_files": [file_identity(path) for path in source_files],
        "artifacts": {
            "libcuntt": file_identity(cuntt),
            "heongpu_pinned_archive": file_identity(original_archive),
            "heongpu_linked_archive": file_identity(archive),
            "gpuntt_linked_archive": file_identity(original_archive.parent.parent / "thirdparty/GPU-NTT/src/libntt-1.0.a"),
            "benchmark_binary": file_identity(Path(command[0])),
        },
        "skip_build": skip_build,
        "stream_ordering": "patched_identically_for_both_backends" if "--stream-ordering-fix" in command else "pinned_upstream",
        "thermal_protocol": ("cool-before-selection-and-measurement-rounds-v2"
                             if "--thermal-guard" in command else "disabled"),
        "source_binary_correspondence": "not_verified",
        "source_binary_note": "Source hashes describe files observed before execution; correspondence with the hashed binary has not been verified.",
    }

    def save_metadata() -> None:
        temporary = metadata_path.with_name(metadata_path.name + ".tmp")
        temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        temporary.replace(metadata_path)

    try:
        gpu_uuid = target_gpu_uuid(env)
    except ValueError as error:
        metadata.update(status="failed", returncode=2, error=str(error), completed_utc=utc_now())
        save_metadata()
        print(str(error), file=sys.stderr, flush=True)
        return 2
    metadata["monitor_scope"]["target_gpu_uuid"] = gpu_uuid
    metadata["nvidia_smi"] = read_only_probe([
        "nvidia-smi", f"--id={gpu_uuid}",
        "--query-gpu=uuid,name,memory.total,driver_version,pci.bus_id",
        "--format=csv,noheader,nounits"])
    before = gpu_observation(gpu_uuid)
    metadata["compute_processes_before"] = before["probe"]
    metadata["gpu_observations"].append(before)
    identity_ok = metadata["nvidia_smi"].get("returncode") == 0
    if not identity_ok or not before["query_ok"] or before["foreign_processes"]:
        busy = identity_ok and before["query_ok"] and bool(before["foreign_processes"])
        metadata.update(status="blocked_gpu_busy" if busy else "failed",
                        returncode=1, completed_utc=utc_now(),
                        error="target GPU is occupied" if busy else before.get("error", "target GPU identity query failed"))
        save_metadata()
        print(metadata["error"], file=sys.stderr, flush=True)
        return 1
    save_metadata()
    print("+", " ".join(command), flush=True)
    process = None
    monitor = None
    monitor_stop = threading.Event()
    monitor_state = {"violation": None}

    def observe_during_run() -> None:
        try:
            while not monitor_stop.wait(GPU_MONITOR_INTERVAL):
                observation = gpu_observation(gpu_uuid, process.pid)
                metadata["gpu_observations"].append(observation)
                if not observation["query_ok"] or observation["foreign_processes"]:
                    monitor_state["violation"] = observation
                    metadata["child_stop"] = stop_own_process(process)
                    save_metadata()
                    return
                save_metadata()
        except Exception as error:
            monitor_state["violation"] = {"query_ok": False, "error": str(error)}
            metadata["child_stop"] = stop_own_process(process)

    try:
        with log_path.open("wb") as log:
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, env=env, bufsize=0)
                metadata["benchmark_pid"] = process.pid
                monitor = threading.Thread(target=observe_during_run, name="ckks-gpu-monitor", daemon=True)
                monitor.start()
                while True:
                    chunk = process.stdout.read(65536)
                    if not chunk:
                        break
                    log.write(chunk)
                    log.flush()
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                metadata["returncode"] = process.wait()
                metadata["status"] = "completed" if process.returncode == 0 else "failed"
            except OSError as error:
                metadata["error"] = str(error)
                metadata["status"] = "failed"
                if process is not None:
                    metadata["child_stop"] = stop_own_process(process)
                metadata["returncode"] = process.returncode if process is not None else 127
                if metadata["returncode"] == 0:
                    metadata["returncode"] = 1
                message = f"benchmark execution failed: {error}\n"
                log.write(message.encode())
                log.flush()
                print(message, file=sys.stderr, end="", flush=True)
    except BaseException as error:
        metadata["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        metadata["error"] = str(error)
        if process is not None:
            metadata["child_stop"] = stop_own_process(process)
        metadata["returncode"] = process.returncode if process is not None else 1
        raise
    finally:
        monitor_stop.set()
        if monitor is not None:
            monitor.join()
        metadata["completed_utc"] = utc_now()
        after = gpu_observation(gpu_uuid, process.pid if process is not None else None)
        metadata["compute_processes_after"] = after["probe"]
        metadata["gpu_observations"].append(after)
        violation = monitor_state["violation"]
        if not after["query_ok"] or after["foreign_processes"]:
            violation = violation or after
        if violation is not None:
            metadata["gpu_violation"] = violation
            metadata["status"] = "invalid_gpu_contention" if violation.get("query_ok") else "failed"
            metadata["error"] = "target GPU occupancy changed or monitoring failed; performance evidence is unusable"
            if not metadata["returncode"]:
                metadata["returncode"] = 1
        metadata["measurement_kind"] = ("preparation-diagnostic" if "--prepare-only" in command else
                                        "profiler-diagnostic" if "--trace-measurement" in command else "ordinary")
        metadata["performance_eligible"] = (metadata["status"] == "completed" and metadata["returncode"] == 0
                                             and metadata["measurement_kind"] == "ordinary")
        for name, path in (("selection_samples", selection_path), ("holdout_samples", samples_path)):
            if path.is_file():
                metadata["artifacts"][name] = file_identity(path)
        save_metadata()
    return metadata["returncode"]


def prepare_archive(original: Path, destination: Path) -> None:
    """Exclude precisely the CKKS operator member, preserving the pinned archive."""
    members = subprocess.check_output(["ar", "t", str(original)], text=True).splitlines()
    matches = []
    with tempfile.TemporaryDirectory(prefix="cubutterfly-ckks-archive-") as directory:
        for occurrence in range(1, members.count("operator.cu.o") + 1):
            subprocess.run(["ar", "xN", str(occurrence), str(original), "operator.cu.o"],
                           cwd=directory, check=True)
            symbols = subprocess.check_output(
                ["nm", "-C", "--defined-only", str(Path(directory) / "operator.cu.o")], text=True)
            if "heongpu::HEOperator<(heongpu::Scheme)1>" in symbols:
                matches.append(occurrence)
    if len(matches) != 1:
        raise RuntimeError(f"expected one CKKS operator archive member, found {matches}")
    shutil.copyfile(original, destination)
    run(["ar", "dN", str(matches[0]), str(destination), "operator.cu.o"])
    run(["ranlib", str(destination)])


def stream_source(source: Path, build: Path, *, prepare: bool) -> Path:
    """Stage the reviewed stream fix without modifying the pinned checkout."""
    patch = ROOT / "adapters/heongpu/patches/ckks_v2_stream_ordering.patch"
    relative = (Path("src/lib/host/ckks/operator.cu"), Path("src/include/heongpu/host/ckks/operator.cuh"))
    staged = build / "wrapper/heongpu-stream-fixed"
    identity = {str(p): file_identity(p)["sha256"] for p in [patch, *(source / p for p in relative)]}
    marker = staged / "stream_ordering.json"
    if not prepare:
        if not marker.is_file() or json.loads(marker.read_text()).get("inputs") != identity:
            raise ValueError("stream ordering patch/source changed or missing; rebuild before --skip-build")
        recorded = json.loads(marker.read_text())["outputs"]
        if any(not (staged / p).is_file() or file_identity(staged / p)["sha256"] != recorded[str(p)] for p in relative):
            raise ValueError("staged stream source changed; rebuild before --skip-build")
        return staged
    for p in relative:
        (staged / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / p, staged / p)
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patch)], cwd=staged, check=True)
    marker.write_text(json.dumps({"inputs": identity, "outputs": {
        str(p): file_identity(staged / p)["sha256"] for p in relative}}, indent=2) + "\n")
    return staged


def cmake_text(source: Path, cached: Path, cuntt: Path, archive: Path, patched: Path | None = None) -> str:
    deps = cached / "_deps"
    prefix = Path("/tmp/heongpu-e13-prefix-20261003")
    gf2x = Path("/tmp/heongpu-e13-gf2x/root/usr/lib/x86_64-linux-gnu/libgf2x.so.3.0.0")
    includes = [source / "src/include", source / "src/include/heongpu",
                source / "src/include/heongpu/host", source / "src/include/heongpu/kernel",
                source / "src/include/heongpu/util", deps / "rmm-src/cpp/include",
                deps / "rmm-build/include", deps / "nvtx3-src/c/include",
                deps / "rapids_logger-src/include", prefix / "usr/include",
                prefix / "usr/include/x86_64-linux-gnu", ROOT / "include",
                ROOT / "adapters/heongpu", archive.parent / "wrapper"]
    if patched is not None:
        includes.insert(0, patched / "src/include")
    lines = ["cmake_minimum_required(VERSION 3.26)",
             "project(cubutterfly_ckks_runner LANGUAGES CXX CUDA)",
             "find_package(CUDAToolkit REQUIRED)", "find_package(Threads REQUIRED)",
             "find_package(ZLIB REQUIRED)",
             "find_package(OpenSSL REQUIRED COMPONENTS SSL Crypto)"]
    for name, project, library in [("ntt", "GPU-NTT", "libntt-1.0.a"),
                                   ("fft", "GPU-FFT", "libfft-1.0.a"),
                                   ("rng", "RNGonGPU", "librngongpu-1.0.a")]:
        includes.append(source / f"thirdparty/{project}/src/include")
        lines += [f"add_library(heongpu_{name} STATIC IMPORTED GLOBAL)",
                  f'set_target_properties(heongpu_{name} PROPERTIES IMPORTED_LOCATION "{cached}/thirdparty/{project}/src/{library}")']
    for name, path in [("rmm", deps / "rmm-build/librmm.so"),
                       ("rapids", deps / "rapids_logger-build/librapids_logger.so"),
                       ("cuntt", cuntt)]:
        lines += [f"add_library({name} SHARED IMPORTED GLOBAL)",
                  f'set_target_properties({name} PROPERTIES IMPORTED_LOCATION "{path}")']
    lines += [f'find_library(HEONGPU_GMP gmp PATHS "{prefix}/usr/lib/x86_64-linux-gnu" REQUIRED)',
              f'find_library(HEONGPU_NTL ntl PATHS "{prefix}/usr/lib/x86_64-linux-gnu" REQUIRED)',
              "add_library(heongpu STATIC IMPORTED GLOBAL)",
              f'set_target_properties(heongpu PROPERTIES IMPORTED_LOCATION "{archive}")',
              "add_executable(cubutterfly_heongpu_ckks",
              f'  "{ROOT}/experiments/workloads/heongpu_ckks_e2e_bench.cu"',
              f'  "{ROOT}/adapters/heongpu/ntt_adapter.cu"',
              f'  "{ROOT}/adapters/heongpu/graph_memory.cpp"',
              f'  "{ROOT}/adapters/heongpu/rns_stage_batch.cu"',
              f'  "{ROOT}/adapters/heongpu/ntt_graph_batch.cpp"',
              f'  "{ROOT}/adapters/heongpu/ckks_operator_overlay.cu")',
              "target_include_directories(cubutterfly_heongpu_ckks PRIVATE " +
              " ".join(f'"{p}"' for p in includes) + ")",
              "target_compile_definitions(cubutterfly_heongpu_ckks PRIVATE CCCL_DISABLE_PDL CUB_DISABLE_NAMESPACE_MAGIC CUB_IGNORE_NAMESPACE_MAGIC_ERROR LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA THRUST_DISABLE_ABI_NAMESPACE THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP THRUST_IGNORE_ABI_NAMESPACE_ERROR)",
              f'target_compile_definitions(cubutterfly_heongpu_ckks PRIVATE CUBUTTERFLY_HEONGPU_CKKS_SOURCE="{patched or source}/src/lib/host/ckks/operator.cu")',
              f'target_compile_definitions(cubutterfly_heongpu_ckks PRIVATE CUBUTTERFLY_HEONGPU_STREAM_ORDERING_FIX={int(patched is not None)})',
              "target_compile_features(cubutterfly_heongpu_ckks PRIVATE cxx_std_17 cuda_std_17)",
              "target_link_options(cubutterfly_heongpu_ckks PRIVATE -Wl,--wrap=_ZNK7heongpu10MemoryPool19get_device_resourceEv -Wl,--wrap=_ZN7heongpu10MemoryPool8instanceEv -Wl,--wrap=cudaMemcpyAsync)",
              "target_link_libraries(cubutterfly_heongpu_ckks PRIVATE heongpu heongpu_ntt heongpu_fft heongpu_rng cuntt rmm rapids ${HEONGPU_GMP} ${HEONGPU_NTL}",
              f'  "{gf2x}" OpenSSL::SSL OpenSSL::Crypto CUDA::curand CUDA::cudart ZLIB::ZLIB Threads::Threads ${{CMAKE_DL_LIBS}})',
              "set_target_properties(cubutterfly_heongpu_ckks PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_RUNTIME_LIBRARY Shared",
              f'  BUILD_RPATH "{deps}/rmm-build;{deps}/rapids_logger-build;{gf2x.parent};{cuntt.parent}")',
              "add_executable(test_heongpu_graph_host_memory",
              f'  "{ROOT}/tests/test_heongpu_graph_host_memory.cu" "{ROOT}/adapters/heongpu/graph_memory.cpp")',
              "get_target_property(ckks_includes cubutterfly_heongpu_ckks INCLUDE_DIRECTORIES)",
              "get_target_property(ckks_libraries cubutterfly_heongpu_ckks LINK_LIBRARIES)",
              "get_target_property(ckks_link_options cubutterfly_heongpu_ckks LINK_OPTIONS)",
              "get_target_property(ckks_definitions cubutterfly_heongpu_ckks COMPILE_DEFINITIONS)",
              "target_include_directories(test_heongpu_graph_host_memory PRIVATE ${ckks_includes})",
              "target_compile_features(test_heongpu_graph_host_memory PRIVATE cxx_std_17 cuda_std_17)",
              "target_compile_definitions(test_heongpu_graph_host_memory PRIVATE ${ckks_definitions})",
              "target_link_libraries(test_heongpu_graph_host_memory PRIVATE ${ckks_libraries})",
              "target_link_options(test_heongpu_graph_host_memory PRIVATE ${ckks_link_options})",
              "set_target_properties(test_heongpu_graph_host_memory PROPERTIES CUDA_RUNTIME_LIBRARY Shared CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)",
              "add_executable(test_heongpu_ntt_adapter",
              f'  "{ROOT}/tests/test_heongpu_ntt_adapter.cu" "{ROOT}/adapters/heongpu/ntt_adapter.cu")',
              "target_include_directories(test_heongpu_ntt_adapter PRIVATE " +
              " ".join(f'"{p}"' for p in includes) + ")",
              "target_compile_features(test_heongpu_ntt_adapter PRIVATE cxx_std_17 cuda_std_17)",
              "target_link_libraries(test_heongpu_ntt_adapter PRIVATE heongpu_ntt cuntt CUDA::cudart)",
              f'set_target_properties(test_heongpu_ntt_adapter PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_RUNTIME_LIBRARY Static BUILD_RPATH "{cuntt.parent}")',
              "add_executable(test_heongpu_stage_batch",
              f'  "{ROOT}/tests/test_heongpu_stage_batch.cu" "{ROOT}/adapters/heongpu/rns_stage_batch.cu")',
              "target_include_directories(test_heongpu_stage_batch PRIVATE " +
              " ".join(f'"{p}"' for p in includes) + ")",
              "target_compile_features(test_heongpu_stage_batch PRIVATE cxx_std_17 cuda_std_17)",
              "target_link_libraries(test_heongpu_stage_batch PRIVATE heongpu_ntt cuntt CUDA::cudart)",
              f'set_target_properties(test_heongpu_stage_batch PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_RUNTIME_LIBRARY Shared BUILD_RPATH "{cuntt.parent}")',
              f'add_executable(test_heongpu_graph_schedule "{ROOT}/tests/test_heongpu_graph_schedule.cu")',
              f'target_include_directories(test_heongpu_graph_schedule PRIVATE "{ROOT}/adapters/heongpu")',
              'target_compile_features(test_heongpu_graph_schedule PRIVATE cxx_std_17 cuda_std_17)',
              'target_link_libraries(test_heongpu_graph_schedule PRIVATE CUDA::cudart)',
              'set_target_properties(test_heongpu_graph_schedule PROPERTIES CUDA_RUNTIME_LIBRARY Shared)',
              f'add_executable(rns_request_compare "{ROOT}/benchmarks/rns_request_compare.cu")',
              "target_include_directories(rns_request_compare PRIVATE " +
              " ".join(f'"{p}"' for p in includes) + ")",
              'target_compile_features(rns_request_compare PRIVATE cxx_std_17 cuda_std_17)',
              'target_link_libraries(rns_request_compare PRIVATE heongpu_ntt cuntt CUDA::cudart ${CMAKE_DL_LIBS})',
              f'set_target_properties(rns_request_compare PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_RUNTIME_LIBRARY Shared BUILD_RPATH "{cuntt.parent}")']
    return "\n".join(lines) + "\n"


def await_preparation(path: Path, command: list[str], env: dict[str, str], timeout: float,
                      cuntt: Path | None = None) -> None:
    """Continue only the identical successful full-call preparation, without rerunning tuning."""
    deadline = time.monotonic() + timeout
    while True:
        meta = json.loads(path.read_text())
        if meta["status"] != "running":
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"preparation still running at deadline: {path}")
        time.sleep(min(15, max(0.01, deadline - time.monotonic())))
    if (meta["status"] != "completed" or meta.get("returncode") != 0 or
            meta.get("measurement_kind") != "preparation-diagnostic"):
        raise ValueError(f"preparation did not complete correctly: {path}")
    before = meta["benchmark_argv"]
    def value(argv, name):
        return argv[argv.index(name) + 1] if name in argv else None
    for name in ("--sizes", "--batches", "--parameter-set", "--aux-primes", "--ntt-execution",
                 "--seed", "--scenes", "--pool-max-fraction", "--stream-mode"):
        if value(before, name) != value(command, name):
            raise ValueError(f"preparation differs in {name}: {path}")
    if (value(before, "--prepare-backend") != "both" or
            ("--stream-ordering-fix" in before) != ("--stream-ordering-fix" in command) or
            meta["CUDA_VISIBLE_DEVICES"] != env.get("CUDA_VISIBLE_DEVICES")):
        raise ValueError(f"preparation backend, stream repair or GPU differs: {path}")
    for name in ("CUBUTTERFLY_RNS_CACHE", "CUBUTTERFLY_RNS_TUNE_REPETITIONS"):
        if not env.get(name) or meta["rns_preparation"].get(name) != env[name]:
            raise ValueError(f"prepared cache protocol differs in {name}: {path}")
    for key in ("benchmark_binary", "libcuntt"):
        expected = meta["artifacts"][key]
        actual = Path(command[0]) if key == "benchmark_binary" else cuntt or Path(expected["path"])
        if not expected.get("sha256") or file_identity(actual)["sha256"] != expected["sha256"]:
            raise ValueError(f"prepared artifact changed: {key}")
    print(f"Preparation verified; continuing cache-only Graph measurement: {path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heongpu-source", type=Path, default=ASSETS / "sources/HEonGPU")
    parser.add_argument("--cached-build", type=Path, default=ASSETS / "heongpu-build-e13")
    parser.add_argument("--build-dir", type=Path, default=ASSETS / "heongpu-build-ckks-runner")
    parser.add_argument("--cuntt-library", type=Path, default=ROOT / "build-contract-20261017/libcuntt.so")
    parser.add_argument("--sizes", default="4096,16384")
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--graph-replays", type=int, default=1)
    parser.add_argument("--schedule-rounds", type=int, default=7)
    parser.add_argument("--schedule-replays", type=int, default=5)
    parser.add_argument("--scenes", help="Explicit comma-separated subset of the selected parameter profile's scenes")
    parser.add_argument("--parameter-set", choices=["short-chain", "bootstrap-v1-small", "bootstrap-v2", "bootstrap-v2-mid", "bootstrap-v2-small"], default="short-chain")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Check full-call repeatability, input mutation and memory phases without timing")
    parser.add_argument("--prepare-backend", choices=["native", "cubutterfly", "both"], default="both")
    parser.add_argument("--stream-mode", choices=["nonblocking", "default"], default="nonblocking",
                        help="Default-stream control is available only with --prepare-only")
    parser.add_argument("--stream-ordering-fix", action="store_true",
                        help="Build both backends with the same reviewed HEonGPU stream-dependency repair")
    parser.add_argument("--stage-batch", action="store_true",
                        help="Compare matched same-stage NTT batching in full independent ciphertext graphs")
    parser.add_argument("--stage-batch-only", action="store_true",
                        help="Retain only the selected stage backend graph and compare it with an eager native call")
    parser.add_argument("--stage-backend", choices=["native", "cubutterfly"], default="cubutterfly",
                        help="Backend for an isolated --stage-batch-only graph; native and cuButterfly runs can be paired offline")
    parser.add_argument("--auto-schedule", action="store_true",
                        help="Measure complete loop/batched candidates, then independently time the selected Graph")
    parser.add_argument("--trace-measurement", action="store_true",
                        help="Emit profiler capture boundary and variant NVTX labels; result is diagnostic only")
    parser.add_argument("--thermal-guard", action="store_true",
                        help="Cool outside timing before each selection/measurement round; reject thermal throttling during samples")
    parser.add_argument("--aux-primes", type=int, choices=[1, 2, 3, 5], default=1)
    parser.add_argument("--pool-max-fraction", type=float, default=0.95,
                        help="HEonGPU device memory-pool maximum as a fraction; default 0.95")
    parser.add_argument("--ntt-execution", choices=["native-rns", "grouped"], default="native-rns")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--arch", default="80")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--after-preparation", type=Path,
                        help="Wait for a running preparation metadata file, verify its contract, then measure with required cache hits")
    parser.add_argument("--preparation-wait-seconds", type=float, default=7200)
    args = parser.parse_args()
    os.environ.setdefault("CUBUTTERFLY_COMPILE_MODE", "research")
    if args.after_preparation and (not args.skip_build or not args.graph or args.prepare_only or
                                   args.build_only or args.preparation_wait_seconds <= 0 or
                                   os.environ.get("CUBUTTERFLY_RNS_CACHE_POLICY") != "require"):
        parser.error("--after-preparation requires --skip-build --graph, cache policy require and a positive wait; no prepare-only/build-only")
    source, cached = args.heongpu_source.resolve(), args.cached_build.resolve()
    build, cuntt = args.build_dir.resolve(), args.cuntt_library.resolve()
    for path in [source / "src/lib/host/ckks/operator.cu", cached / "src/libheongpu.a", cuntt,
                 ROOT / "adapters/heongpu/ntt_adapter.hpp", ROOT / "adapters/heongpu/ntt_adapter.cu",
                 ROOT / "adapters/heongpu/ckks_operator_overlay.cu"]:
        if not path.is_file():
            parser.error(f"required pinned build input is missing: {path}")
    if args.warmup < 0 or args.repetitions < 1 or args.graph_replays < 1:
        parser.error("warmup must be nonnegative and repetitions/graph-replays positive")
    if not (0.0 < args.pool_max_fraction <= 1.0):
        parser.error("--pool-max-fraction must be in (0,1]")
    if not args.prepare_only and (args.prepare_backend != "both" or args.stream_mode != "nonblocking"):
        parser.error("--prepare-backend and default --stream-mode require --prepare-only")
    if args.prepare_only and (args.graph or args.stage_batch or args.auto_schedule or
                              args.trace_measurement or args.thermal_guard):
        parser.error("--prepare-only cannot be combined with Graph/timing options")
    if args.stage_batch and (not args.graph or args.ntt_execution != "native-rns"):
        parser.error("--stage-batch requires --graph --ntt-execution native-rns")
    if args.stage_batch_only and not args.stage_batch:
        parser.error("--stage-batch-only requires --stage-batch")
    if args.stage_backend == "native" and not args.stage_batch_only:
        parser.error("--stage-backend native requires --stage-batch-only")
    if args.stage_batch_only and args.auto_schedule:
        parser.error("--stage-batch-only does not support --auto-schedule")
    if args.thermal_guard and not args.graph:
        parser.error("--thermal-guard currently requires --graph")
    if args.auto_schedule and not args.stage_batch:
        parser.error("--auto-schedule requires --stage-batch")
    if args.trace_measurement and (not args.graph or not args.scenes or "," in args.scenes or
                                   "," in args.sizes or "," in args.batches):
        parser.error("--trace-measurement requires one explicit Graph scene, size and batch")
    if args.schedule_rounds < 1 or args.schedule_replays < 1:
        parser.error("schedule-rounds and schedule-replays must be positive")
    bootstrap_profiles = {
        "bootstrap-v1-small": "bootstrap_v1_small",
        "bootstrap-v2": "primary",
        "bootstrap-v2-mid": "bootstrap_v2_mid",
        "bootstrap-v2-small": "bootstrap_v2_small",
    }
    if args.parameter_set in bootstrap_profiles:
        profile_name = bootstrap_profiles[args.parameter_set]
        profile = bootstrap_profile(profile_name)
        if args.sizes != str(profile["N"]) or args.aux_primes != profile["p_primes"] or args.rotate:
            parser.error(f"{args.parameter_set} requires --sizes {profile['N']} --aux-primes {profile['p_primes']}, without --rotate")
    elif args.aux_primes == 5:
        parser.error("--aux-primes 5 requires a bootstrap-v2 parameter set")
    if args.scenes is not None:
        scenes = args.scenes.split(",")
        allowed = ({"regular_bootstrapping"} if args.parameter_set == "bootstrap-v1-small" else
                   {"regular_bootstrapping_v2"} if args.parameter_set in ("bootstrap-v2", "bootstrap-v2-mid", "bootstrap-v2-small") else
                   {"keyswitch", "square_relinearize_rescale", "square_rescale_keyswitch"})
        if args.rotate:
            allowed.add("rotate_rows_shift1")
        if len(set(scenes)) != len(scenes) or not all(s in allowed for s in scenes):
            parser.error("invalid/duplicate scenes; rotation requires --rotate")
    cmake = str(CONDA_CMAKE) if CONDA_CMAKE.is_file() else "cmake"
    # Keep generated CMake and build commands available for inspection/replay.
    wrapper = build / "wrapper"
    wrapper.mkdir(parents=True, exist_ok=True)
    header_profile = bootstrap_profiles.get(args.parameter_set, "primary")
    (wrapper / "ckks_bootstrap_parameters.hpp").write_text(bootstrap_header(header_profile))
    archive = build / "libheongpu_without_ckks_operator.a"
    patched = stream_source(source, build, prepare=not args.skip_build) if args.stream_ordering_fix else None
    (wrapper / "CMakeLists.txt").write_text(cmake_text(source, cached, cuntt, archive, patched))
    if not args.skip_build:
        prepare_archive(cached / "src/libheongpu.a", archive)
        run([cmake, "-S", str(wrapper), "-B", str(build),
             "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_CUDA_ARCHITECTURES={args.arch}",
             "-DCMAKE_PREFIX_PATH=/tmp/heongpu-e13-prefix-20261003"])
        run([cmake, "--build", str(build), "--target", "cubutterfly_heongpu_ckks",
             "test_heongpu_ntt_adapter", "test_heongpu_stage_batch", "test_heongpu_graph_schedule",
             "test_heongpu_graph_host_memory", "rns_request_compare", "-j", str(args.jobs)])
    if args.build_only:
        return 0
    binary = build / "cubutterfly_heongpu_ckks"
    if not binary.is_file():
        parser.error(f"benchmark binary is missing: {binary}")
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    runtime_paths = [cached / "_deps/rmm-build", cached / "_deps/rapids_logger-build", cuntt.parent,
                     Path("/tmp/heongpu-e13-gf2x/root/usr/lib/x86_64-linux-gnu"),
                     Path("/tmp/heongpu-e13-prefix-20261003/usr/lib/x86_64-linux-gnu")]
    env["LD_LIBRARY_PATH"] = ":".join(map(str, runtime_paths)) + ":" + env.get("LD_LIBRARY_PATH", "")
    command = [str(binary), "--sizes", args.sizes, "--batches", args.batches,
         "--warmup", str(args.warmup), "--repetitions", str(args.repetitions),
         "--graph-replays", str(args.graph_replays),
         "--schedule-rounds", str(args.schedule_rounds), "--schedule-replays", str(args.schedule_replays),
         "--aux-primes", str(args.aux_primes),
         "--pool-max-fraction", str(args.pool_max_fraction),
         "--parameter-set", args.parameter_set,
         "--ntt-execution", args.ntt_execution,
         "--seed", str(args.seed), "--output", str(args.output.resolve()),
         *(["--prepare-only", "--prepare-backend", args.prepare_backend] if args.prepare_only else []),
         "--stream-mode", args.stream_mode,
         *(["--stream-ordering-fix"] if args.stream_ordering_fix else []),
         *(["--rotate"] if args.rotate else []), *(["--graph"] if args.graph else []),
         *(["--stage-batch"] if args.stage_batch else []),
         *(["--stage-batch-only"] if args.stage_batch_only else []),
         "--stage-backend", args.stage_backend,
         *(["--auto-schedule"] if args.auto_schedule else []),
         *(["--trace-measurement"] if args.trace_measurement else []),
         *(["--scenes", args.scenes] if args.scenes else []),
         *(["--thermal-guard"] if args.thermal_guard else [])]
    if args.after_preparation:
        await_preparation(args.after_preparation.resolve(), command, env, args.preparation_wait_seconds, cuntt)
    return run_with_evidence(command,
         env, args.output.resolve(), source, cuntt, archive,
         cached / "src/libheongpu.a", args.skip_build)


if __name__ == "__main__":
    raise SystemExit(main())
