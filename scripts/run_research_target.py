#!/usr/bin/env python3
"""Portable, finite research campaign: CPU preparation, target calibration, comparison.

Run --mode prepare/precompile without a GPU. Run --mode run on the target machine.
Existing calibration/search/benchmark implementations remain the execution engines.
This campaign certifies its declared basis and cells, not the entire design space.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("core", "batch_precision", "crypto", "operator_extensions")
BINARIES = ("cuntt_bench", "cubutterfly_bench", "cubutterfly_plan_probe",
            "cubutterfly_plan_bench", "cubutterfly_stage_microbench",
            "cubutterfly_schedule_microbench", "cubutterfly_pipeline_microbench",
            "cuntt_hardware_microbench")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def freeze(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f"prepared input changed; use a new output directory: {path}")
    else:
        write(path, value)


def fallback_seed(workload):
    n = int(workload["logN"])
    stages = [min(8, n - start) for start in range(0, n, 8)]
    mapping = dict(schema_version=1, kind="butterfly", backend="shared-iterative",
                   stage_partition=stages, tile_threads=128, shared_layout="writer-aligned",
                   compute_unit="radix2", fft_core="scalar", local_exchange="shared")
    if workload["operator"] == "ntt":
        mapping = dict(schema_version=1, kind="ntt", backend="shared-iterative",
                       stage_partition=stages, threads_per_block=128, dataflow_layout="hermes-xor")
    elif workload["precision"] in ("fp16", "bf16"):
        mapping.pop("stage_partition")
        mapping.pop("shared_layout")
        mapping.update(backend="temporal-tile" if n <= 8 else "hierarchical",
                       local_stages=0 if n <= 8 else 8)
    return dict(semantics=workload, mapping=mapping,
                provenance=dict(kind="explicit", reuse="mapping proposal only; target verification required"))


def prepare(args):
    """Project existing mapping proposals without probing CUDA or importing timings."""
    from calibration_seeds import candidates_for_workload, canonical_mapping
    from hardware_registry import canonical_semantics
    import precompile_research as compiler
    from research_compile_requests import compile_request

    contracts = []
    for group in GROUPS:
        path = args.campaign / f"{group}.json"
        if path.exists():
            contracts.extend(read(path)["workloads"])
    # Composition runs cyclic cores; twists are linked in the isolated benchmark.
    for workload in read(args.campaign / "composed.json")["workloads"]:
        contracts.extend(dict(operator="ntt", precision=f"word{workload['word_bits']}",
            logN=workload["logN"], batch=1, modulus=str(p),
            direction=workload.get("direction", "forward")) for p in workload["moduli"])
    # Batch/direction/modulus do not define distinct compiled shared templates.
    # Keep them in the representative point; deduplicate canonical compiler requests.
    seeds = read(args.campaign / "mapping_seeds.json")["seeds"]
    seeds = [dict(s, semantics=canonical_semantics(s["semantics"]),
                  mapping=canonical_mapping(s["mapping"]), provenance={"kind": "explicit"}) for s in seeds]
    seeds += [dict(s, semantics=canonical_semantics(s["semantics"]))
              for s in map(fallback_seed, contracts)]
    snapshot = {"seeds": list({key([s["semantics"], s["mapping"]]): s for s in seeds}.values())}
    modules, seen, deferred = {}, set(), []
    for workload in contracts:
        structural = (workload["operator"], workload["precision"], workload["logN"],
                      workload.get("accumulation", "native"), workload.get("output_order", "natural"))
        if structural in seen:
            continue
        seen.add(structural)
        for candidate in candidates_for_workload(snapshot, workload):
            point = candidate["point"]
            try:
                request = compile_request(point)
            except ValueError as error:
                deferred.append(dict(point=point, reason=str(error)))
                continue
            if request is not None:
                identifier = key(request)
                modules[identifier] = dict(id=identifier, request=request,
                    representative_point=point, representative_cell=key(workload), cells=[key(workload)])
    # The initial calibration is known before device access too. Include both
    # block widths instead of JIT-compiling its 256-thread variants on the GPU window.
    for point in read(args.campaign / "stage_points.json")["points"]:
        projected = dict(point, mapping_json=point["mapping"])
        request = compile_request(projected)
        if request is not None:
            identifier = key(request)
            modules.setdefault(identifier, dict(id=identifier, request=request,
                representative_point=projected, representative_cell=key(point), cells=[key(point)],
                purpose="declared-stage-calibration-basis"))
    directory = args.output_dir / f"precompile-sm{args.cuda_arch}"
    manifest = dict(schema=compiler.SCHEMA, status="ready", complete=True, policy="research",
        candidate_source="seeds", coverage="Finite mapping proposals; runtime legality/search remains target-dependent.",
        root=str(ROOT), mathdx=str(args.mathdx_root), nvcc=str(args.nvcc), cache=str(args.cache),
        profile=dict(sm=args.cuda_arch, compute_capability=f"{args.cuda_arch // 10}.{args.cuda_arch % 10}"),
        device_identity=dict(status="unmeasured-compilation-target", sm=args.cuda_arch),
        source=compiler._source_identity(ROOT), modules=list(modules.values()), module_count=len(modules),
        template_inputs={str(p.relative_to(ROOT)): sha(p)
                         for directory in (ROOT / "src", ROOT / "include")
                         for p in sorted(directory.rglob("*"))
                         if p.is_file() and p.suffix in (".cuh", ".hpp", ".h")},
        deferred_projection=deferred, campaign={p.name: sha(p) for p in sorted(args.campaign.glob("*.json"))})
    freeze(directory / "modules.json", manifest)
    freeze(args.output_dir / "mapping_seeds.json", dict(schema="cubutterfly-mapping-seeds-v1", **snapshot))
    print(json.dumps(dict(prepared_modules=len(modules), deferred_projection=len(deferred),
                         manifest=str(directory / "modules.json"), gpu_access=False)), flush=True)
    return directory


def input_bytes(workload):
    shape = workload.get("shape")
    elements = math.prod(shape_extents(shape)) if shape is not None else 1 << int(workload["logN"])
    precision = workload.get("precision", f"word{workload.get('word_bits', 32)}")
    size = {"fp16": 2, "bf16": 2, "fp32": 4, "fp64": 8, "uint32": 4,
            "word32": 4, "word64": 8, "uint64": 8}[precision]
    if workload.get("operator") == "fft":
        size *= 2
    # Explicit layouts can have storage larger than the logical transform.
    stride = int(workload.get("element_stride", 1))
    distance = int(workload.get("batch_stride", 0)) or elements * stride
    footprint = (int(workload["batch"]) - 1) * distance + (elements - 1) * stride + 1
    return footprint * size * len(workload.get("moduli", [1]))


def shape_extents(shape):
    values = [int(v) for v in shape.split("x")] if isinstance(shape, str) else list(shape)
    if not values or any(type(v) is not int or v <= 0 for v in values):
        raise ValueError("shape extents must be positive integers")
    return values


def partition_memory(workloads, memory_bytes, factor=16, fraction=0.9):
    accepted, deferred = [], []
    for workload in workloads:
        estimated = input_bytes(workload) * factor
        if estimated <= memory_bytes * fraction:
            accepted.append(workload)
        else:
            deferred.append(dict(workload=workload, status="deferred-memory-envelope",
                estimated_bytes=estimated, input_bytes=input_bytes(workload),
                reason="Conservative workspace envelope; not proof of hardware infeasibility. Batch unchanged."))
    return accepted, deferred


def capture(command):
    return subprocess.run(list(map(str, command)), text=True, capture_output=True, check=True).stdout


def target_device(selector):
    rows = list(csv.reader(io.StringIO(capture(["nvidia-smi", "-i", selector,
        "--query-gpu=uuid,name,memory.total,driver_version", "--format=csv,noheader,nounits"])),
        skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 4 or not rows[0][0].startswith("GPU-"):
        raise ValueError("--gpu must select exactly one physical GPU")
    uuid, name, memory, driver = rows[0]
    return dict(uuid=uuid.strip(), name=name.strip(), memory_mib=int(float(memory)), driver=driver.strip())


def target_tag(device):
    name = re.sub(r"[^a-z0-9]+", "-", device["name"].lower()).strip("-")
    return f"{name}-{round(device['memory_mib'] / 1024)}gb-{device['uuid'][4:16]}"


def foreign_clients(uuid, allowed_group=None):
    rows = capture(["nvidia-smi", "-i", uuid, "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    foreign = []
    for row in rows.splitlines():
        if not row.strip():
            continue
        pid = int(row.strip())
        try:
            if allowed_group is not None and os.getpgid(pid) == allowed_group:
                continue
        except ProcessLookupError:
            continue
        foreign.append(pid)
    return foreign


def exclusive(uuid):
    clients = foreign_clients(uuid)
    if clients:
        raise RuntimeError(f"GPU busy ({uuid}): {clients}; rerun with --resume when idle")
    return True


@contextmanager
def target_lock(uuid):
    with (Path("/tmp") / f"cubutterfly-research-{uuid}.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def stop_group(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def guarded(command, uuid, log, quarantine=None):
    """Stop only this invocation if another GPU client arrives; reject its phase."""
    exclusive(uuid)
    log.parent.mkdir(parents=True, exist_ok=True)
    contamination = None
    with log.open("w") as stream:
        process = subprocess.Popen(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            while True:
                try:
                    clients = foreign_clients(uuid, process.pid)
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    contamination = f"GPU exclusivity monitor failed: {error}"
                    raise RuntimeError(contamination) from error
                if clients:
                    contamination = f"foreign GPU clients: {clients}"
                    raise RuntimeError(contamination)
                try:
                    code = process.wait(timeout=1)
                    try:
                        clients = foreign_clients(uuid, process.pid)
                    except (OSError, ValueError, subprocess.SubprocessError) as error:
                        contamination = f"GPU exclusivity monitor failed: {error}"
                        raise RuntimeError(contamination) from error
                    if clients:
                        contamination = f"foreign GPU clients after command: {clients}"
                        raise RuntimeError(contamination)
                    break
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            stop_group(process)
            if contamination and quarantine and quarantine.exists():
                archived = quarantine.with_name(quarantine.name + f".excluded-{time.time_ns()}")
                quarantine.rename(archived)
                write(archived / "excluded.json", dict(reason=contamination, log=str(log),
                    reuse=False, scope="Only the interrupted phase is excluded; earlier complete phases retained."))
            raise
    text = log.read_text()
    if code:
        raise RuntimeError(f"command exited {code}; checkpoint retained, see {log}\n{text[-1500:]}")
    return subprocess.CompletedProcess(command, code, stdout=text, stderr="")


def build_identity(args, device):
    sources = {}
    for directory in (ROOT / "src", ROOT / "include", ROOT / "scripts", ROOT / "benchmarks/crypto_ntt"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".cpp", ".hpp", ".h", ".in", ".txt"):
                sources[str(path.relative_to(ROOT))] = sha(path)
    return dict(schema="cubutterfly-portable-target-v1", device=device,
        binaries={name: sha(args.build_dir / name) for name in BINARIES},
        crypto_binary=sha(args.crypto_binary), sources=sources,
        baselines={name: dict(path=str(path), sha256=sha(path)) if path else None for name, path in
                   (("vkfft", args.vkfft_binary), ("gpuntt", args.gpuntt_binary), ("fht_python", args.fht_python))},
        campaign={p.name: sha(p) for p in sorted(args.campaign.glob("*.json"))},
        seeds_sha256=sha(args.output_dir / "mapping_seeds.json"),
        protocol=dict(trials=args.trials, warmup=args.warmup, repeat=args.repeat,
            search_budget=args.search_budget, finalists=3, seed_budget=4, verify_batches=0,
            stage_warmup=10, stage_repeat=20, stage_trials=3, stage_max_points=args.stage_max_points),
        memory_policy=dict(factor=args.memory_factor, fraction=.9), compile_mode="research",
        cuda_arch=args.cuda_arch, template_root=str(ROOT), build_dir=str(args.build_dir),
        mathdx_root=str(args.mathdx_root), cache=str(args.cache), nvcc=str(args.nvcc))


def calibrate(args, directory, device):
    import schedule_cost_model
    import pipeline_schedule_model
    calibration = directory / "calibration"
    log = directory / "logs"
    profile_path = calibration / "base_profile.json"
    if not profile_path.exists():
        hardware_path = calibration / "caps/hardware_profile.json"
        if not hardware_path.exists():
            guarded([sys.executable, ROOT / "scripts/initialize_hardware_profile.py", "--microbench",
                args.build_dir / "cuntt_hardware_microbench", "--output-dir", calibration / "caps"],
                device["uuid"], log / "capabilities.log", calibration)
        raw = capture([args.build_dir / "cubutterfly_plan_probe", "--hardware"])
        hardware = json.loads(raw)
        hardware = hardware.get("hardware", hardware)
        profile = read(hardware_path)
        measured_device = next(csv.DictReader(io.StringIO(capture(
            [args.build_dir / "cubutterfly_bench", "--device-identity"]))))
        profile.update(hardware=hardware, gpu_uuid=device["uuid"], driver_version=device["driver"],
                       global_memory_bytes=int(measured_device["global_memory_bytes"]))
        if (profile["device"] != hardware["device"] or profile["device"] != measured_device["device"]
                or profile["compute_capability"] != measured_device["compute_capability"]
                or profile["global_memory_bytes"] != int(hardware["memory_bytes"])):
            raise ValueError("hardware capabilities/probe identity mismatch")
        if int(profile["compute_capability"].replace(".", "")) != args.cuda_arch:
            raise ValueError("target SM differs from --cuda-arch; prepare a matching target build")
        write(profile_path, profile)
    profile = read(profile_path)
    def run(command):
        return guarded(command, device["uuid"], log / (Path(command[0]).name + ".log"), calibration)
    require = lambda: exclusive(device["uuid"])
    profile["scheduling"] = schedule_cost_model.calibrate(args.build_dir / "cubutterfly_schedule_microbench",
        calibration / "schedule_calibration.json", profile, run, require)
    profile["pipeline_scheduling"] = pipeline_schedule_model.calibrate(args.build_dir / "cubutterfly_pipeline_microbench",
        calibration / "pipeline_schedule_calibration.json", profile, run, require)
    # Immutable basis input; each acceptance group extends its own compatible checkpoint.
    freeze(calibration / "service_input.json", profile)
    guarded([sys.executable, ROOT / "scripts/stage_service_calibration.py", "--binary",
        args.build_dir / "cubutterfly_stage_microbench", "--checkpoint", calibration / "stage_calibration.json",
        "--profile", calibration / "service_input.json", "--points", args.campaign / "stage_points.json",
        "--mode", "full", "--warmup", "10", "--repeat", "20", "--trials", "3",
        "--max-points", str(args.stage_max_points)], device["uuid"], log / "stage-basis.log", calibration)
    stage = read(calibration / "stage_calibration.json")
    if stage.get("status") != "complete" or not stage.get("coverage", {}).get("validation_complete"):
        raise ValueError("declared stage basis is not validated; comparison has not started")
    return [profile_path, calibration / "stage_calibration.json", calibration / "schedule_calibration.json",
            calibration / "pipeline_schedule_calibration.json", calibration / "service_input.json"]


def acceptance_command(args, directory, group):
    calibration = directory / "calibration"
    command = [sys.executable, ROOT / "scripts/run_research_acceptance.py", "--build-dir", args.build_dir,
        "--output-dir", directory / group, "--workloads", directory / "inputs" / f"{group}.json",
        "--profile", calibration / "base_profile.json", "--stage-checkpoint", calibration / "stage_calibration.json",
        "--schedule-calibration", calibration / "schedule_calibration.json",
        "--pipeline-calibration", calibration / "pipeline_schedule_calibration.json",
        "--mapping-seeds", args.output_dir / "mapping_seeds.json", "--continue-unavailable", "--resume"]
    for name in ("trials", "warmup", "repeat", "search_budget", "vkfft_binary", "fht_python", "gpuntt_binary"):
        value = getattr(args, name)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    return command


def plan_command(args, workload):
    if (workload.get("operator", "fft") != "fft" or workload.get("direction", "forward") != "forward"
            or workload.get("length_mode", "standard") != "standard"
            or workload.get("placement", "out-of-place") != "out-of-place"):
        raise ValueError("portable plan performance currently requires full-output verified forward standard FFT")
    return [args.build_dir / "cubutterfly_plan_bench", "--operator", "fft", "--shape",
        "x".join(map(str, shape_extents(workload["shape"]))), "--batch", str(workload["batch"]),
        "--precision", workload["precision"], "--policy", "measure", "--warmup", str(args.warmup),
        "--repeat", str(args.repeat), "--trials", str(args.trials), "--compare-cufft", "--verify"]


def validate_plan(sample, workload, trials):
    logical = shape_extents(workload["shape"])
    # The public standard arbitrary-length path applies Bluestein to every
    # axis when any extent is non-power-of-two, including its power-of-two axes.
    physical = ([(1 << (2 * n - 2).bit_length()) for n in logical]
                if any(n & (n - 1) for n in logical) else logical)
    if sample.get("algorithm") == "cufft-direct":
        raise ValueError("measured internal plan unexpectedly delegated to direct cuFFT")
    if (sample.get("logical_shape") != "x".join(map(str, logical))
            or sample.get("physical_shape") != "x".join(map(str, physical))
            or int(sample.get("batch", 0)) != workload["batch"]
            or int(sample.get("verified_batches", 0)) != workload["batch"]
            or sample.get("correct") != "1" or int(sample.get("trials", 0)) != trials):
        raise ValueError("plan benchmark semantic/verification contract differs")
    for name in ("kernel_ms", "cufft_ms"):
        if not math.isfinite(float(sample.get(name, 0))) or float(sample.get(name, 0)) <= 0:
            raise ValueError("invalid plan timing")
        values = sample.get("trial_" + name, "").split(":")
        if len(values) != trials or any(not math.isfinite(float(v)) or float(v) <= 0 for v in values):
            raise ValueError("missing/nonpositive repeated plan measurements")


def run_plans(args, directory, device):
    output = directory / "plans"
    path = output / "plans.json"
    rows = read(path) if path.exists() else []
    workloads = read(directory / "inputs/plan_workloads.json")["workloads"]
    seen = set()
    for row in rows:
        workload = row.get("workload")
        if (workload not in workloads or key(workload) in seen or row.get("correct") is not True
                or row.get("gpu_uuid") != device["uuid"]
                or row.get("command") != list(map(str, plan_command(args, workload)))):
            raise ValueError("saved plan result has a different target/contract/protocol")
        validate_plan(row.get("sample", {}), workload, args.trials)
        seen.add(key(workload))
    for workload in workloads:
        if any(row["workload"] == workload for row in rows):
            continue
        command = plan_command(args, workload)
        response = guarded(command, device["uuid"], directory / "logs" / f"plan-{key(workload)[:12]}.log", output)
        line = next(line for line in reversed(response.stdout.splitlines()) if line.startswith("algorithm="))
        sample = dict(field.split("=", 1) for field in line.split(","))
        validate_plan(sample, workload, args.trials)
        rows.append(dict(workload=workload, sample=sample, command=list(map(str, command)),
                         gpu_uuid=device["uuid"], correct=True))
        write(path, rows)
    return [path] if path.exists() else []


def report(directory):
    state = read(directory / "campaign.json")
    result = dict(device=state["identity"]["device"], status=state["status"],
        full_migration_qualified=False, basis_status=state.get("phases", {}).get("calibration", {}).get("status", "pending"),
        groups={}, deferred_memory=read(directory / "inputs/memory.json") if (directory / "inputs/memory.json").exists() else [],
        scope="Finite same-contract campaign; missing baselines and unmeasured cells are not performance wins.")
    for group in GROUPS:
        path = directory / group / "summary.json"
        result["groups"][group] = read(path) if path.exists() else {"status": "pending"}
    for name, relative in (("composed", "composed/summary.json"), ("plans", "plans/plans.json")):
        if (directory / relative).exists():
            result[name] = read(directory / relative)
    write(directory / "report.json", result)
    return result


def run_target(args):
    device = target_device(args.gpu)
    os.environ.update(CUDA_VISIBLE_DEVICES=device["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUBUTTERFLY_COMPILE_MODE="research", CUBUTTERFLY_TEMPLATE_ROOT=str(ROOT),
        CUBUTTERFLY_NVCC=str(args.nvcc), CUBUTTERFLY_MATHDX_ROOT=str(args.mathdx_root),
        CUBUTTERFLY_JIT_CACHE=str(args.cache), CUBUTTERFLY_JIT_PYTHON=sys.executable)
    # Do not allow a user's pre-existing selector registry to seed this target implicitly.
    directory = args.output_dir / target_tag(device)
    os.environ["CUBUTTERFLY_REGISTRY"] = str(directory / "target_registry.json")
    identity = build_identity(args, device)
    state_path = directory / "campaign.json"
    if state_path.exists():
        state = read(state_path)
        if not args.resume or state["identity"] != identity:
            raise ValueError("target campaign exists or inputs changed; use --resume for identical inputs, otherwise a new output directory")
    else:
        state = dict(identity=identity, phases={}, status="prepared")
        write(state_path, state)
    def phase(name, action):
        previous = state["phases"].get(name, {})
        if previous.get("status") == "complete":
            if any(not Path(p).exists() or sha(p) != value for p, value in previous["artifacts"].items()):
                raise ValueError(f"completed phase artifacts changed: {name}")
            return
        state["active_phase"] = name
        state["status"] = "running"
        write(state_path, state)
        print(f"{directory.name}: {name}", flush=True)
        artifacts = action()
        state["phases"][name] = dict(status="complete", artifacts={str(p): sha(p) for p in artifacts})
        write(state_path, state)
    try:
        with target_lock(device["uuid"]):
            exclusive(device["uuid"])
            phase("calibration", lambda: calibrate(args, directory, device))
            memory = int(read(directory / "calibration/base_profile.json")["global_memory_bytes"])
            excluded = []
            for group in (*GROUPS, "composed", "plan_workloads"):
                source = read(args.campaign / f"{group}.json")
                accepted, deferred = partition_memory(source["workloads"], memory, args.memory_factor)
                freeze(directory / "inputs" / f"{group}.json", dict(source, workloads=accepted))
                excluded.extend(dict(group=group, **row) for row in deferred)
            freeze(directory / "inputs/memory.json", excluded)
            for group in GROUPS:
                if not read(directory / "inputs" / f"{group}.json")["workloads"]:
                    continue
                def execute_group(group=group):
                    guarded(acceptance_command(args, directory, group), device["uuid"],
                            directory / "logs" / f"{group}.log", directory / group)
                    return [directory / group / name for name in ("acceptance.json", "summary.json")]
                phase(group, execute_group)
            def composed():
                command = [sys.executable, ROOT / "scripts/run_crypto_composition.py", "--binary", args.crypto_binary,
                    "--build-dir", args.build_dir, "--template-root", ROOT, "--gpu-uuid", device["uuid"],
                    "--workloads", directory / "inputs/composed.json", "--output-dir", directory / "composed",
                    "--preflight", "--resume"]
                for name in ("trials", "warmup", "repeat"):
                    command += ["--" + name, str(getattr(args, name))]
                for group in GROUPS:
                    path = directory / group / "acceptance.json"
                    if path.exists():
                        command += ["--mapping-sources", str(path)]
                guarded(command, device["uuid"], directory / "logs/composed.log", directory / "composed")
                return [directory / "composed/summary.json"]
            phase("composed", composed)
            phase("plans", lambda: run_plans(args, directory, device))
            state["status"] = "resolved-declared-campaign"
            state.pop("active_phase", None)
            state.pop("error", None)
            write(state_path, state)
    except BaseException as error:
        state.update(status="interrupted", error=str(error))
        write(state_path, state)
        report(directory)
        raise
    report(directory)
    print(f"campaign report: {directory / 'report.json'}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "precompile", "run", "report"), required=True)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-v100-sm70")
    parser.add_argument("--campaign", type=Path, default=ROOT / "config/research_campaign")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/research_targets")
    parser.add_argument("--cuda-arch", type=int, default=70)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--mathdx-root", type=Path, default=ROOT / "external/mathdx/nvidia/mathdx")
    parser.add_argument("--nvcc", type=Path, default=Path(shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--crypto-binary", type=Path)
    parser.add_argument("--vkfft-binary", type=Path)
    parser.add_argument("--gpuntt-binary", type=Path)
    parser.add_argument("--fht-python", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--search-budget", type=int, default=16)
    parser.add_argument("--stage-max-points", type=int, default=1024)
    parser.add_argument("--memory-factor", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.jobs, args.trials, args.repeat, args.search_budget, args.memory_factor, args.stage_max_points) <= 0 or args.warmup < 0:
        parser.error("invalid timing/search/memory budget")
    if args.cuda_arch < 70 or args.cuda_arch > 999:
        parser.error("--cuda-arch must be an integer SM target, for example 70 or 80")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    args.cache = args.cache or args.output_dir / f"module-cache-sm{args.cuda_arch}"
    args.crypto_binary = args.crypto_binary or args.build_dir / f"crypto-ntt-sm{args.cuda_arch}/crypto_ntt_bench"
    if args.vkfft_binary is None and (args.build_dir / "vkfft_bench").is_file():
        args.vkfft_binary = args.build_dir / "vkfft_bench"
    return args


def main(argv=None):
    if sys.version_info < (3, 10):
        raise ValueError("Python >= 3.10 is required; activate the cubutterfly conda environment")
    args = parse_args(argv)
    if args.mode == "report":
        paths = sorted(args.output_dir.glob("*/campaign.json"))
        for path in paths:
            print(json.dumps(report(path.parent), ensure_ascii=False))
        return 0
    directory = prepare(args)
    if args.mode == "prepare":
        return 0
    if args.mode == "precompile":
        return subprocess.call([sys.executable, str(ROOT / "scripts/precompile_research.py"), "--mode", "compile",
            "--root", str(ROOT), "--mathdx", str(args.mathdx_root), "--nvcc", str(args.nvcc), "--cache", str(args.cache),
            "--output-dir", str(directory), "--jobs", str(args.jobs), "--policy", "research"])
    compilation = directory / "compilation.json"
    if not compilation.exists() or not read(compilation).get("complete"):
        raise ValueError("run --mode precompile first; compilation is kept outside the GPU measurement window")
    if any(not Path(row["path"]).is_file() for row in read(compilation)["results"]):
        raise ValueError("compiled modules are missing; rerun --mode precompile on this target machine")
    run_target(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
