#!/usr/bin/env python3
"""Prepare a bounded large-FFT extension without touching a GPU.

The 1D queue is deliberately separate from the paper's E1 acceptance journal.
It contains only new large-size A100 cells after excluding six historical exact
semantic points.  A small standalone-module manifest is generated from known
legal candidate families; target-GPU design-point export remains an execution
time check rather than a hidden preparation side effect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from research_compile_requests import compile_request


CONFIG_SCHEMA = "cubutterfly-large-fft-extension-v1"
WORKLOAD_SCHEMA = "cubutterfly-install-search-v1"
PRECOMPILE_SCHEMA = "cubutterfly-research-precompile-v1"
GENERATOR_VERSION = "large-fft-extension-preparation-v1"
STAGE_PROFILE_SCHEMA = "cubutterfly-large-fft-stage-profile-v1"
STAGE_POINTS_SCHEMA = "cubutterfly-large-fft-stage-points-v1"
SUPPORTED_TARGET_IDS = {"a100-80gb-gpu1", "a100-40gb-gpu0"}


def target_slot(target):
    target_id = str(target.get("id", ""))
    if target_id not in SUPPORTED_TARGET_IDS:
        raise ValueError("this bounded extension is currently scoped to A100 GPU0/GPU1")
    return target_id.rsplit("-", 1)[-1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def digest(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    temporary.replace(path)


def resolve(path, root=ROOT):
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (Path(root) / value).resolve()


def cell_key(workload):
    return "/".join((workload["operator"], workload["precision"],
                      str(workload["logN"]), str(workload["batch"]),
                      workload["placement"], workload["direction"],
                      workload["normalization"]))


def make_workload(precision, log_n, batch):
    return {
        "operator": "fft",
        "precision": precision,
        "logN": int(log_n),
        "batch": int(batch),
        "placement": "out-of-place",
        "direction": "forward",
        "normalization": "none",
        "accumulation": "native",
        "element_stride": 1,
        "batch_stride": 1 << int(log_n),
    }


def full_domain(config):
    axes = config["rank1"]
    return [make_workload(precision, log_n, batch)
            for precision in axes["precision"]
            for log_n in axes["logN"]
            for batch in axes["batch"]]


def _sample_correct(sample):
    value = sample.get("correct") if isinstance(sample, dict) else None
    return str(value) == "1"


def _source_cell(document, wanted):
    matches = []
    for cell in document.get("cells", []):
        workload = cell.get("workload", {})
        if (workload.get("operator") == "fft" and
                str(workload.get("precision")) == wanted["precision"] and
                int(workload.get("logN", -1)) == wanted["logN"] and
                int(workload.get("batch", -1)) == wanted["batch"] and
                workload.get("placement", "out-of-place") == "out-of-place" and
                workload.get("normalization", "none") == "none"):
            matches.append(cell)
    return matches


def audit_exclusions(config, root):
    exclusions = []
    seen = set()
    for raw in config["excluded_completed"]:
        wanted = {key: raw[key] for key in ("precision", "logN", "batch")}
        key = (wanted["precision"], int(wanted["logN"]), int(wanted["batch"]))
        if key in seen:
            raise ValueError(f"duplicate historical exclusion: {key}")
        seen.add(key)
        source = resolve(raw["source"], root)
        if not source.is_file():
            raise ValueError(f"historical exclusion source is missing: {source}")
        document, source_identity = read(source)
        matches = _source_cell(document, wanted)
        if len(matches) != 1:
            raise ValueError(f"expected one historical cell for {key}, found {len(matches)} in {source}")
        cell = matches[0]
        measurements = cell.get("measurements", [])
        correct_records = [m for m in measurements if _sample_correct(m.get("sample", m))]
        device = document.get("device", {})
        protocol = document.get("protocol", {})
        exclusions.append({
            **wanted,
            "cell_key": "/".join(("fft", wanted["precision"], str(wanted["logN"]), str(wanted["batch"]))),
            "source": source_identity,
            "historical_device": device,
            "historical_protocol": protocol,
            "historical_measurement_records": len(measurements),
            "historical_correct_records": len(correct_records),
            "historical_all_records_correct": bool(measurements) and len(correct_records) == len(measurements),
            "excluded_from_new_gpu1_workloads": True,
            "reuse_for_new_mean": False,
            "reason": "Exact 1D semantic record exists in historical A100-80GB evidence; it stays outside the new GPU1 journal and mean.",
        })
    return exclusions


def new_workloads(config, exclusions):
    excluded = {(item["precision"], int(item["logN"]), int(item["batch"])) for item in exclusions}
    values = full_domain(config)
    result = [item for item in values
              if (item["precision"], item["logN"], item["batch"]) not in excluded]
    if len(values) != 18 or len(exclusions) != 6 or len(result) != 12:
        raise ValueError(f"expected 18 full points, 6 exclusions and 12 new points; got {len(values)}, {len(exclusions)}, {len(result)}")
    if len({cell_key(item) for item in result}) != len(result):
        raise ValueError("large FFT workload semantics are not unique")
    return result


def memory_record(workload):
    n = 1 << int(workload["logN"])
    element_bytes = {"fp32": 8, "fp64": 16}[workload["precision"]]
    one = n * int(workload["batch"]) * element_bytes
    return {
        "cell": cell_key(workload),
        "N": n,
        "batch": int(workload["batch"]),
        "complex_element_bytes": element_bytes,
        "input_bytes": one,
        "output_bytes": one,
        "input_output_bytes": 2 * one,
        "workspace_bytes": None,
        "workspace_note": "Backend-dependent; target plan creation must report it."
    }


def _mapping_json(mapping):
    return canonical(mapping)


def _point(workload, mapping):
    return {**workload, "mapping_json": _mapping_json(mapping)}


def _family_mappings(log_n):
    tail = int(log_n) - 12
    if tail < 10:
        raise ValueError("large FFT family templates require logN >= 22")
    mappings = [
        ("shared-iterative", [
            {"backend": "shared-iterative", "compute_unit": "radix2",
             "fft_core": "scalar", "local_exchange": "shared",
             "shared_layout": "writer-aligned", "stage_partition": [12, tail],
             "tile_threads": threads, "kind": "butterfly", "schema_version": 1}
            for threads in (128, 256)
        ]),
        ("factor-streamed-cufftdx", [{
            "backend": "factor-streamed", "fft_core": "cufftdx-block",
            "factor_partition": ([10, 12] if log_n == 22 else
                                  [10, 13] if log_n == 23 else [12, 12]),
            "stage_partition": [log_n], "factor_ept": 8,
            "factor_columns": 1, "data_tiles_per_cta": 1,
            "prefetch_depth": 0, "factor_slices": 1,
            "kind": "butterfly", "schema_version": 1,
        }]),
        ("online-register-tile", [{
            "backend": "online-reorder", "fft_core": "register-tile",
            "compute_unit": "auto", "local_exchange": "shared",
            "shared_layout": "writer-aligned", "stage_partition": [local, log_n - local],
            "local_stages": local, "prefix_threads": prefix_threads,
            "prefix_ept": prefix_ept, "suffix_threads": suffix_threads,
            "suffix_ept": suffix_ept, "reorder_columns": 1,
            "cross_twiddle": "recurrence", "direct_boundary": "direct-strided",
            "kind": "butterfly", "schema_version": 1,
        } for local, prefix_threads, prefix_ept, suffix_threads, suffix_ept in (
            (10, 128, 32, 512, 16) if log_n == 22 else
            (10, 128, 32, 512, 32) if log_n == 23 else
            (12, 256, 64, 512, 32),
        )]),
    ]
    return mappings


def limited_modules(workloads, target):
    modules = {}
    points_by_family = {}
    for workload in workloads:
        for family, mappings in _family_mappings(int(workload["logN"])):
            for mapping in mappings:
                point = _point(workload, mapping)
                request = compile_request(point, policy="research")
                if not isinstance(request, dict):
                    raise ValueError(f"family did not project to a module: {family}")
                identifier = hashlib.sha256(canonical(request).encode()).hexdigest()
                entry = modules.setdefault(identifier, {
                    "id": identifier,
                    "request": request,
                    "representative_point": point,
                    "family": family,
                    "cells": [],
                    "policy": "research",
                })
                if cell_key(workload) not in entry["cells"]:
                    entry["cells"].append(cell_key(workload))
                points_by_family.setdefault(family, set()).add(identifier)
    expected = int(target.get("candidate_policy_max_modules", 24))
    if len(modules) > expected:
        raise ValueError(f"bounded module manifest exceeded limit: {len(modules)} > {expected}")
    return modules, {family: len(ids) for family, ids in sorted(points_by_family.items())}


def stage_profile(target):
    """Create an identity-only profile for the fresh large-size stage pass.

    The stage driver needs hardware limits, but importing a prior build's
    calibration profile would make the new checkpoint look like historical
    evidence.  The runtime probe must confirm these declared identity fields;
    no timing or thermal state is carried into this document.
    """
    return {
        "schema": STAGE_PROFILE_SCHEMA,
        "status": "prepared-target-identity-only",
        "device": target["device"],
        "compute_capability": target["compute_capability"],
        "global_memory_bytes": int(target["global_memory_bytes"]),
        "memory_bytes": int(target["global_memory_bytes"]),
        "gpu_uuid": target["gpu_uuid"],
        "sm_count": int(target.get("sm_count", 108)),
        "compile_mode": target["compile_mode"],
        "provenance": {
            "source": "target declaration plus fixed A100 PCIe architecture identity",
            "timings_imported": False,
            "thermal_state_imported": False,
            "runtime_probe_required": True,
        },
    }


def stage_source_points(all_workloads, modules, output):
    """Select one structurally valid, compiled representative per domain.

    Each precision/logN pair gets one shared-iterative source at the largest
    configured batch.  When CPU precompile has already run, failed module
    results are excluded from this source inventory; otherwise their status is
    pending and the execution gate rechecks the binary and module identity.
    """
    compile_status = {}
    compilation = Path(output) / "precompile" / "compilation.json"
    if compilation.is_file():
        document, _ = read(compilation)
        for result in document.get("results", []):
            if isinstance(result, dict) and result.get("id"):
                compile_status[result["id"]] = str(result.get("status", "unknown"))
    domains = sorted({(item["precision"], int(item["logN"])) for item in all_workloads})
    points = []
    selection = []
    for precision, log_n in domains:
        choices = [module for module in modules.values()
                   if module["representative_point"]["precision"] == precision and
                   int(module["representative_point"]["logN"]) == log_n]
        choices.sort(key=lambda module: (
            0 if module["family"] == "shared-iterative" else
            1 if module["family"] == "factor-streamed-cufftdx" else 2,
            module["id"],
        ))
        if not choices:
            raise ValueError(f"no stage source module for {precision}/logN{log_n}")
        eligible = [module for module in choices
                    if compile_status.get(module["id"], "pending") not in {"failed", "error"}]
        if not eligible:
            raise ValueError(f"all stage source modules failed for {precision}/logN{log_n}")
        chosen = eligible[0]
        status = compile_status.get(chosen["id"], "pending")
        point = dict(chosen["representative_point"])
        point["batch"] = max(int(item["batch"]) for item in all_workloads
                              if item["precision"] == precision and int(item["logN"]) == log_n)
        point["source_module_id"] = chosen["id"]
        point["source_family"] = chosen["family"]
        point["source_compile_status"] = status
        point["source_role"] = "large-fft-stage-training"
        points.append(point)
        selection.append({
            "precision": precision,
            "logN": log_n,
            "selected_module_id": chosen["id"],
            "selected_family": chosen["family"],
            "compile_status": status,
            "candidate_module_ids": [module["id"] for module in choices],
            "failed_candidate_module_ids": [module["id"] for module in choices
                                             if compile_status.get(module["id"]) in {"failed", "error"}],
        })
    return points, selection


def _rank2_policy(config):
    policy = config.get("rank2_policy", {})
    if policy.get("schema") != "finite-explicit-composition-v1":
        raise ValueError("rank2_policy must use finite-explicit-composition-v1")
    variants = policy.get("variants", [])
    if policy.get("variant_count") != 4 or len(variants) != 4:
        raise ValueError("rank2 policy must contain exactly four explicit variants")
    ids = [str(variant.get("id", "")) for variant in variants]
    if len(set(ids)) != 4 or any(not value for value in ids):
        raise ValueError("rank2 policy variant IDs must be unique")
    for variant in variants:
        for axis in ("axis0", "axis1"):
            entry = variant.get(axis, {})
            if entry.get("core") not in {"cufftdx-direct", "cufftdx-block"}:
                raise ValueError("rank2 variants must use cufftdx-direct or cufftdx-block")
            if int(entry.get("threads", 0)) not in {128, 256}:
                raise ValueError("rank2 variants must use 128 or 256 threads")
    if policy.get("search_claim") != "finite four-point explicit composition screen; not exhaustive measure":
        raise ValueError("rank2 policy must state its finite search claim")
    return variants


def _rank2_axis_mapping(log_n, entry):
    """Expand a finite policy entry into the complete mapping accepted by C API."""
    core = entry["core"]
    threads = int(entry["threads"])
    common = {
        "schema_version": 1,
        "kind": "butterfly",
        "compute_unit": "auto",
        "complex_multiply": "four-mul",
        "local_exchange": "shared",
        "shared_layout": "linear",
        "stage_overlap": False,
        "batch_tile_count": 1,
        "tile_threads": 128,
    }
    if core == "cufftdx-direct":
        if threads != 256 or log_n not in {11, 12}:
            raise ValueError("rank2 direct mapping requires logN 11/12 and 256 threads")
        return {
            **common,
            "backend": "temporal-tile",
            "fft_core": core,
            "stage_partition": [log_n],
            "direct_boundary": "direct-strided",
            "cross_twiddle": "table",
            "tile_threads": threads,
        }
    if log_n == 11:
        local_stages = 5
        ept = 4 if threads == 256 else 4
        suffix_ept = 4 if threads == 256 else 8
    elif log_n == 12:
        local_stages = 6
        ept = 4 if threads == 256 else 8
        suffix_ept = ept
    else:
        raise ValueError("rank2 block mapping requires logN 11 or 12")
    mapping = {
        **common,
        "backend": "online-reorder",
        "fft_core": core,
        "cross_twiddle": "recurrence",
        "direct_boundary": "direct-strided",
        "shared_layout": "writer-aligned",
        "stage_partition": [local_stages, log_n - local_stages],
        "local_stages": local_stages,
        "reorder_columns": 1,
        "prefix_threads": threads,
        "prefix_ept": ept,
        "suffix_threads": threads,
        "suffix_ept": suffix_ept,
        "segment_mappings": [
            {"core": core, "exchange": "shared", "threads": threads, "ept": ept},
            {"core": core, "exchange": "shared", "threads": threads, "ept": suffix_ept},
        ],
        "execution_group_mappings": [
            {"core": core, "exchange": "shared", "threads": threads, "ept": ept},
            {"core": core, "exchange": "shared", "threads": threads, "ept": suffix_ept},
        ],
        "boundaries": [{"twiddle": "recurrence", "layout": "direct-strided", "residency": "global-scratch"}],
    }
    return mapping


def _rank2_composition(shape, variant):
    logs = []
    for extent in shape:
        value = int(extent)
        if value <= 0 or value & (value - 1):
            raise ValueError("rank2 explicit compositions require power-of-two extents")
        logs.append(value.bit_length() - 1)
    return {
        "schema_version": 1,
        "kind": "composition",
        "axes": [_rank2_axis_mapping(logs[0], variant["axis0"]),
                 _rank2_axis_mapping(logs[1], variant["axis1"])],
    }


def rank2_cases(config, target, output, root):
    cases = []
    variants = _rank2_policy(config)
    slot = target_slot(target)
    plan_bench = resolve(target["build_dir"], root) / "cubutterfly_plan_bench"
    telemetry = root / "experiments/tools/run_with_gpu_telemetry.py"
    python = str(target["python"])
    for raw in config["rank2"]:
        shape = raw["shape"]
        if len(shape) != 2 or any(int(value) <= 0 for value in shape):
            raise ValueError("rank2 shape must contain two positive extents")
        shape_values = [int(shape[0]), int(shape[1])]
        for variant in variants:
            variant_id = str(variant["id"])
            run_id = f"{raw['id']}--{variant_id}"
            composition = _rank2_composition(shape_values, variant)
            algorithm = canonical(composition)
            case = {
                "id": run_id,
                "shape_id": raw["id"],
                "variant_id": variant_id,
                "operator": "fft",
                "shape": shape_values,
                "precision": raw["precision"],
                "batch": int(raw["batch"]),
                "direction": "forward",
                "placement": "out-of-place",
                "length_mode": "standard",
                "compare": "cuFFT",
                "search_scope": "one of four fixed explicit composition mappings",
                "algorithm": algorithm,
                "correctness": {
                    "status": "pending-execution",
                    "method": "full-output-cuFFT-reference",
                    "required": True,
                    "verified_batches": int(raw["batch"]),
                },
                "correctness_note": "--verify compares every output element and every batch with a matching cuFFT forward standard FFT outside CUDA-event timing; mismatch returns nonzero.",
                "binary": {"path": str(plan_bench), "sha256": None, "verified_at_execution": False},
            }
            prefix = output / "telemetry" / run_id
            child = [str(plan_bench), "--operator", "fft", "--shape",
                     f"{shape_values[0]}x{shape_values[1]}", "--batch", str(int(raw["batch"])),
                     "--precision", raw["precision"], "--algorithm", algorithm,
                     "--compare-cufft", "--verify", "--trials", str(config["protocol"]["rank2_trials"]),
                     "--warmup", str(config["protocol"]["warmup"]),
                     "--repeat", str(config["protocol"]["repeat"]), "--cache",
                     str((output / "rank2-cache").resolve())]
            case["command"] = {
                "id": run_id,
                "experiment": "large-fft-rank2-supplement-finite-explicit",
                "argv": [python, str(telemetry), "--gpu-uuid", target["gpu_uuid"],
                         "--output-prefix", str(prefix), "--stop-on-thermal-throttle", "--", *child],
                "cwd": str(root),
                "env": {"CUDA_VISIBLE_DEVICES": target["gpu_uuid"],
                        "CUBUTTERFLY_COMPILE_MODE": target["compile_mode"]},
                "requires_gpu": True,
                "executed": False,
                "prerequisites": [
                    f"Run after the {slot} 1D extension and after the existing {slot} writer exits normally.",
                    "Revalidate the new build binary identity before execution; the preparation did not hash or execute it.",
                    "Keep this rank-2 timing outside the 1D acceptance journal; run this explicit mapping independently with three trials.",
                    "This is a finite four-variant screen, not an exhaustive measure policy; record unavailable, incorrect and thermal stops separately.",
                ],
            }
            cases.append(case)
    return cases


def source_identity(root, generator):
    paths = {
        "generator": generator,
        "research_compile_requests.py": Path(root) / "scripts/research_compile_requests.py",
        "precompile_research.py": Path(root) / "scripts/precompile_research.py",
        "compile_module.py": Path(root) / "scripts/compile_module.py",
        "run_research_acceptance.py": Path(root) / "scripts/run_research_acceptance.py",
        "stage_service_calibration.py": Path(root) / "scripts/stage_service_calibration.py",
        "run_with_gpu_telemetry.py": Path(root) / "experiments/tools/run_with_gpu_telemetry.py",
        "cubutterfly_plan_bench.cpp": Path(root) / "apps/cubutterfly_plan_bench.cpp",
    }
    result = {}
    for name, path in paths.items():
        result[name] = {"path": str(path.resolve()), "exists": path.is_file(),
                        "sha256": digest(path) if path.is_file() else None}
    return result


def _command(command_id, experiment, argv, cwd, env, *, gpu, prerequisites):
    return {
        "id": command_id,
        "experiment": experiment,
        "argv": [str(value) for value in argv],
        "cwd": str(Path(cwd).resolve()),
        "env": {str(key): str(value) for key, value in env.items()},
        "requires_gpu": bool(gpu),
        "prerequisites": list(prerequisites),
        "executed": False,
    }


def build_commands(config, target, output, root, workloads_path, modules_path):
    python = str(target["python"])
    output = Path(output).resolve()
    root = Path(root).resolve()
    slot = target_slot(target)
    compile_env = {"CUDA_VISIBLE_DEVICES": "", "CUBUTTERFLY_COMPILE_MODE": "research"}
    commands = [_command(
        "precompile-compile-cpu", "large-fft-1d",
        [python, root / "scripts/precompile_research.py", "--mode", "compile",
         "--output-dir", modules_path.parent, "--jobs", 2, "--root", root,
         "--mathdx", root / "external/mathdx/nvidia/mathdx",
         "--cache", output / "precompile-cache"],
        root, compile_env, gpu=False,
        prerequisites=[
            "This consumes only the bounded structural modules.json; it does not export the full runtime candidate population.",
            "Run after the source/compiler revision is frozen; successful compilation is not target-GPU legality or correctness evidence.",
            "Compilation and initialization wall time must remain separate from the later CUDA-event timings.",
        ])]

    # These three artifacts are intentionally new.  Passing the target's
    # historical paths would allow a new build to mutate or import old timing
    # evidence during run_research_acceptance.prepare().
    profile = output / f"stage-{slot}" / "stage_profile.json"
    stage = output / f"stage-{slot}" / "stage_calibration.json"
    stage_points = output / f"stage-{slot}" / "stage_source_points.json"
    pipeline = output / f"calibration-{slot}" / "pipeline_schedule_calibration.json"
    schedule = output / f"calibration-{slot}" / "schedule_calibration.json"
    build = resolve(target["build_dir"], root)

    stage_child = [python, root / "scripts/stage_service_calibration.py",
                   "--binary", build / "cubutterfly_stage_microbench",
                   "--checkpoint", stage, "--profile", profile,
                   "--points", stage_points, "--mode", "full",
                   "--warmup", config["protocol"]["warmup"],
                   "--repeat", config["protocol"]["repeat"],
                   "--trials", config["protocol"]["trials"]]
    stage_env = {
        "CUDA_VISIBLE_DEVICES": target["gpu_uuid"],
        "CUBUTTERFLY_COMPILE_MODE": target["compile_mode"],
        "CUBUTTERFLY_JIT_CACHE": str((output / "precompile-cache").resolve()),
    }
    commands.append(_command(
        f"stage-calibration-{slot}", "large-fft-stage-calibration",
        [python, root / "experiments/tools/run_with_gpu_telemetry.py", "--gpu-uuid", target["gpu_uuid"],
         "--output-prefix", output / "telemetry" / f"stage-calibration-{slot}",
         "--stop-on-thermal-throttle",
         "--", *stage_child],
        root, stage_env, gpu=True,
        prerequisites=[
            f"Wait for {slot} to be idle and thermally stable; require the telemetry wrapper's UUID and exclusive-GPU checks.",
            "This is a fresh large-size stage checkpoint. It imports no historical stage checkpoint or thermal state.",
            "The source points contain one structurally selected module per precision/logN domain at batch 16; failed CPU-precompile modules are excluded and any remaining source status is revalidated before launch.",
            "The identity-only stage profile is generated by this preparation and must be matched by the runtime probe; it is not the historical build profile.",
            "Use warmup=1000, repeat=100, trials=3 for this large-size stage cohort. A busy, interrupted or contaminated run stops the queue; do not silently retry.",
        ]))

    acceptance = output / f"acceptance-{slot}"
    telemetry_prefix = output / "telemetry" / f"acceptance-{slot}"
    acceptance_child = [python, root / "scripts/run_research_acceptance.py",
                        "--build-dir", build, "--output-dir", acceptance,
                        "--workloads", workloads_path, "--profile", profile,
                        "--stage-checkpoint", stage, "--pipeline-calibration", pipeline,
                        "--schedule-calibration", schedule, "--mapping-seeds", *
                        [resolve(path, root) for path in target.get("mapping_seeds", [])],
                        "--search-budget", config["protocol"]["search_budget"],
                        "--seed-budget", config["protocol"]["seed_budget"],
                        "--finalists", config["protocol"]["finalists"],
                        "--trials", config["protocol"]["trials"],
                        "--warmup", config["protocol"]["warmup"],
                        "--repeat", config["protocol"]["repeat"],
                        "--verify-batches", config["protocol"]["verify_batches"],
                        "--vkfft-binary", resolve(target["vkfft_binary"], root),
                        "--mode", "all"]
    acceptance_env = {
        "CUDA_VISIBLE_DEVICES": target["gpu_uuid"],
        "CUBUTTERFLY_COMPILE_MODE": target["compile_mode"],
        "CUBUTTERFLY_JIT_CACHE": str((output / "precompile-cache").resolve()),
    }
    commands.append(_command(
        f"acceptance-{slot}", "large-fft-1d",
        [python, root / "experiments/tools/run_with_gpu_telemetry.py", "--gpu-uuid", target["gpu_uuid"],
         "--output-prefix", telemetry_prefix, "--stop-on-thermal-throttle", "--", *acceptance_child],
        root, acceptance_env, gpu=True,
        prerequisites=[
            f"{slot} UUID and exclusive ownership must be verified by the telemetry wrapper.",
            f"Wait for the existing {slot} writer/calibration process to exit normally; an interruption stops this queue.",
            f"Require the fresh stage-{slot} checkpoint to be complete and verify all new stage/pipeline/schedule identities and the newly built benchmark binaries before starting; historical stage/profile paths are forbidden.",
            "The journal contains only the 12 new cells; the six historical exclusions and old results are never merged.",
            "Full-batch correctness is required and every cell gets three trials with 1000 warmups and 100 repeats; stage confirmation inherits the fresh checkpoint protocol.",
            "The registry mapping is candidate seeds only: four seeds plus twelve searched candidates make the exact total screen denominator of 16.",
        ]))
    rank2 = rank2_cases(config, target, output, root)
    for case in rank2:
        commands.append(case["command"])
    return commands, rank2


def prepare(config_path, target_path, output, root=ROOT):
    root = Path(root).resolve()
    output = Path(output).resolve()
    config, config_source = read(config_path)
    target, target_source = read(target_path)
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported large FFT extension config schema")
    if target.get("schema") != "cubutterfly-paper-target-v1":
        raise ValueError("unsupported paper target schema")
    if config.get("target_id") != target.get("id"):
        raise ValueError("large FFT target/config mismatch")
    slot = target_slot(target)
    protocol = config.get("protocol", {})
    required_protocol = ("search_budget", "seed_budget", "screen_budget_total", "finalists",
                         "trials", "warmup", "repeat", "rank2_trials", "verify_batches")
    if any(type(protocol.get(name)) is not int or protocol[name] < 0 for name in required_protocol):
        raise ValueError("invalid extension protocol")
    for name in ("search_budget", "finalists", "trials", "repeat"):
        if protocol[name] < 1:
            raise ValueError(f"{name} must be positive")
    if protocol["rank2_trials"] < 1:
        raise ValueError("rank2_trials must be positive")
    if protocol["screen_budget_total"] != protocol["search_budget"] + protocol["seed_budget"]:
        raise ValueError("screen_budget_total must equal search_budget plus seed_budget")
    if protocol["verify_batches"] != 0:
        raise ValueError("the extension requires full-batch verification via verify-batches=0")
    acceptance = resolve(target["acceptance_dir"], root)
    if (output == root or output == acceptance or output in acceptance.parents or
            acceptance in output.parents):
        raise ValueError("large FFT output must be a separate results directory")
    previous = output / "execution_plan.json"
    generator = Path(__file__).resolve()
    preparation_identity = {
        "generator_version": GENERATOR_VERSION,
        "config": config_source,
        "target": target_source,
        "generator_sha256": digest(generator),
    }
    if previous.exists():
        old, _ = read(previous)
        if old.get("preparation_identity") != preparation_identity:
            raise ValueError("preparation inputs changed; choose a new output directory")
    exclusions = audit_exclusions(config, root)
    all_workloads = full_domain(config)
    workloads = new_workloads(config, exclusions)
    memories = [memory_record(item) for item in workloads]
    modules, family_counts = limited_modules(workloads, target)
    stage_profile_document = stage_profile(target)
    stage_profile_path = output / f"stage-{slot}" / "stage_profile.json"
    write(stage_profile_path, stage_profile_document)
    stage_points, stage_selection = stage_source_points(all_workloads, modules, output)
    stage_points_path = output / f"stage-{slot}" / "stage_source_points.json"
    write(stage_points_path, {
        "schema": STAGE_POINTS_SCHEMA,
        "status": "prepared-not-executed",
        "target": target["id"],
        "training_policy": "one source module per precision/logN domain at configured maximum batch",
        "max_batch": max(int(item["batch"]) for item in all_workloads),
        "points": stage_points,
        "selection": stage_selection,
        "failed_precompile_modules_are_not_retried": True,
    })
    source = source_identity(root, generator)
    source["config"] = config_source
    source["target"] = target_source
    source["stage_profile"] = read(stage_profile_path)[1]
    source["stage_source_points"] = read(stage_points_path)[1]
    build = resolve(target["build_dir"], root)
    binary_requirements = {
        name: {"path": str((build / name).resolve()), "sha256": None,
               "verified_at_execution": False}
        for name in ("cubutterfly_bench", "cubutterfly_plan_bench",
                     "cubutterfly_stage_microbench", "cubutterfly_pipeline_microbench",
                     "cubutterfly_schedule_microbench", "cubutterfly_plan_probe")
    }
    write(output / "full_domain.json", {
        "schema": "cubutterfly-large-fft-domain-v1",
        "status": "prepared-not-executed",
        "target": target["id"],
        "rank1_semantics": "1D FFT; batch is independent transforms, not a dimension",
        "all_points": all_workloads,
        "excluded_old_points": exclusions,
        "new_points": workloads,
        "counts": {"full_domain": len(all_workloads), "excluded": len(exclusions), "new": len(workloads)},
    })
    write(output / "workloads.json", {
        "schema": WORKLOAD_SCHEMA,
        "scope": f"Large FFT exploratory extension; 12 new rank-1 {target['device']} {slot} cells only.",
        "target": target["id"],
        "protocol": protocol,
        "workloads": workloads,
        "limitations": config["limitations"],
    })
    write(output / "excluded_old_points.json", {
        "schema": "cubutterfly-large-fft-exclusions-v1",
        "status": "historical-records-excluded-from-new-mean",
        "target": target["id"],
        "points": exclusions,
        "note": f"These records are provenance only. They are not copied into the {slot} acceptance journal or timing summary.",
    })
    write(output / "memory_budget.json", {
        "schema": "cubutterfly-large-fft-memory-budget-v1",
        "target_memory_bytes": int(target["global_memory_bytes"]),
        "workspace_policy": "unknown until target plan creation; record allocation failure as an explicit negative result",
        "cells": memories,
    })
    manifest_path = output / "precompile" / "modules.json"
    modules_path = manifest_path
    manifest = {
        "schema": PRECOMPILE_SCHEMA,
        "status": "ready",
        "complete": True,
        "policy": target["compile_mode"],
        "target_declared": target["gpu_uuid"],
        "runtime_feasibility_verified": False,
        "structural_only": True,
        "export_basis": "bounded family representatives derived from existing compiler contract; no live --list-design-points query",
        "profile": {"device": target["device"], "compute_capability": target["compute_capability"],
                    "global_memory_bytes": int(target["global_memory_bytes"]),
                    "sm": int(str(target["compute_capability"]).split(".")[0]) * 10 + int(str(target["compute_capability"]).split(".")[1])},
        "workloads": {"path": str((output / "workloads.json").resolve()), "count": len(workloads),
                      "sha256": None},
        "stage_profile": {"path": str(stage_profile_path.resolve()),
                           "sha256": digest(stage_profile_path)},
        "stage_source_points": {"path": str(stage_points_path.resolve()),
                                 "sha256": digest(stage_points_path), "count": len(stage_points)},
        "source": source,
        "family_counts": family_counts,
        "module_count": len(modules),
        "modules_path": str(manifest_path.resolve()),
        "compilation_path": str((output / "precompile" / "compilation.json").resolve()),
        "modules": list(modules.values()),
        "limitations": [
            "This manifest is intentionally finite and does not claim to cover every runtime design point.",
            "The new build and target GPU must revalidate each compiled module before timing.",
        ],
    }
    # The manifest references the workload hash only after the workload file exists.
    manifest["workloads"]["sha256"] = digest(output / "workloads.json")
    write(manifest_path, manifest)
    commands, rank2 = build_commands(config, target, output, root, output / "workloads.json", manifest_path)
    rank2_document = {
        "schema": "cubutterfly-large-fft-rank2-audit-v1",
        "status": "prepared-not-executed",
        "public_api": {
            "rank1": "supported",
            "rank2": "supported by cubutterflySetShape and cubutterfly_plan_bench",
            "rank3": "supported by cubutterflySetShape and the xD lowering for packed power-of-two workloads",
            "batch": "independent transforms; not an extra dimension",
        },
        "adapter": {
            "binary_source": source["cubutterfly_plan_bench.cpp"],
            "cli": "--shape N|RowsxColumns, --algorithm composition-json, --compare-cufft, --verify and --trials 3",
            "acceptance_adapter": False,
            "correctness": "full cuFFT output verification is required for every shape/variant run; these results remain outside the rank-1 acceptance journal",
        },
        "search": {
            "schema": "finite-explicit-composition-v1",
            "shape_count": len(config["rank2"]),
            "variant_count": len(_rank2_policy(config)),
            "run_count": len(rank2),
            "exhaustive_measure": False,
            "claim": config["rank2_policy"]["search_claim"],
        },
        "cases": rank2,
    }
    write(output / "rank2_audit.json", rank2_document)
    pipeline_path = output / f"calibration-{slot}" / "pipeline_schedule_calibration.json"
    schedule_path = output / f"calibration-{slot}" / "schedule_calibration.json"
    queue = {
        "schema": "cubutterfly-large-fft-execution-plan-v1",
        "status": "prepared-not-executed",
        "preparation_identity": preparation_identity,
        "target": target,
        "source": source,
        "calibration_sources": {
            "stage_profile": {"path": str(stage_profile_path.resolve()),
                               "sha256": digest(stage_profile_path),
                               "origin": "fresh target-identity-only profile; no historical build profile imported"},
            "stage_source_points": {"path": str(stage_points_path.resolve()),
                                     "sha256": digest(stage_points_path),
                                     "count": len(stage_points)},
            "stage_checkpoint": {"path": str((output / f"stage-{slot}" / "stage_calibration.json").resolve()),
                                  "sha256": None,
                                  "status": f"created by stage-calibration-{slot}; fresh independent checkpoint required",
                                  "imported_historical_checkpoint": False},
            "pipeline_calibration": {"path": str(pipeline_path.resolve()),
                                      "sha256": None,
                                      "status": "created during acceptance prepare with the new build",
                                      "imported_historical_checkpoint": False},
            "schedule_calibration": {"path": str(schedule_path.resolve()),
                                      "sha256": None,
                                      "status": "created during acceptance prepare with the new build",
                                      "imported_historical_checkpoint": False},
        },
        "binary_requirements": binary_requirements,
        "gpu_probe_performed": False,
        "scope": {
            "rank1_full_domain_cells": len(all_workloads),
            "rank1_excluded_historical_cells": len(exclusions),
            f"rank1_new_{slot}_cells": len(workloads),
            "rank2_supplement_cases": len(config["rank2"]),
            "rank2_explicit_runs": len(rank2),
            f"{slot}_priority": True,
            f"{'gpu0' if slot == 'gpu1' else 'gpu1'}_mirror_required": False,
        },
        "protocol": protocol,
        "candidate_modules": {"path": str(manifest_path.resolve()), "count": len(modules),
                              "family_counts": family_counts, "runtime_export_deferred": True},
        "commands": commands,
        "rank2_cases": [case["id"] for case in rank2],
        "limitations": config["limitations"],
    }
    write(output / "execution_plan.json", queue)
    lines = [
        "# Large FFT extension runbook",
        "",
        f"This is a prepared, not executed, queue for {target['device']} {slot}.",
        "The 1D acceptance journal contains exactly 12 new cells. Six historical exact points are documented separately and never merged into its mean.",
        "All commands below are structured in execution_plan.json with argv/env/cwd; this document is a shell rendering.",
        "",
        "## Scope and gates",
        "",
        "- Rank 1 only for run_research_acceptance; batch means independent transforms.",
        "- Rank 2 API support is audited separately through cubutterfly_plan_bench with four fixed explicit compositions per shape, full cuFFT output verification and three trials; rank 3 is unsupported.",
        f"- {slot} UUID/exclusivity, checkpoint identity and new-build binary hashes must be checked at execution.",
        "- Precompile compile is finite structural preparation. It does not replace target-GPU design-point export or correctness.",
        "- Keep compilation, plan creation and command wall time outside CUDA-event kernel timing; retain telemetry and negative results.",
        "- Do not use an older precompile directory as this queue's manifest or timing evidence.",
        "- Do not pass the target's historical calibration_device.json or stage_calibration.json: the stage profile and source points are fresh preparation inputs, and the stage command must complete first.",
        "",
    ]
    for command in commands:
        lines.extend([f"## {command['id']} ({command['experiment']}; {'GPU' if command['requires_gpu'] else 'CPU'})", ""])
        lines.extend([f"- {item}" for item in command["prerequisites"]])
        lines.extend(["", "```bash", "cd " + shlex.quote(command["cwd"]),
                      shlex.join(["env", *[f"{key}={value}" for key, value in command["env"].items()], *command["argv"]]),
                      "```", ""])
    (output / "RUNBOOK.md").parent.mkdir(parents=True, exist_ok=True)
    (output / "RUNBOOK.md").write_text("\n".join(lines) + "\n")
    write(output / "preparation.json", {
        "schema": "cubutterfly-large-fft-preparation-v1",
        "status": "prepared-not-executed",
        "target": target["id"],
        "new_cells": len(workloads),
        "excluded_cells": len(exclusions),
        "full_domain_cells": len(all_workloads),
        "module_count": len(modules),
        "stage_source_points": len(stage_points),
        "stage_source_profile": str(stage_profile_path.resolve()),
        "rank2_cases": len(config["rank2"]),
        "rank2_explicit_runs": len(rank2),
        "gpu_probe_performed": False,
        "execution_plan": str((output / "execution_plan.json").resolve()),
    })
    return queue


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/large_fft_extension.json")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.config, args.target, args.output_dir)
    print(json.dumps({"status": result["status"], "scope": result["scope"],
                      "modules": result["candidate_modules"]["count"],
                      "output": str(args.output_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
