#!/usr/bin/env python3
"""Bounded current-build calibration, confirmation and public AUTO comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
from pathlib import Path
import subprocess
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ROOT / "config/current_recalibration_workloads.json"
ARTIFACTS = ("libcuntt.so", "cubutterfly_bench", "cuntt_bench", "cuntt_hardware_microbench",
             "cubutterfly_plan_probe", "cubutterfly_stage_microbench",
             "cubutterfly_schedule_microbench", "cubutterfly_pipeline_microbench",
             "cubutterfly_plan_bench", "generated/build_fingerprint.hpp", "CMakeCache.txt")
SOURCE_INPUTS = ("run_current_recalibration.py", "calibrate_local_hardware.py",
                 "initialize_hardware_profile.py", "check_hardware_profile.py",
                 "static_hardware_profile.py", "rank_static_candidates.py",
                 "calibration_cpu_cache.py", "capture_calibration_cpu_inputs.py",
                 "prepare_calibration_cpu.py", "research_compile_requests.py",
                 "prepare_refresh_calibration.py",
                 "calibration_seeds.py",
                 "finalize_recalibration.py", "run_research_acceptance.py", "research_baselines.py",
                 "calibration_space.py", "stage_cost_model.py", "stage_service_model.py",
                 "stage_service_calibration.py", "stage_schedule_native.py", "stage_schedule_native.cpp",
                 "stage_composition.py", "calibrate_composition.py", "verify_local_selector.py",
                 "hardware_registry.py", "local_selector_data.py", "benchmark_semantics.py")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def validate_build(build):
    missing = [name for name in ARTIFACTS if not (build / name).is_file()]
    if missing:
        raise ValueError("build is missing required artifacts: " + ", ".join(missing))
    cache = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if "=" in line and ":" in line.split("=", 1)[0]:
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    for name in ("CUBUTTERFLY_BUILD_SHARED", "CUBUTTERFLY_ENABLE_CUFFTDX",
                 "CUBUTTERFLY_ENABLE_XD_CUFFTDX"):
        if cache.get(name) not in ("ON", "TRUE", "1"):
            raise ValueError("current retest requires " + name + "=ON")
    targets = cache.get("CMAKE_CUDA_ARCHITECTURES", "").split(";")
    if not targets or any(not re.fullmatch(r"[0-9]+(?:-real|-virtual)?", target)
                          for target in targets):
        raise ValueError("CMAKE_CUDA_ARCHITECTURES requires explicit numeric SM targets")
    architectures = {int(target.split("-", 1)[0]) for target in targets}
    if len(architectures) != 1 or next(iter(architectures)) <= 0:
        raise ValueError("CMAKE_CUDA_ARCHITECTURES must specify one unique positive SM")
    return architectures.pop()


def identity(args):
    result = dict(schema="cubutterfly-current-recalibration-v1", build_dir=str(args.build_dir),
                gpu_uuid=args.gpu_uuid, compile_mode="research",
                jit_cache_environment={name: os.environ.get(name) for name in
                                       ("CUBUTTERFLY_JIT_CACHE", "XDG_CACHE_HOME")},
                workloads_path=str(args.workloads), workloads_sha256=digest(args.workloads),
                binaries={name: digest(args.build_dir / name) for name in ARTIFACTS},
                scripts={name: digest(ROOT / "scripts" / name) for name in SOURCE_INPUTS},
                baselines={name: (dict(path=str(getattr(args, name)),
                                     sha256=digest(getattr(args, name)))
                                   if getattr(args, name) else None)
                           for name in ("vkfft_binary", "gpuntt_binary", "fht_python")},
                protocol=dict(trials=3, warmup=20, repeat=50, verify_batches=2,
                              search_budget=args.search_budget, finalists=2, seed_budget=8,
                              stage_point_budget=args.stage_point_budget, compile_seconds=1200, search_seconds=0),
                  include_xd=args.include_xd)
    if args.include_xd:
        result["xd_script_sha256"] = digest(ROOT / "scripts/calibrate_xd_public.py")
    return result


def environment(args, finalized=False):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUBUTTERFLY_COMPILE_MODE="research",
               CUBUTTERFLY_REGISTRY=str(args.output_dir / ("finalized" if finalized else "initial") / "registry.json"))
    env.pop("CUBUTTERFLY_HARDWARE_MODEL", None)
    env.pop("CUBUTTERFLY_PROFILE_ROOT", None)
    if finalized:
        hardware = args.output_dir / "hardware/hardware_model.json"
        if hardware.is_file():
            env["CUBUTTERFLY_HARDWARE_MODEL"] = str(hardware)
    env.pop("CUBUTTERFLY_HARDWARE_PROFILE", None)
    env.pop("CUBUTTERFLY_COST_MODEL", None)
    env.pop("LD_PRELOAD", None)
    env["LD_LIBRARY_PATH"] = str(args.build_dir) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def commands(args):
    python = sys.executable
    initial, finalized = args.output_dir / "initial", args.output_dir / "finalized"
    hardware = args.output_dir / "hardware"
    calibrate = [python, str(ROOT / "scripts/calibrate_local_hardware.py"),
                 "--build-dir", str(args.build_dir), "--profile-dir", str(hardware),
                 "--output-dir", str(initial), "--trials", "3", "--operator-trials", "3",
                 "--operator-warmup", "20", "--operator-repeat", "50", "--verify-batches", "2",
                 "--search-only", "--search-workloads", str(args.workloads), "--cost-model", "staged",
                 "--stage-calibration", "full", "--stage-point-budget", str(args.stage_point_budget),
                 "--search-budget", str(args.search_budget), "--search-finalists", "2", "--seed-budget", "8",
                 "--search-stage-projection", "full", "--search-strategy", "model",
                 "--compile-seconds", "1200", "--search-seconds", "0"]
    finalize = [python, str(ROOT / "scripts/finalize_recalibration.py"),
                "--build-dir", str(args.build_dir), "--input-profile", str(initial),
                "--output-dir", str(finalized), "--hardware-profile", str(hardware / "hardware_profile.json"),
                "--trials", "3", "--warmup", "20", "--repeat", "50"]
    compare = [python, str(Path(__file__).resolve()), "--build-dir", str(args.build_dir),
               "--output-dir", str(args.output_dir), "--gpu-uuid", args.gpu_uuid, "--compare-worker",
               "--workloads", str(args.workloads), "--stage-point-budget", str(args.stage_point_budget),
               "--search-budget", str(args.search_budget)]
    for name in ("vkfft_binary", "gpuntt_binary", "fht_python"):
        if getattr(args, name):
            compare += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    if args.resume:
        calibrate += ["--resume-search"]
        finalize += ["--resume"]
        compare += ["--resume"]
    plan = [("calibrate", calibrate), ("finalize", finalize), ("compare", compare)]
    if args.include_xd:
        plan.append(("xd", [python, str(ROOT / "scripts/calibrate_xd_public.py"),
                            "--library", str(args.build_dir / "libcuntt.so"),
                            "--output", str(args.output_dir / "xd/public_calibration.json"),
                            "--gpu-uuid", args.gpu_uuid]))
    return plan


def require_stage_ready(profile, declared_sampling=None):
    """Require validation of the declared inventory, never a usable subset."""
    service = profile.get("stage_service", {})
    coverage = service.get("coverage", {})
    if (not service.get("curves") or coverage.get("coverage_complete") is not True
            or coverage.get("validation_complete") is not True
            or service.get("validation", {}).get("passed") is not True):
        raise ValueError("stage calibration has not validated its declared scope")
    for field in ("pending_description_count", "adapter_gap_count", "unsupported_group_count", "omitted_count"):
        if coverage.get(field, 0):
            raise ValueError("stage calibration has unresolved coverage: " + field)
    if coverage.get("one_point_curves"):
        raise ValueError("stage calibration has one-point curves")
    if declared_sampling is not None:
        if not isinstance(declared_sampling, dict) or not declared_sampling:
            raise ValueError("stage calibration requires a declared sampling scope")
        recorded = service.get("stage_point_sampling", {})
        # The sidecar may add its schema; all declared acquisition fields must match.
        if any(recorded.get(key) != value for key, value in declared_sampling.items() if key != "schema"):
            raise ValueError("stage calibration differs from the declared sampling scope")
        selected = declared_sampling.get("selected")
        if (type(selected) is not int or selected < 1
                or coverage.get("requested_static_count") != selected
                or coverage.get("described_static_count") != selected):
            raise ValueError("stage calibration does not cover all declared selected points")
        if declared_sampling.get("inventory_complete") is False:
            if (service.get("calibration_status") != "complete-sampled"
                    or service.get("selected_scope_status") != "complete"
                    or service.get("selected_scope_validated") is not True):
                raise ValueError("finite stage sampling scope is not complete and validated")
            return
        if declared_sampling.get("inventory_complete") is not True:
            raise ValueError("stage inventory scope must be explicitly declared")
        if service.get("inventory_complete") is not True:
            raise ValueError("stage calibration does not cover the global inventory")
    # Confirmation has its own finite requested mappings, represented by coverage.
    if service.get("calibration_status") != "complete" or service.get("validated") is not True:
        raise ValueError("stage calibration is not complete and validated")


def require_release_ready(profile, composition, service_gaps):
    require_stage_ready(profile)
    if service_gaps:
        raise ValueError("confirmed mapping stage services have unresolved gaps")
    if composition.get("status") != "complete-observed":
        raise ValueError("composition calibration is not ready: expected complete-observed")


def gate(phase, output, workloads=WORKLOADS):
    requested_count=len(read(workloads)["workloads"])
    if phase == "calibrate":
        manifest = read(output / "initial/calibration_manifest.json")
        coverage = read(output / "initial/search_coverage.json")
        if manifest.get("operator_result", {}).get("status") != "complete":
            raise ValueError("operator calibration did not complete")
        if len(coverage.get("workloads", [])) != requested_count or not all(c.get("complete") for c in coverage["workloads"]):
            raise ValueError(f"all {requested_count} search cells must complete before confirmation")
        require_stage_ready(read(output / "initial/calibration_device.json"),
                            manifest.get("stage_point_sampling_summary", {}))
    elif phase == "finalize":
        manifest = read(output / "finalized/calibration_manifest.json")
        if manifest.get("status") != "ready-with-measured-mappings" or manifest.get("selector_replay", {}).get("status") != "passed":
            raise ValueError("finalized public selector replay is not ready")
        if not (output / "finalized/registry.json").is_file() or not manifest.get("registry_promoted_records"):
            raise ValueError("no released finalized registry")
        if not manifest.get("hardware_rates_remeasured"):
            raise ValueError("finalizer must use the fresh hardware-rate profile")
        require_release_ready(read(output / "finalized/calibration_device.json"),
                              manifest.get("composition_calibration", {}),
                              manifest.get("service_gaps", ["not recorded"]))
    elif phase == "compare":
        report = read(output / "comparison.json")
        if report.get("status") != "complete" or len(report.get("cells", [])) != requested_count:
            raise ValueError("all requested public AUTO comparison cells must complete")
    elif phase == "xd":
        report = read(output / "xd/public_calibration.json")
        cases = report.get("cases", [])
        if report.get("status") != "complete" or len(cases) != 3 or not all(
                c.get("status") == "measured" and c.get("cache_gate", {}).get("passed") for c in cases):
            raise ValueError("xD public calibration and DEFAULT user-cache replay are not ready")
    else:
        raise ValueError("unknown phase: " + phase)


def comparison_cells(records, workloads):
    from local_selector_data import select_points, semantic_key
    selected = {semantic_key(r["samples"][0]): r for r in select_points({"candidates": records})}
    cells = []
    for index, requested in enumerate(workloads):
        workload = {"operator": "fft", **requested}
        normalized = dict(workload)
        if workload.get("stage_matrix"):
            normalized["stage_matrices"] = ":".join(format(float(v), ".17g") for v in workload["stage_matrix"].split(","))
        winner = selected.get(semantic_key(normalized))
        if winner is None:
            raise ValueError("no confirmed selected mapping for requested cell " + str(index))
        cells.append(dict(id=index, workload=workload, selected=winner, status="pending", measurements=[]))
    return cells


def write_comparison_csv(path, document):
    output = io.StringIO()
    fields = ("cell", "operator", "precision", "accumulation", "logN", "batch", "direction",
              "normalization", "placement", "modulus", "input_order", "output_order", "reference",
              "external", "status", "cubutterfly_ms", "reference_ms", "speedup", "reason")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for cell in document["cells"]:
        times = {}
        for row in cell["measurements"]:
            times.setdefault(row["name"], []).append(float(row["sample"]["kernel_ms"]))
        own = statistics.median(times["cuButterfly"])
        common = {key: cell["workload"].get(key, "") for key in fields[:12]}
        common.update(cell=cell["id"], cubutterfly_ms=own)
        for spec in cell["baseline_plan"]["available"]:
            reference = statistics.median(times[spec["name"]])
            writer.writerow(dict(common, reference=spec["name"], external=spec["external"],
                                 status="measured", reference_ms=reference, speedup=reference / own))
        for gap in cell["baseline_plan"]["unavailable"]:
            writer.writerow(dict(common, reference=gap["name"], external=True, status="unavailable", reason=gap["reason"]))
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    temporary.write_text(output.getvalue())
    temporary.replace(path)


def compare_worker(args):
    from research_baselines import baseline_identity
    from run_research_acceptance import compare_cell
    profile = read(args.output_dir / "finalized/calibration_device.json")
    records = read(args.output_dir / "finalized/operator_calibration.json")
    path = args.output_dir / "comparison.json"
    paths = {name: getattr(args, name) for name in ("build_dir", "vkfft_binary", "gpuntt_binary", "fht_python")}
    comparison_identity = dict(device=profile, calibration_sha256=digest(args.output_dir / "finalized/calibration_manifest.json"),
                               baseline_identity=baseline_identity(paths), workloads_sha256=digest(args.workloads),
                               search_budget=args.search_budget, stage_point_budget=args.stage_point_budget)
    if path.exists():
        document = read(path)
        if not args.resume or document.get("identity") != comparison_identity:
            raise ValueError("comparison resume identity mismatch")
    else:
        cells = comparison_cells(records, read(args.workloads)["workloads"])
        document = dict(schema="cubutterfly-current-comparison-v1", identity=comparison_identity,
                        status="running", cells=cells,
                        timing_excludes="compilation, plan construction and correctness references")
    def save():
        write(path, document)
    save()
    protocol = SimpleNamespace(**paths, trials=3, warmup=20, repeat=50, verify_batches=2)
    for cell in document["cells"]:
        if cell["status"] != "complete":
            compare_cell(cell, protocol, document, save)
    document["status"] = "complete"
    document["gaps"] = [dict(cell=cell["id"], **gap) for cell in document["cells"]
                        for gap in cell["baseline_plan"]["unavailable"]]
    save()
    write_comparison_csv(args.output_dir / "comparison.csv", document)
    print(json.dumps(dict(status=document["status"], cells=len(document["cells"]),
                          unmatched_baselines=len(document["gaps"]))))


def orchestrate(args, run=subprocess.run):
    validate_build(args.build_dir)
    current = identity(args)
    path = args.output_dir / "status.json"
    if path.exists():
        state = read(path)
        if not args.resume or state.get("identity") != current:
            raise ValueError("resume requires the same build, source, GPU, workloads and baseline identity")
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError("use a fresh output directory or a matching --resume")
        state = dict(identity=current, status="pending", phases={}, gaps=[
            ("xD real/complex has a separate three-cell user-cache calibration scope" if args.include_xd else
             "xD real/complex calibration and replay are a separate pending GPU campaign"),
            "application component timing is a separate campaign",
            f"{args.stage_point_budget} stage templates are a finite sample, not complete inventory qualification"])
    plan = commands(args)
    write(args.output_dir / "manifest.json", dict(identity=current,
          commands=[dict(phase=name, argv=command) for name, command in plan],
          scope=read(args.workloads)["scope"], comparison_policy="finalized registry public AUTO; alternating paired trials"))
    if args.dry_run:
        state["status"] = "dry-run" if not any(p.get("status") == "complete" for p in state["phases"].values()) else state["status"]
        write(path, state)
        print(json.dumps(read(args.output_dir / "manifest.json"), indent=2))
        return 0
    try:
        for name, command in plan:
            if identity(args) != current:
                raise ValueError("measurement inputs changed during campaign")
            if state["phases"].get(name, {}).get("status") == "complete":
                gate(name, args.output_dir, args.workloads)
                continue
            state["status"] = "running"
            state["phases"][name] = dict(status="running", command=command)
            write(path, state)
            if name == "xd":
                (args.output_dir / "xd").mkdir(exist_ok=True)
            with (args.output_dir / (name + ".log")).open("a") as log:
                completed = run(command, cwd=ROOT, env=environment(args, name != "calibrate"),
                                stdout=log, stderr=subprocess.STDOUT, check=False)
            if completed.returncode:
                raise RuntimeError(name + " exited " + str(completed.returncode))
            gate(name, args.output_dir, args.workloads)
            if identity(args) != current:
                raise ValueError("measurement inputs changed during " + name)
            state["phases"][name]["status"] = "complete"
            write(path, state)
        finalized = read(args.output_dir / "finalized/calibration_manifest.json")
        state["calibration_gaps"] = dict(service_gaps=finalized.get("service_gaps", []),
                                        composition=finalized.get("composition_calibration"),
                                        cost_model_status=finalized.get("cost_model_status"))
        state["baseline_gaps"] = read(args.output_dir / "comparison.json").get("gaps", [])
        state["status"] = "complete-finite-with-gaps" if args.include_xd else "complete-finite-1d-with-gaps"
        state.pop("error", None)
        write(path, state)
        return 0
    except Exception as error:
        state["status"] = "failed"
        state["error"] = str(error)
        if "name" in locals():
            state["phases"].setdefault(name, {})["status"] = "failed"
        write(path, state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--workloads", type=Path, default=WORKLOADS)
    parser.add_argument("--stage-point-budget", type=int, default=8)
    parser.add_argument("--search-budget", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--vkfft-binary", type=Path)
    parser.add_argument("--gpuntt-binary", type=Path)
    parser.add_argument("--fht-python", type=Path)
    parser.add_argument("--include-xd", action="store_true", help="run the separate finite public xD cache calibration after 1D comparison")
    parser.add_argument("--compare-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ("build_dir", "output_dir", "workloads", "vkfft_binary", "gpuntt_binary", "fht_python"):
        if getattr(args, name):
            setattr(args, name, getattr(args, name).resolve())
    if not args.gpu_uuid.startswith("GPU-") or "," in args.gpu_uuid:
        parser.error("select one GPU UUID")
    if args.stage_point_budget < 1 or args.search_budget < 1:
        parser.error("stage and search budgets must be positive")
    try:
        workloads=read(args.workloads)
        if not isinstance(workloads.get("workloads"), list) or not workloads["workloads"]:
            raise ValueError("workloads must contain a nonempty workload list")
        if args.compare_worker:
            compare_worker(args)
            return 0
        return orchestrate(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
