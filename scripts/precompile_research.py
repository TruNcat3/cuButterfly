#!/usr/bin/env python3
"""Export and precompile the finite research specialization population.

Export asks the linked benchmark for its candidate inventory and delegates the
research-specific legality decision to ``research_compile_requests``.  Compile
then consumes only the exported module manifest and calls the standalone module
compiler with bounded process concurrency.  Neither phase promotes a mapping
or claims runtime correctness; those remain benchmark/selector evidence.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

from calibration_space import candidate_id, runtime_candidates


SCHEMA = "cubutterfly-research-precompile-v1"
COMPILATION_SCHEMA = "cubutterfly-research-compilation-v1"
POLICY = "research"
ROOT = Path(__file__).resolve().parents[1]
MODULES_NAME = "modules.json"
COMPILATION_NAME = "compilation.json"
SEED_SNAPSHOT_NAME = "mapping_seed_snapshot.json"
CANDIDATE_SOURCES = ("runtime", "seeds")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path):
    path = Path(path).expanduser().resolve()
    result = {"path": str(path), "exists": path.is_file(), "sha256": None}
    if result["exists"]:
        result["sha256"] = _sha256(path)
    return result


def _atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")
    os.replace(temporary, path)


def _as_path(value, base):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def _compute_sm(value):
    text = str(value).strip().lower().replace("compute_", "")
    if text.startswith("sm"):
        text = text[2:].lstrip("_")
    if "." in text:
        major, minor = text.split(".", 1)
        if major.isdigit() and minor.isdigit() and minor:
            return int(major) * 10 + int(minor[0])
    if text.isdigit():
        number = int(text)
        return number * 10 if number < 10 else number
    raise ValueError(f"unsupported compute capability: {value}")


def _profile_compute_capability(profile):
    hardware = profile.get("hardware") if isinstance(profile.get("hardware"), dict) else {}
    value = profile.get("compute_capability", hardware.get("compute_capability"))
    if value in (None, ""):
        raise ValueError("profile is missing compute_capability")
    return str(value), _compute_sm(value)


def _load_workloads(path):
    specification = json.loads(Path(path).read_text())
    if specification.get("schema") != "cubutterfly-install-search-v1":
        raise ValueError("unsupported research workload schema")
    workloads = specification.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("research workload file has no workloads")
    return specification, workloads


def _binary_name(workload):
    return "cuntt_bench" if workload.get("operator", "fft") == "ntt" else "cubutterfly_bench"


def _cell_id(workload):
    return "cell-" + hashlib.sha256(_canonical(workload).encode()).hexdigest()[:20]


def _run_candidate_command(command, root):
    return subprocess.run([str(item) for item in command], cwd=str(root), text=True,
                          capture_output=True, check=True)


def _device_identity(binary, root):
    completed = _run_candidate_command([binary, "--device-identity"], root)
    try:
        row = next(csv.DictReader(io.StringIO(completed.stdout)))
    except (StopIteration, csv.Error) as error:
        raise ValueError("cubutterfly_bench returned no device identity") from error
    required = ("device", "compute_capability", "global_memory_bytes")
    if any(not row.get(field) for field in required):
        raise ValueError("cubutterfly_bench returned an incomplete device identity")
    try:
        memory = int(row["global_memory_bytes"])
    except (TypeError, ValueError) as error:
        raise ValueError("cubutterfly_bench returned invalid global_memory_bytes") from error
    if memory <= 0:
        raise ValueError("cubutterfly_bench returned non-positive global_memory_bytes")
    return {
        "device": row["device"],
        "compute_capability": row["compute_capability"],
        "global_memory_bytes": memory,
        "source": "cubutterfly_bench",
    }


def _profile_identity(profile):
    hardware = profile.get("hardware") if isinstance(profile.get("hardware"), dict) else {}
    identity = {}
    for field in ("device", "compute_capability", "global_memory_bytes"):
        value = profile.get(field, hardware.get(field))
        if value not in (None, ""):
            identity[field] = value
    if "global_memory_bytes" in identity:
        try:
            identity["global_memory_bytes"] = int(identity["global_memory_bytes"])
        except (TypeError, ValueError) as error:
            raise ValueError("profile has invalid global_memory_bytes") from error
    return identity


def _validate_device_identity(profile, actual):
    expected = _profile_identity(profile)
    for field, value in expected.items():
        if field == "compute_capability":
            matches = _compute_sm(value) == _compute_sm(actual[field])
        elif field == "global_memory_bytes":
            matches = int(value) == int(actual[field])
        else:
            matches = str(value) == str(actual[field])
        if not matches:
            raise ValueError(f"calibration profile differs from target: {field}")


def _candidate_coverage(candidate_source):
    if candidate_source == "runtime":
        return {
            "source": "runtime",
            "scope": "runtime-enumerated",
            "full_space": False,
            "full_runtime_inventory": True,
            "limitation": "runtime candidate enumeration is used",
        }
    return {
        "source": "seeds",
        "scope": "mapping-seeds-snapshot",
        "full_space": False,
        "full_runtime_inventory": False,
        "limitation": "seed candidates are a bounded subset; runtime enumeration is omitted",
    }


def _seed_snapshot(args, output_dir):
    candidate_source = getattr(args, "candidate_source", "runtime")
    if candidate_source == "seeds" and not args.mapping_seeds:
        raise ValueError("candidate-source seeds requires at least one explicit --mapping-seeds file")
    if not args.mapping_seeds:
        return {"seeds": []}, None
    import calibration_seeds

    path = Path(output_dir) / SEED_SNAPSHOT_NAME
    seed_args = SimpleNamespace(mapping_seeds=args.mapping_seeds, resume_search=False)
    return calibration_seeds.seed_snapshot(seed_args, path), path


def _candidate_points(binary, workload, root, snapshot, candidate_source="runtime"):
    workload = {"operator": "fft", **workload}
    if candidate_source not in CANDIDATE_SOURCES:
        raise ValueError(f"unsupported candidate source: {candidate_source}")
    if candidate_source == "runtime":
        points = list(runtime_candidates(binary, workload,
                                         lambda command: _run_candidate_command(command, root)))
    else:
        points = []
    by_id = {}
    for point in points:
        by_id[candidate_id(point)] = point
    if snapshot.get("seeds"):
        import calibration_seeds

        for candidate in calibration_seeds.candidates_for_workload(snapshot, workload):
            point = candidate["point"]
            by_id.setdefault(candidate_id(point), point)
    return list(by_id.values())


def _compile_request(point, policy=POLICY):
    """Load the policy module only when export actually reaches projection."""
    projector = importlib.import_module("research_compile_requests")
    request = projector.compile_request(point, policy=policy)
    if request is not None and not isinstance(request, dict):
        raise ValueError("compile_request must return a mapping or None")
    return request


def _module_id(request):
    return hashlib.sha256(_canonical(request).encode()).hexdigest()


def _reason_count(reasons, reason):
    reasons[reason] = reasons.get(reason, 0) + 1


def _source_identity(root):
    scripts = Path(root) / "scripts"
    return {
        "precompile_research.py": _file_identity(scripts / "precompile_research.py"),
        "compile_module.py": _file_identity(scripts / "compile_module.py"),
        "research_compile_requests.py": _file_identity(scripts / "research_compile_requests.py"),
        "resident_mapping.py": _file_identity(scripts / "resident_mapping.py"),
    }


def _export(args):
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    specification, workloads = _load_workloads(args.workloads)
    profile = json.loads(args.profile.read_text())
    compute_capability, sm = _profile_compute_capability(profile)
    device_identity = _device_identity(args.build_dir / "cubutterfly_bench", args.root)
    _validate_device_identity(profile, device_identity)
    snapshot, snapshot_path = _seed_snapshot(args, output_dir)
    coverage = _candidate_coverage(args.candidate_source)

    modules = {}
    cells = []
    binary_identities = {}
    export_complete = True
    previous_policy = os.environ.get("CUBUTTERFLY_COMPILE_MODE")
    os.environ["CUBUTTERFLY_COMPILE_MODE"] = args.policy
    try:
        for index, raw_workload in enumerate(workloads, 1):
            workload = {"operator": "fft", **raw_workload}
            cell = {
                "id": _cell_id(workload),
                "workload": workload,
                "binary": _binary_name(workload),
                "candidate_source": args.candidate_source,
                "coverage": coverage,
                "candidate_count": 0,
                "projected_count": 0,
                "module_count": 0,
                "no_jit_count": 0,
                "cannot_project_count": 0,
                "cannot_project_reasons": {},
                "cannot_project": [],
                "module_ids": [],
            }
            binary = args.build_dir / cell["binary"]
            if cell["binary"] not in binary_identities:
                binary_identities[cell["binary"]] = _file_identity(binary)
            try:
                points = _candidate_points(binary, workload, args.root, snapshot,
                                           args.candidate_source)
            except Exception as error:
                cell["enumeration_error"] = str(error)
                export_complete = False
                cells.append(cell)
                print(f"exported cell {index}/{len(workloads)}: {cell['id']}", flush=True)
                continue

            cell["candidate_count"] = len(points)
            if args.candidate_source == "seeds" and not points:
                cell["enumeration_error"] = (
                    "candidate-source seeds produced no candidates from the "
                    "--mapping-seeds snapshot"
                )
                export_complete = False
            cell_modules = set()
            for point in points:
                try:
                    request = _compile_request(point, args.policy)
                except ValueError as error:
                    reason = str(error) or type(error).__name__
                    cell["cannot_project_count"] += 1
                    _reason_count(cell["cannot_project_reasons"], reason)
                    cell["cannot_project"].append({"candidate_id": candidate_id(point),
                                                     "reason": reason})
                    export_complete = False
                    continue
                if request is None:
                    cell["no_jit_count"] += 1
                    continue
                identifier = _module_id(request)
                cell["projected_count"] += 1
                cell_modules.add(identifier)
                if identifier not in modules:
                    modules[identifier] = {
                        "id": identifier,
                        "request": request,
                        "representative_point": point,
                        "representative_cell": cell["id"],
                        "cells": [cell["id"]],
                        "policy": args.policy,
                        "candidate_source": args.candidate_source,
                        "coverage": coverage,
                    }
                elif cell["id"] not in modules[identifier]["cells"]:
                    modules[identifier]["cells"].append(cell["id"])
            cell["module_ids"] = sorted(cell_modules)
            cell["module_count"] = len(cell_modules)
            cells.append(cell)
            print(f"exported cell {index}/{len(workloads)}: {cell['id']}", flush=True)
    finally:
        if previous_policy is None:
            os.environ.pop("CUBUTTERFLY_COMPILE_MODE", None)
        else:
            os.environ["CUBUTTERFLY_COMPILE_MODE"] = previous_policy

    manifest = {
        "schema": SCHEMA,
        "status": "ready" if export_complete else "incomplete",
        "complete": export_complete,
        "policy": args.policy,
        "candidate_source": args.candidate_source,
        "coverage": coverage,
        "root": str(args.root),
        "mathdx": str(args.mathdx),
        "nvcc": str(_nvcc_path(args)),
        "cache": str(args.cache),
        "profile": {
            "path": str(args.profile),
            "sha256": _sha256(args.profile),
            "compute_capability": compute_capability,
            "sm": sm,
        },
        "device_identity": device_identity,
        "workloads": {
            "path": str(args.workloads),
            "sha256": _sha256(args.workloads),
            "schema": specification.get("schema"),
            "count": len(workloads),
        },
        "binaries": binary_identities,
        "source": _source_identity(args.root),
        "mapping_seed_snapshot": ({
            "path": str(snapshot_path),
            "sha256": _sha256(snapshot_path),
        } if snapshot_path else None),
        "cells": cells,
        "module_count": len(modules),
        "modules_path": str(output_dir / MODULES_NAME),
        "compilation_path": str(output_dir / COMPILATION_NAME),
        "modules": list(modules.values()),
    }
    _atomic_write(output_dir / MODULES_NAME, manifest)
    return manifest


def _load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported precompile module manifest schema")
    modules = manifest.get("modules")
    if not isinstance(modules, list):
        raise ValueError("module manifest has no module list")
    for module in modules:
        request = module.get("request") if isinstance(module, dict) else None
        identifier = module.get("id") if isinstance(module, dict) else None
        if not isinstance(request, dict) or identifier != _module_id(request):
            raise ValueError("module manifest contains a non-canonical module id")
    return manifest


def _nvcc_path(args):
    if args.nvcc is not None:
        return args.nvcc
    value = os.environ.get("CUBUTTERFLY_NVCC")
    if value:
        return Path(value).expanduser().resolve()
    found = shutil.which("nvcc")
    return Path(found).resolve() if found else Path("nvcc")


def _compile_one(task):
    started = time.monotonic()
    try:
        compiler = importlib.import_module("compile_module")
        path = compiler.compile_mapping(
            task["request"], root=Path(task["root"]), mathdx=Path(task["mathdx"]),
            nvcc=Path(task["nvcc"]), sm=int(task["sm"]), cache=Path(task["cache"]),
            timeout=task["timeout"],
        )
        if path in (None, ""):
            raise RuntimeError("compile_mapping returned no module path")
        return {
            "id": task["id"],
            "path": str(path),
            "status": "compiled",
            "elapsed_seconds": time.monotonic() - started,
            "error": None,
        }
    except Exception as error:
        return {
            "id": task["id"],
            "path": None,
            "status": "failed",
            "elapsed_seconds": time.monotonic() - started,
            "error": str(error) or type(error).__name__,
        }


def _journal(manifest_path, manifest, results, status, interrupted=False):
    ordered = [results[key] for key in sorted(results)]
    successful = sum(row["status"] == "compiled" for row in ordered)
    failed = sum(row["status"] == "failed" for row in ordered)
    complete = (status == "complete" and manifest.get("complete", True)
                and successful == len(manifest["modules"]))
    return {
        "schema": COMPILATION_SCHEMA,
        "status": status,
        "complete": complete,
        "interrupted": interrupted,
        "manifest": str(manifest_path),
        "policy": manifest.get("policy", POLICY),
        "candidate_source": manifest.get("candidate_source", "runtime"),
        "coverage": manifest.get(
            "coverage",
            _candidate_coverage(manifest.get("candidate_source", "runtime")),
        ),
        "expected_modules": len(manifest["modules"]),
        "completed_modules": len(ordered),
        "successful_modules": successful,
        "failed_modules": failed,
        "results": ordered,
    }


def _hide_cuda_visible_devices():
    marker = object()
    previous = os.environ.pop("CUDA_VISIBLE_DEVICES", marker)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return marker, previous


def _restore_cuda_visible_devices(marker, previous):
    if previous is marker:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = previous


def _compile_manifest(args, manifest):
    output_dir = args.output_dir
    manifest_path = output_dir / MODULES_NAME
    results = {}
    journal_path = output_dir / COMPILATION_NAME
    _atomic_write(journal_path, _journal(manifest_path, manifest, results, "running"))
    tasks = [{
        "id": module["id"],
        "request": module["request"],
        "root": args.root,
        "mathdx": args.mathdx,
        "nvcc": _nvcc_path(args),
        "sm": manifest["profile"]["sm"],
        "cache": args.cache,
        "timeout": args.timeout,
    } for module in manifest["modules"]]
    if not tasks:
        status = "complete" if manifest.get("complete", True) else "failed"
        _atomic_write(journal_path, _journal(manifest_path, manifest, results, status))
        return 0 if status == "complete" else 1

    marker, previous = _hide_cuda_visible_devices()
    executor = None
    futures = {}
    interrupted = False
    try:
        executor = ProcessPoolExecutor(max_workers=args.jobs)
        for task in tasks:
            futures[executor.submit(_compile_one, task)] = task["id"]
        for future in as_completed(futures):
            identifier = futures[future]
            try:
                result = future.result()
            except KeyboardInterrupt:
                raise
            except Exception as error:
                result = {
                    "id": identifier,
                    "path": None,
                    "status": "failed",
                    "elapsed_seconds": 0.0,
                    "error": str(error) or type(error).__name__,
                }
            results[identifier] = result
            _atomic_write(journal_path, _journal(manifest_path, manifest, results, "running"))
    except KeyboardInterrupt:
        interrupted = True
        for future in futures:
            future.cancel()
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)
        _atomic_write(journal_path, _journal(manifest_path, manifest, results,
                                             "partial", interrupted=True))
        return 130
    finally:
        if executor is not None and not interrupted:
            executor.shutdown(wait=True)
        _restore_cuda_visible_devices(marker, previous)

    status = ("complete" if manifest.get("complete", True)
              and all(row["status"] == "compiled" for row in results.values())
              else "failed")
    _atomic_write(journal_path, _journal(manifest_path, manifest, results, status))
    return 0 if status == "complete" else 1


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("export", "compile", "all"), default="all")
    parser.add_argument("--policy", choices=("research", "auto", "on-demand", "precompiled"))
    parser.add_argument("--candidate-source", choices=CANDIDATE_SOURCES, default="runtime",
                        help="candidate inventory source for export (default: runtime)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--workloads", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--mapping-seeds", type=Path, action="append", default=[])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--mathdx", type=Path, default=Path("external/mathdx/nvidia/mathdx"))
    parser.add_argument("--nvcc", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--timeout", type=float, default=0)
    return parser


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    if args.jobs < 1 or args.timeout < 0:
        raise ValueError("jobs must be positive and timeout must be nonnegative")
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.policy is None:
        args.policy = os.environ.get("CUBUTTERFLY_COMPILE_MODE", "research") or "research"
    args.root = args.root.expanduser().resolve()
    args.mathdx = _as_path(args.mathdx, args.root)
    if args.build_dir is not None:
        args.build_dir = _as_path(args.build_dir, args.root)
    if args.workloads is not None:
        args.workloads = _as_path(args.workloads, args.root)
    if args.profile is not None:
        args.profile = _as_path(args.profile, args.root)
    args.mapping_seeds = [_as_path(path, args.root) for path in args.mapping_seeds]
    if args.nvcc is not None:
        args.nvcc = args.nvcc.expanduser().resolve()
    if args.cache is not None:
        args.cache = _as_path(args.cache, args.root)
    elif os.environ.get("CUBUTTERFLY_JIT_CACHE"):
        args.cache = Path(os.environ["CUBUTTERFLY_JIT_CACHE"]).expanduser().resolve()
    elif os.environ.get("XDG_CACHE_HOME"):
        args.cache = (Path(os.environ["XDG_CACHE_HOME"]).expanduser()
                      / "cubutterfly" / "modules").resolve()
    else:
        args.cache = (Path.home() / ".cache" / "cubutterfly" / "modules").resolve()
    return args


def _require_export_args(args):
    missing = [name for name in ("build_dir", "workloads", "profile")
               if getattr(args, name) is None]
    if missing:
        raise ValueError("export mode requires --" + ", --".join(name.replace("_", "-") for name in missing))
    if args.candidate_source == "seeds" and not args.mapping_seeds:
        raise ValueError("candidate-source seeds requires at least one explicit --mapping-seeds file")


def main(argv=None):
    args = parse_args(argv)
    if args.mode in ("export", "all"):
        _require_export_args(args)
        manifest = _export(args)
        export_code = 0 if manifest["complete"] else 1
        if args.mode == "export":
            print(json.dumps({"status": manifest["status"], "module_count": manifest["module_count"],
                              "path": manifest["modules_path"]}, indent=2))
            return export_code
        compile_code = _compile_manifest(args, manifest)
        return 0 if export_code == 0 and compile_code == 0 else 1

    manifest = _load_manifest(args.output_dir / MODULES_NAME)
    code = _compile_manifest(args, manifest)
    print(json.dumps({"status": "complete" if code == 0 else "failed",
                      "module_count": manifest["module_count"],
                      "path": str(args.output_dir / COMPILATION_NAME)}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
