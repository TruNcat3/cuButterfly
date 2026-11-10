#!/usr/bin/env python3
"""Run one bounded xD coverage matrix for every public butterfly operator.

This is a coverage campaign rather than a claim of global optimality. It
records both the selected xD plan and a resident-t128 control. The historical
xor-zeta spelling is a compatibility alias for subset-zeta, not a seventh
kernel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import statistics
from datetime import datetime, timezone
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY = ROOT / "build-e13-shared" / "libcuntt.so"
DEFAULT_BENCH = ROOT / "build-e13-shared" / "cubutterfly_plan_bench"
SCHEMA = 3

OPERATORS: tuple[dict[str, Any], ...] = (
    {"name": "fft", "precisions": ("fp32", "fp64")},
    {"name": "fwht", "precisions": ("fp32", "fp64")},
    {"name": "ntt", "precisions": ("uint32", "uint64")},
    {"name": "subset-zeta", "precisions": ("uint32",)},
    {"name": "superset-zeta", "precisions": ("uint32",)},
    {"name": "structured-2x2", "precisions": ("fp32", "fp64")},
)
PROFILE_SHAPES: dict[str, tuple[tuple[int, ...], ...]] = {
    # The original bounded matrix is kept stable for quick regression checks.
    "smoke": ((4, 8), (2, 4, 8)),
    # xD plan_bench accepts rank-2/rank-3 shapes. Rank-1 is covered by the
    # 74/284 semantic acceptance matrices and is intentionally not sent to
    # this rank-aware entry point.
    "coverage": ((8, 8), (32, 32), (4, 4, 4), (8, 8, 8)),
    # Adds the large working sets used by the paper's scaling discussion.
    "scale": ((128, 128), (16, 16, 16), (32, 32, 32),
               (64, 64, 64)),
}
PROFILE_BATCHES: dict[str, tuple[int, ...]] = {
    "smoke": (1, 4),
    "coverage": (1, 4, 16),
    "scale": (1, 4, 16),
}
# Backward-compatible names for callers that imported the smoke matrix.
SHAPES = PROFILE_SHAPES["smoke"]
BATCHES = PROFILE_BATCHES["smoke"]
DIRECTIONS: tuple[bool, ...] = (False, True)
LINE_RE = re.compile(
    r"^algorithm=(?P<algorithm>.*?),logical_shape=(?P<shape>[^,]+),"
    r"physical_shape=(?P<physical>[^,]+),batch=(?P<batch>\d+),"
    r"workspace_bytes=(?P<workspace>\d+),trials=(?P<trials>\d+),"
    r"kernel_ms=(?P<kernel>[-+0-9.eE]+)(?P<rest>.*)$"
)


def _timing_protocol(warmup: int, repeat: int, trials: int) -> str:
    return (
        "plan_bench CUDA events; planning/calibration/warmup excluded; "
        f"warmup={warmup}; repeat={repeat}; trials={trials}"
    )


# Keep the historical constant available to callers that imported it, while
# making manifests use the actual CLI values through _timing_protocol.
REPORT_PROTOCOL = _timing_protocol(warmup=2, repeat=25, trials=1)
RUNTIME_LIBRARY_SONAME = "libcuntt.so"


def _median(values: Iterable[float]) -> float | None:
    samples = [float(value) for value in values]
    return float(statistics.median(samples)) if samples else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _shape_text(shape: Iterable[int]) -> str:
    return "x".join(str(int(value)) for value in shape)


def _default_gpu_selector() -> str:
    for variable in ("CUBUTTERFLY_GPU_UUID", "CUDA_VISIBLE_DEVICES"):
        value = os.environ.get(variable, "")
        selector = value.split(",", 1)[0].strip()
        if selector and selector != "-1":
            return selector
    # Preserve the old no-argument invocation without tying it to one
    # machine's physical UUID.  Callers can pass an explicit UUID or index.
    return "0"


def _gpu_info(gpu_uuid: str | None = None) -> dict[str, Any]:
    selector = (gpu_uuid or "").strip() or None
    command = ["nvidia-smi"]
    if selector is not None:
        command.append(f"--id={selector}")
    command += [
        "--query-gpu=uuid,name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=False)
    except OSError as exc:
        return {"uuid": selector, "selector": selector, "query_error": str(exc)}
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode == 0 and len(lines) == 1:
        fields = [field.strip() for field in lines[0].split(",", 3)]
        try:
            if len(fields) >= 4:
                return {"uuid": fields[0], "selector": selector,
                        "model": fields[1], "memory_mib": int(fields[2]),
                        "compute_capability": fields[3]}
        except ValueError:
            pass
    return {"uuid": selector, "selector": selector,
            "query_error": result.stderr.strip() or "nvidia-smi failed"}


def _parse_output(stdout: str) -> dict[str, Any]:
    match: re.Match[str] | None = None
    for line in stdout.splitlines():
        candidate = LINE_RE.match(line.strip())
        if candidate is not None:
            match = candidate
    if match is None:
        raise ValueError("plan benchmark returned no timing record")
    values = match.groupdict()
    result: dict[str, Any] = {
        "algorithm": values["algorithm"],
        "reported_shape": values["shape"],
        "physical_shape": values["physical"],
        "batch": int(values["batch"]),
        "workspace_bytes": int(values["workspace"]),
        "trials": int(values["trials"]),
        "kernel_ms": float(values["kernel"]),
    }
    if not math.isfinite(result["kernel_ms"]) or result["kernel_ms"] <= 0:
        raise ValueError(f"invalid kernel_ms={result['kernel_ms']!r}")
    for field in ("speedup_vs_cufft", "cufft_ms"):
        found = re.search(rf",{re.escape(field)}=([-+0-9.eE]+)", values["rest"])
        if found:
            result[field] = float(found.group(1))
            if not math.isfinite(result[field]) or result[field] <= 0:
                raise ValueError(f"invalid {field}={result[field]!r}")
    correctness = re.search(r",correct=([01])(?:,|$)", values["rest"])
    if correctness is not None:
        result["correct"] = correctness.group(1) == "1"
        verified = re.search(r",verified_batches=(\d+)(?:,|$)", values["rest"])
        if verified is not None:
            result["verified_batches"] = int(verified.group(1))
    # mapping_json is emitted last because it is a complete JSON object whose
    # nested fields contain commas. Keep the original compact text so a
    # downstream correctness run can pass it back as an explicit mapping.
    mapping = re.search(r",mapping_json=(?P<mapping>\{.*\})\s*$",
                        values["rest"])
    if mapping is not None:
        mapping_text = mapping.group("mapping")
        try:
            mapping_value = json.loads(mapping_text)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid mapping_json in plan benchmark output") from exc
        if not isinstance(mapping_value, dict):
            raise ValueError("mapping_json in plan benchmark output must be a JSON object")
        result["mapping_json"] = mapping_text
    return result


def _case_id(operator: str, precision: str, shape: tuple[int, ...], batch: int,
             inverse: bool, configuration: str) -> str:
    direction = "inv" if inverse else "fwd"
    return f"{operator}_{precision}_{_shape_text(shape)}_b{batch}_{direction}_{configuration}"


def _read_dynamic_tags(path: Path) -> dict[str, Any]:
    """Read the loader contract without executing the benchmark or library."""
    try:
        result = subprocess.run(
            ["readelf", "-d", str(path)], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
    except OSError as exc:
        raise ValueError(f"cannot inspect ELF dynamic tags for {path}: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or "readelf failed"
        raise ValueError(f"cannot inspect ELF dynamic tags for {path}: {detail}")
    needed = re.findall(r"\(NEEDED\).*?\[([^]]+)\]", result.stdout)
    soname_match = re.search(r"\(SONAME\).*?\[([^]]+)\]", result.stdout)
    runpath_match = re.search(r"\(RUNPATH\).*?\[([^]]*)\]", result.stdout)
    rpath_match = re.search(r"\(RPATH\).*?\[([^]]*)\]", result.stdout)
    return {
        "needed": needed,
        "soname": soname_match.group(1) if soname_match else None,
        "runpath": runpath_match.group(1) if runpath_match else "",
        "rpath": rpath_match.group(1) if rpath_match else "",
    }


def _loader_path_entries(value: str, origin: Path) -> list[Path]:
    entries: list[Path] = []
    for item in value.split(":"):
        if not item:
            continue
        expanded = item.replace("${ORIGIN}", str(origin)).replace("$ORIGIN", str(origin))
        entries.append(Path(expanded))
    return entries


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return left.resolve() == right.resolve()


def _verify_runtime_library(bench: Path, library: Path,
                            environment: dict[str, str]) -> dict[str, Any]:
    """Verify that the benchmark loader will resolve the requested library.

    The benchmark links against the library SONAME, not the path supplied to
    this script.  LD_LIBRARY_PATH therefore has to contain a same-file
    SONAME entry; this also handles a versioned filename with a SONAME symlink.
    """
    bench_tags = _read_dynamic_tags(bench)
    library_tags = _read_dynamic_tags(library)
    if RUNTIME_LIBRARY_SONAME not in bench_tags["needed"]:
        raise ValueError(
            f"benchmark {bench} does not require {RUNTIME_LIBRARY_SONAME}; "
            "cannot verify the requested runtime library"
        )
    if library_tags["soname"] != RUNTIME_LIBRARY_SONAME:
        raise ValueError(
            f"requested library {library} has SONAME {library_tags['soname']!r}; "
            f"expected {RUNTIME_LIBRARY_SONAME!r}"
        )
    if not bench_tags["runpath"] and not bench_tags["rpath"]:
        raise ValueError(
            f"benchmark {bench} has neither DT_RUNPATH nor DT_RPATH; "
            "the CMake build-local loader contract is missing"
        )

    origin = bench.parent.resolve()
    library_path_dirs = _loader_path_entries(
        environment.get("LD_LIBRARY_PATH", ""), origin
    )
    if bench_tags["runpath"]:
        # DT_RUNPATH is searched after LD_LIBRARY_PATH.
        search_dirs = library_path_dirs + _loader_path_entries(
            bench_tags["runpath"], origin
        )
    else:
        # A legacy DT_RPATH with no DT_RUNPATH is searched before
        # LD_LIBRARY_PATH by the dynamic loader.
        search_dirs = _loader_path_entries(bench_tags["rpath"], origin) + library_path_dirs
    resolved: Path | None = None
    for directory in search_dirs:
        candidate = directory / RUNTIME_LIBRARY_SONAME
        if candidate.is_file():
            resolved = candidate.resolve()
            break
    if resolved is None:
        raise ValueError(
            f"cannot resolve {RUNTIME_LIBRARY_SONAME} for {bench} through "
            "LD_LIBRARY_PATH/RUNPATH"
        )
    requested = library.resolve()
    if not _same_file(resolved, requested):
        raise ValueError(
            f"runtime loader resolves {RUNTIME_LIBRARY_SONAME} to {resolved}, "
            f"not requested library {requested}"
        )
    return {
        "soname": RUNTIME_LIBRARY_SONAME,
        "requested": str(requested),
        "resolved": str(resolved),
        "search_path": str(environment.get("LD_LIBRARY_PATH", "")),
    }


def _runtime_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    if args.gpu_uuid:
        environment["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    # Research runs must be able to materialize factor I/O and other
    # on-demand templates.  Keep this explicit in every child process so a
    # clean shell cannot silently exercise a different compilation policy.
    environment["CUBUTTERFLY_COMPILE_MODE"] = getattr(
        args, "compile_mode", "research"
    )

    # plan_bench has no --library option.  Make the requested shared object
    # the one resolved by the child process instead of loading it through
    # LD_PRELOAD.  Preloading a CUDA-bearing libcuntt image changes its
    # loading/symbol-binding path and was observed to alter backend
    # availability.
    environment.pop("LD_PRELOAD", None)
    library_dir = str(args.library.resolve().parent)
    inherited_path = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = (
        f"{library_dir}:{inherited_path}" if inherited_path else library_dir
    )
    return environment


def _run_case(args: argparse.Namespace, operator: str, precision: str,
              shape: tuple[int, ...], batch: int, inverse: bool,
              configuration: str) -> dict[str, Any]:
    command = [
        str(args.bench), "--operator", operator, "--shape", _shape_text(shape),
        "--batch", str(batch), "--precision", precision,
        "--execution-mode", "xd", "--warmup", str(args.warmup),
        "--repeat", str(args.repeat), "--trials", str(args.trials),
        "--policy", args.policy,
    ]
    if configuration == "resident-t128":
        command += ["--algorithm", "xd:resident-t128"]
    if inverse:
        command.append("--inverse")
    if operator == "ntt":
        command += ["--modulus", str(args.modulus)]
    if operator == "structured-2x2":
        command += ["--matrix", args.matrix]
    if operator == "fft" and not inverse and args.compare_cufft:
        command.append("--compare-cufft")
    if operator == "fft" and not inverse and getattr(args, "verify", False):
        command.append("--verify")
    environment = _runtime_environment(args)
    result = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=False)
    record: dict[str, Any] = {
        "id": _case_id(operator, precision, shape, batch, inverse, configuration),
        "operator": operator, "precision": precision, "shape": list(shape),
        "batch": batch, "inverse": inverse, "configuration": configuration,
        "command": command, "returncode": result.returncode,
    }
    if result.returncode == 0:
        try:
            parsed = _parse_output(result.stdout)
            record.update(parsed)
            if "mapping_json" in parsed:
                record["mapping_evidence"] = {
                    "status": "observed",
                    "bound": True,
                    "reason": "plan_bench emitted the resolved mapping_json on the timing record",
                }
            else:
                # Old plan_bench binaries remain readable, but their timing
                # cannot be claimed to identify the mapping used by a later
                # independent correctness plan.
                record["mapping_evidence"] = {
                    "status": "unavailable",
                    "bound": False,
                    "reason": "plan_bench output has no mapping_json; measured timing is unbound",
                }
            # An explicit resident control is only a control if the plan
            # actually used that kernel. plan_bench intentionally falls back
            # to a legal composite/flattened plan for unsupported shapes.
            if (configuration == "resident-t128" and
                    "resident-t128" not in parsed.get("algorithm", "")):
                record.update({
                    "status": "unavailable",
                    "error": "resident-t128 unavailable; plan selected "
                             + parsed.get("algorithm", "unknown"),
                })
            elif "correct" not in parsed:
                # A timing-only row is useful evidence, but it is not a
                # correctness pass.  The runner does not request --verify
                # for operators without a complete plan_bench oracle.
                record["status"] = "measured"
            elif parsed["correct"]:
                record["status"] = "passed"
            else:
                record.update({"status": "failed",
                               "error": "benchmark correctness check failed"})
        except ValueError as exc:
            record.update({"status": "failed", "error": str(exc)})
    else:
        error = result.stderr.strip()[-2000:]
        # A valid plan can be inapplicable to a particular shape. Keep this
        # separate from a real execution or parsing failure.
        unavailable_markers = ("unsupported", "not available", "unavailable",
                               "cannot launch", "no candidate", "not supported")
        status = ("unavailable" if any(marker in error.lower()
                                       for marker in unavailable_markers)
                  else "failed")
        record.update({"status": status, "error": error})
    if args.keep_logs:
        log_path = args.output / "logs" / f"{record['id']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(result.stdout + "\n--- stderr ---\n" + result.stderr,
                            encoding="utf-8")
        record["log"] = str(log_path)
    return record


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["id", "operator", "precision", "shape", "batch", "inverse",
              "configuration", "status", "algorithm", "kernel_ms",
              "cufft_ms", "speedup_vs_cufft", "workspace_bytes", "correct",
              "verified_batches", "mapping_json", "mapping_evidence", "error"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["shape"] = _shape_text(output.get("shape", ()))
            writer.writerow(output)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [row for row in rows if row.get("status") == "passed"]
    measured = [row for row in rows if row.get("status") == "measured"]
    completed = passed + measured
    unavailable = [row for row in rows if row.get("status") == "unavailable"]
    failed = [row for row in rows if row.get("status") == "failed"]
    summary: dict[str, Any] = {
        "requested": len(rows), "completed": len(completed),
        "passed": len(passed), "measured": len(measured),
        "unavailable": len(unavailable), "failed": len(failed), "operators": {},
        "configurations": {}, "semantic_aliases": {"xor-zeta": "subset-zeta"},
    }
    for operator in [item["name"] for item in OPERATORS]:
        op_rows = [row for row in rows if row["operator"] == operator]
        op_passed = [row for row in op_rows if row.get("status") == "passed"]
        op_measured = [row for row in op_rows if row.get("status") in
                       ("passed", "measured")]
        op_unavailable = [row for row in op_rows
                          if row.get("status") == "unavailable"]
        op_failed = [row for row in op_rows if row.get("status") == "failed"]
        values = [float(row["kernel_ms"]) for row in op_measured]
        summary["operators"][operator] = {
            "requested": len(op_rows), "completed": len(op_measured),
            "passed": len(op_passed), "measured": sum(
                row.get("status") == "measured" for row in op_measured
            ), "unavailable": len(op_unavailable), "failed": len(op_failed),
            "median_kernel_ms": _median(values),
            "selected_rows": sum(row["configuration"] == "selected" for row in op_measured),
            "resident_t128_rows": sum(row["configuration"] == "resident-t128" for row in op_measured),
        }
    for configuration in ("selected", "resident-t128"):
        config_rows = [row for row in completed if row["configuration"] == configuration]
        values = [float(row["kernel_ms"]) for row in config_rows]
        summary["configurations"][configuration] = {
            "rows": len(config_rows),
            "median_kernel_ms": _median(values),
        }
    paired: list[float] = []
    selected_by_key = {(row["operator"], row["precision"], tuple(row["shape"]),
                       row["batch"], row["inverse"]): row for row in completed
                      if row["configuration"] == "selected"}
    resident_by_key = {(row["operator"], row["precision"], tuple(row["shape"]),
                        row["batch"], row["inverse"]): row for row in completed
                       if row["configuration"] == "resident-t128"}
    for key in sorted(selected_by_key):
        if key in resident_by_key:
            paired.append(float(resident_by_key[key]["kernel_ms"]) /
                          float(selected_by_key[key]["kernel_ms"]))
    summary["selector_vs_resident_t128"] = {
        "paired_rows": len(paired),
        "median_speedup": _median(paired),
        "wins": sum(value > 1.0 for value in paired), "speedups": paired,
    }
    by_operator: dict[str, dict[str, Any]] = {}
    for operator in [item["name"] for item in OPERATORS]:
        values = []
        for key, selected in selected_by_key.items():
            if key[0] != operator or key not in resident_by_key:
                continue
            values.append(float(resident_by_key[key]["kernel_ms"]) /
                          float(selected["kernel_ms"]))
        by_operator[operator] = {
            "paired_rows": len(values),
            "median_speedup": _median(values),
            "wins": sum(value > 1.0 for value in values),
        }
    summary["selector_vs_resident_by_operator"] = by_operator
    fft_external = [row["speedup_vs_cufft"] for row in completed
                    if row["configuration"] == "selected" and "speedup_vs_cufft" in row]
    summary["fft_forward_vs_cufft"] = {
        "rows": len(fft_external),
        "geomean_speedup": math.exp(sum(math.log(float(value)) for value in fft_external) /
                                    len(fft_external)) if fft_external else None,
        "wins": sum(float(value) > 1.0 for value in fft_external),
    }
    return summary


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", default=_default_gpu_selector(),
                        help="GPU UUID or local selector (default: selected visible GPU)")
    parser.add_argument("--policy", choices=("default", "measure"), default="measure")
    parser.add_argument(
        "--compile-mode", choices=("research", "auto", "precompiled"),
        default=os.environ.get("CUBUTTERFLY_COMPILE_MODE", "research"),
        help="template compilation policy passed to the benchmark child",
    )
    parser.add_argument("--profile", choices=tuple(PROFILE_SHAPES), default="smoke",
                        help="coverage matrix: smoke, coverage, or scale")
    parser.add_argument("--operators", default="all",
                        help="comma-separated operator subset for a resumable slice")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=25)
    parser.add_argument("--trials", type=int, default=1)
    # 257 - 1 is divisible by every power-of-two axis through 128.  The
    # previous smoke-only modulus 97 silently made 64/128 NTT rows invalid.
    parser.add_argument("--modulus", type=int, default=257)
    parser.add_argument("--matrix", default="1,0.25,-0.5,1")
    parser.add_argument("--compare-cufft", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="request plan_bench verification where supported (FFT forward)")
    parser.add_argument("--skip-resident-control", action="store_true")
    parser.add_argument("--keep-logs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.bench = args.bench.resolve()
    args.library = args.library.resolve()
    args.output = args.output.resolve()
    requested_operators = [value.strip() for value in args.operators.split(",")
                           if value.strip() and value.strip() != "all"]
    known_operators = {item["name"] for item in OPERATORS}
    unknown = sorted(set(requested_operators) - known_operators)
    if unknown:
        parser.error(f"unknown operators: {', '.join(unknown)}")
    args.selected_operators = (
        list(OPERATORS) if not requested_operators
        else [item for item in OPERATORS if item["name"] in requested_operators]
    )
    if not args.dry_run and (not args.bench.is_file() or not args.library.is_file()):
        parser.error("--bench and --library must exist")
    if args.repeat <= 0 or args.trials <= 0 or args.warmup < 0:
        parser.error("warmup must be >= 0; repeat and trials must be positive")
    if not args.dry_run:
        try:
            args.runtime_library_resolution = _verify_runtime_library(
                args.bench, args.library, _runtime_environment(args)
            )
        except ValueError as exc:
            parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    gpu = _gpu_info(args.gpu_uuid)
    shapes = PROFILE_SHAPES[args.profile]
    batches = PROFILE_BATCHES[args.profile]
    manifest = {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "library": str(args.library),
        "library_sha256": _sha256(args.library) if args.library.is_file() else None,
        "bench": str(args.bench),
        "bench_sha256": _sha256(args.bench) if args.bench.is_file() else None,
        "gpu": gpu,
        "gpu_selector": args.gpu_uuid,
        "gpu_uuid": gpu.get("uuid", args.gpu_uuid),
        "timing_protocol": _timing_protocol(
            warmup=args.warmup, repeat=args.repeat, trials=args.trials
        ),
        "timing": {"warmup": args.warmup, "repeat": args.repeat,
                   "trials": args.trials},
        "compile_mode": args.compile_mode,
        "configurations": (["selected"] if args.skip_resident_control else
                           ["selected", "resident-t128"]),
        "modulus": args.modulus,
        "matrix": args.matrix,
        "runtime_library": str(args.library),
        "runtime_library_resolution": getattr(args, "runtime_library_resolution", None),
        "profile": args.profile, "policy": args.policy,
        "shapes": [list(shape) for shape in shapes],
        "batches": list(batches), "directions": ["forward", "inverse"],
        "operators": [{"name": item["name"], "precisions": list(item["precisions"])}
                      for item in args.selected_operators],
        "semantic_aliases": {"xor-zeta": "subset-zeta"},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for item in args.selected_operators:
        for precision in item["precisions"]:
            for shape in shapes:
                for batch in batches:
                    for inverse in DIRECTIONS:
                        configurations = ["selected"]
                        if not args.skip_resident_control:
                            configurations.append("resident-t128")
                        for configuration in configurations:
                            if args.dry_run:
                                rows.append({
                                    "id": _case_id(item["name"], precision, shape, batch, inverse, configuration),
                                    "operator": item["name"], "precision": precision,
                                    "shape": list(shape), "batch": batch, "inverse": inverse,
                                    "configuration": configuration, "status": "dry-run",
                                })
                                continue
                            print(f"{len(rows) + 1}: {item['name']} {precision} {_shape_text(shape)} "
                                  f"b{batch} {'inverse' if inverse else 'forward'} {configuration}",
                                  flush=True)
                            rows.append(_run_case(args, item["name"], precision, shape, batch,
                                                  inverse, configuration))
                            # Preserve completed cells if the experiment is interrupted.
                            (args.output / "rows.json").write_text(
                                json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (args.output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.output / "rows.csv", rows)
    summary = _summary(rows) if not args.dry_run else {"requested": len(rows), "dry_run": True}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
