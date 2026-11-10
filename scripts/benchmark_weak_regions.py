#!/usr/bin/env python3
"""Run a small, reproducible benchmark/trace matrix for weak regions.

The JSON file is the source of truth for shapes, batch sizes, precision and
trace selection.  Plan construction and calibration are recorded separately
from the steady CUDA-event timing emitted by ``cubutterfly_plan_bench``.
Nsight Compute is optional and only runs for explicitly marked cases.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

try:
    from benchmark.runner_registry import resolve as resolve_runner
except ModuleNotFoundError:  # direct ``python scripts/...`` invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmark.runner_registry import resolve as resolve_runner

ROOT = Path(__file__).resolve().parents[1]
TIMING_RE = re.compile(
    r"^algorithm=(?P<algorithm>.*?),logical_shape=(?P<logical>[^,]+),"
    r"physical_shape=(?P<physical>[^,]+),batch=(?P<batch>\d+),"
    r"workspace_bytes=(?P<workspace>\d+),trials=(?P<trials>\d+),"
    r"kernel_ms=(?P<kernel>[-+0-9.eE]+)(?P<rest>.*)$"
)


def _shape_text(shape: list[int]) -> str:
    return "x".join(str(int(value)) for value in shape)


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") not in (1, 2):
        raise ValueError("config must be an object with schema=1 or schema=2")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("config.cases must be a non-empty list")
    ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each case must be an object")
        required = {"id", "operator", "shape", "batch", "precision"}
        missing = sorted(required - set(case))
        if missing:
            raise ValueError(f"case is missing fields: {', '.join(missing)}")
        case_id = str(case["id"])
        if case_id in ids:
            raise ValueError(f"duplicate case id: {case_id}")
        ids.add(case_id)
        if not isinstance(case["shape"], list) or not case["shape"]:
            raise ValueError(f"{case_id}: shape must be a non-empty list")
        if any(int(value) <= 0 for value in case["shape"]):
            raise ValueError(f"{case_id}: shape extents must be positive")
        if int(case["batch"]) <= 0:
            raise ValueError(f"{case_id}: batch must be positive")
        if case["operator"] == "fft" and case.get("verify") and case.get("inverse"):
            raise ValueError(f"{case_id}: plan_bench verification only supports forward FFT")
        execution_mode = str(case.get("execution_mode", "auto" if len(case["shape"]) == 1 else "xd"))
        if execution_mode not in {"auto", "xd", "composite"}:
            raise ValueError(f"{case_id}: unsupported execution_mode {execution_mode!r}")
        if execution_mode == "xd" and len(case["shape"]) == 1:
            raise ValueError(f"{case_id}: rank-1 cases must use execution_mode=auto or composite")
        runner = str(case.get("runner", "plan"))
        if runner not in {"plan", "ntt"}:
            raise ValueError(f"{case_id}: unsupported primary runner {runner!r}")
        if runner != "plan" and case.get("trace"):
            raise ValueError(f"{case_id}: trace is currently supported only for the plan runner")
        baselines = case.get("baselines", [])
        if not isinstance(baselines, list) or any(not isinstance(item, str) or not item for item in baselines):
            raise ValueError(f"{case_id}: baselines must be a list of names")
    runners = value.get("runners", {})
    if "search_budget" in value and int(value["search_budget"]) < 0:
        raise ValueError("search_budget must be non-negative (zero means unbounded)")
    if "xd_tune_max_bytes" in value and int(value["xd_tune_max_bytes"]) < 0:
        raise ValueError("xd_tune_max_bytes must be non-negative")
    if runners and (not isinstance(runners, dict) or any(
            not isinstance(name, str) or not isinstance(spec, dict)
            for name, spec in runners.items())):
        raise ValueError("config.runners must map names to objects")
    for name, spec in runners.items():
        if not isinstance(spec.get("executable"), str) or not spec["executable"]:
            raise ValueError(f"runners.{name}.executable must be a non-empty string")
        kind = spec.get("kind", name)
        if kind not in {"plan", "ntt", "gpuntt"}:
            raise ValueError(f"runners.{name}.kind is unsupported: {kind!r}")
    timing = value.get("timing", {})
    if not isinstance(timing, dict):
        raise ValueError("config.timing must be an object")
    for field in ("warmup", "repeat", "trials"):
        if int(timing.get(field, 0 if field == "warmup" else 1)) < 0:
            raise ValueError(f"timing.{field} must be non-negative")
    return value


def _parse_timing(stdout: str) -> dict[str, Any]:
    match: re.Match[str] | None = None
    for line in stdout.splitlines():
        candidate = TIMING_RE.match(line.strip())
        if candidate:
            match = candidate
    if match is None:
        raise ValueError("benchmark emitted no timing record")
    values = match.groupdict()
    result: dict[str, Any] = {
        "algorithm": values["algorithm"],
        "logical_shape": values["logical"],
        "physical_shape": values["physical"],
        "batch": int(values["batch"]),
        "workspace_bytes": int(values["workspace"]),
        "trials": int(values["trials"]),
        "kernel_ms": float(values["kernel"]),
    }
    rest = values["rest"]
    for field in ("cufft_ms", "speedup_vs_cufft", "relative_error",
                  "relative_l2_error", "max_absolute_error", "tolerance"):
        found = re.search(rf",{re.escape(field)}=([-+0-9.eE]+)(?:,|$)", rest)
        if found:
            result[field] = float(found.group(1))
    found = re.search(r",correct=([01])(?:,|$)", rest)
    if found:
        result["correct"] = found.group(1) == "1"
    found = re.search(r",verified_batches=(\d+)(?:,|$)", rest)
    if found:
        result["verified_batches"] = int(found.group(1))
    mapping = re.search(r",mapping_json=(?P<mapping>\{.*\})\s*$", rest)
    if mapping:
        try:
            result["mapping_json"] = json.loads(mapping.group("mapping"))
        except json.JSONDecodeError as exc:
            raise ValueError("benchmark emitted invalid mapping_json") from exc
    return result


def _parse_csv_timing(stdout: str) -> dict[str, Any]:
    """Parse the single CSV record emitted by cuntt_bench.

    The parser intentionally keeps the complete row as provenance instead of
    reducing it to a timing scalar.  This makes a configured NTT baseline
    auditable for backend, order and correctness fields.
    """
    lines = [line for line in stdout.splitlines() if line.strip()]
    for index, line in enumerate(lines[:-1]):
        if "kernel_ms" not in line or "," not in line:
            continue
        rows = list(csv.DictReader(io.StringIO(f"{line}\n{lines[index + 1]}\n")))
        if len(rows) != 1:
            continue
        row = rows[0]
        if not row.get("kernel_ms"):
            continue
        result: dict[str, Any] = {
            "kernel_ms": float(row["kernel_ms"]),
            "correct": row.get("correct", "1") == "1",
            "backend": row.get("backend", ""),
        }
        if row.get("mapping_json"):
            try:
                result["mapping_json"] = json.loads(row["mapping_json"])
            except json.JSONDecodeError:
                result["mapping_json"] = row["mapping_json"]
        for field in ("logN", "N", "batch", "modulus", "modulus_bits", "word_bits",
                      "input_order", "output_order", "end_to_end_ntt_s"):
            if row.get(field) not in (None, ""):
                result[field] = row[field]
        return result
    raise ValueError("benchmark emitted no CSV timing record")


def _environment(config: Mapping[str, Any], library: Path) -> dict[str, str]:
    env = os.environ.copy()
    visible = str(config.get("gpu", "")).strip()
    if visible:
        env["CUDA_VISIBLE_DEVICES"] = visible
    env["CUBUTTERFLY_COMPILE_MODE"] = str(config.get("compile_mode", "research"))
    # Candidate enumeration is part of the benchmark protocol.  Keep it in
    # the manifest so a quick replay cannot accidentally turn into an
    # unbounded installation-time search.
    if "search_budget" in config:
        env["CUBUTTERFLY_SEARCH_BUDGET"] = str(int(config["search_budget"]))
    if "xd_tune_max_bytes" in config:
        env["CUBUTTERFLY_XD_TUNE_MAX_BYTES"] = str(int(config["xd_tune_max_bytes"]))
    library_dir = str(library.resolve().parent)
    inherited = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{library_dir}:{inherited}" if inherited else library_dir
    env.pop("LD_PRELOAD", None)
    return env


def _benchmark_command(args: argparse.Namespace, config: Mapping[str, Any],
                       case: Mapping[str, Any], *, trace: bool = False,
                       mapping: Mapping[str, Any] | None = None,
                       cache_path: Path | None = None) -> list[str]:
    timing = config.get("timing", {})
    if trace:
        timing = config.get("trace", {})
    execution_mode = str(case.get("execution_mode", "auto" if len(case["shape"]) == 1 else "xd"))
    command = [
        str(args.benchmark), "--operator", str(case["operator"]),
        "--shape", _shape_text([int(value) for value in case["shape"]]),
        "--batch", str(int(case["batch"])), "--precision", str(case["precision"]),
        "--execution-mode", execution_mode, "--warmup", str(int(timing.get("warmup", 0))),
        "--repeat", str(int(timing.get("repeat", 1))),
        "--trials", str(int(timing.get("trials", 1))),
        "--policy", "default" if trace else str(timing.get("policy", "measure")),
    ]
    if cache_path is not None:
        command += ["--cache", str(cache_path)]
    if trace and mapping:
        command += ["--algorithm", json.dumps(mapping, sort_keys=True, separators=(",", ":"))]
    if case.get("inverse"):
        command.append("--inverse")
    if case.get("modulus") is not None:
        command += ["--modulus", str(int(case["modulus"]))]
    if case.get("matrix") is not None:
        command += ["--matrix", str(case["matrix"])]
    # Comparison and correctness verification execute extra plans and copies.
    # They belong to the event-timed benchmark, but must not be nested inside
    # an Nsight replay, otherwise the trace measures the comparison harness.
    if (not trace and case.get("compare_cufft") and not case.get("inverse") and
            case["operator"] == "fft"):
        command.append("--compare-cufft")
    if (not trace and case.get("verify") and not case.get("inverse") and
            case["operator"] == "fft"):
        command.append("--verify")
    return command


def _runner_executable(args: argparse.Namespace, config: Mapping[str, Any], name: str) -> Path:
    if name == "plan" and not config.get("runners"):
        return args.benchmark
    if name == "plan" and name not in config.get("runners", {}):
        return args.benchmark
    try:
        return resolve_runner(config, name, ROOT)["path"]
    except ValueError as exc:
        raise ValueError(str(exc)) from exc


def _ntt_command(args: argparse.Namespace, config: Mapping[str, Any],
                 case: Mapping[str, Any]) -> list[str]:
    shape = [int(value) for value in case["shape"]]
    if len(shape) != 1 or shape[0] & (shape[0] - 1):
        raise ValueError(f"{case['id']}: the ntt runner requires one power-of-two extent")
    precision = str(case["precision"]).lower()
    word_bits = {"uint32": 32, "word32": 32, "uint64": 64, "word64": 64}.get(precision)
    if word_bits is None:
        raise ValueError(f"{case['id']}: NTT precision must be uint32/uint64 or word32/word64")
    timing = config.get("timing", {})
    command = [str(_runner_executable(args, config, "ntt")),
               "--logN", str(shape[0].bit_length() - 1), "--batch", str(int(case["batch"])),
               "--word-bits", str(word_bits), "--warmup", str(int(timing.get("warmup", 0))),
               "--repeat", str(int(timing.get("repeat", 1))), "--csv"]
    if case.get("backend"):
        command += ["--backend", str(case["backend"])]
    if case.get("modulus") is not None:
        command += ["--modulus", str(int(case["modulus"]))]
    command += ["--input-order", str(case.get("input_order", "natural")),
                "--output-order", str(case.get("output_order", "natural"))]
    if case.get("inverse"):
        command.append("--inverse")
    if case.get("verify"):
        command.append("--verify")
    return command


def _gpuntt_command(args: argparse.Namespace, config: Mapping[str, Any],
                    case: Mapping[str, Any]) -> list[str]:
    shape = [int(value) for value in case["shape"]]
    if len(shape) != 1 or shape[0] & (shape[0] - 1):
        raise ValueError(f"{case['id']}: GPU-NTT requires one power-of-two extent")
    precision = str(case["precision"]).lower()
    word_bits = {"uint32": 32, "word32": 32, "uint64": 64, "word64": 64}.get(precision)
    if word_bits is None or case.get("modulus") is None:
        raise ValueError(f"{case['id']}: GPU-NTT requires a word precision and one modulus")
    timing = config.get("timing", {})
    command = [str(_runner_executable(args, config, "gpuntt")),
               "--log-n", str(shape[0].bit_length() - 1), "--batch", str(int(case["batch"])),
               "--word-bits", str(word_bits), "--moduli", str(int(case["modulus"])),
               "--mode", str(case.get("ntt_mode", "cyclic")),
               "--direction", "inverse" if case.get("inverse") else "forward",
               "--warmup", str(int(timing.get("warmup", 0))),
               "--repeat", str(int(timing.get("repeat", 1))), "--json"]
    if case.get("coset_generator") is not None:
        command += ["--coset-generator", str(int(case["coset_generator"]))]
    return command


def _run_configured_baseline(args: argparse.Namespace, config: Mapping[str, Any],
                             case: Mapping[str, Any], baseline: str,
                             library: Path, output: Path) -> dict[str, Any]:
    if baseline != "gpuntt":
        return {"id": baseline, "status": "unsupported",
                "reason": f"no configured adapter for baseline {baseline!r}"}
    executable = _runner_executable(args, config, "gpuntt")
    if not executable.is_file():
        return {"id": baseline, "status": "unavailable", "reason": f"executable not found: {executable}"}
    command = _gpuntt_command(args, config, case)
    env = _environment(config, library)
    try:
        completed = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=int(args.timeout), check=False)
    except subprocess.TimeoutExpired:
        return {"id": baseline, "status": "timeout", "command": command}
    log = output / "logs" / f"{case['id']}-{baseline}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout + "\n--- stderr ---\n" + completed.stderr, encoding="utf-8")
    result: dict[str, Any] = {"id": baseline, "command": command, "log": str(log)}
    if completed.returncode != 0:
        result.update(status="failed", reason=completed.stderr.strip()[-4000:])
        return result
    try:
        payload = json.loads(completed.stdout)
        result.update(status="passed", kernel_ms=float(payload["kernel_ms"]),
                      correct=bool(payload.get("correct", False)),
                      implementation=payload.get("implementation", "GPU-NTT"),
                      input_order=payload.get("input_order", ""),
                      output_order=payload.get("output_order", ""),
                      device=payload.get("device_name", ""),
                      source=payload.get("source", ""))
        if not result["correct"]:
            result["status"] = "incorrect"
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        result.update(status="failed", reason=f"invalid GPU-NTT JSON: {exc}")
    return result


def _run_benchmark(args: argparse.Namespace, config: Mapping[str, Any],
                   case: Mapping[str, Any], library: Path, output: Path,
                   *, trace: bool = False) -> dict[str, Any]:
    (output / "tuning").mkdir(parents=True, exist_ok=True)
    runner = str(case.get("runner", "plan"))
    if runner == "plan":
        command = _benchmark_command(
            args, config, case, trace=trace,
            cache_path=output / "tuning" / "cache.json")
    elif runner == "ntt":
        if trace:
            raise ValueError(f"{case['id']}: NTT trace runner is not configured")
        command = _ntt_command(args, config, case)
    else:
        raise ValueError(f"{case['id']}: unsupported runner {runner!r}")
    env = _environment(config, library)
    timeout = int(args.timeout)
    try:
        completed = subprocess.run(
            command, cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False,
        )
        returncode = completed.returncode
        stdout, stderr = completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nbenchmark timeout after {timeout}s"
    record: dict[str, Any] = {
        "id": str(case["id"]),
        "operator": case["operator"], "shape": case["shape"],
        "batch": int(case["batch"]), "precision": case["precision"],
        "trace_run": bool(trace), "command": command, "returncode": returncode,
    }
    log = output / "logs" / f"{case['id']}{'-trace' if trace else ''}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(stdout + "\n--- stderr ---\n" + stderr, encoding="utf-8")
    record["log"] = str(log)
    if returncode != 0:
        record["status"] = "timeout" if returncode == 124 else "failed"
        record["error"] = stderr.strip()[-4000:]
        return record
    try:
        record.update(_parse_timing(stdout) if runner == "plan" else _parse_csv_timing(stdout))
        record["status"] = "passed" if record.get("correct", True) else "incorrect"
    except ValueError as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
    return record


def _run_ncu(args: argparse.Namespace, config: Mapping[str, Any],
             case: Mapping[str, Any], library: Path, output: Path,
             mapping: Mapping[str, Any] | None = None) -> dict[str, Any]:
    trace = config.get("trace", {})
    executable = Path(str(trace.get("executable", "ncu")))
    if not executable.is_absolute():
        executable = Path(str(executable))
    metrics = trace.get("metrics", [])
    if not executable.exists() and executable.name != "ncu":
        return {"id": case["id"], "status": "unavailable", "error": f"ncu not found: {executable}"}
    env = _environment(config, library)
    (output / "tuning").mkdir(parents=True, exist_ok=True)
    trace_dir = output / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    stem = trace_dir / str(case["id"])
    command = [
        str(executable), "--target-processes", "all", "--replay-mode", "kernel",
        "--cache-control", "none", "--clock-control", "base",
        "--page", "raw", "--csv", "--force-overwrite",
        "--log-file", str(stem.with_suffix(".csv")),
    ]
    if metrics:
        command += ["--metrics", ",".join(str(item) for item in metrics)]
    command += _benchmark_command(
        args, config, case, trace=True, mapping=mapping,
        cache_path=output / "tuning" / "cache.json")
    try:
        completed = subprocess.run(
            command, cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=int(args.trace_timeout), check=False,
        )
        status = "passed" if completed.returncode == 0 else "failed"
        error = completed.stderr.strip()[-4000:] if completed.returncode else None
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        error = f"trace timeout after {args.trace_timeout}s: {exc}"
    result: dict[str, Any] = {
        "id": case["id"], "status": status, "command": command,
        "csv": str(stem.with_suffix(".csv")),
    }
    if error:
        result["error"] = error
    (trace_dir / f"{case['id']}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _run_nsys(args: argparse.Namespace, config: Mapping[str, Any],
              case: Mapping[str, Any], library: Path, output: Path,
              mapping: Mapping[str, Any] | None = None) -> dict[str, Any]:
    trace = config.get("trace", {})
    executable = Path(str(trace.get("executable", "nsys")))
    if not executable.exists() and executable.name != "nsys":
        return {"id": case["id"], "status": "unavailable", "error": f"nsys not found: {executable}"}
    env = _environment(config, library)
    (output / "tuning").mkdir(parents=True, exist_ok=True)
    trace_dir = output / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    stem = trace_dir / str(case["id"])
    profile_command = [
        str(executable), "profile", "--force-overwrite=true", "--sample=none",
        "--cpuctxsw=none", "--trace=cuda,nvtx", "--output", str(stem),
    ] + _benchmark_command(
        args, config, case, trace=True, mapping=mapping,
        cache_path=output / "tuning" / "cache.json")
    result: dict[str, Any] = {
        "id": case["id"], "tool": "nsys", "command": profile_command,
        "report": str(stem.with_suffix(".nsys-rep")),
    }
    try:
        completed = subprocess.run(
            profile_command, cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=int(args.trace_timeout), check=False,
        )
    except subprocess.TimeoutExpired as exc:
        result.update({"status": "timeout", "error": f"trace timeout after {args.trace_timeout}s: {exc}"})
        (trace_dir / f"{case['id']}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        return result
    profile_log = trace_dir / f"{case['id']}-profile.log"
    profile_log.write_text(completed.stdout + "\n--- stderr ---\n" + completed.stderr,
                           encoding="utf-8")
    result["profile_log"] = str(profile_log)
    if completed.returncode != 0:
        result.update({"status": "failed", "error": completed.stderr.strip()[-4000:]})
        (trace_dir / f"{case['id']}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        return result
    stats: dict[str, Any] = {}
    for report in ("gpukernsum", "gputrace"):
        stats_stem = trace_dir / f"{case['id']}-{report}"
        stats_command = [
            str(executable), "stats", "--force-overwrite=true",
            "--force-export=true", "--report", report, "--format", "csv",
            "--output", str(stats_stem), str(stem.with_suffix(".nsys-rep")),
        ]
        stats_result = subprocess.run(
            stats_command, cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=int(args.trace_timeout), check=False,
        )
        generated_csv = stats_stem.with_name(stats_stem.name + f"_{report}.csv")
        if not generated_csv.is_file():
            generated_csv = stats_stem.with_suffix(".csv")
        stats[report] = {
            "command": stats_command,
            "returncode": stats_result.returncode,
            "csv": str(generated_csv),
            "log": stats_result.stderr.strip()[-2000:],
        }
    result.update({"status": "passed", "stats": stats})
    (trace_dir / f"{case['id']}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _run_trace(args: argparse.Namespace, config: Mapping[str, Any],
               case: Mapping[str, Any], library: Path, output: Path,
               mapping: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if str(case.get("runner", "plan")) != "plan":
        return {"id": case["id"], "status": "unsupported",
                "error": "trace adapter is not configured for this runner"}
    tool = str(config.get("trace", {}).get("tool", "ncu")).lower()
    if tool == "nsys":
        return _run_nsys(args, config, case, library, output, mapping)
    return _run_ncu(args, config, case, library, output, mapping)


def _write_summary(output: Path, manifest: Mapping[str, Any], rows: list[dict[str, Any]],
                   traces: list[dict[str, Any]]) -> None:
    summary = {
        "schema": 1, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": manifest, "rows": rows, "traces": traces,
        "counts": {
            "requested": len(rows),
            "passed": sum(row.get("status") == "passed" for row in rows),
            "failed_or_timeout": sum(row.get("status") not in ("passed",) for row in rows),
            "traces_passed": sum(row.get("status") == "passed" for row in traces),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fields = ["id", "operator", "shape", "batch", "precision", "status",
              "algorithm", "kernel_ms", "cufft_ms", "speedup_vs_cufft",
              "workspace_bytes", "correct", "relative_l2_error",
              "baselines_json", "speedup_vs_gpuntt", "log"]
    with (output / "rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["shape"] = _shape_text([int(value) for value in item.get("shape", [])])
            writer.writerow(item)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--trace-timeout", type=int, default=180)
    parser.add_argument("--only", action="append", default=[],
                        help="run only these case IDs (repeat or use comma-separated IDs)")
    parser.add_argument("--skip-trace", action="store_true")
    parser.add_argument("--trace-only", action="store_true",
                        help="run only the explicitly selected NCU cases")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    config = _load_config(args.config)
    if args.only:
        requested = {
            item.strip() for value in args.only for item in str(value).split(",") if item.strip()
        }
        available = {str(case["id"]) for case in config["cases"]}
        missing = sorted(requested - available)
        if missing:
            parser.error("unknown case ID(s): " + ", ".join(missing))
        if not requested:
            parser.error("--only requires at least one case ID")
        config = dict(config)
        config["cases"] = [case for case in config["cases"] if str(case["id"]) in requested]
    args.benchmark = (args.benchmark or ROOT / str(config["benchmark"])).resolve()
    args.library = (args.library or ROOT / str(config["library"])).resolve()
    if not args.dry_run and (not args.benchmark.is_file() or not args.library.is_file()):
        parser.error("benchmark and library must exist")
    if not args.dry_run:
        required_runners = {
            "ntt" if str(case.get("runner", "plan")) == "ntt" else "plan"
            for case in config["cases"]
        }
        if any("gpuntt" in case.get("baselines", []) for case in config["cases"]):
            required_runners.add("gpuntt")
        missing = []
        for name in sorted(required_runners):
            path = args.benchmark if name == "plan" else _runner_executable(args, config, name)
            if not path.is_file():
                missing.append(f"{name}: {path}")
        if missing:
            parser.error("configured benchmark executable(s) missing: " + "; ".join(missing))
    args.loaded_config = config
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = args.loaded_config
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        for case in config["cases"]:
            runner = str(case.get("runner", "plan"))
            command = (_benchmark_command(
                args, config, case,
                cache_path=args.output / "tuning" / "cache.json")
                       if runner == "plan" else _ntt_command(args, config, case))
            print(" ".join(command))
            for baseline in case.get("baselines", []):
                if baseline == "gpuntt":
                    print(" ".join(_gpuntt_command(args, config, case)))
        return 0
    if args.trace_only:
        traces: list[dict[str, Any]] = []
        trace_ids = set(config.get("trace", {}).get("cases", []))
        existing: list[dict[str, Any]] = []
        summary_path = args.output / "summary.json"
        if summary_path.is_file():
            try:
                loaded = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("rows"), list):
                    existing = loaded["rows"]
            except (OSError, json.JSONDecodeError):
                existing = []
        mapping_by_id = {
            str(row.get("id")): row.get("mapping_json")
            for row in existing if isinstance(row, dict) and row.get("mapping_json")
        }
        for case in config["cases"]:
            if not config.get("trace", {}).get("enabled"):
                continue
            if not (case.get("trace") or case["id"] in trace_ids):
                continue
            trace = _run_trace(args, config, case, args.library, args.output,
                               mapping_by_id.get(str(case["id"])))
            traces.append(trace)
            print(f"{case['id']}: {config.get('trace', {}).get('tool', 'trace')} "
                  f"{trace['status']}", flush=True)
        manifest = {
            "config_path": str(args.config), "benchmark": str(args.benchmark),
            "library": str(args.library), "gpu": config.get("gpu"),
            "compile_mode": config.get("compile_mode", "research"),
        }
        _write_summary(args.output, manifest, existing, traces)
        return 0 if all(item.get("status") == "passed" for item in traces) else 1
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    trace_ids = set(config.get("trace", {}).get("cases", []))
    for case in config["cases"]:
        row = _run_benchmark(args, config, case, args.library, args.output)
        baselines: list[dict[str, Any]] = []
        if row.get("status") == "passed" and not row.get("trace_run"):
            for baseline in case.get("baselines", []):
                result = _run_configured_baseline(args, config, case, baseline,
                                                  args.library, args.output)
                if result.get("status") == "passed" and result.get("correct"):
                    result["speedup_vs_primary"] = float(result["kernel_ms"]) / float(row["kernel_ms"])
                    if baseline == "gpuntt":
                        row["speedup_vs_gpuntt"] = result["speedup_vs_primary"]
                elif result.get("status") == "passed":
                    row["status"] = "incorrect"
                baselines.append(result)
            if baselines:
                row["baselines"] = baselines
                row["baselines_json"] = json.dumps(baselines, sort_keys=True, separators=(",", ":"))
        rows.append(row)
        print(f"{case['id']}: {row['status']} {row.get('kernel_ms', 'n/a')} ms "
              f"{row.get('algorithm', '')}", flush=True)
        for baseline in baselines:
            print(f"{case['id']}: {baseline['id']} {baseline['status']} "
                  f"{baseline.get('kernel_ms', 'n/a')} ms", flush=True)
        if (not args.skip_trace and config.get("trace", {}).get("enabled") and
                (case.get("trace") or case["id"] in trace_ids)):
            trace = _run_trace(args, config, case, args.library, args.output,
                               row.get("mapping_json"))
            traces.append(trace)
            print(f"{case['id']}: {config.get('trace', {}).get('tool', 'trace')} "
                  f"{trace['status']}", flush=True)
    manifest = {
        "config_path": str(args.config), "benchmark": str(args.benchmark),
        "library": str(args.library), "gpu": config.get("gpu"),
        "compile_mode": config.get("compile_mode", "research"),
    }
    _write_summary(args.output, manifest, rows, traces)
    return 0 if all(row.get("status") == "passed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
