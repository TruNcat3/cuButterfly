#!/usr/bin/env python3
"""Bounded public xD radix-4 experiment; run only on an exclusive GPU UUID.

Errors against same-precision CPU SciPy FFT are report-only (no error threshold).
Nonfinite outputs, input mutations, missing cache hits, and replay mismatches
are failures. Timings cover complete preallocated plans in serial graph chains.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes as ct
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ("auto", "radix4", "cufftdx", "torch_cufft")
FAMILIES = {"radix4": "xd:pair-tiled-radix4", "cufftdx": "xd:pair-tiled-cufftdx",
            "cufft_whole_external": "xd:cufft-whole"}


def providers_for(case):
    whole = case["fft_type"] != "c2c" or len(case["shape"]) == 2
    return PROVIDERS + (("cufft_whole_external",) if whole else ())


def family_for(provider, case):
    if provider == "cufft_whole_external" and case["fft_type"] != "c2c":
        return "xd:cufft-whole-real"
    return FAMILIES.get(provider, "")


def cases(precisions=("fp32",), suite="c2c"):
    result = []
    for precision in precisions:
        matrices = []
        if suite in ("c2c", "all"):
            matrices += [("c2c", shape, batch) for shape, batch in
                         [([32, 32], 1), ([32, 32], 64), ([32, 32], 256), ([8, 32, 32], 2)]]
        if suite in ("compact-real", "all"):
            matrices += [("compact-real", [32, 32, 64], batch) for batch in (1, 4)]
        for kind, shape, batch in matrices:
            for inverse in (False, True):
                name = f"rank{len(shape)}_b{batch}_{'inverse' if inverse else 'forward'}"
                if kind != "c2c":
                    name = "compact_real_" + name
                if precision != "fp32":
                    name += "_" + precision
                result.append(dict(name=name, shape=shape, batch=batch, inverse=inverse,
                                   normalize_inverse=inverse, precision=precision, suite=kind,
                                   fft_type="c2c" if kind == "c2c" else "c2r" if inverse else "r2c"))
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def cpu_arrays(case, seed):
    import numpy as np
    from scipy import fft
    real_dtype = np.float64 if case["precision"] == "fp64" else np.float32
    complex_dtype = np.complex128 if case["precision"] == "fp64" else np.complex64
    rng = np.random.default_rng(seed)
    shape = (case["batch"], *case["shape"])
    axes = tuple(range(1, len(shape)))
    real = rng.uniform(-1, 1, shape).astype(real_dtype)
    if case["fft_type"] == "c2c":
        data = (real + 1j * rng.uniform(-1, 1, shape).astype(real_dtype)).astype(complex_dtype)
        transform = fft.ifftn if case["inverse"] else fft.fftn
        reference = transform(data, axes=axes, norm="backward", workers=1)
    elif case["fft_type"] == "r2c":
        data = real
        reference = fft.rfftn(data, axes=axes, norm="backward", workers=1)
    else:
        # Never use an arbitrary half spectrum: multidimensional endpoint planes
        # must satisfy Hermitian symmetry as well.
        data = fft.rfftn(real, axes=axes, norm="backward", workers=1)
        reference = fft.irfftn(data, s=case["shape"], axes=axes, norm="backward", workers=1)
    expected = real_dtype if case["fft_type"] == "c2r" else complex_dtype
    if reference.dtype != expected or not np.isfinite(reference).all():
        raise RuntimeError("CPU reference changed precision or contains nonfinite values")
    return np.ascontiguousarray(data), np.ascontiguousarray(reference)


def prepare_cpu(output, precisions=("fp32",), suite="c2c"):
    import numpy as np
    from scipy import fft
    prepared = []
    for index, case in enumerate(cases(precisions, suite)):
        case["cpu_seed"] = 20261024 + index
        data, reference = cpu_arrays(case, 20261024 + index)
        np.savez_compressed(output / (case["name"] + ".cpu.npz"),
                            input=data, cpu_reference=reference)
        prepared.append((case, data, reference))
    # One small boundary case exercises input and output permutations together.
    # This legacy C2C semantic-boundary probe is separate from the selected matrix.
    real_dtype = np.float64 if precisions[0] == "fp64" else np.float32
    complex_dtype = np.complex128 if precisions[0] == "fp64" else np.complex64
    case = dict(name="bit_reversed_boundary_inverse", shape=[32, 32], batch=2,
                precision=precisions[0], suite="c2c", fft_type="c2c",
                inverse=True, normalize_inverse=True, input_orders=[1, 0], output_orders=[0, 1])
    rng = np.random.default_rng(20261024 + len(prepared))
    logical = (rng.uniform(-1, 1, (2, 32, 32)).astype(real_dtype) +
               1j * rng.uniform(-1, 1, (2, 32, 32)).astype(real_dtype)).astype(complex_dtype)
    reverse = [int(f"{index:05b}"[::-1], 2) for index in range(32)]
    data = np.ascontiguousarray(np.take(logical, reverse, axis=1))
    reference = np.ascontiguousarray(np.take(
        fft.ifftn(logical, axes=(1, 2), norm="backward", workers=1), reverse, axis=2))
    if reference.dtype != complex_dtype:
        raise RuntimeError("boundary CPU reference changed precision")
    np.savez_compressed(output / (case["name"] + ".cpu.npz"), input=data, cpu_reference=reference)
    return prepared, (case, data, reference)


def errors(actual, reference):
    import numpy as np
    absolute = np.abs(actual.astype(np.complex128) - reference.astype(np.complex128))
    valid = absolute[np.isfinite(absolute)]
    result = dict(elements=int(actual.size), nonfinite=int((~np.isfinite(actual)).sum()),
                  absolute_error_quantiles=None, relative_l2=None)
    if valid.size:
        result["absolute_error_quantiles"] = dict(zip(
            ("min", "median", "p95", "p99", "max"),
            map(float, np.quantile(valid, [0, .5, .95, .99, 1]))))
    if np.isfinite(absolute).all():
        norm = np.linalg.norm(reference.astype(np.complex128).ravel())
        result["relative_l2"] = float(np.linalg.norm(absolute.ravel()) / norm) if norm else None
    return result, absolute


def gpu_state(uuid):
    def query(arguments):
        return subprocess.check_output(["nvidia-smi", *arguments], text=True).strip()
    details = query(["-i", uuid,
        "--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu,driver_version",
        "--format=csv,noheader,nounits"]).split(", ")
    apps = query(["--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"])
    if len(details) != 6 or details[0] != uuid:
        raise RuntimeError("requested GPU UUID did not match identity query")
    if int(details[4]) or any(line.split(",")[0] == uuid for line in apps.splitlines()):
        raise RuntimeError(f"GPU is busy; no experiment started: {details}; {apps}")
    return dict(zip(("uuid", "name", "memory_mib", "memory_used_mib", "utilization_pct", "driver"), details))


def check_contention(uuid, phase, row, report, output):
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits"], text=True).strip()
    others = []
    for line in apps.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            raise RuntimeError("unrecognized compute-app identity row: " + line)
        if fields[0] == uuid and int(fields[1]) != os.getpid():
            others.append(dict(uuid=fields[0], pid=int(fields[1]), process_name=fields[2]))
    check = dict(phase=phase, checked_utc=datetime.now(timezone.utc).isoformat(),
                 runner_pid=os.getpid(), other_processes=others)
    row.setdefault("contention_checks", []).append(check)
    if others:
        row.update(status="contaminated", contamination=check)
        save(output / "summary.json", report)
        raise RuntimeError(f"contaminated GPU measurement at {phase}: {others}")


def metadata(plan):
    source = ct.c_int()
    plan.library._call("PlanGetSelectionSource", plan.native_handle, ct.byref(source))
    return dict(mapping=json.loads(plan.mapping_json()), descriptor=json.loads(plan.descriptor_json()),
                selection_source_id=source.value,
                selection_source={0: "measured_table", 1: "user_cache", 2: "runtime_measured",
                                  3: "static_model", 4: "explicit"}.get(source.value, "unknown"),
                selection_reason=plan.selection_reason(), algorithm=plan.algorithm_name(),
                workspace_bytes=plan.workspace_size())


def assert_radix4(mapping):
    while mapping.get("kind") in ("semantic-boundary", "real-boundary"):
        mapping = mapping["child"]
    physical = mapping["physical"]
    if physical.get("pair_core") != "radix4-register":
        raise AssertionError("forced radix4 plan did not preserve pair_core")
    axes = physical.get("axes", [])
    if not any(axis.get("core") == "radix4-register-pair" for axis in axes):
        raise AssertionError("forced radix4 plan contains no actual radix4-register-pair")
    if physical.get("pair_groups", 0) < 1:
        raise AssertionError("forced radix4 plan contains no executable pair group")


def run_gpu(args, prepared, boundary, report):
    # Visibility must be set before importing torch or loading the public library.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / "python"))
    from cubutterfly import (AlgorithmPolicy, ComputeType, DataType, Error, ExecutionMode, FftType,
                             Library, Plan, PlanOptions, Transform)

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    stream = torch.cuda.Stream(device=device)
    library = Library(args.library)
    properties = torch.cuda.get_device_properties(device)
    report["torch"] = torch.__version__
    report["torch_cuda"] = torch.version.cuda
    report["runtime_device"] = dict(name=properties.name, memory_bytes=properties.total_memory,
                                    uuid=str(getattr(properties, "uuid", args.gpu)))

    def transform_for(case):
        fp64 = case["precision"] == "fp64"
        storage = (DataType.COMPLEX_FP64 if fp64 else DataType.COMPLEX_FP32) if case["fft_type"] == "c2c" else (DataType.FP64 if fp64 else DataType.FP32)
        return Transform(storage=storage,
                         compute=ComputeType.FP64 if fp64 else ComputeType.FP32,
                         fft_type={"c2c": FftType.C2C, "r2c": FftType.R2C, "c2r": FftType.C2R}[case["fft_type"]],
                         extents=case["shape"], batch=case["batch"], inverse=case["inverse"],
                         normalize_inverse=case["normalize_inverse"], execution_mode=ExecutionMode.XD,
                         input_orders=case.get("input_orders", []), output_orders=case.get("output_orders", []))

    def make_plan(transform, mapping, cache):
        options = PlanOptions(policy=AlgorithmPolicy.EXPLICIT if mapping else AlgorithmPolicy.DEFAULT,
                              mapping=mapping, execution_mode=ExecutionMode.XD)
        start = time.perf_counter()
        plan = Plan(library, transform, options, stream=stream.cuda_stream, cache_path=str(cache))
        stream.synchronize()
        return plan, 1000 * (time.perf_counter() - start)

    def bind_workspace(plan):
        workspace = torch.empty(plan.workspace_size(), dtype=torch.uint8, device=device)
        if workspace.numel():
            plan.set_workspace(workspace, workspace.numel())
        return workspace

    base_mapping = base_case = None
    for case, data, reference in prepared:
        providers = providers_for(case)
        whole_cufft = "cufft_whole_external" in providers
        public_providers = tuple(name for name in providers if name != "torch_cufft")
        row = dict(case, status="running", providers={})
        row["direct_cufft_coverage"] = dict(
            covered=whole_cufft,
            reason="forced external whole-transform cuFFT" if whole_cufft else
                   "DirectFftPlan C2C supports rank<=2; matching torch baseline retained")
        row.update(input_shape=list(data.shape), output_shape=list(reference.shape),
                   input_dtype=str(data.dtype), output_dtype=str(reference.dtype),
                   cpu_reference_dtype=str(reference.dtype))
        report["cases"].append(row)
        check_contention(args.gpu, "case_start", row, report, args.output)
        save(args.output / "summary.json", report)
        cache = args.output / (case["name"] + ".cache.jsonl")
        with ExitStack() as owned:
            transform = transform_for(case)
            plans = {}
            workspaces = []
            for name in public_providers:
                requested_mapping = family_for(name, case)
                # Persist the exact rejected request if construction fails; never
                # substitute another operator, precision or provider.
                row["providers"][name] = dict(status="constructing", requested_mapping=requested_mapping)
                save(args.output / "summary.json", report)
                try:
                    plan, init_ms = make_plan(transform, requested_mapping, cache)
                except Exception as error:
                    row["providers"][name].update(status="failed", error=str(error),
                                                  error_type=type(error).__name__)
                    row["status"] = "failed"
                    save(args.output / "summary.json", report)
                    raise
                owned.enter_context(plan)
                plans[name] = plan
                workspaces.append(bind_workspace(plan))
                row["providers"][name] = dict(metadata(plan), requested_mapping=requested_mapping,
                                               plan_initialization_ms=init_ms)
                if name == "cufft_whole_external":
                    row["providers"][name].update(
                        role="external baseline; entire transform delegated to cuFFT, not method result",
                        normalization_implementation="provider mapping; all required scaling included",
                        plan_many_initialization_included_in_timing=False)
                if name == "radix4":
                    assert_radix4(row["providers"][name]["mapping"])
                    if base_mapping is None:
                        base_mapping, base_case = row["providers"][name]["mapping"], case
            with torch.cuda.stream(stream):
                source = torch.from_numpy(data).to(device)
                output_template = torch.from_numpy(reference)
                outputs = {name: torch.empty(reference.shape, dtype=output_template.dtype, device=device)
                           for name in providers}
            axes = tuple(range(1, data.ndim))
            fft = (torch.fft.ifftn if case["inverse"] else torch.fft.fftn) if case["fft_type"] == "c2c" else (torch.fft.irfftn if case["inverse"] else torch.fft.rfftn)
            launches = {name: (lambda name=name: plans[name].execute(source, outputs[name]))
                        for name in public_providers}
            launches["torch_cufft"] = lambda: fft(source, s=case["shape"], dim=axes, norm="backward", out=outputs["torch_cufft"])
            eager = {}
            graphs = {}
            row["providers"]["torch_cufft"] = dict(
                backend="torch.fft CUDA cuFFT", norm="backward", role="framework out-path baseline",
                interpretation="includes torch out-path overhead; not isolated cuFFT core latency")
            for name in providers:
                start = time.perf_counter()
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        launches[name]()
                stream.synchronize()
                row["providers"][name]["execution_preparation_ms"] = 1000 * (time.perf_counter() - start)
                eager[name] = outputs[name].cpu().numpy().copy()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(args.chain):
                        launches[name]()
                graphs[name] = graph
                row["providers"][name]["trial_us"] = []
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for trial in range(args.trials):
                order = providers[trial % len(providers):] + providers[:trial % len(providers)]
                for name in order:
                    with torch.cuda.stream(stream):
                        for _ in range(args.warmup):
                            graphs[name].replay()
                        start.record(stream)
                        for _ in range(args.repeats):
                            graphs[name].replay()
                        stop.record(stream)
                    stop.synchronize()
                    elapsed = 1000 * start.elapsed_time(stop) / (args.repeats * args.chain)
                    if not np.isfinite(elapsed) or elapsed <= 0:
                        raise RuntimeError("invalid event timing")
                    row["providers"][name]["trial_us"].append(elapsed)
                row.setdefault("trial_provider_orders", []).append(list(order))
            stream.synchronize()
            check_contention(args.gpu, "timing_end", row, report, args.output)
            distributions = {}
            failures = []
            for name in providers:
                actual = outputs[name].cpu().numpy().copy()
                metrics, absolute = errors(actual, reference)
                details = row["providers"][name]
                details.update(cpu_reference_error=metrics, median_us=statistics.median(details["trial_us"]),
                               graph_matches_eager=bool(np.array_equal(actual, eager[name])))
                details["cpu_" + case["precision"] + "_error"] = metrics
                distributions[name + "_output"] = actual
                distributions[name + "_absolute_error"] = absolute
                if metrics["nonfinite"] or not details["graph_matches_eager"]:
                    failures.append(name + ": nonfinite output or graph/eager mismatch")
            row["input_preserved"] = bool(np.array_equal(source.cpu().numpy(), data))
            if not row["input_preserved"]:
                failures.append("timed execution mutated input")
            row["speedup_vs_torch_cufft"] = {
                name: row["providers"]["torch_cufft"]["median_us"] / row["providers"][name]["median_us"]
                for name in providers if name != "torch_cufft"}
            if "cufft_whole_external" in providers:
                row["speedup_vs_cufft_whole"] = {
                    name: row["providers"]["cufft_whole_external"]["median_us"] / row["providers"][name]["median_us"]
                    for name in providers if name != "cufft_whole_external"}
            np.savez_compressed(args.output / (case["name"] + ".errors.npz"), **distributions)
            save(args.output / "summary.json", report)

            # Verify default cache reuse and exact forced replay outside timing.
            validations = [("auto_cache", "auto", "")]
            validations += [(name + "_replay", name, json.dumps(row["providers"][name]["mapping"]))
                            for name in ("radix4", "cufftdx")]
            row["validations"] = {}
            for label, original, mapping in validations:
                validation, init_ms = make_plan(transform, mapping, cache)
                with validation:
                    workspace = bind_workspace(validation)
                    details = dict(metadata(validation), plan_initialization_ms=init_ms)
                    with torch.cuda.stream(stream):
                        out = torch.empty_like(outputs[original])
                        validation.execute(source, out)
                    stream.synchronize()
                    actual = out.cpu().numpy().copy()
                    details["mapping_equal"] = details["mapping"] == row["providers"][original]["mapping"]
                    details["output_equal"] = bool(np.array_equal(actual, eager[original]))
                    details["input_preserved"] = bool(np.array_equal(source.cpu().numpy(), data))
                    details["cpu_reference_error"], absolute = errors(actual, reference)
                    details["cpu_" + case["precision"] + "_error"] = details["cpu_reference_error"]
                    distributions[label + "_output"] = actual
                    distributions[label + "_absolute_error"] = absolute
                    if label == "auto_cache":
                        details["cache_hit"] = details["selection_source_id"] == 1
                    row["validations"][label] = details
                    np.savez_compressed(args.output / (case["name"] + ".errors.npz"), **distributions)
                    save(args.output / "summary.json", report)
                    if (not details["mapping_equal"] or not details["output_equal"] or
                            not details["input_preserved"] or details["cpu_reference_error"]["nonfinite"] or
                            (label == "auto_cache" and not details["cache_hit"])):
                        failures.append(label + ": cache/replay contract failed")
                    del workspace
            row["cache_sha256"] = digest(cache) if cache.is_file() else None
            np.savez_compressed(args.output / (case["name"] + ".errors.npz"), **distributions)
            row.update(status="failed" if failures else "measured_and_errors_reported", failures=failures)
            save(args.output / "summary.json", report)
            # Graphs retain plan/workspace addresses until their final replay.
            graphs.clear()
            if failures:
                raise RuntimeError("; ".join(failures))
        print(f"{case['name']}: measured, cache and exact forced replay checked", flush=True)

    case, data, reference = boundary
    row = dict(case, status="running", variants=[])
    report["semantic_boundary"] = row
    cache = args.output / "boundary.cache.jsonl"
    mapping_text = FAMILIES["radix4"]
    original_output = original_mapping = None
    distributions = {}
    for label in ("forced", "exact_json_replay"):
        try:
            plan, init_ms = make_plan(transform_for(case), mapping_text, cache)
        except Error as error:
            # The API exposes per-axis orders. Preserve any implementation
            # rejection explicitly instead of substituting natural order.
            if label != "forced":
                raise
            row.update(status="unsupported", error=str(error), api_exposes_orders=True)
            save(args.output / "summary.json", report)
            break
        with plan:
            workspace = bind_workspace(plan)
            details = dict(metadata(plan), label=label, plan_initialization_ms=init_ms)
            assert_radix4(details["mapping"])
            if original_mapping is None:
                original_mapping = details["mapping"]
                mapping_text = json.dumps(original_mapping)
            details["mapping_equal"] = details["mapping"] == original_mapping
            with torch.cuda.stream(stream):
                source = torch.from_numpy(data).to(device)
                out = torch.empty_like(source)
                plan.execute(source, out)
            stream.synchronize()
            eager = out.cpu().numpy().copy()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(args.chain):
                    plan.execute(source, out)
            with torch.cuda.stream(stream):
                graph.replay()
            stream.synchronize()
            actual = out.cpu().numpy().copy()
            details["cpu_reference_error"], absolute = errors(actual, reference)
            details["cpu_" + case["precision"] + "_error"] = details["cpu_reference_error"]
            details["graph_matches_eager"] = bool(np.array_equal(actual, eager))
            details["input_preserved"] = bool(np.array_equal(source.cpu().numpy(), data))
            if original_output is None:
                original_output = actual.copy()
            details["output_equal"] = bool(np.array_equal(actual, original_output))
            distributions[label + "_output"] = actual
            distributions[label + "_absolute_error"] = absolute
            row["variants"].append(details)
            np.savez_compressed(args.output / (case["name"] + ".errors.npz"), **distributions)
            save(args.output / "summary.json", report)
            if (details["cpu_reference_error"]["nonfinite"] or not details["mapping_equal"] or
                    not details["graph_matches_eager"] or not details["input_preserved"] or
                    not details["output_equal"]):
                raise AssertionError("semantic boundary execution/replay contract failed")
            del graph, workspace
    else:
        row["status"] = "errors_reported_and_exact_replay_checked"
    save(args.output / "summary.json", report)

    invalids = [("shape_unsupported", dict(base_case, shape=[3, 5]), FAMILIES["radix4"])]
    bad_mapping = json.loads(json.dumps(base_mapping))
    physical = bad_mapping
    while "child" in physical:
        physical = physical["child"]
    physical["physical"]["pair_core"] = "unknown-pair-core"
    invalids.append(("unknown_pair_core", base_case, json.dumps(bad_mapping)))
    for name, case, mapping in invalids:
        try:
            accepted, _ = make_plan(transform_for(case), mapping, args.output / "invalid.cache.jsonl")
        except Error as error:
            report["invalid_cases"].append(dict(name=name, request=case, requested_mapping=mapping,
                                                rejected=True, error=str(error),
                                                status_code=getattr(error, "status", None)))
        else:
            accepted.close()
            report["invalid_cases"].append(dict(name=name, request=case,
                                                requested_mapping=mapping, rejected=False))
            save(args.output / "summary.json", report)
            raise AssertionError("unsupported explicit mapping accepted: " + name)
        save(args.output / "summary.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--gpu", required=True, help="exclusive GPU UUID, never an index")
    parser.add_argument("--output", required=True, type=Path, help="new result directory")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--chain", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--precisions", nargs="+", choices=("fp32", "fp64"), default=["fp32"])
    parser.add_argument("--suite", choices=("c2c", "compact-real", "all"), default="c2c")
    parser.add_argument("--prepare-only", action="store_true", help="save CPU references without querying or using GPU")
    args = parser.parse_args()
    if len(set(args.precisions)) != len(args.precisions):
        parser.error("--precisions must not contain duplicates")
    if not args.gpu.startswith("GPU-"):
        parser.error("--gpu must be a full GPU UUID")
    if not (1 <= args.trials <= 101 and 1 <= args.chain <= 1024 and
            1 <= args.repeats <= 100000 and 0 <= args.warmup <= 100000):
        parser.error("invalid bounded timing parameters")
    args.library = args.library.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="preparing_cpu", started_utc=datetime.now(timezone.utc).isoformat(),
                  protocol="public_xd_serial_graph_chain_v3_precision_real", trials=args.trials, chain=args.chain,
                  precisions=args.precisions, suite=args.suite,
                  argv=sys.argv, prepare_only=args.prepare_only,
                  semantic_boundary_scope="separate legacy C2C probe, first requested precision",
                  repeats=args.repeats, warmup=args.warmup, cases=[], invalid_cases=[],
                  numerical_policy="report-only same-precision finite errors; no error threshold",
                  auto_policy="DEFAULT with XD execution, fresh per-case cache; environment preserved",
                  timing_scope="complete preallocated plan; same input replayed; us per entire batch",
                  normalization="forward unscaled, inverse normalized by transform volume",
                  library=str(args.library), library_sha256=digest(args.library),
                  runner_sha256=digest(__file__), requested_gpu=args.gpu,
                  environment={key: value for key, value in os.environ.items()
                               if key.startswith("CUBUTTERFLY_")})
    save(args.output / "summary.json", report)
    try:
        prepared, boundary = prepare_cpu(args.output, args.precisions, args.suite)
        import numpy as np
        import scipy
        report.update(numpy=np.__version__, scipy=scipy.__version__,
                      cpu_reference="scipy.fft same storage precision, workers=1, norm=backward")
        report["prepared_cases"] = [dict(case, input_shape=list(data.shape),
                                         output_shape=list(reference.shape), input_dtype=str(data.dtype),
                                         output_dtype=str(reference.dtype),
                                         cpu_artifact_sha256=digest(args.output / (case["name"] + ".cpu.npz")))
                                    for case, data, reference in prepared]
        if args.prepare_only:
            report["status"] = "cpu_prepared"
        else:
            report["gpu"] = gpu_state(args.gpu)
            report["status"] = "running"
            save(args.output / "summary.json", report)
            run_gpu(args, prepared, boundary, report)
            report["status"] = "measured_and_errors_reported"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save(args.output / "summary.json", report)
    print(json.dumps(dict(status=report["status"], output=str(args.output / "summary.json"))))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
