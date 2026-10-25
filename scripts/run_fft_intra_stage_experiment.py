#!/usr/bin/env python3
"""Bounded forward FFT study: register control, four factor schedules, whole cuFFT.

Only comparisons within the factor family isolate intra-stage organization.
CPU SciPy errors are report-only; nonfinite data and replay mismatches fail.
Prepare all CPU/JIT artifacts first; resume uses a fail-closed module lookup.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
import traceback

import resident_mapping
from compile_module import compile_mapping
from research_compile_requests import compile_request
from run_register_radix4_experiment import mapping_for as register_mapping
from run_public_radix4_pair_experiment import (
    ROOT, check_contention, cpu_arrays, digest, errors, gpu_state, metadata, save,
)

PROVIDERS = ("register_radix4", "factor_t1_p0", "factor_t4_p0", "factor_t4_p1",
             "factor_t4_p2", "cufft")
FACTOR_POINTS = ((1, 0), (4, 0), (4, 1), (4, 2))
COMPLETE = "measured_and_errors_reported"


def cases():
    return [dict(name=f"log{n}_b16_{precision}_forward", logN=n, shape=[1 << n],
                 batch=16, precision=precision, storage_precision=precision,
                 compute_precision=precision, compute_family="fft-h", fft_type="c2c", inverse=False,
                 normalize_inverse=False)
            for precision in ("fp32", "fp64") for n in (16, 20)]


def mappings_for(case):
    result = {"register_radix4": register_mapping(case, "radix4")}
    for tiles, depth in FACTOR_POINTS:
        result[f"factor_t{tiles}_p{depth}"] = dict(
            kind="butterfly", backend="factor-streamed", fft_core="cufftdx-block",
            compute_unit="auto", stage_partition=[case["logN"]],
            factor_partition=[case["logN"] - 10, 10], factor_columns=4,
            factor_ept=8 if case["logN"] == 16 else 16,
            shared_layout="writer-aligned", cross_twiddle="recurrence",
            data_tiles_per_cta=tiles, prefetch_depth=depth, stage_overlap=False,
            factor_overlap=False, factor_slices=1)
    result["cufft"] = "cufft"
    return result


def resource_lower_bounds(case, mapping):
    """CPU-only necessary condition, not a claim that the kernel can launch."""
    if not isinstance(mapping, dict) or mapping.get("backend") != "factor-streamed":
        return None
    width = 16 if case["precision"] == "fp64" else 8
    depth = mapping["prefetch_depth"]
    stages = [dict(log_size=size, dynamic_shared_lower_bound_bytes=
                   (1 << size) * mapping["factor_columns"] * width *
                   (depth + 2 if depth else 1))
              for size in mapping["factor_partition"]]
    return dict(stages=stages, max_dynamic_shared_lower_bound_bytes=max(
        stage["dynamic_shared_lower_bound_bytes"] for stage in stages),
        basis="(Depth + 1) * tile + max(tile, FFT::shared_memory_size); excludes static shared",
        status="necessary_condition_only")


def resource_rejection(case, mapping, shared_limit):
    requirement = resource_lower_bounds(case, mapping)
    if requirement and requirement["max_dynamic_shared_lower_bound_bytes"] > shared_limit:
        return dict(status="infeasible_resource", requested_mapping=mapping,
                    resource_requirement=requirement, device_shared_limit_bytes=shared_limit,
                    reason="dynamic shared memory lower bound exceeds device opt-in per-block limit",
                    trial_us=[])
    return None


def canonical_request(request):
    result = json.loads(json.dumps(request))
    if result.get("backend") == "factor-streamed":
        result.setdefault("factor_io_policies", ["dynamic"] * len(result["factor_partition"]))
    else:
        for key, default in dict(local_stage_partitions=[], exchange_chunks=[],
                                 prefix_codelet="native", prefix_shared_layout="linear",
                                 prefix_codelet_lanes=1, suffix_core="cufftdx-block").items():
            result.setdefault(key, default)
    return result


def module_lookup(argv):
    """Public JIT subprocess adapter: return only an already prepared module."""
    parser = argparse.ArgumentParser()
    parser.add_argument("compiler_script")
    parser.add_argument("--mapping", required=True, type=json.loads)
    parser.add_argument("--sm", required=True, type=int)
    for key in ("root", "mathdx", "nvcc", "cache", "timeout"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args(argv)
    report = json.loads(Path(os.environ["CUBUTTERFLY_INTRA_STAGE_PREPARED"]).read_text())
    if args.sm != report["identity"]["sm"]:
        raise RuntimeError("prepared module architecture mismatch")
    request = canonical_request(args.mapping)
    for module in report["modules"]:
        if canonical_request(module["request"]) == request:
            if digest(module["module"]) != module["sha256"]:
                raise RuntimeError("prepared module hash mismatch")
            print(module["module"])
            return 0
    raise RuntimeError("runtime requested an unprepared module; JIT is disabled during measurement")


def identity(args):
    sources = [Path(__file__), *[ROOT / "scripts" / name for name in (
        "run_register_radix4_experiment.py", "run_public_radix4_pair_experiment.py",
        "research_compile_requests.py", "compile_module.py", "resident_mapping.py",
        "cufftdx_capabilities.py")]]
    for directory in ("src", "include", "python/cubutterfly"):
        sources.extend(path for path in (ROOT / directory).rglob("*")
                       if path.is_file() and path.suffix in (".py", ".h", ".hpp", ".cpp", ".cu", ".cuh"))
    return dict(schema="fft-intra-stage-v1", gpu=args.gpu, sm=args.sm,
                python=sys.executable, python_version=sys.version,
                packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch")},
                library=str(args.library), library_sha256=digest(args.library),
                compile_policy=args.compile_policy, mathdx=str(args.mathdx), nvcc=str(args.nvcc),
                jit_cache=str(args.jit_cache), chain=args.chain, repeats=args.repeats,
                trials=args.trials, warmup=args.warmup,
                sources={str(path.relative_to(ROOT)): digest(path) for path in sorted(set(sources))},
                requests={case["name"]: mappings_for(case) for case in cases()})


def prepare(args, report):
    import numpy as np
    import scipy
    report.update(numpy=np.__version__, scipy=scipy.__version__, prepared_cases=[], modules=[])
    for index, case in enumerate(cases()):
        data, reference = cpu_arrays(case, 20261024 + index)
        artifact = args.output / (case["name"] + ".cpu.npz")
        np.savez_compressed(artifact, input=data, cpu_reference=reference)
        report["prepared_cases"].append(dict(case, artifact=str(artifact),
            artifact_sha256=digest(artifact), cpu_seed=20261024 + index,
            input_dtype=str(data.dtype), reference_dtype=str(reference.dtype), requests={}))
        del data, reference
        save(args.output / "summary.json", report)
    report["status"] = "preparing_jit"
    modules = {}
    for case in report["prepared_cases"]:
        for provider, mapping in mappings_for(case).items():
            request = None if provider == "cufft" else compile_request(dict(
                operator="fft", precision=case["precision"], logN=case["logN"],
                mapping_json=mapping), args.compile_policy)
            case["requests"][provider] = dict(mapping=mapping, compile_request=request,
                                              resource_lower_bounds=resource_lower_bounds(case, mapping))
            if request is not None:
                key = json.dumps(canonical_request(request), sort_keys=True)
                if key not in modules:
                    start = time.perf_counter()
                    module = compile_mapping(request, root=ROOT, mathdx=args.mathdx,
                        nvcc=args.nvcc, sm=args.sm, cache=args.jit_cache, timeout=args.compile_timeout)
                    modules[key] = str(module)
                    report["modules"].append(dict(request=request, module=str(module),
                        sha256=digest(module), preparation_ms=1000 * (time.perf_counter() - start)))
                case["requests"][provider]["module"] = modules[key]
            save(args.output / "summary.json", report)
    report["status"] = "cpu_and_jit_prepared"
    report["preparation_complete"] = True
    save(args.output / "summary.json", report)


def validate_artifacts(report):
    if not report.get("preparation_complete"):
        raise RuntimeError("preparation is incomplete")
    if report.get("identity_sha256") != hashlib.sha256(json.dumps(
            report["identity"], sort_keys=True).encode()).hexdigest():
        raise RuntimeError("prepared identity hash mismatch")
    if len(report["prepared_cases"]) != 4:
        raise RuntimeError("prepared case matrix is incomplete")
    for index, (entry, expected) in enumerate(zip(report["prepared_cases"], cases())):
        if (any(entry.get(key) != value for key, value in expected.items()) or
                entry.get("cpu_seed") != 20261024 + index):
            raise RuntimeError("prepared CPU case semantics or seed changed")
        if digest(entry["artifact"]) != entry["artifact_sha256"]:
            raise RuntimeError("CPU artifact hash mismatch: " + entry["name"])
    for entry in report["modules"]:
        if digest(entry["module"]) != entry["sha256"]:
            raise RuntimeError("prepared module hash mismatch: " + entry["module"])


def validate_prepared(args, report):
    if report.get("identity") != identity(args):
        raise RuntimeError("resume requires complete preparation with identical build, requests and protocol")
    validate_artifacts(report)


def reuse_preparation(args):
    """Adopt untouched CPU/modules after a runner-only edit; retain old provenance."""
    source = args.reuse_preparation_from.resolve(strict=True) / "summary.json"
    report = json.loads(source.read_text())
    if report.get("cases"):
        raise RuntimeError("runner-only preparation reuse requires no prior measurements")
    validate_artifacts(report)
    expected = identity(args)
    previous = json.loads(json.dumps(report["identity"]))
    runner_key = str(Path(__file__).relative_to(ROOT))
    previous["sources"][runner_key] = expected["sources"][runner_key]
    if previous != expected:
        raise RuntimeError("preparation reuse allows only a runner source change; build, requests and protocol must match")
    # Re-derive requests too: a matching identity cannot hide a mutated prepared request.
    for case in report["prepared_cases"]:
        for provider, mapping in mappings_for(case).items():
            entry = case["requests"][provider]
            request = None if provider == "cufft" else compile_request(dict(
                operator="fft", precision=case["precision"], logN=case["logN"],
                mapping_json=mapping), args.compile_policy)
            if entry["mapping"] != mapping or entry["compile_request"] != request:
                raise RuntimeError("prepared request changed")
            entry["resource_lower_bounds"] = resource_lower_bounds(case, mapping)
    report["preparation_reused_from"] = dict(summary=str(source), sha256=digest(source),
        identity=report["identity"], identity_sha256=report["identity_sha256"],
        prior_status=report["status"], reason="runner-only revision; unchanged CPU inputs, modules, requests and protocol")
    report.update(identity=expected, status="cpu_and_jit_prepared",
                  prepared_revision_utc=datetime.now(timezone.utc).isoformat())
    report["identity_sha256"] = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    for key in ("error", "error_type", "traceback", "finished_utc"):
        report.pop(key, None)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "summary.json", report)
    return report


def design_summary(mapping):
    if mapping.get("backend") == "cufft":
        return dict(core="whole cuFFT", stage_partition="unknown", factor_partition="unknown",
                    physical_stage_count="unknown", stages="unknown", overlap_scope="unknown",
                    parallelism="unknown")
    factor = mapping.get("backend") == "factor-streamed"
    partition = mapping["factor_partition"] if factor else mapping["stage_partition"]
    groups = mapping.get("execution_group_mappings") or mapping.get("segment_mappings") or []
    geometry = None if factor else resident_mapping.register_geometry(dict(mapping, logN=sum(partition)), partition)
    stages = []
    for index, log_size in enumerate(partition):
        group = groups[index] if index < len(groups) else {}
        stages.append(dict(log_size=log_size, factor_size=1 << log_size,
            core=group.get("core", "unknown"), threads_per_cta=group.get("threads", "unknown"),
            ept=group.get("ept", mapping.get("factor_ept", "unknown") if factor else
                          geometry["prefix_ept" if index == 0 else "suffix_ept"]),
            columns=mapping.get("factor_columns", "unknown") if factor else
                    geometry["prefix_columns" if index == 0 else "suffix_columns"],
            tilegroups_per_cta=mapping.get("data_tiles_per_cta", "unknown"),
            prefetch_depth=mapping.get("prefetch_depth", "unknown")))
    return dict(core=mapping.get("fft_core", "unknown"),
                prefix_codelet=mapping.get("prefix_codelet", "unknown"),
                stage_partition=mapping.get("stage_partition", "unknown"),
                factor_partition=mapping.get("factor_partition", []),
                physical_stage_count=len(partition), stages=stages,
                stage_overlap=mapping.get("stage_overlap", "unknown"),
                factor_overlap=mapping.get("factor_overlap", "unknown"),
                factor_slices=mapping.get("factor_slices", "unknown"),
                overlap_scope="within each factor only; full factor completion before next factor" if factor
                              else "serial prefix/suffix boundary", parallelism="unknown")


def validate_mapping(provider, actual, requested):
    if provider == "cufft":
        if actual.get("backend") != "cufft":
            raise AssertionError("whole cuFFT baseline was not preserved")
        return
    for key, value in requested.items():
        # The default suffix core is omitted by the public canonical serializer.
        observed = actual.get(key, "cufftdx-block" if key == "suffix_core" else None)
        if observed != value:
            raise AssertionError(f"{provider}: requested {key}={value!r}, actual={observed!r}")


def check_factor_differences(providers):
    normalized = []
    for provider in PROVIDERS[1:-1]:
        if providers[provider].get("status") == "infeasible_resource":
            # This candidate has no actual mapping; never pretend it launched.
            continue
        mapping = dict(providers[provider]["mapping"])
        for key in ("data_tiles_per_cta", "prefetch_depth"):
            mapping.pop(key, None)
        normalized.append(mapping)
    if normalized and any(value != normalized[0] for value in normalized[1:]):
        raise AssertionError("factor mappings differ beyond data_tiles_per_cta and prefetch_depth")


def save_provider_errors(output, case_name, provider, distributions):
    import numpy as np
    artifact = output / f"{case_name}.{provider}.errors.npz"
    np.savez_compressed(artifact, **distributions)
    return dict(path=str(artifact), sha256=digest(artifact), arrays=list(distributions))


def run_gpu(args, report):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUBUTTERFLY_JIT_PYTHON"] = str(Path(__file__).resolve())
    os.environ["CUBUTTERFLY_INTRA_STAGE_PREPARED"] = str(args.output / "summary.json")
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / "python"))
    from cubutterfly import (AlgorithmPolicy, ComputeType, DataType, ExecutionMode,
                             Library, Plan, PlanOptions, Transform)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    if props.major * 10 + props.minor != args.sm:
        raise RuntimeError("prepared architecture differs from selected GPU")
    # CUDA runtime enum 97 is cudaDevAttrMaxSharedMemoryPerBlockOptin.
    import ctypes
    runtime = ctypes.CDLL(str(args.nvcc.parent.parent / "lib64/libcudart.so"))
    query = runtime.cudaDeviceGetAttribute
    query.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    query.restype = ctypes.c_int
    shared_limit = ctypes.c_int()
    if query(ctypes.byref(shared_limit), 97, 0) != 0 or shared_limit.value <= 0:
        raise RuntimeError("could not query actual device opt-in shared memory limit")
    report.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                  runtime_device=dict(name=props.name, memory_bytes=props.total_memory,
                                      max_shared_per_block_optin_bytes=shared_limit.value,
                                      uuid=str(getattr(props, "uuid", args.gpu))))
    stream, library = torch.cuda.Stream(device=device), Library(args.library)

    def open_plan(owned, transform, mapping, cache):
        start = time.perf_counter()
        plan = owned.enter_context(Plan(library, transform, PlanOptions(policy=AlgorithmPolicy.EXPLICIT,
            mapping=mapping if isinstance(mapping, str) else json.dumps(mapping),
            execution_mode=ExecutionMode.COMPOSITE), stream=stream.cuda_stream, cache_path=str(cache)))
        buffer = torch.empty(plan.workspace_size(), dtype=torch.uint8, device=device)
        if buffer.numel():
            plan.set_workspace(buffer, buffer.numel())
        stream.synchronize()
        return plan, buffer, 1000 * (time.perf_counter() - start)

    def execute(plan, source, out):
        with torch.cuda.stream(stream):
            plan.execute(source, out)
        stream.synchronize()
        return out.cpu().numpy().copy()

    for case in report["prepared_cases"]:
        row = next((entry for entry in report["cases"] if entry["name"] == case["name"]), None)
        if row is not None and row["status"] == COMPLETE:
            continue
        if row is None:
            row = dict(name=case["name"], compute_family=case["compute_family"],
                       storage_precision=case["storage_precision"], compute_precision=case["compute_precision"],
                       providers={}, validations={})
            report["cases"].append(row)
        row.update(status="running", failures=[])
        check_contention(args.gpu, "case_start", row, report, args.output)
        with np.load(case["artifact"], allow_pickle=False) as arrays:
            data, reference = arrays["input"], arrays["cpu_reference"]
        fp64 = case["precision"] == "fp64"
        transform = Transform(extents=case["shape"], batch=case["batch"], inverse=False,
            normalize_inverse=False, storage=DataType.COMPLEX_FP64 if fp64 else DataType.COMPLEX_FP32,
            compute=ComputeType.FP64 if fp64 else ComputeType.FP32, execution_mode=ExecutionMode.COMPOSITE)
        cache = args.output / (case["name"] + ".cache.jsonl")
        with ExitStack() as owned:
            plans, buffers, graphs, eager, outputs = {}, [], {}, {}, {}
            for provider, requested in mappings_for(case).items():
                rejected = resource_rejection(case, requested, shared_limit.value)
                if rejected:
                    if row["providers"].get(provider, {}).get("trial_us"):
                        raise RuntimeError("resource-infeasible candidate has prior timing samples")
                    row["providers"][provider] = rejected
                    save(args.output / "summary.json", report)
                    continue
                plan, buffer, init_ms = open_plan(owned, transform, requested, cache)
                plans[provider] = plan
                buffers.append(buffer)
                details = metadata(plan)
                validate_mapping(provider, details["mapping"], requested)
                prior = row["providers"].get(provider, {})
                if prior.get("mapping", details["mapping"]) != details["mapping"]:
                    raise RuntimeError("actual mapping changed on resume")
                details.update(requested_mapping=requested, plan_initialization_ms=init_ms,
                    trial_us=prior.get("trial_us", []), design=design_summary(details["mapping"]))
                module = case["requests"][provider].get("module")
                details["actual_module"] = (dict(path=module, sha256=digest(module)) if module else
                                             dict(kind="linked library", sha256=report["identity"]["library_sha256"]))
                row["providers"][provider] = details
                save(args.output / (case["name"] + f".{provider}.mapping.json"), details["mapping"])
            check_factor_differences(row["providers"])
            row["factor_allowed_mapping_differences"] = ["data_tiles_per_cta", "prefetch_depth"]
            active_providers = tuple(provider for provider in PROVIDERS if provider in plans)
            row["candidate_counts"] = dict(requested=len(PROVIDERS), executable=len(active_providers),
                                          infeasible=len(PROVIDERS) - len(active_providers))
            with torch.cuda.stream(stream):
                source = torch.from_numpy(data).to(device)
                for provider in active_providers:
                    outputs[provider] = torch.empty_like(source)
            stream.synchronize()
            for provider in active_providers:
                for _ in range(3):
                    eager[provider] = execute(plans[provider], source, outputs[provider])
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(args.chain):
                        plans[provider].execute(source, outputs[provider])
                graphs[provider] = graph
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for trial in range(args.trials):
                offset = trial % len(active_providers)
                order = active_providers[offset:] + active_providers[:offset]
                row.setdefault("trial_provider_orders", {})[str(trial)] = list(order)
                check_contention(args.gpu, f"trial_{trial}_start", row, report, args.output)
                for provider in order:
                    values = row["providers"][provider]["trial_us"]
                    if len(values) > trial:
                        continue
                    with torch.cuda.stream(stream):
                        for _ in range(args.warmup):
                            graphs[provider].replay()
                        start.record(stream)
                        for _ in range(args.repeats):
                            graphs[provider].replay()
                        stop.record(stream)
                    stop.synchronize()
                    elapsed = 1000 * start.elapsed_time(stop) / (args.chain * args.repeats)
                    if not np.isfinite(elapsed) or elapsed <= 0:
                        raise RuntimeError("invalid CUDA event duration")
                    # Commit each sample only after checking its exclusive-use boundary.
                    check_contention(args.gpu, f"trial_{trial}_{provider}_end", row, report, args.output)
                    values.append(elapsed)
                    save(args.output / "summary.json", report)
            with torch.cuda.stream(stream):
                for provider in active_providers:
                    graphs[provider].replay()
            stream.synchronize()
            for provider in active_providers:
                actual = outputs[provider].cpu().numpy().copy()
                metrics, absolute = errors(actual, reference)
                details = row["providers"][provider]
                details.update(cpu_reference_error=metrics, median_us=statistics.median(details["trial_us"]),
                               graph_matches_eager=bool(np.array_equal(actual, eager[provider])))
                distributions = dict(actual=actual, absolute_error=absolute)
                if metrics["nonfinite"] or not details["graph_matches_eager"]:
                    row["failures"].append(provider + ": nonfinite output or Graph/eager mismatch")
                # Exact actual JSON replay is outside all timed regions.
                try:
                    with ExitStack() as replay_owned:
                        replay, buffer, init_ms = open_plan(replay_owned, transform, details["mapping"], cache)
                        out = torch.empty_like(source)
                        replay_actual = execute(replay, source, out)
                        checks = dict(metadata(replay), plan_initialization_ms=init_ms,
                            mapping_equal=json.loads(replay.mapping_json()) == details["mapping"],
                            output_equal=bool(np.array_equal(replay_actual, eager[provider])),
                            input_preserved=bool(np.array_equal(source.cpu().numpy(), data)))
                        checks["cpu_reference_error"], replay_error = errors(replay_actual, reference)
                        row["validations"][provider + "_json_replay"] = checks
                        distributions.update(replay_output=replay_actual, replay_absolute_error=replay_error)
                        if checks["cpu_reference_error"]["nonfinite"] or not all(checks[key] for key in
                                ("mapping_equal", "output_equal", "input_preserved")):
                            row["failures"].append(provider + ": exact JSON replay failed")
                finally:
                    details["error_artifact"] = save_provider_errors(args.output, case["name"], provider, distributions)
                    save(args.output / "summary.json", report)
                    distributions.clear()
                del actual, absolute, replay_actual, replay_error
            row["input_preserved"] = bool(np.array_equal(source.cpu().numpy(), data))
            if not row["input_preserved"]:
                row["failures"].append("input mutated")
            row["speedup_vs_cufft"] = {provider: row["providers"]["cufft"]["median_us"] /
                row["providers"][provider]["median_us"] for provider in active_providers if provider != "cufft"}
            row["status"] = "failed" if row["failures"] else COMPLETE
            save(args.output / "summary.json", report)
            graphs.clear()
            del graph
            if row["failures"]:
                raise RuntimeError("; ".join(row["failures"]))
        print(case["name"] + ": measured and exact JSON replay checked", flush=True)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--gpu", required=True, help="exclusive full GPU UUID")
    parser.add_argument("--output", required=True, type=Path)
    for key, default in (("chain", 8), ("repeats", 20), ("trials", 5), ("warmup", 5),
                         ("sm", 80), ("compile-timeout", 600)):
        parser.add_argument("--" + key, type=int, default=default)
    parser.add_argument("--mathdx", type=Path, default=Path(os.environ.get(
        "CUBUTTERFLY_MATHDX_ROOT", ROOT / "external/mathdx/nvidia/mathdx")))
    parser.add_argument("--nvcc", type=Path, default=Path(os.environ.get("CUBUTTERFLY_NVCC", "/usr/local/cuda/bin/nvcc")))
    parser.add_argument("--jit-cache", type=Path, default=Path(os.environ.get("CUBUTTERFLY_JIT_CACHE",
        str(Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "cubutterfly/modules"))))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="verify artifacts and reuse prepared CPU/JIT work and saved trials")
    parser.add_argument("--reuse-preparation-from", type=Path,
                        help="new output reuses unmeasured CPU/JIT artifacts after a runner-only revision")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.resume and args.reuse_preparation_from:
        parser.error("--resume and --reuse-preparation-from are mutually exclusive")
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", args.gpu):
        parser.error("--gpu must be a full GPU UUID")
    if not (1 <= args.chain <= 1024 and 1 <= args.repeats <= 100000 and
            1 <= args.trials <= 101 and 0 <= args.warmup <= 100000 and
            args.sm >= 70 and args.compile_timeout > 0):
        parser.error("invalid bounded experiment parameters")
    for key in ("library", "output", "jit_cache", "mathdx", "nvcc"):
        setattr(args, key, getattr(args, key).resolve(strict=key == "library"))
    for key, value in dict(CUBUTTERFLY_JIT_CACHE=args.jit_cache,
                           CUBUTTERFLY_MATHDX_ROOT=args.mathdx, CUBUTTERFLY_NVCC=args.nvcc,
                           CUBUTTERFLY_TEMPLATE_ROOT=ROOT).items():
        os.environ[key] = str(value)
    os.environ.setdefault("CUBUTTERFLY_COMPILE_MODE", "research")
    args.compile_policy = os.environ["CUBUTTERFLY_COMPILE_MODE"]
    report, ready = None, False
    try:
        if args.resume:
            report = json.loads((args.output / "summary.json").read_text())
            validate_prepared(args, report)
            ready = True
        elif args.reuse_preparation_from:
            report = reuse_preparation(args)
            validate_prepared(args, report)
        else:
            args.output.mkdir(parents=True, exist_ok=False)
            report = dict(status="preparing_cpu", started_utc=datetime.now(timezone.utc).isoformat(),
                identity=identity(args), cases=[], protocol="public_long1d_serial_cuda_graph_chain_v1",
                scope="four forward B16 requests; 24 candidates; two distinct core/organization families",
                compute_family="fft-h", storage_precisions=["fp32", "fp64"],
                compute_precisions=["fp32", "fp64"],
                attribution="only factor-family comparisons isolate intra-stage organization",
                factor_geometry="columns=4; logN16 EPT8 (threads 32,512); logN20 EPT16 (threads 256,256)",
                numerical_policy="same-precision scipy.fft workers=1 norm=backward; finite errors report-only",
                timing_scope="complete preallocated Plan CUDA Graph; us per full batch; fixed input serial chain",
                normalization="forward unscaled", baseline="whole-transform cuFFT",
                argv=sys.argv if argv is None else argv)
            report["identity_sha256"] = hashlib.sha256(json.dumps(
                report["identity"], sort_keys=True).encode()).hexdigest()
            save(args.output / "summary.json", report)
            prepare(args, report)
            validate_prepared(args, report)
        if not args.prepare_only and report["status"] != COMPLETE:
            state = gpu_state(args.gpu)
            if report.get("gpu") and any(report["gpu"][key] != state[key] for key in
                                          ("uuid", "name", "memory_mib", "driver")):
                raise RuntimeError("resume GPU identity or driver changed")
            report.update(gpu=state, status="running")
            for key in ("error", "error_type", "traceback"):
                report.pop(key, None)
            report.setdefault("measurement_sessions", []).append(datetime.now(timezone.utc).isoformat())
            save(args.output / "summary.json", report)
            run_gpu(args, report)
            report["status"] = COMPLETE
    except Exception as error:
        if report is None:
            raise
        # An incompatible resume must not overwrite the original checkpoint.
        failure = dict(error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
        if args.resume and not ready:
            save(args.output / "resume_failure.json", dict(status="failed", **failure))
            print(json.dumps(dict(status="failed", **failure)), file=sys.stderr)
            return 1
        for row in report["cases"]:
            if row["status"] == "running":
                row.update(status="failed", **failure)
        report.update(status="failed", **failure)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save(args.output / "summary.json", report)
    print(json.dumps(dict(status=report["status"], output=str(args.output / "summary.json"))))
    return int(report["status"] == "failed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and Path(sys.argv[1]).name == "compile_module.py":
        raise SystemExit(module_lookup(sys.argv[1:]))
    raise SystemExit(main())
