#!/usr/bin/env python3
"""Fixed-mapping long-1D prefix core replacement, not automatic-library evidence.

Prepare same-precision SciPy references and all JIT modules before GPU execution.
Finite numerical errors are reported without an acceptance threshold.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
import traceback

from compile_module import compile_mapping
from research_compile_requests import compile_request
from run_public_radix4_pair_experiment import (
    ROOT, check_contention, cpu_arrays, digest, errors, gpu_state, metadata, save,
)

PROVIDERS = ("native", "radix4", "cufft")


def cases(precisions, batches, logns):
    for precision in precisions:
        for log_n in logns:
            for batch in batches:
                for inverse in (False, True):
                    yield dict(name=f"log{log_n}_b{batch}_{precision}_{'inverse' if inverse else 'forward'}",
                               logN=log_n, shape=[1 << log_n], batch=batch,
                               precision=precision, fft_type="c2c", inverse=inverse,
                               normalize_inverse=inverse)


def mapping_for(case, codelet):
    local = case["logN"] - 10
    radius = 1 << (local // 2)
    return dict(kind="butterfly", backend="online-reorder", fft_core="register-tile",
                compute_unit="auto", local_stages=local, stage_partition=[local, 10],
                prefix_threads=radius * 4, prefix_ept=radius, prefix_codelet=codelet,
                prefix_shared_layout="linear", prefix_codelet_lanes=1,
                suffix_threads=256, suffix_ept=16, suffix_core="cufftdx-block",
                shared_layout="writer-aligned", cross_twiddle="recurrence",
                local_exchange="shared", direct_boundary="direct-strided",
                complex_multiply="four-mul", stage_overlap=False, batch_tile_count=1,
                boundaries=[dict(twiddle="recurrence", layout="direct-strided",
                                 residency="global-scratch")])


def prepare(args, report):
    import numpy as np
    import scipy
    prepared, modules = [], {}
    report.update(numpy=np.__version__, scipy=scipy.__version__, prepared_cases=[], modules=[])
    for index, case in enumerate(cases(args.precisions, args.batches, args.logns)):
        artifact = args.output / (case["name"] + ".cpu.npz")
        data, reference = cpu_arrays(case, 20261024 + index)
        np.savez_compressed(artifact, input=data, cpu_reference=reference)
        entry = dict(case, cpu_seed=20261024 + index, artifact=str(artifact),
                     artifact_sha256=digest(artifact), input_dtype=str(data.dtype),
                     reference_dtype=str(reference.dtype), requests={})
        prepared.append(entry)
        report["prepared_cases"].append(entry)
        del data, reference
        save(args.output / "summary.json", report)
    # Every reference exists before compilation, and compilation uses only the
    # explicit architecture. No CUDA context or GPU identity query is needed.
    report["status"] = "preparing_jit"
    for entry in prepared:
        for provider in PROVIDERS[:2]:
            mapping = mapping_for(entry, provider)
            point = dict(operator="fft", precision=entry["precision"], logN=entry["logN"],
                         mapping_json=mapping)
            request = compile_request(point, args.compile_policy)
            entry["requests"][provider] = dict(mapping=mapping, compile_request=request)
            if request is None:
                continue
            key = json.dumps(request, sort_keys=True)
            if key not in modules:
                start = time.perf_counter()
                module = compile_mapping(request, root=ROOT, mathdx=args.mathdx,
                                         nvcc=args.nvcc, sm=args.sm, cache=args.jit_cache,
                                         timeout=args.compile_timeout)
                modules[key] = str(module)
                report["modules"].append(dict(request=request, module=str(module),
                                              sha256=digest(module),
                                              preparation_ms=1000 * (time.perf_counter() - start)))
            entry["requests"][provider]["module"] = modules[key]
            save(args.output / "summary.json", report)
    return prepared


def normalized_core_mapping(mapping):
    result = json.loads(json.dumps(mapping))
    result.pop("prefix_codelet", None)
    for field in ("segment_mappings", "execution_group_mappings"):
        if result.get(field):
            result[field][0].pop("codelet", None)
    return result


def run_gpu(args, prepared, report):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / "python"))
    from cubutterfly import (AlgorithmPolicy, ComputeType, DataType, ExecutionMode,
                             Library, Plan, PlanOptions, Transform)

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    if props.major * 10 + props.minor != args.sm:
        raise RuntimeError("prepared JIT architecture differs from selected GPU")
    stream = torch.cuda.Stream(device=device)
    library = Library(args.library)
    report.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                  runtime_device=dict(name=props.name, memory_bytes=props.total_memory,
                                      uuid=str(getattr(props, "uuid", args.gpu))))

    def make_plan(transform, mapping, cache):
        start = time.perf_counter()
        plan = Plan(library, transform, PlanOptions(policy=AlgorithmPolicy.EXPLICIT,
                    mapping=mapping, execution_mode=ExecutionMode.COMPOSITE),
                    stream=stream.cuda_stream, cache_path=str(cache))
        stream.synchronize()
        return plan, 1000 * (time.perf_counter() - start)

    def workspace(plan):
        buffer = torch.empty(plan.workspace_size(), dtype=torch.uint8, device=device)
        if buffer.numel():
            plan.set_workspace(buffer, buffer.numel())
        return buffer

    for case in prepared:
        row = dict(case, status="running", providers={}, validations={})
        report["cases"].append(row)
        check_contention(args.gpu, "case_start", row, report, args.output)
        with np.load(case["artifact"]) as arrays:
            data, reference = arrays["input"], arrays["cpu_reference"]
        fp64 = case["precision"] == "fp64"
        transform = Transform(extents=case["shape"], batch=case["batch"],
                              storage=DataType.COMPLEX_FP64 if fp64 else DataType.COMPLEX_FP32,
                              compute=ComputeType.FP64 if fp64 else ComputeType.FP32,
                              inverse=case["inverse"], normalize_inverse=case["inverse"],
                              execution_mode=ExecutionMode.COMPOSITE)
        cache = args.output / (case["name"] + ".cache.jsonl")
        distributions, failures = {}, []
        with ExitStack() as owned:
            plans, buffers, outputs, eager, graphs = {}, [], {}, {}, {}
            for provider in PROVIDERS:
                requested = "cufft" if provider == "cufft" else json.dumps(mapping_for(case, provider))
                row["providers"][provider] = dict(status="constructing", requested_mapping=requested)
                save(args.output / "summary.json", report)
                plan, init_ms = make_plan(transform, requested, cache)
                owned.enter_context(plan)
                plans[provider] = plan
                buffers.append(workspace(plan))
                details = dict(metadata(plan), requested_mapping=requested,
                               plan_initialization_ms=init_ms, trial_us=[])
                row["providers"][provider] = details
                mapping = details["mapping"]
                if provider != "cufft":
                    if mapping.get("prefix_codelet") != provider or mapping.get("fft_core") != "register-tile":
                        raise AssertionError("explicit prefix codelet was not preserved")
                elif mapping.get("backend") != "cufft":
                    raise AssertionError("whole-transform cuFFT baseline was not preserved")
                save(args.output / (case["name"] + f".{provider}.mapping.json"), mapping)
            if normalized_core_mapping(row["providers"]["native"]["mapping"]) != normalized_core_mapping(
                    row["providers"]["radix4"]["mapping"]):
                raise AssertionError("actual mappings differ beyond the prefix codelet identity")
            row["allowed_mapping_differences"] = ["prefix_codelet", "segment_mappings[0].codelet",
                                                    "execution_group_mappings[0].codelet"]
            with torch.cuda.stream(stream):
                source = torch.from_numpy(data).to(device)
                outputs = {provider: torch.empty_like(source) for provider in PROVIDERS}
            stream.synchronize()
            for provider in PROVIDERS:
                start = time.perf_counter()
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        plans[provider].execute(source, outputs[provider])
                stream.synchronize()
                eager[provider] = outputs[provider].cpu().numpy().copy()
                row["providers"][provider]["execution_preparation_ms"] = 1000 * (time.perf_counter() - start)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(args.chain):
                        plans[provider].execute(source, outputs[provider])
                graphs[provider] = graph
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for trial in range(args.trials):
                order = PROVIDERS[trial % 3:] + PROVIDERS[:trial % 3]
                row.setdefault("trial_provider_orders", []).append(list(order))
                check_contention(args.gpu, f"trial_{trial}_start", row, report, args.output)
                for provider in order:
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
                    row["providers"][provider]["trial_us"].append(elapsed)
                check_contention(args.gpu, f"trial_{trial}_end", row, report, args.output)
            for provider in PROVIDERS:
                actual = outputs[provider].cpu().numpy().copy()
                metrics, absolute = errors(actual, reference)
                details = row["providers"][provider]
                details.update(cpu_reference_error=metrics,
                               graph_matches_eager=bool(np.array_equal(actual, eager[provider])),
                               median_us=statistics.median(details["trial_us"]))
                distributions[provider + "_output"] = actual
                distributions[provider + "_absolute_error"] = absolute
                if metrics["nonfinite"] or not details["graph_matches_eager"]:
                    failures.append(provider + ": nonfinite output or graph/eager mismatch")
            row["input_preserved"] = bool(np.array_equal(source.cpu().numpy(), data))
            if not row["input_preserved"]:
                failures.append("input mutated")
            row["radix4_speedup_vs_native"] = row["providers"]["native"]["median_us"] / row["providers"]["radix4"]["median_us"]
            row["speedup_vs_cufft"] = {name: row["providers"]["cufft"]["median_us"] /
                                         row["providers"][name]["median_us"] for name in PROVIDERS[:2]}
            np.savez_compressed(args.output / (case["name"] + ".errors.npz"), **distributions)
            save(args.output / "summary.json", report)
            # Reconstruct each exact public mapping outside all timed regions.
            for provider in PROVIDERS:
                original = row["providers"][provider]
                replay, init_ms = make_plan(transform, json.dumps(original["mapping"]), cache)
                with replay:
                    buffer = workspace(replay)
                    with torch.cuda.stream(stream):
                        out = torch.empty_like(source)
                        replay.execute(source, out)
                    stream.synchronize()
                    actual = out.cpu().numpy().copy()
                    details = dict(metadata(replay), plan_initialization_ms=init_ms,
                                   mapping_equal=json.loads(replay.mapping_json()) == original["mapping"],
                                   output_equal=bool(np.array_equal(actual, eager[provider])),
                                   input_preserved=bool(np.array_equal(source.cpu().numpy(), data)))
                    details["cpu_reference_error"], absolute = errors(actual, reference)
                    row["validations"][provider + "_json_replay"] = details
                    distributions[provider + "_replay_output"] = actual
                    distributions[provider + "_replay_absolute_error"] = absolute
                    if (not all(details[key] for key in ("mapping_equal", "output_equal", "input_preserved")) or
                            details["cpu_reference_error"]["nonfinite"]):
                        failures.append(provider + ": explicit JSON replay failed")
                    del buffer
            np.savez_compressed(args.output / (case["name"] + ".errors.npz"), **distributions)
            row.update(status="failed" if failures else "measured_and_errors_reported", failures=failures)
            save(args.output / "summary.json", report)
            graphs.clear()
            del graph
            if failures:
                raise RuntimeError("; ".join(failures))
        print(case["name"] + ": measured and exact JSON replay checked", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--gpu", required=True, help="exclusive full GPU UUID")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--precisions", nargs="+", choices=("fp32", "fp64"), default=["fp32", "fp64"])
    parser.add_argument("--batches", nargs="+", type=int, default=[4, 16])
    parser.add_argument("--logns", nargs="+", type=int, choices=(16, 20), default=[16, 20])
    parser.add_argument("--chain", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--sm", type=int, default=80, help="explicit compilation architecture; no GPU query")
    parser.add_argument("--mathdx", type=Path, default=Path(os.environ.get(
        "CUBUTTERFLY_MATHDX_ROOT", ROOT / "external/mathdx/nvidia/mathdx")))
    parser.add_argument("--nvcc", type=Path, default=Path(os.environ.get("CUBUTTERFLY_NVCC", "/usr/local/cuda/bin/nvcc")))
    parser.add_argument("--jit-cache", type=Path, default=Path(os.environ.get("CUBUTTERFLY_JIT_CACHE",
        str(Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "cubutterfly/modules"))))
    parser.add_argument("--compile-timeout", type=int, default=600)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", args.gpu):
        parser.error("--gpu must be a full GPU UUID")
    if (any(value < 1 or value > 256 for value in args.batches) or
            not 1 <= args.chain <= 1024 or not 1 <= args.repeats <= 100000 or
            not 1 <= args.trials <= 101 or not 0 <= args.warmup <= 100000 or
            args.sm < 70 or args.compile_timeout <= 0):
        parser.error("invalid bounded experiment parameters")
    if any(len(values) != len(set(values)) for values in (args.precisions, args.batches, args.logns)):
        parser.error("case axes must not contain duplicates")
    args.library = args.library.resolve(strict=True)
    args.output = args.output.resolve()
    args.jit_cache = args.jit_cache.resolve()
    args.mathdx = args.mathdx.resolve()
    args.nvcc = args.nvcc.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["CUBUTTERFLY_JIT_CACHE"] = str(args.jit_cache)
    os.environ["CUBUTTERFLY_MATHDX_ROOT"] = str(args.mathdx)
    os.environ["CUBUTTERFLY_NVCC"] = str(args.nvcc)
    os.environ.setdefault("CUBUTTERFLY_COMPILE_MODE", "research")
    args.compile_policy = os.environ["CUBUTTERFLY_COMPILE_MODE"]
    report = dict(status="preparing_cpu", started_utc=datetime.now(timezone.utc).isoformat(),
                  scope="fixed mapping prefix core replacement; not automatic selection or full-library comparison",
                  protocol="public_long1d_serial_cuda_graph_chain_v1", argv=sys.argv,
                  requested_gpu=args.gpu, architecture=args.sm, chain=args.chain,
                  repeats=args.repeats, trials=args.trials, warmup=args.warmup,
                  prepare_only=args.prepare_only, cases=[], library=str(args.library),
                  library_sha256=digest(args.library), runner_sha256=digest(__file__),
                  helper_sha256=digest(Path(__file__).with_name("run_public_radix4_pair_experiment.py")),
                  numerical_policy="same-precision scipy.fft, workers=1, norm=backward; finite errors report-only",
                  normalization="forward unscaled; inverse normalized by N",
                  timing_scope="complete preallocated public Plan, us per full batch; fixed input, serial graph chain",
                  baseline="public explicit algorithm cufft; entire true 1D transform delegated to cuFFT",
                  environment={key: value for key, value in os.environ.items() if key.startswith("CUBUTTERFLY_")})
    save(args.output / "summary.json", report)
    try:
        prepared = prepare(args, report)
        report["status"] = "cpu_and_jit_prepared"
        save(args.output / "summary.json", report)
        if not args.prepare_only:
            report["gpu"] = gpu_state(args.gpu)
            report["status"] = "running"
            run_gpu(args, prepared, report)
            report["status"] = "measured_and_errors_reported"
    except Exception as error:
        for row in report["cases"]:
            if row["status"] == "running":
                row.update(status="failed", error=str(error), error_type=type(error).__name__)
        report.update(status="failed", error=str(error), error_type=type(error).__name__,
                      traceback=traceback.format_exc())
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save(args.output / "summary.json", report)
    print(json.dumps(dict(status=report["status"], output=str(args.output / "summary.json"))))
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
