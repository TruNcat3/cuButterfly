#!/usr/bin/env python3
"""FP32 periodic spectral Poisson solve: -Laplacian(u)=f on [0,2*pi)^2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "experiments/tools"))
from run_phase_correlation_graph_20261013 import CupyGraph, digest, errors, measure, tensor_digest

SOURCE = "https://github.com/google/jax-cfd/blob/main/jax_cfd/spectral/utils.py"


def spectral_coefficients(torch, size):
    frequencies = torch.arange(size, dtype=torch.float32)
    frequencies = torch.where(frequencies <= size // 2, frequencies, frequencies - size)
    half = torch.arange(size // 2 + 1, dtype=torch.float32)
    eigenvalues = frequencies[:, None].square() + half[None, :].square()
    multiplier = eigenvalues.clamp_min(1).reciprocal()
    multiplier[0, 0] = 0
    return eigenvalues, multiplier


def solve(rhs, forward, inverse, multiplier):
    return inverse(forward(rhs) * multiplier)


def make_inputs(torch, size, batch, seed=20261013):
    coordinate = torch.arange(size, dtype=torch.float32) * (2 * torch.pi / size)
    y, x = coordinate[:, None], coordinate[None, :]
    mode_a = torch.sin(2 * y) * torch.cos(3 * x)
    mode_b = torch.cos(5 * y) * torch.sin(x)
    amplitude = torch.arange(1, batch + 1, dtype=torch.float32)[:, None, None] / batch
    analytical = amplitude * mode_a + (1 - amplitude / 2) * mode_b
    manufactured_rhs = 13 * amplitude * mode_a + 26 * (1 - amplitude / 2) * mode_b
    generator = torch.Generator().manual_seed(seed)
    random_inputs = [torch.randn(batch, size, size, generator=generator, dtype=torch.float32)
                     for _ in range(2)]
    inputs = [manufactured_rhs, *random_inputs]
    inputs = [value - value.mean(dim=(-2, -1), keepdim=True) for value in inputs]
    analytical = analytical - analytical.mean(dim=(-2, -1), keepdim=True)
    return inputs, analytical


def cpu_operators(torch, size, norm="backward"):
    forward = lambda x: torch.fft.rfftn(x, dim=(-2, -1), norm=norm)
    inverse = lambda x: torch.fft.irfftn(x, s=(size, size), dim=(-2, -1), norm=norm)
    return forward, inverse


def residual_diagnostic(torch, solution, rhs, eigenvalues, norm="backward"):
    forward, inverse = cpu_operators(torch, rhs.shape[-1], norm)
    return errors(torch, inverse(forward(solution) * eigenvalues), rhs)


def summarize_complete_call_timings(timings, batch):
    """Use full-call device intervals as primary; retain synchronized wall diagnostics."""
    speedup_event, speedup_wall = {}, {}
    for backend in ("torch", "cupy"):
        for mode in ("eager", "graph"):
            name, own = backend + "_" + mode, timings["cubutterfly_" + mode]
            speedup_event[name] = timings[name]["cuda_event_median_ms"] / own["cuda_event_median_ms"]
            speedup_wall[name] = timings[name]["wall_median_ms"] / own["wall_median_ms"]
    for timing in timings.values():
        timing.update(batch_throughput_per_second=batch * 1000 / timing["cuda_event_median_ms"],
                      batch_cuda_event_throughput_per_second=batch * 1000 / timing["cuda_event_median_ms"],
                      batch_wall_throughput_per_second=batch * 1000 / timing["wall_median_ms"],
                      per_rhs_cuda_event_ms=timing["cuda_event_median_ms"] / batch,
                      per_rhs_wall_ms=timing["wall_median_ms"] / batch)
    return dict(speedup=speedup_event, speedup_cuda_event=speedup_event,
                speedup_wall=speedup_wall)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=ROOT / "build-e2e-20261013/libcuntt.so")
    parser.add_argument("--size", type=int, choices=(1024, 2048), default=1024)
    parser.add_argument("--batch", type=int, choices=(4, 16), default=4)
    parser.add_argument("--norm", choices=("backward", "forward"), default="backward")
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--process-trial-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("periodic_poisson.json"))
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--transform-diagnostic", action="store_true")
    args = parser.parse_args()
    if min(args.calls, args.warmup) < 1:
        parser.error("calls and warmup must be positive")
    import torch
    torch.set_num_threads(min(8, torch.get_num_threads()))
    cpu_inputs, analytical = make_inputs(torch, args.size, args.batch)
    eigenvalues, cpu_multiplier = spectral_coefficients(torch, args.size)
    cpu_forward, cpu_inverse = cpu_operators(torch, args.size, args.norm)
    references = [solve(x, cpu_forward, cpu_inverse, cpu_multiplier) for x in cpu_inputs]
    report = dict(schema="periodic-poisson-complete-call-event-v2",
        application="periodic 2D spectral Poisson RHS-to-solution solve", source=SOURCE,
        source_scope="JAX-CFD vorticity_to_velocity uses an inverse Fourier Laplacian to solve a stream-function Poisson equation",
        equation="-Laplacian(u)=f on [0,2*pi)^2; mean-zero solution", norm=args.norm,
        zero_mode="inverse multiplier[0,0]=0; CPU FP32 inputs projected to mean zero before device transfer",
        size=args.size, batch=args.batch, seed=20261013, torch=torch.__version__,
        runner_sha256=digest(__file__), graph_timing_helpers_sha256=digest(ROOT / "experiments/tools/run_phase_correlation_graph_20261013.py"),
        input_sha256=[tensor_digest(x) for x in cpu_inputs], multiplier_sha256=tensor_digest(cpu_multiplier),
        input_roles=["manufactured Fourier modes (2,3) and (5,1)", "independent seeded random RHS 0", "independent seeded random RHS 1"],
        timed_input_roles=["independent seeded random RHS 0", "independent seeded random RHS 1"],
        input_means=[x.mean(dim=(-2, -1)).tolist() for x in cpu_inputs],
        cpu_finite=[bool(torch.isfinite(x).all()) for x in references],
        cpu_manufactured_errors=errors(torch, references[0], analytical),
        cpu_residual_diagnostics=[residual_diagnostic(torch, u, f, eigenvalues, args.norm) for u, f in zip(references, cpu_inputs)],
        residual_scope="CPU FP32 spectral -Laplacian of returned solution; derivative amplifies high-frequency rounding; diagnostic only",
        error_acceptance="finite required; error distributions reported without a new numerical threshold",
        timing_scope="device input copy + complete RHS FFT/multiplier/inverse FFT or replay + independent full solution output copy + synchronization",
        primary_timing="cuda_event_median_ms",
        timing_contract={"cuda_event": "same-stream events bracket input device refresh + complete solve/replay + independent output copy",
                         "wall": "host wall interval brackets event recording + the same complete call + completion synchronization; diagnostic",
                         "graph_call": ["device_input_refresh", "rhs_fft", "frequency_multiplier", "inverse_fft", "independent_output_copy"],
                         "aggregation": "direct complete-call intervals; no sum of operator or stage timings",
                         "excluded": ["host_device_transfer", "CPU_validation", "plan_setup", "search", "graph_capture", "warmup"]},
        setup_search_capture_excluded=True, quality={}, timings={}, setup={})
    if not all(report["cpu_finite"]):
        raise RuntimeError("non-finite CPU solution")
    report_dir = Path(os.environ.get("XD_REPORT_DIR", str(args.output.parent))).resolve()
    if not args.cpu_only:
        import cupy as cp
        from cubutterfly.real_fft import RealFFT
        from cubutterfly.torch_graph import InferenceGraph
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        report_dir.mkdir(parents=True, exist_ok=True)
        cache = Path(os.environ.get("CUBUTTERFLY_PLAN_CACHE", str(report_dir / "plans.cache"))).resolve()
        os.environ["CUBUTTERFLY_PLAN_CACHE"] = str(cache)
        os.environ.setdefault("CUBUTTERFLY_COMPILE_MODE", "research")
        properties = torch.cuda.get_device_properties(device)
        report.update(library_sha256=digest(args.library), public_real_fft_sha256=digest(ROOT / "python/cubutterfly/real_fft.py"),
            gpu_uuid=str(properties.uuid), gpu_name=properties.name, gpu_memory_bytes=properties.total_memory,
            cuda=torch.version.cuda, cupy=cp.__version__, cupy_cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
            cupy_cuda_driver=cp.cuda.runtime.driverGetVersion(),
            environment={k: v for k, v in os.environ.items() if k.startswith(("CUBUTTERFLY_", "CUDA_", "XD_"))})
        backends = ["torch", "cubutterfly", "cupy"]
        offset = args.process_trial_index % len(backends)
        backends = backends[offset:] + backends[:offset]
        report["backend_order"] = backends
        for backend in backends:
            start = time.perf_counter()
            adapter = None
            stream = cp.cuda.Stream(non_blocking=True) if backend == "cupy" else torch.cuda.Stream(device=device)
            context = stream if backend == "cupy" else torch.cuda.stream(stream)
            try:
                with context, torch.inference_mode():
                    if backend == "cupy":
                        inputs = [cp.asarray(x.numpy()) for x in cpu_inputs]
                        multiplier = cp.asarray(cpu_multiplier.numpy())
                        forward = lambda x: cp.fft.rfftn(x, axes=(-2, -1), norm=args.norm)
                        inverse = lambda x: cp.fft.irfftn(x, s=(args.size, args.size), axes=(-2, -1), norm=args.norm)
                        clone, copyto = lambda x: x.copy(), cp.copyto
                        host = lambda x: torch.from_numpy(cp.asnumpy(x))
                        pointer = lambda x: x.data.ptr
                        factory, elapsed = lambda: cp.cuda.Event(), cp.cuda.get_elapsed_time
                    else:
                        inputs = [x.to(device) for x in cpu_inputs]
                        multiplier = cpu_multiplier.to(device)
                        if backend == "cubutterfly":
                            adapter = RealFFT(str(args.library.resolve()), (args.size, args.size), args.batch,
                                device, norm=args.norm, reuse_buffers=True, cache_path=str(cache))
                            forward, inverse = adapter.rfftn, adapter.irfftn
                        else:
                            forward, inverse = cpu_operators(torch, args.size, args.norm)
                        clone, copyto = lambda x: x.clone(), lambda dest, src: dest.copy_(src)
                        host, pointer = lambda x: x.cpu(), lambda x: x.data_ptr()
                        factory = lambda: torch.cuda.Event(enable_timing=True)
                        elapsed = lambda begin, end: begin.elapsed_time(end)
                    function = lambda x: solve(x, forward, inverse, multiplier)
                    buffer = clone(inputs[1])
                    def eager(value):
                        copyto(buffer, value)
                        return clone(function(buffer))
                    graph = (CupyGraph(cp, function, inputs[1], stream, args.warmup) if backend == "cupy"
                             else InferenceGraph(function, inputs[1], stream=stream, warmup=args.warmup))
                    stream.synchronize()
                    report["setup"][backend] = dict(elapsed_ms=(time.perf_counter() - start) * 1000,
                        fixed_multiplier_precomputed=True, plans=adapter.plan_metadata() if adapter else None)
                    validation_inputs = inputs + [inputs[0]]
                    validation_references = references + [references[0]]
                    for mode, call in (("eager", eager), ("graph", graph)):
                        outputs = [call(x) for x in validation_inputs]
                        stream.synchronize()
                        returned = [host(x) for x in outputs]
                        eager_outputs = [eager(x) for x in validation_inputs]
                        stream.synchronize()
                        same_backend = [host(x) for x in eager_outputs]
                        quality = dict(vs_cpu_fp32=[errors(torch, y, ref) for y, ref in zip(returned, validation_references)],
                            vs_same_backend_eager=[errors(torch, y, ref) for y, ref in zip(returned, same_backend)],
                            exact_vs_same_backend_eager=[bool(torch.equal(y, ref)) for y, ref in zip(returned, same_backend)],
                            manufactured_errors=errors(torch, returned[0], analytical),
                            output_sha256=[tensor_digest(x) for x in returned],
                            output_means=[x.mean(dim=(-2, -1)).tolist() for x in returned],
                            independent_outputs=len({pointer(x) for x in outputs}) == len(outputs),
                            retained_first_after_changed_and_restored=bool(torch.equal(returned[0], returned[-1])),
                            residual_diagnostics=[residual_diagnostic(torch, u, f, eigenvalues, args.norm) for u, f in zip(returned[:3], cpu_inputs)])
                        report["quality"][backend + "_" + mode] = quality
                        if (not quality["independent_outputs"] or not quality["retained_first_after_changed_and_restored"]
                                or not all(quality["exact_vs_same_backend_eager"])
                                or not all(x["finite"] for x in quality["vs_cpu_fp32"] + quality["vs_same_backend_eager"])):
                            raise RuntimeError("Poisson finite/output-retention validation failed")
                        report["timings"][backend + "_" + mode] = measure(call, inputs[1:], stream,
                            factory, args.warmup, args.calls, elapsed)
                        if args.profile and mode == "graph":
                            from profile_graph_application import profile_call
                            report.setdefault("gpu_traces", {})[backend] = profile_call(lambda: call(inputs[1]),
                                report_dir / f"{backend}.trace.json", synchronize=stream.synchronize)
                    if args.transform_diagnostic:
                        pair = lambda x: inverse(forward(x))
                        diagnostic = (CupyGraph(cp, pair, inputs[1], stream, args.warmup) if backend == "cupy"
                                      else InferenceGraph(pair, inputs[1], stream=stream, warmup=args.warmup))
                        report.setdefault("transform_pair_diagnostic", {})[backend] = measure(diagnostic,
                            inputs[1:], stream, factory, args.warmup, args.calls, elapsed)
                    stream.synchronize()
            finally:
                stream.synchronize()
                if adapter:
                    adapter.close()
        report["cache"] = dict(path=str(cache), sha256=digest(cache) if cache.is_file() else None)
        report["diagnostic_interpretation"] = "isolated transform-pair graphs are nonadditive; profiler attribution does not replace ordinary complete-call timing"
        report.update(summarize_complete_call_timings(report["timings"], args.batch))
    report["cpu_only"] = args.cpu_only
    output = report_dir / args.output if "XD_REPORT_DIR" in os.environ and not args.output.is_absolute() else args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(report=str(output), cpu_finite=report["cpu_finite"], speedup=report.get("speedup"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
