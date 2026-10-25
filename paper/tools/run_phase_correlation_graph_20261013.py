#!/usr/bin/env python3
"""Bounded FP32 integer-translation image registration, including complete graphs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
EPSILON = 1.1920928955078125e-7
SOURCE = "https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.phase_cross_correlation"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(x):
    return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()


def torch_registration(torch, reference_spectrum, moving, forward, inverse):
    cross = reference_spectrum * forward(moving).conj()
    cross = cross / cross.abs().clamp_min(EPSILON)
    correlation = inverse(cross)
    peak = correlation.abs().flatten(1).argmax(dim=1)
    height, width = moving.shape[-2:]
    y, x = peak // width, peak % width
    shifts = torch.stack((torch.where(y > height // 2, y - height, y),
                          torch.where(x > width // 2, x - width, x)), dim=1)
    return shifts, correlation


def cupy_registration(cp, reference_spectrum, moving, forward, inverse):
    cross = reference_spectrum * forward(moving).conj()
    cross = cross / cp.maximum(cp.abs(cross), cp.float32(EPSILON))
    correlation = inverse(cross)
    peak = cp.argmax(cp.abs(correlation).reshape(moving.shape[0], -1), axis=1)
    height, width = moving.shape[-2:]
    y, x = peak // width, peak % width
    shifts = cp.stack((cp.where(y > height // 2, y - height, y),
                       cp.where(x > width // 2, x - width, x)), axis=1)
    return shifts, correlation


def make_inputs(torch, size, batch, count, seed=20261013):
    generator = torch.Generator().manual_seed(seed)
    reference = torch.randn(batch, size, size, generator=generator, dtype=torch.float32)
    moving, expected = [], []
    for index in range(count):
        translations = [((13 * (b + 1) + 7 * index) % (size // 2 - 1) + 1,
                         -((19 * (b + 1) + 11 * index) % (size // 2 - 1) + 1))
                        for b in range(batch)]
        moving.append(torch.stack([torch.roll(reference[b], shift, (-2, -1))
                                   for b, shift in enumerate(translations)]))
        expected.append(torch.tensor([[-y, -x] for y, x in translations], dtype=torch.int64))
    return reference, moving, expected


def errors(torch, actual, reference):
    actual, reference = actual.cpu(), reference.cpu()
    difference = (actual - reference).abs()
    return dict(finite=bool(torch.isfinite(actual).all()), max_abs=float(difference.max()),
                mean_abs=float(difference.mean()),
                relative_l2=float(torch.linalg.vector_norm(difference) /
                                  torch.linalg.vector_norm(reference).clamp_min(EPSILON)))


def cpu_reference(torch, reference, inputs):
    spectrum = torch.fft.rfftn(reference, dim=(-2, -1), norm="backward")
    forward = lambda x: torch.fft.rfftn(x, dim=(-2, -1), norm="backward")
    inverse = lambda x: torch.fft.irfftn(x, s=reference.shape[-2:], dim=(-2, -1), norm="backward")
    return [torch_registration(torch, spectrum, x, forward, inverse) for x in inputs]


class CupyGraph:
    def __init__(self, cp, function, example, stream, warmup):
        self.cp, self.stream = cp, stream
        self.input = example.copy()
        # Keep captured temporary addresses out of the ordinary allocation pool.
        self.pool = cp.cuda.MemoryPool()
        with cp.cuda.using_allocator(self.pool.malloc):
            for _ in range(warmup):
                function(self.input)
            stream.synchronize()
            stream.begin_capture()
            self.output = function(self.input)
            self.graph = stream.end_capture()
        stream.synchronize()

    def __call__(self, value):
        self.cp.copyto(self.input, value)
        self.graph.launch(stream=self.stream)
        return self.output.copy()


def measure(function, inputs, stream, event_factory, warmup, calls, elapsed):
    for index in range(warmup):
        function(inputs[index % len(inputs)])
    stream.synchronize()
    wall, gpu = [], []
    for index in range(calls):
        begin, end = event_factory(), event_factory()
        start = time.perf_counter()
        begin.record(stream)
        result = function(inputs[index % len(inputs)])
        end.record(stream)
        end.synchronize()
        wall.append((time.perf_counter() - start) * 1000)
        gpu.append(elapsed(begin, end))
    return dict(wall_ms=wall, cuda_event_ms=gpu, wall_median_ms=statistics.median(wall),
                cuda_event_median_ms=statistics.median(gpu))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=ROOT / "build-e2e-20261013/libcuntt.so")
    parser.add_argument("--size", type=int, choices=(1024, 2048), default=1024)
    parser.add_argument("--batch", type=int, choices=(4, 16), default=4)
    parser.add_argument("--input-batches", type=int, default=2)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--process-trial-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("phase_correlation.json"))
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--transform-diagnostic", action="store_true")
    parser.add_argument("--profile", action="store_true",
                        help="Record one additional graph kernel timeline per backend")
    args = parser.parse_args()
    if args.input_batches < 2 or min(args.calls, args.warmup) < 1:
        parser.error("at least two input batches and positive calls/warmup required")
    import torch
    torch.set_num_threads(min(8, torch.get_num_threads()))
    reference, cpu_inputs, expected = make_inputs(torch, args.size, args.batch, args.input_batches)
    cpu_results = cpu_reference(torch, reference, cpu_inputs)
    report = dict(application="batched 2D integer-translation phase correlation", source=SOURCE,
                  scope="integer circular translations; fixed cached reference; no subpixel refinement",
                  size=args.size, batch=args.batch, norm="backward", epsilon=EPSILON,
                  seed=20261013, runner_sha256=digest(__file__), torch=torch.__version__,
                  reference_sha256=tensor_digest(reference), input_sha256=[tensor_digest(x) for x in cpu_inputs],
                  expected_registration_shifts=[x.tolist() for x in expected],
                  cpu_peak_correct=[bool(torch.equal(x[0], y)) for x, y in zip(cpu_results, expected)],
                  cpu_correlation_finite=[bool(torch.isfinite(x[1]).all()) for x in cpu_results],
                  error_acceptance="finite required; errors reported with no new numerical threshold",
                  timing_scope="device input copy + complete registration/replay + independent shift output copy + synchronization",
                  setup_search_capture_excluded=True, quality={}, timings={}, setup={})
    if not all(report["cpu_peak_correct"]) or not all(report["cpu_correlation_finite"]):
        raise RuntimeError("CPU reference does not recover known integer translations")
    if not args.cpu_only:
        import cupy as cp
        from cubutterfly.real_fft import RealFFT
        from cubutterfly.torch_graph import InferenceGraph
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        report_dir = Path(os.environ.get("XD_REPORT_DIR", str(args.output.parent))).resolve()
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
            if backend == "cupy":
                stream = cp.cuda.Stream(non_blocking=True)
                context = stream
            else:
                stream = torch.cuda.Stream(device=device)
                context = torch.cuda.stream(stream)
            try:
                with context, torch.inference_mode():
                    if backend == "cupy":
                        template = cp.asarray(reference.numpy())
                        inputs = [cp.asarray(x.numpy()) for x in cpu_inputs]
                        forward = lambda x: cp.fft.rfftn(x, axes=(-2, -1), norm="backward")
                        inverse = lambda x: cp.fft.irfftn(x, s=(args.size, args.size), axes=(-2, -1), norm="backward")
                        spectrum = forward(template)
                        registration = lambda x: cupy_registration(cp, spectrum, x, forward, inverse)
                        clone = lambda x: x.copy()
                        copyto = cp.copyto
                        host = lambda x: torch.from_numpy(cp.asnumpy(x))
                        factory = lambda: cp.cuda.Event()
                        elapsed = cp.cuda.get_elapsed_time
                    else:
                        template = reference.to(device)
                        inputs = [x.to(device) for x in cpu_inputs]
                        if backend == "cubutterfly":
                            adapter = RealFFT(str(args.library.resolve()), (args.size, args.size), args.batch,
                                              device, norm="backward", reuse_buffers=True, cache_path=str(cache))
                            forward, inverse = adapter.rfftn, adapter.irfftn
                        else:
                            forward = lambda x: torch.fft.rfftn(x, dim=(-2, -1), norm="backward")
                            inverse = lambda x: torch.fft.irfftn(x, s=(args.size, args.size), dim=(-2, -1), norm="backward")
                        spectrum = forward(template).clone()
                        registration = lambda x: torch_registration(torch, spectrum, x, forward, inverse)
                        clone = lambda x: x.clone()
                        copyto = lambda dest, src: dest.copy_(src)
                        host = lambda x: x.cpu()
                        factory = lambda: torch.cuda.Event(enable_timing=True)
                        elapsed = lambda begin, end: begin.elapsed_time(end)
                    buffer = clone(inputs[0])
                    def eager(value):
                        copyto(buffer, value)
                        return clone(registration(buffer)[0])
                    function = lambda x: registration(x)[0]
                    graph = (CupyGraph(cp, function, inputs[0], stream, args.warmup) if backend == "cupy"
                             else InferenceGraph(function, inputs[0], stream=stream, warmup=args.warmup))
                    stream.synchronize()
                    report["setup"][backend] = dict(elapsed_ms=(time.perf_counter() - start) * 1000,
                        plans=adapter.plan_metadata() if adapter else None, reference_spectrum_cached=True)
                    for mode, call in (("eager", eager), ("graph", graph)):
                        validation_inputs = inputs + [inputs[0]]
                        validation_expected = expected + [expected[0]]
                        outputs = [call(x) for x in validation_inputs]
                        stream.synchronize()
                        shifts = [host(x) for x in outputs]
                        eager_outputs = [eager(x) for x in validation_inputs]
                        stream.synchronize()
                        eager_shifts = [host(x) for x in eager_outputs]
                        correlations = []
                        for x, (_, ref_corr) in zip(inputs, cpu_results):
                            _, corr = registration(x)
                            stream.synchronize()
                            correlations.append(errors(torch, host(corr), ref_corr))
                        quality = dict(shifts=[x.tolist() for x in shifts],
                            peak_correct=[bool(torch.equal(x, y)) for x, y in zip(shifts, validation_expected)],
                            exact_vs_same_backend_eager=[bool(torch.equal(x, y)) for x, y in zip(shifts, eager_shifts)],
                            validation_sequence="distinct inputs, then restored original; outputs retained across all calls",
                            eager_correlation_diagnostic_vs_cpu_fp32=correlations,
                            correlation_diagnostic_scope="same-backend eager intermediate; graph correctness checks actual returned shifts",
                            output_sha256=[tensor_digest(x) for x in shifts],
                            independent_outputs=len({x.data_ptr() if backend != "cupy" else x.data.ptr for x in outputs}) == len(outputs))
                        report["quality"][backend + "_" + mode] = quality
                        if (not quality["independent_outputs"] or not all(quality["peak_correct"])
                                or not all(quality["exact_vs_same_backend_eager"])
                                or not all(x["finite"] for x in correlations)):
                            raise RuntimeError("registration correctness/ownership validation failed")
                        report["timings"][backend + "_" + mode] = measure(call, inputs, stream, factory,
                            args.warmup, args.calls, elapsed)
                        if args.profile and mode == 'graph':
                            from profile_graph_application import profile_call
                            report.setdefault('gpu_traces', {})[backend] = profile_call(
                                lambda: call(inputs[0]), report_dir / f'{backend}.trace.json',
                                synchronize=stream.synchronize)
                    if args.transform_diagnostic:
                        pair = lambda x: inverse(forward(x))
                        diagnostic = (CupyGraph(cp, pair, inputs[0], stream, args.warmup) if backend == "cupy"
                                      else InferenceGraph(pair, inputs[0], stream=stream, warmup=args.warmup))
                        report.setdefault("transform_pair_diagnostic", {})[backend] = measure(diagnostic,
                            inputs, stream, factory, args.warmup, args.calls, elapsed)
                    stream.synchronize()
            finally:
                stream.synchronize()
                if adapter:
                    adapter.close()
        report["cache"] = dict(path=str(cache), sha256=digest(cache) if cache.is_file() else None)
        report["diagnostic_interpretation"] = "isolated transform-pair graphs are nonadditive diagnostics, not exact application time fractions"
        report["speedup"] = {backend + "_" + mode: report["timings"][backend + "_" + mode]["wall_median_ms"] /
            report["timings"]["cubutterfly_" + mode]["wall_median_ms"] for backend in ("torch", "cupy") for mode in ("eager", "graph")}
        for timing in report["timings"].values():
            timing.update(batch_throughput_per_second=args.batch * 1000 / timing["wall_median_ms"],
                          per_image_wall_ms=timing["wall_median_ms"] / args.batch)
    report["cpu_only"] = args.cpu_only
    output = args.output
    if "XD_REPORT_DIR" in os.environ and not output.is_absolute():
        output = Path(os.environ["XD_REPORT_DIR"]) / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(report=str(output), cpu_peak_correct=report["cpu_peak_correct"], speedup=report.get("speedup"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
