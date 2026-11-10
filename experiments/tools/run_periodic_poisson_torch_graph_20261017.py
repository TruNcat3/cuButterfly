#!/usr/bin/env python3
"""Current-build PyTorch/cuFFT versus cuButterfly Graph Poisson control.

This runner intentionally omits CuPy so the current-build Graph control can
run in the shared PyTorch environment.  Both backends use the same complete
RHS-to-solution function, input refresh, output clone, stream and CUDA Graph
protocol.  Plan construction, search and capture are excluded from timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_digest(value) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def errors(torch, actual, reference):
    actual, reference = actual.detach().cpu(), reference.detach().cpu()
    difference = (actual - reference).abs()
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "relative_l2": float(torch.linalg.vector_norm(difference) /
                              torch.linalg.vector_norm(reference).clamp_min(1.1920929e-7)),
    }


def make_inputs(torch, size, batch, seed):
    generator = torch.Generator().manual_seed(seed)
    values = [torch.randn(batch, size, size, generator=generator, dtype=torch.float32)
              for _ in range(2)]
    return [value - value.mean(dim=(-2, -1), keepdim=True) for value in values]


def multiplier(torch, size, device):
    frequency = torch.arange(size, dtype=torch.float32, device=device)
    frequency = torch.where(frequency <= size // 2, frequency, frequency - size)
    half = torch.arange(size // 2 + 1, dtype=torch.float32, device=device)
    eigenvalues = frequency[:, None].square() + half[None, :].square()
    result = eigenvalues.clamp_min(1).reciprocal()
    result[0, 0] = 0
    return result


def solve(torch, value, forward, inverse, scale):
    return inverse(forward(value) * scale)


def measure(torch, function, values, stream, warmup, calls):
    for index in range(warmup):
        function(values[index % len(values)])
    stream.synchronize()
    wall, device = [], []
    for index in range(calls):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        begin.record(stream)
        function(values[index % len(values)])
        end.record(stream)
        end.synchronize()
        wall.append((time.perf_counter() - start) * 1000)
        device.append(begin.elapsed_time(end))
    return {
        "wall_ms": wall,
        "cuda_event_ms": device,
        "wall_median_ms": statistics.median(wall),
        "cuda_event_median_ms": statistics.median(device),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(1024, 2048), default=1024)
    parser.add_argument("--batch", type=int, choices=(4, 16), default=16)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.calls, args.warmup) < 1:
        parser.error("calls and warmup must be positive")

    import torch
    from cubutterfly.real_fft import RealFFT
    from cubutterfly.torch_graph import InferenceGraph

    torch.set_num_threads(min(8, torch.get_num_threads()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    inputs_cpu = make_inputs(torch, args.size, args.batch, args.seed)
    cpu_scale = multiplier(torch, args.size, torch.device("cpu"))
    cpu_forward = lambda value: torch.fft.rfftn(value, dim=(-2, -1), norm="backward")
    cpu_inverse = lambda value: torch.fft.irfftn(value, s=(args.size, args.size),
                                                  dim=(-2, -1), norm="backward")
    references = [solve(torch, value, cpu_forward, cpu_inverse, cpu_scale) for value in inputs_cpu]
    stream = torch.cuda.Stream(device=device)
    report = {
        "schema": "cubutterfly-periodic-poisson-torch-graph-v1",
        "application": "batched 2D spectral Poisson RHS-to-solution solve",
        "size": args.size,
        "batch": args.batch,
        "seed": args.seed,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "library_sha256": digest(args.library),
        "runner_sha256": digest(Path(__file__)),
        "gpu_name": torch.cuda.get_device_name(device),
        "timing_scope": "device input refresh + complete two-dimensional FFT solve + independent output clone",
        "setup_search_capture_excluded": True,
        "error_acceptance": "finite required; full error distributions reported without a new threshold",
        "input_sha256": [tensor_digest(value) for value in inputs_cpu],
        "quality": {},
        "timings": {},
        "setup": {},
    }
    scale = multiplier(torch, args.size, device)
    for backend in ("torch", "cubutterfly"):
        adapter = None
        with torch.cuda.stream(stream), torch.inference_mode():
            inputs = [value.to(device) for value in inputs_cpu]
            if backend == "cubutterfly":
                cache = args.output.parent / "plans.cache"
                adapter = RealFFT(str(args.library.resolve()), (args.size, args.size), args.batch,
                                  device, norm="backward", reuse_buffers=True,
                                  cache_path=str(cache))
                forward, inverse = adapter.rfftn, adapter.irfftn
            else:
                forward = lambda value: torch.fft.rfftn(value, dim=(-2, -1), norm="backward")
                inverse = lambda value: torch.fft.irfftn(value, s=(args.size, args.size),
                                                         dim=(-2, -1), norm="backward")
            function = lambda value: solve(torch, value, forward, inverse, scale)
            buffer = inputs[1].clone()

            def eager(value):
                buffer.copy_(value)
                return function(buffer).clone()

            start = time.perf_counter()
            graph = InferenceGraph(function, inputs[1], stream=stream, warmup=args.warmup)
            stream.synchronize()
            report["setup"][backend] = {
                "elapsed_ms": (time.perf_counter() - start) * 1000,
                "plans": adapter.plan_metadata() if adapter else None,
            }
            for mode, call in (("eager", eager), ("graph", graph)):
                outputs = [call(value) for value in inputs]
                stream.synchronize()
                returned = [value.detach().cpu() for value in outputs]
                report["quality"][backend + "_" + mode] = {
                    "vs_cpu_fp32": [errors(torch, value, reference)
                                    for value, reference in zip(returned, references)],
                    "finite": all(bool(torch.isfinite(value).all()) for value in returned),
                    "independent_outputs": len({value.data_ptr() for value in outputs}) == len(outputs),
                }
                report["timings"][backend + "_" + mode] = measure(
                    torch, call, inputs, stream, args.warmup, args.calls)
        if adapter:
            adapter.close()
    report["speedup"] = {
        backend + "_" + mode: report["timings"][backend + "_" + mode]["wall_median_ms"] /
        report["timings"]["cubutterfly_" + mode]["wall_median_ms"]
        for backend in ("torch",)
        for mode in ("eager", "graph")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.output), "speedup": report["speedup"]}))


if __name__ == "__main__":
    main()
