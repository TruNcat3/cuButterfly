#!/usr/bin/env python3
"""Changing-input FNO inference: matched public auto-plan eager and CUDA Graph calls."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "experiments/tools"))
from profile_application_boundaries_20261012 import digest, errors, load_module


def variant_order(index):
    variants = ["torch_eager", "cubutterfly_eager", "torch_graph", "cubutterfly_graph"]
    offset = index % len(variants)
    return variants[offset:] + variants[:offset]


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def weight_digest(model):
    def named_callable(value):
        if callable(value) and hasattr(value, "__qualname__"):
            return dict(callable=f"{value.__module__}.{value.__qualname__}")
        raise TypeError(f"unsupported model extra state: {type(value).__name__}")

    identity = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        identity.update(name.encode())
        if hasattr(tensor, "detach"):
            identity.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            identity.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        else:
            identity.update(json.dumps(tensor, sort_keys=True, default=named_callable).encode())
    return identity.hexdigest()


def public_patch(base, torch):
    from cubutterfly.real_fft import RealFFT

    class PublicPatch(base):
        def _adapter(self, shape, batch):
            key = (shape, batch)
            if key not in self.adapters:
                begin = time.perf_counter()
                adapter = RealFFT(self.library, shape, batch, self.device,
                                  norm="forward", reuse_buffers=True)
                torch.cuda.synchronize(self.device)
                elapsed = (time.perf_counter() - begin) * 1000
                self.plan_create_ms += elapsed
                self.plan_create_events.append(dict(shape=list(shape), batch=batch,
                    elapsed_ms=elapsed, plans=adapter.plan_metadata()))
                self.adapters[key] = adapter
            return self.adapters[key]

        def rfftn(self, x, s=None, dim=None, norm=None):
            if not (x.is_cuda and x.dtype == torch.float32 and x.ndim >= 3
                    and dim in ([-2, -1], (-2, -1)) and s is None and norm == "forward"):
                raise ValueError("rfftn outside frozen FNO adapter contract")
            self.calls["rfftn"] += 1
            shape = tuple(int(v) for v in x.shape[-2:])
            batch = math.prod(int(v) for v in x.shape[:-2])
            return self._adapter(shape, batch).rfftn(x)

        def irfftn(self, x, s=None, dim=None, norm=None):
            if not (x.is_cuda and x.dtype == torch.complex64 and x.ndim >= 3
                    and dim in ([-2, -1], (-2, -1)) and s is not None
                    and len(s) == 2 and norm == "forward"
                    and tuple(int(v) for v in x.shape[-2:]) ==
                    (int(s[0]), int(s[1]) // 2 + 1)):
                raise ValueError("irfftn outside frozen FNO adapter contract")
            self.calls["irfftn"] += 1
            shape = (int(s[0]), int(s[1]))
            batch = math.prod(int(v) for v in x.shape[:-2])
            return self._adapter(shape, batch).irfftn(x)
    return PublicPatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(64, 128), default=64)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--process-trial-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input-batches", type=int, default=3)
    parser.add_argument("--calls", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.batch, args.calls, args.warmup) < 1 or args.input_batches < 2:
        parser.error("positive batch/calls/warmup and at least two input batches required")
    import torch
    from cubutterfly.torch_graph import InferenceGraph
    runner = load_module("fno_graph_frozen_runner", ROOT / "results/e13_assets/run_fno_end_to_end.py")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261002)
    report_dir = Path(os.environ.get("XD_REPORT_DIR", str(args.output.parent))).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    if not args.output.is_absolute() and "XD_REPORT_DIR" in os.environ:
        args.output = report_dir / args.output
    cache = Path(os.environ.get("CUBUTTERFLY_PLAN_CACHE", str(report_dir / "plans.cache"))).resolve()
    os.environ["CUBUTTERFLY_PLAN_CACHE"] = str(cache)
    data_root = args.data_root or ROOT / f"results/e13_assets/data/extracted{args.size}"
    dataset = runner.DarcyDataset(root_dir=data_root, n_train=1,
        n_tests=[args.batch * args.input_batches], batch_size=1,
        test_batch_sizes=[args.batch], train_resolution=args.size,
        test_resolutions=[args.size], encode_input=False, encode_output=False, download=False)
    loader = runner.DataLoader(dataset.test_dbs[args.size], batch_size=args.batch, shuffle=False)
    inputs = [item["x"].to(device=device, dtype=torch.float32) for item in loader]
    if len(inputs) < args.input_batches or any(tuple(x.shape) != tuple(inputs[0].shape) for x in inputs):
        raise RuntimeError("payload must provide multiple full fixed-shape batches")
    input_hashes = [tensor_digest(x) for x in inputs]
    if len(set(input_hashes)) != len(inputs):
        raise RuntimeError("changing-input validation requires distinct real batches")
    model = runner.FNO(n_modes=(16, 16), in_channels=1, out_channels=1,
        hidden_channels=24, n_layers=4, positional_embedding="grid",
        projection_channel_ratio=2, enforce_hermitian_symmetry=False).to(device).eval()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    treatment = copy.deepcopy(model).eval()
    before = weight_digest(model)
    patch = public_patch(runner.AdapterPatch, torch)(str(args.library.resolve()), device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    functions, graphs, eager_inputs, setup = {}, {}, {}, {}
    quality = {}
    try:
        with torch.cuda.stream(stream), torch.inference_mode():
            for backend, selected in (("torch", model), ("cubutterfly", treatment)):
                context = runner.patched_fft(patch) if backend == "cubutterfly" else nullcontext()
                start = time.perf_counter()
                with context:
                    eager_inputs[backend] = inputs[0].clone()
                    def eager(x, selected=selected, buffer=eager_inputs[backend]):
                        buffer.copy_(x)
                        return selected(buffer).clone()
                    functions[backend + "_eager"] = eager
                    for x in inputs:
                        eager(x)
                    graphs[backend] = InferenceGraph(selected, inputs[0], stream=stream, warmup=args.warmup)
                    functions[backend + "_graph"] = graphs[backend]
                stream.synchronize()
                setup[backend + "_ms"] = (time.perf_counter() - start) * 1000
            plans_before = len(patch.adapters)
            references = [functions["torch_eager"](x) for x in inputs]
            for variant, function in functions.items():
                context = runner.patched_fft(patch) if variant.startswith("cubutterfly") else nullcontext()
                with context:
                    outputs = [function(x) for x in inputs]
                    eager_function = functions[variant.split("_")[0] + "_eager"]
                    eager_outputs = [eager_function(x) for x in inputs]
                stream.synchronize()
                quality[variant] = dict(vs_torch=[errors(torch, y, ref) for y, ref in zip(outputs, references)],
                    vs_same_backend_eager=[errors(torch, y, ref) for y, ref in zip(outputs, eager_outputs)],
                    output_sha256=[tensor_digest(y) for y in outputs])
                if not all(item["finite"] for key in ("vs_torch", "vs_same_backend_eager") for item in quality[variant][key]):
                    raise RuntimeError("non-finite FNO output")
            order = variant_order(args.process_trial_index)
            timings = {}
            for variant in order:
                context = runner.patched_fft(patch) if variant.startswith("cubutterfly") else nullcontext()
                with context:
                    function = functions[variant]
                    for i in range(args.warmup):
                        function(inputs[i % len(inputs)])
                    stream.synchronize()
                    wall, gpu = [], []
                    for i in range(args.calls):
                        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                        start = time.perf_counter()
                        begin.record(stream)
                        output = function(inputs[i % len(inputs)])
                        end.record(stream)
                        end.synchronize()
                        wall.append((time.perf_counter() - start) * 1000)
                        gpu.append(begin.elapsed_time(end))
                    timings[variant] = dict(wall_ms=wall, cuda_event_ms=gpu,
                        wall_median_ms=statistics.median(wall), cuda_event_median_ms=statistics.median(gpu),
                        batch_throughput_per_second=args.batch * 1000 / statistics.median(wall),
                        per_sample_wall_ms=statistics.median(wall) / args.batch)
            if len(patch.adapters) != plans_before:
                raise RuntimeError("plans were created inside measured calls")
        after = weight_digest(model)
        if before != after or before != weight_digest(treatment):
            raise RuntimeError("model weights changed")
        report = dict(test="fno_reusable_graph_integration", size=args.size, batch=args.batch,
            process_trial_index=args.process_trial_index, variant_order=order,
            library_sha256=digest(args.library), runner_sha256=digest(__file__),
            public_real_fft_sha256=digest(ROOT / "python/cubutterfly/real_fft.py"),
            graph_wrapper_sha256=digest(ROOT / "python/cubutterfly/torch_graph.py"),
            frozen_runner_sha256=digest(ROOT / "results/e13_assets/run_fno_end_to_end.py"),
            checkpoint_sha256=digest(args.checkpoint) if args.checkpoint else None,
            data_sha256=digest(data_root / f"darcy_test_{args.size}.pt"),
            model_weights_sha256=before, model_weights_unchanged=True,
            weights="checkpoint" if args.checkpoint else "seeded random; not model quality evidence",
            input_sha256=input_hashes, changing_input_count=len(inputs), device=str(device),
            gpu_uuid=str(torch.cuda.get_device_properties(device).uuid),
            gpu_name=torch.cuda.get_device_name(device), torch=torch.__version__, cuda=torch.version.cuda,
            gpu_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
            environment={k: v for k, v in os.environ.items() if k.startswith(("CUBUTTERFLY_", "CUDA_", "XD_"))},
            graph_contract=graphs["torch"].contract(), timings=timings, quality=quality,
            error_acceptance="finite required; distributions reported without numeric acceptance threshold",
            setup=dict(setup, plan_create_ms=patch.plan_create_ms, plans=patch.plan_create_events,
                       compilation_search_capture_excluded=True),
            plan_count_before_timing=plans_before, adapter_calls=patch.calls,
            cache=dict(path=str(cache), sha256=digest(cache) if cache.is_file() else None),
            timing_scope="device-resident input copy + complete forward/replay + independent output clone + synchronization; setup excluded",
            speedup={mode: timings["torch_" + mode]["wall_median_ms"] /
                          timings["cubutterfly_" + mode]["wall_median_ms"] for mode in ("eager", "graph")},
            pass_finite_and_weights=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(dict(report=str(args.output), speedup=report["speedup"])))
    finally:
        stream.synchronize()
        patch.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
