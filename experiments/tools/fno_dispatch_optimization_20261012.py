#!/usr/bin/env python3
"""Controlled FNO adapter simplification and equal CUDA-graph controls.

The fixed-shape treatment removes duplicated flattening/validation from the
frozen application adapter. It retains boundary validation, stream tracking,
and the identical public FFT plans. Graph replay is a separate application
execution mode applied to both libraries, never mixed into eager speedups.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/tools"))
from profile_application_boundaries_20261012 import (
    LIB, TARGETS, digest, effective_environment, errors, load_module, save, timing,
)


def simplified_patch(base, torch):
    class SingleBoundaryPatch(base):
        """Same public contract; each tensor validated once by _execute."""

        def _call(self, x, shape, inverse):
            leading = tuple(x.shape[:-2])
            tail = (shape[0], shape[1] // 2 + 1) if inverse else shape
            flat = x.contiguous().reshape(-1, *tail)
            adapter = self._adapter(shape, flat.shape[0])
            output_tail = shape if inverse else adapter._half_shape
            attribute = "_inverse_output" if inverse else "_forward_output"
            output = getattr(adapter, attribute)
            if output is None:
                output = torch.empty((adapter.batch, *output_tail), device=adapter.device,
                                     dtype=torch.float32 if inverse else torch.complex64)
                setattr(adapter, attribute, output)
            plan = adapter._inverse if inverse else adapter._forward
            result = adapter._execute(plan, flat, output)
            return result.reshape(*leading, *output_tail)

        def rfftn(self, x, s=None, dim=None, norm=None):
            if not (x.is_cuda and x.dtype == torch.float32 and x.ndim >= 3
                    and dim in ([-2, -1], (-2, -1)) and s is None and norm == "forward"):
                raise ValueError("rfftn outside frozen FNO adapter contract")
            self.calls["rfftn"] += 1
            return self._call(x, tuple(x.shape[-2:]), False)

        def irfftn(self, x, s=None, dim=None, norm=None):
            if not (x.is_cuda and x.dtype == torch.complex64 and x.ndim >= 3
                    and dim in ([-2, -1], (-2, -1)) and s is not None
                    and len(s) == 2 and norm == "forward"
                    and tuple(x.shape[-2:]) == (int(s[0]), int(s[1]) // 2 + 1)):
                raise ValueError("irfftn outside frozen FNO adapter contract")
            self.calls["irfftn"] += 1
            return self._call(x, tuple(int(v) for v in s), True)

    return SingleBoundaryPatch


def graph_control(torch, function):
    reference = function().clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            function()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = function()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    quality = errors(torch, output, reference)
    if not quality["finite"]:
        raise RuntimeError("non-finite graph output")
    measured = timing(torch, graph.replay, warmup=5, repeats=20, samples=9)
    measured["mode"] = "whole-model CUDA graph; both backends; fixed device input"
    return dict(timing=measured, error_vs_eager=quality)


def child(output, resolution, variant):
    import torch
    import numpy as np
    runner_path = ROOT / "results/e13_assets/run_fno_end_to_end.py"
    runner = load_module("fno_dispatch_frozen_runner", runner_path)
    if variant == "single_boundary":
        runner.AdapterPatch = simplified_patch(runner.AdapterPatch, torch)
    original = runner.synchronized_forward
    seen = set()
    graphs = {}

    def measured(model, x, device):
        label = ("cubutterfly" if isinstance(getattr(torch.fft.rfftn, "__self__", None),
                                            runner.AdapterPatch) else "torch")
        if label not in seen:
            seen.add(label)
            graphs[label] = graph_control(torch, lambda: model(x))
            np.save(output / f"{label}_output.npy", model(x).detach().cpu().numpy())
            save(output / "graph_control.json", graphs)
        return original(model, x, device)

    runner.synchronized_forward = measured
    os.environ["CUBUTTERFLY_PLAN_CACHE"] = str(output / "plans.cache")
    os.environ["CUBUTTERFLY_E13_ALGORITHM_POLICY"] = "default"
    sys.argv = [str(runner_path), "--data-root", str(ROOT / f"results/e13_assets/data/extracted{resolution}"),
                "--resolution", str(resolution), "--n-test", "4", "--batch-size", "4",
                "--trials", "1", "--calls", "30", "--warmup", "10", "--library", str(LIB),
                "--output", str(output / "eager.json")]
    result = runner.main()
    if result or seen != {"torch", "cubutterfly"}:
        raise RuntimeError("incomplete FNO optimization measurement")
    graphs["torch_over_cubutterfly"] = (graphs["torch"]["timing"]["median_us"] /
                                       graphs["cubutterfly"]["timing"]["median_us"])
    save(output / "graph_control.json", graphs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("stock", "single_boundary"), required=True)
    parser.add_argument("--resolution", choices=(64, 128), type=int, default=64)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--cache-seed", type=Path)
    parser.add_argument("--until", default="2026-10-12T18:00:00+08:00")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    uuid, calibration_path = TARGETS["fft_filter"]
    calibration = json.loads(calibration_path.read_text())
    if digest(LIB) != calibration["measurement_identity"]["library_sha256"]:
        raise RuntimeError("library/calibration mismatch")
    env, identity = effective_environment(calibration, uuid, 512 * 1024**2)
    if not args.child:
        command = [sys.executable, str(ROOT / "scripts/run_isolated_gpu.py"),
                   "--gpu-uuid", uuid, "--report-dir", str(args.report_dir.resolve()),
                   "--until", args.until, "--", sys.executable, str(Path(__file__).resolve()),
                   "--variant", args.variant, "--resolution", str(args.resolution),
                   "--report-dir", str(args.report_dir.resolve()), "--child"]
        if args.cache_seed:
            command += ["--cache-seed", str(args.cache_seed.resolve())]
        print(json.dumps(dict(command=command, environment=identity), indent=2), flush=True)
        return subprocess.call(command, env=env, cwd=ROOT) if args.run else 0
    output = Path(os.environ["XD_REPORT_DIR"])
    seed_hash = digest(args.cache_seed) if args.cache_seed else None
    if args.cache_seed:
        shutil.copyfile(args.cache_seed, output / "plans.cache")
    save(output / "identity.json", dict(library_sha256=digest(LIB), runner_sha256=digest(__file__),
        frozen_runner_sha256=digest(ROOT / "results/e13_assets/run_fno_end_to_end.py"),
        frozen_adapter_sha256=digest(ROOT / "results/e13_assets/cubutterfly_real_fft.py"),
        cache_seed_sha256=seed_hash,
        variant=args.variant, resolution=args.resolution, gpu_uuid=uuid,
        calibration_sha256=digest(calibration_path), environment=identity,
        methodology="eager and graph separate; identical models/data; setup excluded; full output saved",
        mechanism_claim=False, diagnostic_process_trials=1))
    child(output, args.resolution, args.variant)
    final_hash = digest(output / "plans.cache")
    save(output / "cache_receipt.json", dict(seed_sha256=seed_hash, final_sha256=final_hash))
    if seed_hash and seed_hash != final_hash:
        raise RuntimeError("cached plan changed; comparison needs a fresh declared cohort")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
