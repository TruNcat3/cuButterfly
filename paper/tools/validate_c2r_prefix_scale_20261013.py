#!/usr/bin/env python3
"""Finite public-Plan correctness matrix for inverse C2R prefix scale fusion.

Run under the allocated CUDA UUID. CPU Torch keeps FP32/FP64 reference precision;
this runner makes no performance claims and does not invoke automatic tuning.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[2]


def cases():
    rows = []
    for precision in ("fp32", "fp64"):
        for index, (shape, normalize, scale_kind) in enumerate([
            ((64, 64), True, "one"),
            ((64, 64, 32), False, "one"),
            ((64, 64), False, "reciprocal-volume"),
            ((64, 64, 32), True, "sqrt-volume"),
        ]):
            volume = math.prod(shape)
            scale = {"one": 1.0, "reciprocal-volume": 1.0 / volume,
                     "sqrt-volume": math.sqrt(volume)}[scale_kind]
            rows.append(dict(precision=precision, shape=list(shape), batch=2,
                             normalize_inverse=normalize, output_scale=scale,
                             scale_kind=scale_kind, seed=20261013 + index))
    return rows


def prefix_mapping(base, effective_scale):
    mapping = copy.deepcopy(base)
    physical = mapping["physical"]
    physical["c2r_prefix_scale"] = True
    physical["c2r_scale_placement"] = "last-complex-axis"
    cost = physical["cost"]
    prior = cost["scale_kernel_launches"]
    if prior != int(effective_scale != 1.0):
        raise AssertionError("legacy mapping scale-launch count disagrees with semantics")
    cost["scale_kernel_launches"] = 0
    cost["total_launches"] -= prior
    return mapping


def run(args, report):
    import torch
    sys.path.insert(0, str(ROOT / "python"))
    from cubutterfly import (AlgorithmPolicy, ComputeType, DataType, Error,
                            ExecutionMode, FftType, Library, Plan, PlanOptions, Transform)

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    stream = torch.cuda.Stream(device=device)
    library = Library(args.library)
    properties = torch.cuda.get_device_properties(device)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    uuid = getattr(properties, "uuid", None)
    if uuid is None and visible and visible.startswith("GPU-") and "," not in visible:
        uuid = visible
    report.update(schema="xd-c2r-prefix-scale-validation-20261013-v1", status="running",
                  library=str(args.library.resolve()),
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cuda_visible_devices=visible, gpu_uuid=str(uuid) if uuid is not None else None,
                  gpu_name=torch.cuda.get_device_name(device),
                  gpu_memory_bytes=properties.total_memory,
                  reference="CPU torch.fft in matching FP32/FP64 precision",
                  numerical_acceptance="report-only; no floating error threshold introduced",
                  cases=[], invalid_cases=[])

    with tempfile.TemporaryDirectory(prefix="cubutterfly-prefix-scale-") as cache:
        def plan_for(transform, mapping="axis-tiled-local"):
            options = PlanOptions(policy=AlgorithmPolicy.EXPLICIT,
                                  mapping=json.dumps(mapping) if isinstance(mapping, dict) else mapping,
                                  execution_mode=ExecutionMode.XD)
            return Plan(library, transform, options, stream=stream.cuda_stream,
                        cache_path=str(Path(cache) / "plans.json"))

        def check(actual, reference):
            actual = actual.detach().cpu()
            delta = actual - reference
            finite = bool(torch.isfinite(actual).all())
            max_abs = float(delta.abs().max())
            reference_max = float(reference.abs().max())
            relative_l2 = float(torch.linalg.vector_norm(delta.reshape(-1)) /
                                torch.linalg.vector_norm(reference.reshape(-1)))
            quantiles = torch.quantile(delta.abs().reshape(-1),
                                     torch.tensor([.5, .9, .99], dtype=delta.dtype))
            metrics = dict(finite=finite, max_abs=max_abs, reference_max_abs=reference_max,
                           relative_l2=relative_l2, p50_abs=float(quantiles[0]),
                           p90_abs=float(quantiles[1]), p99_abs=float(quantiles[2]))
            if not finite:
                raise AssertionError(f"nonfinite numerical output: {metrics}")
            return metrics

        for case in cases():
            shape = tuple(case["shape"])
            fp64 = case["precision"] == "fp64"
            real_dtype = torch.float64 if fp64 else torch.float32
            transform = Transform(storage=DataType.FP64 if fp64 else DataType.FP32,
                                  compute=ComputeType.FP64 if fp64 else ComputeType.FP32,
                                  extents=list(shape), batch=case["batch"], inverse=True,
                                  normalize_inverse=case["normalize_inverse"], fft_type=FftType.C2R,
                                  fft_output_scale=case["output_scale"], execution_mode=ExecutionMode.XD)
            dimensions = tuple(range(1, len(shape) + 1))
            generator = torch.Generator().manual_seed(case["seed"])
            spectra = [torch.fft.rfftn(torch.randn((case["batch"], *shape), dtype=real_dtype,
                                                  generator=generator), dim=dimensions)
                       for _ in range(2)]
            effective_scale = case["output_scale"] / (math.prod(shape) if case["normalize_inverse"] else 1)
            references = [torch.fft.irfftn(spectrum, s=shape, dim=dimensions, norm="forward") * effective_scale
                          for spectrum in spectra]
            with plan_for(transform) as seed:
                base = json.loads(seed.mapping_json())
            if base["physical"].get("c2r_prefix_scale", False):
                raise AssertionError("explicit legacy local-family seed unexpectedly enabled prefix scaling")
            mappings = [("legacy", base), ("prefix", prefix_mapping(base, effective_scale))]
            row = dict(case, variants=[])
            report["cases"].append(row)
            for label, requested in mappings:
                with plan_for(transform, requested) as plan:
                    resolved = json.loads(plan.mapping_json())
                    if resolved != requested:
                        raise AssertionError("explicit mapping replay changed the requested mapping")
                    workspace_bytes = plan.workspace_size()
                    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
                    if workspace_bytes:
                        plan.set_workspace(workspace, workspace_bytes)
                    variant = dict(label=label, mapping=resolved, eager=[])
                    row["variants"].append(variant)
                    with torch.cuda.stream(stream):
                        source = spectra[0].to(device)
                        output = torch.empty((case["batch"], *shape), device=device, dtype=real_dtype)
                    eager_outputs = []
                    for index in (0, 1):
                        with torch.cuda.stream(stream):
                            source.copy_(spectra[index])
                            plan.execute(source, output)
                        stream.synchronize()
                        actual = output.detach().cpu().clone()
                        variant["eager"].append(check(actual, references[index]))
                        eager_outputs.append(actual)
                        if not torch.equal(source.cpu(), spectra[index]):
                            raise AssertionError("eager C2R mutated its input")
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        plan.execute(source, output)
                    graph_metrics = []
                    saved = None
                    for index in (0, 1, 0):
                        with torch.cuda.stream(stream):
                            source.copy_(spectra[index])
                            graph.replay()
                        stream.synchronize()
                        actual = output.detach().cpu().clone()
                        metrics = check(actual, references[index])
                        metrics["exactly_matches_same_backend_eager"] = torch.equal(actual, eager_outputs[index])
                        graph_metrics.append(metrics)
                        variant["graph_original_changed_restored"] = graph_metrics
                        if not metrics["exactly_matches_same_backend_eager"]:
                            raise AssertionError("graph output differs from same-backend eager output")
                        if not torch.equal(source.cpu(), spectra[index]):
                            raise AssertionError("graph C2R mutated its input")
                        if saved is None:
                            saved = output.detach().cpu().clone()
                    variant["saved_result"] = check(saved, references[0])
                    variant["restored_output_exact"] = torch.equal(actual, eager_outputs[0])
                    variant["saved_result_exact"] = torch.equal(saved, eager_outputs[0])
                    if not variant["saved_result_exact"]:
                        raise AssertionError("saved graph result changed after later replays")
                    variant["input_preserved"] = True

        for kind in ("forward", "complex", "unfused"):
            transform = Transform(storage=DataType.COMPLEX_FP32 if kind == "complex" else DataType.FP32,
                                  compute=ComputeType.FP32, extents=[64, 64], batch=1,
                                  inverse=kind != "forward", normalize_inverse=False,
                                  fft_type=FftType.C2C if kind == "complex" else
                                  FftType.R2C if kind == "forward" else FftType.C2R,
                                  fft_output_scale=1.0 if kind == "complex" else .5,
                                  execution_mode=ExecutionMode.XD)
            with plan_for(transform, "xd:axis-tiled-local" if kind == "complex" else "axis-tiled-local") as seed:
                mapping = json.loads(seed.mapping_json())
            physical = mapping["physical"]
            physical["c2r_prefix_scale"] = True
            if kind == "unfused":
                physical["fused_axes"] = False
            try:
                with plan_for(transform, mapping):
                    pass
            except Error as error:
                if "scale" not in str(error).lower() and "c2r" not in str(error).lower():
                    raise AssertionError(f"invalid {kind} rejected for an unrelated reason: {error}") from error
                report["invalid_cases"].append(dict(kind=kind, rejected=True, error=str(error)))
            else:
                raise AssertionError(f"prefix scale accepted invalid {kind} transform")
    report["status"] = "pass"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--list-cases", action="store_true", help="CPU-only finite matrix inspection")
    args = parser.parse_args(argv)
    if args.list_cases:
        print(json.dumps(dict(cases=cases(), invalid_cases=["forward", "complex", "unfused"]), indent=2))
        return 0
    if not args.library or not args.output:
        parser.error("GPU validation requires --library and --output")
    if not args.output.is_absolute() and os.environ.get("XD_REPORT_DIR"):
        args.output = Path(os.environ["XD_REPORT_DIR"]) / args.output
    result = dict(schema="xd-c2r-prefix-scale-validation-20261013-v1", status="running",
                  library=str(args.library.resolve()),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    try:
        result["library_sha256"] = hashlib.sha256(args.library.read_bytes()).hexdigest()
        run(args, result)
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], cases=len(result.get("cases", [])),
                          invalid_cases=len(result.get("invalid_cases", [])),
                          error=result.get("error"), output=str(args.output)), indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
