#!/usr/bin/env python3
"""Finite public axis-vkfft matrix with SciPy references and exact mapping replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[2]


def cases():
    rows = []
    modes = [("c2c", False, False, 1.0), ("c2c", True, True, 1.0),
             ("c2c", True, False, 1.0), ("r2c", False, False, .5),
             ("c2r", True, True, .5), ("c2r", True, False, 1.0)]
    for precision in ("fp32", "fp64"):
        for shape in ((128, 32), (64, 16, 32), (2048, 32)):
            for kind, inverse, normalize, scale in modes:
                rows.append(dict(precision=precision, shape=list(shape), batch=2,
                    kind=kind, inverse=inverse, normalize_inverse=normalize,
                    output_scale=scale, seed=20261013 + len(rows)))
        rows.append(dict(precision=precision, shape=[128, 32], batch=2, kind="r2c",
            inverse=False, normalize_inverse=False, output_scale=1.0, seed=20261013 + len(rows)))
    return rows


def reference_inputs(case):
    import numpy as np
    from scipy import fft

    real = np.float64 if case["precision"] == "fp64" else np.float32
    complex_type = np.complex128 if case["precision"] == "fp64" else np.complex64
    shape = tuple(case["shape"])
    dimensions = tuple(range(1, len(shape) + 1))
    generator = np.random.RandomState(case["seed"])
    inputs, references = [], []
    for _ in range(2):
        source = generator.standard_normal((case["batch"], *shape)).astype(real)
        if case["kind"] == "c2c":
            source = (source + 1j * generator.standard_normal(source.shape).astype(real)).astype(complex_type)
            if case["inverse"]:
                reference = fft.ifftn(source, axes=dimensions,
                    norm="backward" if case["normalize_inverse"] else "forward")
            else:
                reference = fft.fftn(source, axes=dimensions)
        elif case["kind"] == "r2c":
            reference = fft.rfftn(source, axes=dimensions)
        else:
            source = fft.rfftn(source, axes=dimensions)
            reference = fft.irfftn(source, s=shape, axes=dimensions,
                norm="backward" if case["normalize_inverse"] else "forward")
        reference = reference * real(case["output_scale"])
        if source.dtype != (complex_type if case["kind"] != "r2c" else real):
            raise AssertionError("SciPy reference input changed precision")
        if reference.dtype != (real if case["kind"] == "c2r" else complex_type):
            raise AssertionError("SciPy reference output changed precision")
        inputs.append(np.ascontiguousarray(source))
        references.append(np.ascontiguousarray(reference))
    return inputs, references


def check(actual, reference):
    import numpy as np
    actual = actual.detach().cpu().numpy()
    absolute = np.abs(actual.astype(np.complex128) - reference.astype(np.complex128)).reshape(-1)
    finite = bool(np.isfinite(actual).all())
    denominator = float(np.linalg.norm(reference.reshape(-1).astype(np.complex128)))
    quantiles = np.quantile(absolute, [.5, .9, .99])
    metrics = dict(finite=finite, max_abs=float(absolute.max()),
        reference_max_abs=float(np.abs(reference).max()),
        relative_l2=float(np.linalg.norm(absolute) / denominator) if denominator else 0.0,
        p50_abs=float(quantiles[0]), p90_abs=float(quantiles[1]), p99_abs=float(quantiles[2]),
        quantile_element_count=int(absolute.size))
    if not finite:
        raise AssertionError("nonfinite output: " + str(metrics))
    return metrics


def run(args, report):
    import torch
    import scipy
    sys.path.insert(0, str(ROOT / "python"))
    from cubutterfly import (AlgorithmPolicy, ComputeType, DataType, Error,
                            ExecutionMode, FftType, Library, Plan, PlanOptions, Transform)

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    stream = torch.cuda.Stream(device=device)
    library = Library(args.library)
    properties = torch.cuda.get_device_properties(device)
    report.update(status="running", gpu_name=properties.name,
        gpu_uuid=str(getattr(properties, "uuid", os.environ.get("CUDA_VISIBLE_DEVICES"))),
        reference="CPU scipy.fft matching FP32/FP64 precision", scipy_version=scipy.__version__,
        numerical_acceptance="report-only; finite required; no error threshold",
        cases=[], invalid_cases=[])

    with tempfile.TemporaryDirectory(prefix="cubutterfly-vkfft-public-") as cache:
        def plan_for(transform, mapping):
            return Plan(library, transform,
                PlanOptions(policy=AlgorithmPolicy.EXPLICIT, mapping=mapping,
                            execution_mode=ExecutionMode.XD),
                stream=stream.cuda_stream, cache_path=str(Path(cache) / "plans.json"))

        def transform_for(case):
            fp64 = case["precision"] == "fp64"
            storage = (DataType.COMPLEX_FP64 if fp64 else DataType.COMPLEX_FP32) if case["kind"] == "c2c" else (
                DataType.FP64 if fp64 else DataType.FP32)
            return Transform(storage=storage, compute=ComputeType.FP64 if fp64 else ComputeType.FP32,
                extents=case["shape"], batch=case["batch"], inverse=case["inverse"],
                normalize_inverse=case["normalize_inverse"], fft_type=getattr(FftType, case["kind"].upper()),
                fft_output_scale=case["output_scale"], execution_mode=ExecutionMode.XD)

        for case in cases():
            inputs, references = reference_inputs(case)
            transform = transform_for(case)
            row = dict(case, variants=[])
            report["cases"].append(row)
            mapping_json = None
            original_outputs = None
            for label in ("original", "exact-json-replay"):
                with plan_for(transform, "xd:axis-vkfft" if mapping_json is None else mapping_json) as plan:
                    resolved_json = plan.mapping_json()
                    if mapping_json is None:
                        mapping_json = resolved_json
                    elif resolved_json != mapping_json:
                        raise AssertionError("exact public mapping JSON replay changed serialization")
                    variant = dict(label=label, mapping=json.loads(resolved_json), eager=[], graph=[])
                    row["variants"].append(variant)
                    workspace_bytes = plan.workspace_size()
                    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
                    if workspace_bytes:
                        plan.set_workspace(workspace, workspace_bytes)
                    variant["workspace_bytes"] = workspace_bytes
                    with torch.cuda.stream(stream):
                        source = torch.from_numpy(inputs[0]).to(device)
                        output = torch.empty(references[0].shape, dtype=torch.from_numpy(references[0]).dtype,
                                             device=device)
                    eager_outputs = []
                    for index in (0, 1):
                        with torch.cuda.stream(stream):
                            source.copy_(torch.from_numpy(inputs[index]))
                            plan.execute(source, output)
                        stream.synchronize()
                        actual = output.detach().cpu().clone()
                        variant["eager"].append(check(actual, references[index]))
                        eager_outputs.append(actual)
                        if not torch.equal(source.cpu(), torch.from_numpy(inputs[index])):
                            raise AssertionError("eager execution mutated input")
                        if original_outputs is not None and not torch.equal(actual, original_outputs[index]):
                            raise AssertionError("mapping replay differs from original eager output")
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        plan.execute(source, output)
                    saved = None
                    for index in (0, 1, 0):
                        with torch.cuda.stream(stream):
                            source.copy_(torch.from_numpy(inputs[index]))
                            graph.replay()
                        stream.synchronize()
                        actual = output.detach().cpu().clone()
                        metrics = check(actual, references[index])
                        metrics["input_index"] = index
                        metrics["exactly_matches_eager"] = torch.equal(actual, eager_outputs[index])
                        variant["graph"].append(metrics)
                        if not metrics["exactly_matches_eager"]:
                            raise AssertionError("graph replay differs from eager output")
                        if not torch.equal(source.cpu(), torch.from_numpy(inputs[index])):
                            raise AssertionError("graph replay mutated input")
                        if saved is None:
                            saved = actual.clone()
                    if not torch.equal(saved, eager_outputs[0]):
                        raise AssertionError("saved graph result changed")
                    variant.update(input_preserved=True, saved_result_exact=True, restored_output_exact=True)
                    if original_outputs is None:
                        original_outputs = eager_outputs
            row["exact_json_replay"] = True

        for kind in ("c2c", "r2c", "c2r"):
            invalid = dict(precision="fp32", shape=[96, 32], batch=1, kind=kind,
                          inverse=kind == "c2r", normalize_inverse=False, output_scale=1.0)
            try:
                with plan_for(transform_for(invalid), "xd:axis-vkfft"):
                    pass
            except Error as error:
                report["invalid_cases"].append(dict(kind=kind, shape=invalid["shape"],
                    reason="non-power-of-two axis", rejected=True, error=str(error)))
            else:
                raise AssertionError("axis-vkfft accepted non-power-of-two " + kind)
    report["status"] = "pass"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--cpu-check", action="store_true", help="generate all SciPy inputs/references without CUDA")
    args = parser.parse_args()
    if args.list_cases or args.cpu_check:
        summary = dict(cases=cases(), invalid_cases=[dict(kind=kind, shape=[96, 32])
                                                   for kind in ("c2c", "r2c", "c2r")])
        if args.cpu_check:
            for case in cases():
                reference_inputs(case)
            summary["cpu_reference_checks"] = len(cases())
        print(json.dumps(summary, indent=2))
        return 0
    if not args.library or not args.output:
        parser.error("GPU validation requires --library and --output")
    if not args.output.is_absolute() and os.environ.get("XD_REPORT_DIR"):
        args.output = Path(os.environ["XD_REPORT_DIR"]) / args.output
    report = dict(schema="vkfft-public-validation-20261013-v1", status="running",
        library=str(args.library.resolve()), runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    try:
        report["library_sha256"] = hashlib.sha256(args.library.read_bytes()).hexdigest()
        run(args, report)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], cases=len(report.get("cases", [])),
        invalid_cases=len(report.get("invalid_cases", [])), error=report.get("error"), output=str(args.output)), indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
