#!/usr/bin/env python3
"""Finite full-lowering correctness checks for explicit long-axis CTA groups.

The C ABI bridge calls XdRealFftPlan directly and reports its canonical mapping.
This is component correctness evidence, not public automatic-selection evidence.
CPU Torch references retain the requested FP32/FP64 precision. No timing claims
or floating error thresholds are introduced.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[2]
CPP = Path(__file__).with_suffix(".cpp")


def cases():
    rows = []
    for (shape, batch), precision, kind, ept, group, exchange in itertools.product(
        [([2048, 32], 3), ([4096, 16], 2)], ["fp32", "fp64"],
        ["r2c", "c2r", "c2c-forward", "c2c-inverse"], [8, 16], [2, 4], [0, 2, 3]
    ):
        rows.append(dict(shape=shape, batch=batch, precision=precision, kind=kind,
                         ept=ept, group=group, exchange=exchange,
                         normalize_inverse=kind in ("c2r", "c2c-inverse"),
                         output_scale=1.0, seed=20261013))
    rows.append(dict(shape=[2048, 8, 16], batch=2, precision="fp32", kind="c2r",
                     ept=16, group=4, exchange=3, normalize_inverse=True,
                     output_scale=0.5, seed=20261014))
    for kind in ("r2c", "c2r", "c2c-forward", "c2c-inverse"):
        for ept, group, exchange in ((16, 2, 2), (8, 4, 0)):
            rows.append(dict(shape=[2048, 2048], batch=1, precision="fp32", kind=kind,
                             ept=ept, group=group, exchange=exchange,
                             normalize_inverse=kind in ("c2r", "c2c-inverse"),
                             output_scale=1.0, seed=20261015))
    return rows


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_bridge(args, directory):
    output = directory / "xd_grouped_validation_bridge.so"
    argv = [args.cxx, "-std=c++17", "-O2", "-fPIC", "-shared",
            "-I" + str(ROOT / "src"), "-I" + str(args.cuda_include),
            str(CPP), str(args.library.resolve()),
            "-L" + str(args.cuda_include.parent / "lib64" / "stubs"), "-lcuda",
            "-Wl,-rpath," + str(args.library.resolve().parent), "-o", str(output)]
    subprocess.run(argv, check=True, capture_output=True, text=True)
    return output, argv


class Bridge:
    def __init__(self, path):
        self.dll = ct.CDLL(str(path))
        signatures = {
            "create": ([ct.POINTER(ct.c_size_t), ct.c_uint, ct.c_size_t,
                        ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_double,
                        ct.c_uint, ct.c_uint, ct.c_uint], ct.c_void_p),
            "error": ([], ct.c_char_p),
            "mapping": ([ct.c_void_p], ct.c_char_p),
            "workspace_size": ([ct.c_void_p], ct.c_size_t),
            "set_workspace": ([ct.c_void_p, ct.c_void_p, ct.c_size_t], ct.c_int),
            "execute": ([ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p], ct.c_int),
            "capture_resources": ([ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p], ct.c_char_p),
            "destroy": ([ct.c_void_p], None),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.dll, "xd_grouped_" + name)
            function.argtypes, function.restype = arguments, result
            setattr(self, name, function)

    def checked(self, function, *arguments):
        if function(*arguments):
            raise RuntimeError(self.error().decode())


def error_metrics(torch, actual, reference):
    delta = (actual - reference).abs()
    denominator = torch.linalg.vector_norm(reference.reshape(-1))
    relative = torch.linalg.vector_norm((actual - reference).reshape(-1)) / denominator
    quantiles = torch.quantile(delta.reshape(-1), torch.tensor([.5, .9, .99], dtype=delta.dtype))
    metrics = dict(finite=bool(torch.isfinite(actual).all()), max_abs=float(delta.max()),
                   relative_l2=float(relative), reference_max_abs=float(reference.abs().max()),
                   p50_abs=float(quantiles[0]), p90_abs=float(quantiles[1]), p99_abs=float(quantiles[2]))
    if not metrics["finite"]:
        raise AssertionError(f"nonfinite result: {metrics}")
    return metrics


def inputs_and_references(torch, case):
    shape = tuple(case["shape"])
    dimensions = tuple(range(1, len(shape) + 1))
    dtype = torch.float64 if case["precision"] == "fp64" else torch.float32
    generator = torch.Generator().manual_seed(case["seed"])
    inputs, references = [], []
    for _ in range(2):
        real = torch.randn((case["batch"], *shape), dtype=dtype, generator=generator)
        if case["kind"] == "r2c":
            source = real
            reference = torch.fft.rfftn(source, dim=dimensions)
        elif case["kind"] == "c2r":
            source = torch.fft.rfftn(real, dim=dimensions)
            reference = torch.fft.irfftn(source, s=shape, dim=dimensions,
                                        norm="backward" if case["normalize_inverse"] else "forward")
        else:
            imag = torch.randn(real.shape, dtype=dtype, generator=generator)
            source = torch.complex(real, imag)
            if case["kind"] == "c2c-inverse":
                # CPU Torch on this host applies an extra 1/N in the large
                # C2C backward-normalized path. Compare matching-precision
                # unnormalized IFFT + explicit division against an independent
                # CPU pocketfft reference; preserve both error distributions.
                reference = torch.fft.ifftn(source, dim=dimensions, norm="forward")
                if case["normalize_inverse"]:
                    reference = reference / math.prod(shape)
            else:
                reference = torch.fft.fftn(source, dim=dimensions)
        inputs.append(source)
        references.append(reference * case["output_scale"])
    return inputs, references


def validate_case(torch, bridge, case, stream, device, probe_resources=False):
    shape = (ct.c_size_t * len(case["shape"]))(*case["shape"])
    handle = bridge.create(shape, len(shape), case["batch"],
                           int(case["kind"] in ("c2r", "c2c-inverse")),
                           int(case["normalize_inverse"]), int(case["precision"] == "fp64"),
                           int(case["kind"].startswith("c2c")), case["output_scale"],
                           case["ept"], case["group"], case["exchange"])
    if not handle:
        reason = bridge.error().decode()
        if reason == "unsupported grouped axis descriptor or device resources":
            return dict(case, status="unsupported", reason=reason)
        raise RuntimeError(f"plan construction failed: {reason}")
    try:
        mapping = json.loads(bridge.mapping(handle))
        long_axis = next(axis for axis in mapping["axes"] if axis["axis"] == 0)
        resource = mapping["axis_resources"][0]
        if (long_axis["core"] != "cufftdx-axis-tiled" or
                long_axis["ffts_per_block"] != case["group"] or
                long_axis["elements_per_thread"] != case["ept"] or
                resource.get("ffts_per_block") != case["group"] or
                resource["exchange"] != case["exchange"]):
            raise AssertionError(f"requested grouped resource did not reach the long axis: {mapping}")
        row = dict(case, status="running", mapping=mapping,
                   long_axis_suffix=long_axis["axis_suffix"],
                   long_axis_tail_lines=(case["batch"] * math.prod(
                       case["shape"][1:-1] + [case["shape"][-1] // 2 + 1]
                       if not case["kind"].startswith("c2c") else case["shape"][1:])) % case["group"],
                   eager=[], graph_original_changed_restored=[])
        sources, references = inputs_and_references(torch, case)
        if case["kind"] == "c2c-inverse":
            import scipy.fft
            dimensions = tuple(range(1, len(case["shape"]) + 1))
            row["cpu_reference_crosscheck"] = []
            for source, reference in zip(sources, references):
                independent = torch.from_numpy(scipy.fft.ifftn(
                    source.numpy(), axes=dimensions,
                    norm="backward" if case["normalize_inverse"] else "forward"))
                independent = independent * case["output_scale"]
                row["cpu_reference_crosscheck"].append(error_metrics(torch, reference, independent))
            row["reference_normalization"] = "CPU torch unnormalized IFFT followed by same-precision 1/N; crosschecked CPU scipy.fft"
        workspace_bytes = bridge.workspace_size(handle)
        with torch.cuda.stream(stream):
            workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
            source = sources[0].to(device)
            output = torch.empty_like(references[0], device=device)
        if workspace_bytes:
            bridge.checked(bridge.set_workspace, handle, workspace.data_ptr(), workspace_bytes)

        def execute():
            bridge.checked(bridge.execute, handle, source.data_ptr(), output.data_ptr(), stream.cuda_stream)

        eager_outputs = []
        for index in (0, 1):
            with torch.cuda.stream(stream):
                source.copy_(sources[index])
                execute()
            stream.synchronize()
            actual = output.cpu().clone()
            eager_outputs.append(actual)
            row["eager"].append(error_metrics(torch, actual, references[index]))
            if not torch.equal(source.cpu(), sources[index]):
                raise AssertionError("eager execution mutated its input")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            execute()
        saved = None
        for index in (0, 1, 0):
            with torch.cuda.stream(stream):
                source.copy_(sources[index])
                graph.replay()
            stream.synchronize()
            actual = output.cpu().clone()
            metrics = error_metrics(torch, actual, references[index])
            metrics["exactly_matches_same_backend_eager"] = torch.equal(actual, eager_outputs[index])
            row["graph_original_changed_restored"].append(metrics)
            if not metrics["exactly_matches_same_backend_eager"]:
                raise AssertionError("Graph output differs from same-backend eager")
            if not torch.equal(source.cpu(), sources[index]):
                raise AssertionError("Graph execution mutated its input")
            if saved is None:
                saved = actual.clone()
        row.update(status="pass", input_preserved=True, restored_output_exact=True,
                   saved_result_exact=torch.equal(saved, eager_outputs[0]), workspace_bytes=workspace_bytes)
        if not row["saved_result_exact"]:
            raise AssertionError("saved output changed after later Graph replays")
        resources = (bridge.capture_resources(handle, source.data_ptr(), output.data_ptr(), stream.cuda_stream)
                     if probe_resources else None)
        if resources is None:
            row["captured_kernel_resources"] = {"status": "unavailable",
                                                 "reason": bridge.error().decode() if probe_resources else "resource probe disabled"}
        else:
            row["captured_kernel_resources"] = json.loads(resources)
        return row
    finally:
        bridge.destroy(handle)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--cuda-include", type=Path, default=Path("/usr/local/cuda/include"))
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--probe-resources", action="store_true")
    args = parser.parse_args(argv)
    if args.list_cases:
        print(json.dumps(dict(cases=cases(), attempts=len(cases())), indent=2))
        return 0
    if not args.library or not args.output:
        parser.error("--library and --output are required")
    if not args.output.is_absolute() and os.environ.get("XD_REPORT_DIR"):
        args.output = Path(os.environ["XD_REPORT_DIR"]) / args.output
    report = dict(schema="xd-grouped-axis-validation-20261013-v1", status="running",
                  scope="full XdRealFftPlan explicit resources; component correctness, no automatic-library or timing claim",
                  library=str(args.library.resolve()), runner_sha256=sha256(__file__),
                  bridge_source_sha256=sha256(CPP), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  reference="CPU torch.fft at matching FP32/FP64 precision",
                  numerical_acceptance="finite required; error distributions only, no floating threshold",
                  cases=[])
    try:
        report["library_sha256"] = sha256(args.library)
        with tempfile.TemporaryDirectory(prefix="cubutterfly-grouped-axis-") as directory:
            path, command = compile_bridge(args, Path(directory))
            report.update(bridge_compile_argv=command, bridge_sha256=sha256(path))
            if args.compile_only:
                report["status"] = "compiled"
            else:
                import torch
                torch.cuda.set_device(args.device)
                device = torch.device("cuda", args.device)
                properties = torch.cuda.get_device_properties(device)
                report.update(gpu_name=properties.name, gpu_uuid=str(getattr(properties, "uuid", "unknown")),
                              gpu_memory_bytes=properties.total_memory, torch_version=torch.__version__)
                stream = torch.cuda.Stream(device=device)
                bridge = Bridge(path)
                for case in cases():
                    report["current_case"] = case
                    try:
                        report["cases"].append(validate_case(torch, bridge, case, stream, device, args.probe_resources))
                    except Exception as error:
                        report["cases"].append(dict(case, status="failed",
                                                     error_type=type(error).__name__,
                                                     error=str(error)))
                report.pop("current_case", None)
                report["status"] = "pass" if not any(row["status"] == "failed" for row in report["cases"]) else "failed"
    except Exception as error:
        report.update(status="failed", error=str(error), error_type=type(error).__name__,
                      traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], attempted=len(report["cases"]),
                          passed=sum(row["status"] == "pass" for row in report["cases"]),
                          unsupported=sum(row["status"] == "unsupported" for row in report["cases"]),
                          error=report.get("error"), output=str(args.output)), indent=2))
    return 0 if report["status"] in ("pass", "compiled") else 1


if __name__ == "__main__":
    raise SystemExit(main())
