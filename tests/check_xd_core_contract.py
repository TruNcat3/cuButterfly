#!/usr/bin/env python3
"""Bounded public replay checks for the xD core resource contract.

GPU execution is intentionally opt-in. ``--compile-only`` prepares the bridge and
CPU references without selecting a device; the GPU path is limited to the cases
returned by ``cases()`` and reports floating-point errors without thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import traceback
import ctypes as ct
import sys

ROOT = Path(__file__).resolve().parents[1]
CPP = Path(__file__).with_name("xd_core_contract_bridge.cpp")
sys.path.insert(0, str(ROOT / "experiments" / "tools"))
from validate_vkfft_public_20261013 import check, reference_inputs


def cases():
    rows = []
    for precision in ("fp32", "fp64"):
        for inverse in (False, True):
            for twiddle in (0, 1):
                rows.append(dict(name=f"local-{precision}-{'inv' if inverse else 'fwd'}-tw{twiddle}",
                                 precision=precision, shape=[64, 64], batch=1,
                                 inverse=inverse, normalize=inverse, complex_only=True,
                                 core="local", cap=32, twiddle=twiddle, exchange=0))
                rows.append(dict(name=f"compactreal-{precision}-{'inv' if inverse else 'fwd'}-tw{twiddle}",
                                 precision=precision, shape=[64, 64], batch=1,
                                 inverse=inverse, normalize=inverse, complex_only=False,
                                 core="local", cap=32, twiddle=twiddle, exchange=0))
    for precision in ("fp32", "fp64"):
        rows.append(dict(name=f"block-exchange4-{precision}", precision=precision,
                         shape=[64, 64], batch=1, inverse=False, normalize=False,
                         complex_only=True, core="block", cap=0, twiddle=0, exchange=4))
    return rows


def invalid_cases():
    return [
        {"name": "axis128-cap32", "shape": [128, 64], "core": "local", "cap": 32},
        {"name": "negative-resource", "axis_resources": [{"core": "thread-shared", "ept": -1}]},
        {"name": "oversized-resource", "axis_resources": [{"core": "thread-shared", "ept": 4294967304}]},
        {"name": "float-resource", "axis_resources": [{"core": "thread-shared", "ept": 8.0}]},
        {"name": "vkfft-ept", "axis_resources": [{"core": "vkfft", "ept": 16, "register_lanes": 4}]},
        {"name": "vkfft-lanes", "axis_resources": [{"core": "vkfft", "ept": 8, "register_lanes": 8}]},
    ]


def cpu_references():
    return [reference_inputs(dict(case,
        kind="c2c" if case["complex_only"] else "c2r" if case["inverse"] else "r2c",
        seed=20261013 + index, normalize_inverse=case["normalize"], output_scale=1.0))
        for index, case in enumerate(cases())]


def compile_bridge(args, directory):
    output = directory / "xd_core_contract_bridge.so"
    command = [args.cxx, "-std=c++17", "-O2", "-fPIC", "-shared",
               "-I" + str(ROOT / "src"), "-I" + str(args.cuda_include), str(CPP),
               str(args.library.resolve()), "-Wl,-rpath," + str(args.library.resolve().parent),
               "-o", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output, command


class Bridge:
    def __init__(self, path):
        self.dll = ct.CDLL(str(path))
        self.create = self.dll.xd_contract_create
        self.create.argtypes = [ct.POINTER(ct.c_size_t), ct.c_uint, ct.c_size_t,
                                ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_double,
                                ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint]
        self.create.restype = ct.c_void_p
        self.mapping = self.dll.xd_contract_mapping
        self.mapping.argtypes, self.mapping.restype = [ct.c_void_p], ct.c_char_p
        self.error = self.dll.xd_contract_error
        self.error.restype = ct.c_char_p
        self.destroy = self.dll.xd_contract_destroy
        self.destroy.argtypes = [ct.c_void_p]

    def canonical(self, case):
        shape = (ct.c_size_t * len(case["shape"]))(*case["shape"])
        handle = self.create(shape, len(case["shape"]), case["batch"], int(case["inverse"]),
                             int(case["normalize"]), int(case["precision"] == "fp64"),
                             int(case["complex_only"]), 1.0, int(case["core"] == "block"),
                             case["cap"], case["twiddle"], case["exchange"])
        if not handle:
            raise RuntimeError(self.error().decode())
        try:
            return self.mapping(handle).decode()
        finally:
            self.destroy(handle)


def run_gpu(args, bridge_path, prepared, report):
    import torch
    sys.path.insert(0, str(ROOT / "python"))
    from cubutterfly import (AlgorithmPolicy, ComputeType, DataType, Error, ExecutionMode,
                             FftType, Library, Plan, PlanOptions, Transform)
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    stream = torch.cuda.Stream(device=device)
    library = Library(args.library)
    bridge = Bridge(bridge_path)
    properties = torch.cuda.get_device_properties(device)
    report.update(gpu_name=properties.name,
        gpu_uuid=str(getattr(properties, "uuid", os.environ.get("CUDA_VISIBLE_DEVICES"))),
        cases=[], invalid_cases=[])
    with tempfile.TemporaryDirectory(prefix="xd-core-public-") as cache:
        def plan_for(transform, mapping):
            return Plan(library, transform, PlanOptions(policy=AlgorithmPolicy.EXPLICIT,
                mapping=mapping, execution_mode=ExecutionMode.XD),
                stream=stream.cuda_stream, cache_path=str(Path(cache) / "plans.json"))

        for case, (inputs, references) in zip(cases(), prepared):
            fp64 = case["precision"] == "fp64"
            complex_only = case["complex_only"]
            if complex_only:
                storage = DataType.COMPLEX_FP64 if fp64 else DataType.COMPLEX_FP32
                fft_type = FftType.C2C
            else:
                storage = DataType.FP64 if fp64 else DataType.FP32
                fft_type = FftType.C2R if case["inverse"] else FftType.R2C
            transform = Transform(storage=storage, compute=ComputeType.FP64 if fp64 else ComputeType.FP32,
                                  extents=case["shape"], batch=case["batch"], inverse=case["inverse"],
                                  normalize_inverse=case["normalize"], fft_type=fft_type,
                                  execution_mode=ExecutionMode.XD)
            # Obtain the public semantic/target envelope, then request an exact
            # physical resource vector. The bridge only constructs that vector.
            with plan_for(transform, "xd:axis-tiled-local") as seed:
                mapping = json.loads(seed.mapping_json())
            mapping["physical"] = json.loads(bridge.canonical(case))
            mapping["lowering"] = ("axis-tiled-mixed" if complex_only else
                "xd-compact-c2r[axis-tiled-mixed]" if case["inverse"] else
                "xd-compact-r2c[axis-tiled-mixed]")
            if case == cases()[0]:
                base, base_transform = mapping, transform
            row = dict(case, status="running", eager=[], graph=[])
            report["cases"].append(row)
            with plan_for(transform, json.dumps(mapping)) as plan:
                if json.loads(plan.mapping_json()) != mapping:
                    raise AssertionError("public mapping replay changed the resource contract")
                row["mapping"] = mapping
                workspace = torch.empty(plan.workspace_size(), dtype=torch.uint8, device=device)
                if workspace.numel(): plan.set_workspace(workspace, workspace.numel())
                with torch.cuda.stream(stream):
                    src = torch.from_numpy(inputs[0]).to(device)
                    out = torch.empty(references[0].shape,
                        dtype=torch.from_numpy(references[0]).dtype, device=device)
                eager_outputs = []
                for index in (0, 1):
                    with torch.cuda.stream(stream):
                        src.copy_(torch.from_numpy(inputs[index]))
                        plan.execute(src, out)
                    stream.synchronize()
                    eager_outputs.append(out.cpu().clone())
                    row["eager"].append(check(eager_outputs[-1], references[index]))
                    if not torch.equal(src.cpu(), torch.from_numpy(inputs[index])):
                        raise AssertionError("eager execution mutated input")
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream): plan.execute(src, out)
                for index in (0, 1, 0):
                    with torch.cuda.stream(stream):
                        src.copy_(torch.from_numpy(inputs[index]))
                        graph.replay()
                    stream.synchronize()
                    actual = out.cpu().clone()
                    metrics = check(actual, references[index])
                    metrics["input_index"] = index
                    row["graph"].append(metrics)
                    if not torch.equal(actual, eager_outputs[index]):
                        raise AssertionError("graph differs from eager")
                    if not torch.equal(src.cpu(), torch.from_numpy(inputs[index])):
                        raise AssertionError("graph execution mutated input")
                row.update(status="pass", input_preserved=True, graph_matches_eager=True,
                           exact_json_replay=True)
        shape = (ct.c_size_t * 2)(128, 64)
        rejected = bridge.create(shape, 2, 1, 0, 0, 0, 1, 1.0, 0, 32, 0, 0)
        if rejected:
            bridge.destroy(rejected)
            raise AssertionError("128-axis cap32 plan was accepted")
        error = bridge.error().decode()
        if "thread-shared resource is unavailable for this axis" not in error:
            raise AssertionError("wrong cap32 rejection: " + error)
        report["invalid_cases"].append(dict(name="axis128-cap32", rejected=True, error=error))
        for invalid in invalid_cases()[1:]:
            bad = json.loads(json.dumps(base))
            resource = bad["physical"]["axis_resources"][0]
            resource.update(invalid["axis_resources"][0])
            if resource["core"] == "vkfft":
                resource.update(block_cap=0, local_twiddle_policy=0, exchange=0, ffts_per_block=0)
            expected = ("invalid xD axis resources for vkfft" if resource["core"] == "vkfft" else
                "xD axis parameter exceeds unsigned range: ept" if invalid["name"] == "oversized-resource" else
                "xD axis parameter must be an unsigned integer: ept")
            try:
                with plan_for(base_transform, json.dumps(bad)):
                    pass
            except Error as error:
                if expected not in str(error):
                    raise AssertionError("wrong resource rejection: " + str(error)) from error
                report["invalid_cases"].append(dict(name=invalid["name"], rejected=True, error=str(error)))
            else:
                raise AssertionError("invalid public resource accepted: " + invalid["name"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--cuda-include", type=Path, default=Path("/usr/local/cuda/include"))
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()
    if args.list_cases:
        print(json.dumps({"cases": cases(), "invalid_cases": invalid_cases()}, indent=2))
        return 0
    if not args.library or not args.output:
        parser.error("--library and --output are required")
    if not args.output.is_absolute() and os.environ.get("XD_REPORT_DIR"):
        args.output = Path(os.environ["XD_REPORT_DIR"]) / args.output
    report = dict(schema="xd-core-contract-20261013-v1", status="running",
                  scope="N64 cap32 local cores, block exchange4, and public JSON rejection",
                  cases=cases(), invalid_cases=invalid_cases(), device=args.device,
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  bridge_source_sha256=hashlib.sha256(CPP.read_bytes()).hexdigest(),
                  numerical_acceptance="report-only; finite required; no error threshold")
    try:
        prepared = cpu_references()
        report["cpu_reference_count"] = len(prepared)
        with tempfile.TemporaryDirectory(prefix="cubutterfly-xd-core-contract-") as directory:
            bridge, command = compile_bridge(args, Path(directory))
            report.update(bridge_compile_argv=command,
                          bridge_sha256=hashlib.sha256(bridge.read_bytes()).hexdigest(),
                          library=str(args.library.resolve()),
                          library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest())
            if args.compile_only:
                report["status"] = "compiled"
            else:
                run_gpu(args, bridge, prepared, report)
                report["status"] = "pass"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}, indent=2))
    return 0 if report["status"] in ("compiled", "pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
