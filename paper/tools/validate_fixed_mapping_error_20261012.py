#!/usr/bin/env python3
"""Record report-only FP32 error distribution for an explicit public mapping.

The host reference is generated independently from the benchmark input; this
does not claim bitwise identity with benchmark random inputs unless supplied.
"""
import argparse
import ctypes as ct
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from cubutterfly import Library, Plan, PlanOptions, Transform, Operator, DataType, AlgorithmPolicy, ExecutionMode


def fwht_reference(values):
    result = np.array(values, dtype=np.float32, copy=True)
    width = result.shape[-1]
    stride = 1
    while stride < width:
        for offset in range(0, width, 2 * stride):
            left = result[..., offset:offset + stride].copy()
            right = result[..., offset + stride:offset + 2 * stride].copy()
            result[..., offset:offset + stride] = left + right
            result[..., offset + stride:offset + 2 * stride] = left - right
        stride *= 2
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logN", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args(argv)
    mapping = json.loads(args.mapping.read_text())
    rng = np.random.default_rng(20261012)
    host_input = rng.standard_normal((args.batch, 1 << args.logN), dtype=np.float32)
    expected = fwht_reference(host_input)
    library = Library(args.library)
    transform = Transform(op=Operator.FWHT, storage=DataType.FP32, compute=0,
                          extents=[1 << args.logN], batch=args.batch,
                          normalize_inverse=False, execution_mode=ExecutionMode.COMPOSITE)
    cuda = ct.CDLL("libcudart.so.12")
    cuda.cudaMalloc.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t]
    cuda.cudaMemcpy.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int]
    cuda.cudaDeviceSynchronize.argtypes = []
    cuda.cudaFree.argtypes = [ct.c_void_p]
    allocations = []
    def call(name, *values):
        code = getattr(cuda, name)(*values)
        if code:
            raise RuntimeError(f"{name}: CUDA status {code}")
    def allocate(size):
        pointer = ct.c_void_p()
        call("cudaMalloc", ct.byref(pointer), max(size, 1))
        allocations.append(pointer)
        return pointer
    try:
        with Plan(library, transform, PlanOptions(policy=AlgorithmPolicy.EXPLICIT,
                                                  mapping=json.dumps(mapping), execution_mode=ExecutionMode.COMPOSITE)) as plan:
            resolved = json.loads(plan.mapping_json())
            source, destination = allocate(plan.input_size()), allocate(plan.output_size())
            workspace = allocate(plan.workspace_size()) if plan.workspace_size() else None
            if workspace is not None:
                plan.set_workspace(workspace, plan.workspace_size())
            flat = (ct.c_float * host_input.size)(*host_input.reshape(-1))
            call("cudaMemcpy", source, ct.cast(flat, ct.c_void_p), plan.input_size(), 1)
            plan.execute(source, destination)
            call("cudaDeviceSynchronize")
            raw = (ct.c_float * expected.size)()
            call("cudaMemcpy", ct.cast(raw, ct.c_void_p), destination, plan.output_size(), 2)
            actual = np.ctypeslib.as_array(raw).reshape(expected.shape).copy()
            error = np.abs(actual - expected)
            report = dict(schema="cubutterfly-fixed-mapping-error-report-v1", status="report-only",
                          operator="fwht", shape=[args.batch, 1 << args.logN], seed=20261012,
                          input_source="new deterministic NumPy input; not benchmark timing input",
                          mapping_requested=mapping, mapping_resolved=resolved,
                          error_distribution=dict(max_abs=float(error.max()), mean_abs=float(error.mean()),
                                                  p50=float(np.quantile(error, .5)), p95=float(np.quantile(error, .95)),
                                                  p99=float(np.quantile(error, .99)), nonfinite=int(np.count_nonzero(~np.isfinite(actual)))),
                          user_numerical_acceptance="unassessed; no threshold introduced")
    finally:
        for pointer in reversed(allocations):
            call("cudaFree", pointer)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
