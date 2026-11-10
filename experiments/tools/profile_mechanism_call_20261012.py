#!/usr/bin/env python3
"""Capture exactly one warmed logical call through the public C API."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_controlled_experiment as controlled
from cubutterfly import Library, Plan, PlanOptions, Transform, Operator, DataType, AlgorithmPolicy, ExecutionMode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    journal = json.loads(args.journal.read_text())
    case = next(item for item in journal['cases'] if item['id'] == args.case)
    variant = next(item for item in case['variants'] if item['id'] == args.variant)
    if variant['preflight']['status'] != 'accepted':
        raise RuntimeError('requires accepted full-batch preflight')
    workload = case['workload']
    mapping = variant['mapping']
    dtype = {'fp32': DataType.FP32, 'word64': DataType.UINT64}[workload['precision']]
    op = {'fft': Operator.FFT, 'fwht': Operator.FWHT, 'ntt': Operator.NTT}[workload['operator']]
    if op == Operator.FFT:
        dtype = DataType.COMPLEX_FP32
    transform = Transform(op=op, storage=dtype, extents=[1 << workload['logN']], batch=workload['batch'],
                          normalize_inverse=False, modulus=workload.get('modulus', 1152921504606584833),
                          execution_mode=ExecutionMode.COMPOSITE)
    libpath = ROOT / 'results/search_budget_20261011/frozen_runtime/libcuntt.so'
    library = Library(libpath)
    cuda = ct.CDLL('libcudart.so.12')
    cuda.cudaMalloc.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t]
    cuda.cudaMemcpy.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int]
    cuda.cudaMemset.argtypes = [ct.c_void_p, ct.c_int, ct.c_size_t]
    cuda.cudaFree.argtypes = [ct.c_void_p]
    allocations = []
    def call(name, *values):
        code = getattr(cuda, name)(*values)
        if code:
            raise RuntimeError(f'{name}: CUDA status {code}')
    def allocate(size):
        pointer = ct.c_void_p()
        call('cudaMalloc', ct.byref(pointer), max(size, 1))
        allocations.append(pointer)
        return pointer
    try:
        with Plan(library, transform, PlanOptions(policy=AlgorithmPolicy.EXPLICIT, mapping=json.dumps(mapping),
                                                  execution_mode=ExecutionMode.COMPOSITE)) as plan:
            resolved = json.loads(plan.mapping_json())
            expected = variant['preflight']['resolved_mapping']
            # Public C API may wrap the same one-dimensional kernel mapping.
            physical = resolved.get('mapping', resolved)
            if 'axes' in physical:
                physical = physical['axes']
            physical = dict(physical)
            physical.pop('contract', None)
            differences = controlled._mapping_diff(expected, physical, include_derived=False)
            if differences:
                args.output.write_text(json.dumps(dict(status='mapping-mismatch', resolved=resolved,
                                                       expected=expected, differences=sorted(differences)), indent=2) + '\n')
                raise RuntimeError('C API mapping differs from benchmark; inspect record')
            source, destination = allocate(plan.input_size()), allocate(plan.output_size())
            if plan.workspace_size():
                workspace = allocate(plan.workspace_size())
                plan.set_workspace(workspace, plan.workspace_size())
            elements = (1 << workload['logN']) * workload['batch']
            if op == Operator.FFT:
                elements *= 2
            if op == Operator.NTT:
                host = (ct.c_uint64 * elements)(*(index % 257 for index in range(elements)))
            else:
                host = (ct.c_float * elements)(*((index % 257 - 128) / 128 for index in range(elements)))
            call('cudaMemcpy', source, ct.cast(host, ct.c_void_p), plan.input_size(), 1)
            call('cudaMemset', destination, 0, plan.output_size())
            for _ in range(100):
                plan.execute(source, destination)
            call('cudaDeviceSynchronize')
            call('cudaProfilerStart')
            plan.execute(source, destination)
            call('cudaDeviceSynchronize')
            call('cudaProfilerStop')
            args.output.write_text(json.dumps(dict(status='captured', logical_calls=1, warmup_calls=100,
                capture='cudaProfilerStart/Stop, synchronized full-call range', mapping=resolved,
                normal_latency_source=str(args.journal.resolve()), normal_timing_accepted=False), indent=2) + '\n')
    finally:
        for pointer in reversed(allocations):
            call('cudaFree', pointer)


if __name__ == '__main__':
    main()
