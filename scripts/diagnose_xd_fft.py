#!/usr/bin/env python3
"""Bounded xD FFT diagnosis without changing the production selector.

Uses the public C API, records the loaded cuFFT binaries, and separates graph
timing from profiler collection. Plan construction and correctness checking
are outside both regions. --profile-only brackets warmed execution with the
CUDA profiler API for ncu/nsys --profile-from-start off. To explain an existing
measurement, use --lowerings replay --mapping-json with its saved winner;
attaching a profiler can perturb a fresh empirical selection.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import benchmark_xd as bench

xd = bench.xd


class NativeCufft:
    """Independent vendor reference using the same loaded cuFFT as the library."""

    def __init__(self, torch, shape, batch, fp64, inverse, stream, source, output):
        self.lib = ct.CDLL('libcufft.so.11')
        self.handle = ct.c_int()
        self.inverse, self.source, self.output = inverse, source, output
        self.points = xd._prod(shape)
        self.check(self.lib.cufftCreate(ct.byref(self.handle)))
        self.check(self.lib.cufftSetAutoAllocation(self.handle, 0))
        dims = (ct.c_int * len(shape))(*shape)
        work = ct.c_size_t()
        try:
            self.check(self.lib.cufftMakePlanMany(
                self.handle, len(shape), dims, dims, 1, self.points,
                dims, 1, self.points, 0x69 if fp64 else 0x29, batch, ct.byref(work)))
            self.workspace = torch.empty(max(1, work.value), device='cuda', dtype=torch.uint8)
            self.check(self.lib.cufftSetWorkArea(self.handle, ct.c_void_p(self.workspace.data_ptr())))
            self.check(self.lib.cufftSetStream(self.handle, ct.c_void_p(stream.cuda_stream)))
            self.exec_fn = self.lib.cufftExecZ2Z if fp64 else self.lib.cufftExecC2C
        except Exception:
            self.close()
            raise

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError('native cuFFT error ' + str(status))

    def __call__(self):
        self.check(self.exec_fn(self.handle, ct.c_void_p(self.source.data_ptr()),
                               ct.c_void_p(self.output.data_ptr()), 1 if self.inverse else -1))
        if self.inverse:
            self.output.mul_(1.0 / self.points)

    def close(self):
        if self.handle.value:
            self.lib.cufftDestroy(self.handle)
            self.handle = ct.c_int()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', required=True)
    parser.add_argument('--shape', type=int, nargs='+', required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--precision', choices=('fp32', 'fp64'), default='fp32')
    parser.add_argument('--inverse', action='store_true')
    parser.add_argument('--vkfft-library', type=Path,
                        help='optional VkFFT xD bridge for the independent vkfft baseline')
    parser.add_argument('--lowerings', nargs='+',
                        default=['selected', 'packed-axis-cufft', 'cufft-native', 'torch'])
    parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--mapping-json', type=Path,
                        help='exact mapping JSON used by the replay lowering')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch

    torch.manual_seed(42)
    fp64 = args.precision == 'fp64'
    dtype = torch.complex128 if fp64 else torch.complex64
    library = Path(args.library).resolve()
    api = xd.Api(library)
    version = ct.c_int()
    NativeCufft.check(ct.CDLL('libcufft.so.11').cufftGetVersion(ct.byref(version)))
    report = {
        'shape': args.shape, 'batch': args.batch, 'precision': args.precision,
        'inverse': args.inverse, 'gpu': bench._gpu_identity(torch),
        'library': str(library), 'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
        'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
        'cufft_version': version.value,
        'python': sys.executable, 'compile_mode': os.getenv('CUBUTTERFLY_COMPILE_MODE'),
        'protocol': ('one warmed call bracketed by CUDA profiler API; no graph timing'
                     if args.profile_only else
                     '20-call CUDA graph; 7 replays; planning, warmup, checking excluded'),
        'rows': [],
    }
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), xd.Handle(api, int(stream.cuda_stream)) as handle:
        source = bench._make_fft_source(torch, args.shape, args.batch, dtype)
        source_copy = source.clone()
        output = torch.empty_like(source)
        dims = tuple(range(1, len(args.shape) + 1))
        reference = torch.fft.ifftn if args.inverse else torch.fft.fftn
        expected = reference(source, dim=dims)
        torch.cuda.synchronize()
        for lowering in args.lowerings:
            row = {'requested': lowering}
            plan = None
            native = None
            try:
                if lowering == 'torch':
                    def execute():
                        reference(source, dim=dims, out=output)
                elif lowering == 'cufft-native':
                    native = NativeCufft(torch, args.shape, args.batch, fp64,
                                         args.inverse, stream, source, output)
                    execute = native
                elif lowering == 'vkfft':
                    if args.vkfft_library is None:
                        raise ValueError('vkfft requires --vkfft-library')
                    from vkfft_xd import NativeVkfft
                    native = NativeVkfft(torch, args.shape, args.batch, fp64,
                                         args.inverse, stream, source, output, args.vkfft_library)
                    row['baseline_metadata'] = native.metadata()
                    execute = native
                else:
                    mapping = None if lowering == 'selected' else 'xd:' + lowering
                    if lowering == 'replay':
                        if args.mapping_json is None:
                            raise ValueError('replay requires --mapping-json')
                        mapping = json.dumps(json.loads(args.mapping_json.read_text()))
                    plan = xd.make_plan(
                        api, handle, torch, op=xd.OP_FFT,
                        storage=xd.DATA_COMPLEX_FP64 if fp64 else xd.DATA_COMPLEX_FP32,
                        compute=xd.COMPUTE_FP64 if fp64 else xd.COMPUTE_FP32,
                        shape=args.shape, batch=args.batch, mode=xd.EXEC_XD,
                        direction=xd.INVERSE if args.inverse else xd.FORWARD,
                        mapping=mapping,
                    )
                    row.update(bench._metadata(plan))
                    def execute():
                        plan.execute(source, output)
                execute()
                torch.cuda.synchronize()
                error = bench._max_error(torch, output, expected)
                if not math.isfinite(error['max_abs']):
                    raise AssertionError(lowering + ': non-finite output')
                bench._check_error(error, atol=5e-11 if fp64 else 5e-4,
                                   rtol=5e-11 if fp64 else 5e-4, label=lowering)
                row['correctness'] = error
                if not torch.equal(source, source_copy):
                    raise AssertionError(lowering + ': out-of-place input was modified')
                if args.profile_only:
                    for _ in range(10):
                        execute()
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStart()
                    torch.cuda.nvtx.range_push('xd_diagnostic/' + lowering)
                    execute()
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()
                else:
                    # Verify that the external-library callback really enters
                    # the captured stream before accepting its graph timing.
                    probe = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(probe, stream=stream):
                        execute()
                    output.fill_(complex(float('nan'), float('nan')))
                    probe.replay()
                    stream.synchronize()
                    graph_error = bench._max_error(torch, output, expected)
                    if not math.isfinite(graph_error['max_abs']):
                        raise AssertionError(lowering + ': graph replay did not produce finite output')
                    bench._check_error(graph_error, atol=5e-11 if fp64 else 5e-4,
                                       rtol=5e-11 if fp64 else 5e-4, label=lowering+' graph')
                    row['graph_correctness'] = graph_error
                    del probe
                    row['graph'] = bench._graph_median_ms(
                        torch, execute, stream, warmup=10, replays=7, capture_calls=20)
                if not torch.equal(source, source_copy):
                    raise AssertionError(lowering + ': repeated execution modified the input')
                row['input_preserved'] = True
                row['status'] = 'passed'
            except Exception as exc:
                row.update(status='failed', error=str(exc))
            finally:
                if plan is not None:
                    plan.close()
                if native is not None:
                    native.close()
            report['rows'].append(row)
            print(lowering, row['status'], row.get('algorithm', ''),
                  row.get('graph', {}).get('median_ms', ''), row.get('error', ''), flush=True)
    report['loaded_cuda_libraries'] = sorted({
        line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
        if 'libcufft' in line or 'libcudart' in line
    })
    report['status'] = 'passed' if all(r['status'] == 'passed' for r in report['rows']) else 'failed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
