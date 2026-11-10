#!/usr/bin/env python3
"""Separate cached real-FFT operator dispatch, graph replay and profiler evidence.

This is an operator diagnostic, not an application end-to-end benchmark.
The cache and library must match a completed calibrate_xd_public.py record.
"""
from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import statistics
import sys
import time
from pathlib import Path

import calibrate_xd_public as cal


def intervals(torch, function, repeats=50, trials=5):
    for _ in range(20):
        result = function()
    torch.cuda.synchronize()
    samples, wall = [], []
    for _ in range(trials):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        for _ in range(repeats):
            result = function()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - begin) * 1e6 / repeats)
        samples.append(start.elapsed_time(end) * 1000 / repeats)
    return dict(event_us=statistics.median(samples), synchronized_wall_us=statistics.median(wall),
                event_samples_us=samples, wall_samples_us=wall, repeats=repeats, trials=trials)


def graph_intervals(torch, function, calls=20):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            result = function()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls):
            result = function()
    torch.cuda.current_stream().wait_stream(stream)
    timing = intervals(torch, graph.replay)
    timing['event_us'] /= calls
    timing['event_samples_us'] = [v / calls for v in timing['event_samples_us']]
    timing['synchronized_wall_us'] /= calls
    timing['wall_samples_us'] = [v / calls for v in timing['wall_samples_us']]
    timing['captured_calls'] = calls
    timing['scope'] = 'CUDA graph throughput per captured operator; excludes Python dispatch and allocation'
    return timing


def profile(torch, function, path):
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.profiler.record_function('operator_diagnostic'):
            function()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    events = json.loads(path.read_text())['traceEvents']
    kernels = [dict(name=e['name'], duration_us=e['dur'], start_us=e['ts'], args=e.get('args', {}))
               for e in events if e.get('cat') == 'kernel' and 'dur' in e]
    return dict(kernels=kernels, kernel_count=len(kernels),
                sum_kernel_us=sum(k['duration_us'] for k in kernels),
                gpu_span_us=(max(k['start_us'] + k['duration_us'] for k in kernels)
                             - min(k['start_us'] for k in kernels)) if kernels else None,
                headline=False, note='instrumented observations; not additive to unprofiled timings')


def cpu_profile(torch, function):
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(200):
        function()
    profiler.disable()
    torch.cuda.synchronize()
    rows = []
    for (file, line, name), (primitive, count, self_s, total_s, callers) in pstats.Stats(profiler).stats.items():
        rows.append(dict(file=file, line=line, name=name, calls=count,
                         self_us_per_invocation=self_s * 1e6 / 200,
                         cumulative_us_per_invocation=total_s * 1e6 / 200))
    return sorted(rows, key=lambda r: r['self_us_per_invocation'], reverse=True)[:20]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--backend', choices=('public', 'torch'), default='public')
    parser.add_argument('--operation', choices=('forward', 'inverse', 'pair'), default='forward')
    parser.add_argument('--case', type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    record = json.loads(args.calibration.read_text())
    if record['status'] != 'complete':
        raise ValueError('requires a completed calibration and cache replay record')
    if args.output.exists():
        raise ValueError('use a fresh output directory')
    args.output.mkdir(parents=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = record['gpu_uuid']
    os.environ['CUBUTTERFLY_COMPILE_MODE'] = 'research'
    os.environ['CUBUTTERFLY_E13_ALGORITHM_POLICY'] = 'default'
    cal.require_library_identity(record['library'], record['library_sha256'])
    cal.exclusive_gpu(record['gpu_uuid'])
    import torch
    sys.path.insert(0, str(cal.ROOT / 'results/e13_assets'))
    import cubutterfly_real_fft as adapter_module
    device = torch.device('cuda:0')
    report = dict(scope='cached xD operator diagnostics; not application end-to-end',
                  calibration=str(args.calibration.resolve()), calibration_sha256=cal.sha256(args.calibration),
                  library_sha256=record['library_sha256'], gpu_uuid=record['gpu_uuid'],
                  adapter_sha256=cal.sha256(adapter_module.__file__), cases=[])
    for i, cell in enumerate(record['cases']):
        if args.case is not None and i != args.case:
            continue
        shape, batch = tuple(cell['shape']), cell['batch']
        adapter = adapter_module.RealFFTAdapter(record['library'], shape, batch, device,
                                                cache_path=record['cache_path'])
        try:
            gate = cal.cache_gate(cell['calibration']['plans'], adapter.plan_metadata())
            x, half, forward_ref, inverse_ref = cal.inputs(shape, batch, device)
            dims = tuple(range(-len(shape), 0))
            functions = {
                'forward': {'public': lambda: adapter.rfftn(x),
                            'torch': lambda: torch.fft.rfftn(x, dim=dims, norm='forward')},
                'inverse': {'public': lambda: adapter.irfftn(half),
                            'torch': lambda: torch.fft.irfftn(half, s=shape, dim=dims, norm='forward')},
                'pair': {'public': lambda: adapter.irfftn(adapter.rfftn(x)),
                         'torch': lambda: torch.fft.irfftn(torch.fft.rfftn(x, dim=dims, norm='forward'),
                                                         s=shape, dim=dims, norm='forward')},
            }
            if args.profile_only:
                fn = functions[args.operation][args.backend]
                for _ in range(50):
                    fn()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                result = fn()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                continue
            row = dict(shape=shape, batch=batch, cache_gate=gate,
                       validation=cal.validate(adapter, shape, batch, device), measurements={})
            cal.exclusive_gpu(record['gpu_uuid'])
            for operation, backends in functions.items():
                row['measurements'][operation] = {}
                for backend, function in backends.items():
                    row['measurements'][operation][backend] = dict(
                        eager=intervals(torch, function), graph=graph_intervals(torch, function),
                        profiler=profile(torch, function, args.output / f'case{i}-{operation}-{backend}.trace.json'),
                        cpu_profile=cpu_profile(torch, function))
            cal.exclusive_gpu(record['gpu_uuid'])
            report['cases'].append(row)
            cal.checkpoint(args.output / 'report.json', report)
            print(json.dumps(dict(shape=shape, batch=batch, timings={op:{be:dict(eager=v['eager']['event_us'],
                 graph=v['graph']['event_us'], kernels=v['profiler']['kernel_count']) for be,v in entries.items()}
                 for op,entries in row['measurements'].items()})), flush=True)
        finally:
            torch.cuda.synchronize()
            adapter.close()
    cal.require_library_identity(record['library'], record['library_sha256'])
    report['status'] = 'complete'
    cal.checkpoint(args.output / 'report.json', report)


if __name__ == '__main__':
    main()
