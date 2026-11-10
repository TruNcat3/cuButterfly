#!/usr/bin/env python3
"""Interleaved adapter A/B retest with identical calibrated GPU mappings."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

import calibrate_xd_public as cal
from diagnose_real_xd_dispatch import graph_intervals, intervals


def load_adapter(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_views(torch, before, after, x, half, shape, batch):
    checks = {}
    for direction, value in (('forward', x), ('inverse', half)):
        old_call = before.rfftn if direction == 'forward' else before.irfftn
        new_call = after.rfftn if direction == 'forward' else after.irfftn
        expected, actual = old_call(value), new_call(value)
        saved = actual.clone()
        second = new_call(value)
        checks[direction + '_unchanged_bits'] = cal.input_exact(expected, actual)
        checks[direction + '_independent_output'] = actual.data_ptr() != second.data_ptr()
        new_call(value * 0.5)
        checks[direction + '_preserved_output'] = cal.input_exact(actual, saved)
        wide = torch.empty((*value.shape[:-1], value.shape[-1] * 2),
                           device=value.device, dtype=value.dtype)
        strided = wide[..., ::2]
        strided.copy_(value)
        checks[direction + '_noncontiguous'] = cal.input_exact(new_call(strided), expected)
        leading = value.reshape(2, batch // 2, *value.shape[1:])
        nested = new_call(leading)
        checks[direction + '_nested_batch'] = (
            nested.shape == (2, batch // 2, *expected.shape[1:])
            and cal.input_exact(nested.reshape(expected.shape), expected))
    if not all(checks.values()):
        raise RuntimeError(f'adapter A/B correctness failed: {checks}')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.calibration.read_text())
    if record['status'] != 'complete':
        raise ValueError('requires completed calibration')
    if args.output.exists():
        raise ValueError('use a fresh output directory')
    args.output.mkdir(parents=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = record['gpu_uuid']
    os.environ['CUBUTTERFLY_COMPILE_MODE'] = 'research'
    os.environ['CUBUTTERFLY_E13_ALGORITHM_POLICY'] = 'default'
    cal.require_library_identity(record['library'], record['library_sha256'])
    cal.exclusive_gpu(record['gpu_uuid'])
    import torch
    after_path = cal.ROOT / 'results/e13_assets/cubutterfly_real_fft.py'
    old = load_adapter('dispatch_before', args.before)
    new = load_adapter('dispatch_after', after_path)
    device = torch.device('cuda:0')
    report = dict(status='running', scope='full cached operator calls; not application end-to-end',
                  calibration=str(args.calibration.resolve()),
                  calibration_sha256=cal.sha256(args.calibration),
                  library_sha256=record['library_sha256'], gpu_uuid=record['gpu_uuid'],
                  before_sha256=cal.sha256(args.before), after_sha256=cal.sha256(after_path),
                  protocol=dict(warmup=20, repeats=100, trials=6,
                                order='rotate three backends, reverse on alternate trials',
                                allocations='independent fresh output for every eager call',
                                excluded=['plan creation', 'calibration', 'compilation', 'validation'],
                                graph='20 captured operators; per-operator throughput, not eager latency'),
                  cases=[])
    for cell in record['cases']:
        shape, batch = tuple(cell['shape']), cell['batch']
        adapters = {name: module.RealFFTAdapter(record['library'], shape, batch, device,
                                                cache_path=record['cache_path'])
                    for name, module in (('before', old), ('after', new))}
        try:
            row = dict(shape=shape, batch=batch, measurements={},
                       cache_gates={name: cal.cache_gate(cell['calibration']['plans'], a.plan_metadata())
                                    for name, a in adapters.items()},
                       validation={name: cal.validate(a, shape, batch, device) for name, a in adapters.items()})
            if not all(cal.valid(v) for v in row['validation'].values()):
                raise RuntimeError('nonfinite output or input mutation')
            x, half, _, _ = cal.inputs(shape, batch, device)
            x_saved, half_saved = x.clone(), half.clone()
            row['view_checks'] = validate_views(torch, adapters['before'], adapters['after'], x, half, shape, batch)
            dims = tuple(range(-len(shape), 0))
            forward = lambda: torch.fft.rfftn(x, dim=dims, norm='forward')
            inverse = lambda: torch.fft.irfftn(half, s=shape, dim=dims, norm='forward')
            for operation in ('forward', 'inverse', 'pair'):
                functions = {name: (lambda a=a: a.rfftn(x)) if operation == 'forward' else
                             (lambda a=a: a.irfftn(half)) if operation == 'inverse' else
                             (lambda a=a: a.irfftn(a.rfftn(x))) for name, a in adapters.items()}
                functions['torch'] = forward if operation == 'forward' else inverse if operation == 'inverse' else (
                    lambda: torch.fft.irfftn(forward(), s=shape, dim=dims, norm='forward'))
                names = list(functions)
                measurements = {name: dict(event_samples_us=[], wall_samples_us=[]) for name in names}
                orders = []
                cal.exclusive_gpu(record['gpu_uuid'])
                for trial in range(6):
                    order = names[trial % 3:] + names[:trial % 3]
                    if trial % 2:
                        order = order[::-1]
                    orders.append(order)
                    for name in order:
                        sample = intervals(torch, functions[name], repeats=100, trials=1)
                        measurements[name]['event_samples_us'].extend(sample['event_samples_us'])
                        measurements[name]['wall_samples_us'].extend(sample['wall_samples_us'])
                for name in names:
                    measured = measurements[name]
                    measured['event_us'] = statistics.median(measured['event_samples_us'])
                    measured['wall_us'] = statistics.median(measured['wall_samples_us'])
                    measured['graph'] = graph_intervals(torch, functions[name])
                cal.exclusive_gpu(record['gpu_uuid'])
                row['measurements'][operation] = dict(backends=measurements, orders=orders)
                print(json.dumps(dict(shape=shape, batch=batch, operation=operation,
                                      eager_us={n: m['event_us'] for n, m in measurements.items()},
                                      graph_us={n: m['graph']['event_us'] for n, m in measurements.items()})), flush=True)
            row['timing_input_exact'] = cal.input_exact(x_saved, x) and cal.input_exact(half_saved, half)
            if not row['timing_input_exact']:
                raise RuntimeError('timing mutated inputs')
            report['cases'].append(row)
            cal.checkpoint(args.output / 'report.json', report)
        finally:
            torch.cuda.synchronize()
            for adapter in adapters.values():
                adapter.close()
    cal.require_library_identity(record['library'], record['library_sha256'])
    report['status'] = 'complete'
    cal.checkpoint(args.output / 'report.json', report)
    with (args.output / 'timings.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('shape', 'batch', 'operation', 'before_us', 'after_us', 'torch_us',
                         'after_over_before_speedup', 'torch_over_after_speedup',
                         'graph_before_us', 'graph_after_us', 'graph_torch_us'))
        for row in report['cases']:
            for op, values in row['measurements'].items():
                b, a, t = (values['backends'][n] for n in ('before', 'after', 'torch'))
                writer.writerow(('x'.join(map(str, row['shape'])), row['batch'], op,
                                 b['event_us'], a['event_us'], t['event_us'], b['event_us'] / a['event_us'],
                                 t['event_us'] / a['event_us'], *(m['graph']['event_us'] for m in (b, a, t))))


if __name__ == '__main__':
    main()
