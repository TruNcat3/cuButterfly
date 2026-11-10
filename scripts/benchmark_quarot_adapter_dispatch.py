#!/usr/bin/env python3
"""Measure the serialized QuaRot FWHT adapter boundary with and without reuse."""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--extent', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--samples', type=int, default=7)
    parser.add_argument('--compute-contract', choices=('fp32_composed', 'public_fp32_final'),
                        default='fp32_composed')
    parser.add_argument('--algorithm-policy', choices=('default', 'measure'), default='default',
                        help='FWHT plan selection policy; measure performs bounded local tuning')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output must be fresh')
    os.environ.setdefault('CUBUTTERFLY_COMPILE_MODE', 'research')
    os.environ['CUBUTTERFLY_E13_ALGORITHM_POLICY'] = args.algorithm_policy
    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/workloads/e13'))
    from quarot_adapter import QuaRotFWHTAdapter
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(20261012)
    values = torch.randn((args.batch, args.extent), generator=generator,
                         device=device, dtype=torch.float16)
    values_before = values.clone()
    reference = torch.empty_like(values)
    rows = []
    for reuse in (False, True):
        adapter = QuaRotFWHTAdapter(args.library, device=device,
                                    compute_contract=args.compute_contract,
                                    reuse_buffers=reuse)
        try:
            first = adapter.matmul_hadU_cuda(values, None, 1)
            torch.cuda.synchronize(device)
            reference.copy_(first)
            samples = []
            for _ in range(args.warmup):
                adapter.matmul_hadU_cuda(values, None, 1)
            torch.cuda.synchronize(device)
            for _ in range(args.samples):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.repeats):
                    result = adapter.matmul_hadU_cuda(values, None, 1)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000.0 / args.repeats)
            torch.cuda.synchronize(device)
            rows.append({
                'reuse_buffers': reuse,
                'event_us': samples,
                'median_us': statistics.median(samples),
                'plans': adapter.plan_statistics(),
                'finite': bool(torch.isfinite(result).all()),
                'input_unchanged': bool(torch.equal(values, values_before)),
            })
        finally:
            adapter.close()
    report = {
        'scope': 'serialized FWHT adapter call; not application E2E',
        'library': str(args.library.resolve()),
        'device': str(device), 'batch': args.batch, 'extent': args.extent,
        'protocol': {'warmup': args.warmup, 'repeats': args.repeats, 'samples': args.samples,
                     'setup_excluded': True, 'compute_contract': args.compute_contract,
                     'algorithm_policy': args.algorithm_policy},
        'rows': rows,
        'speedup_reuse_over_fresh': rows[0]['median_us'] / rows[1]['median_us'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'median_us': [r['median_us'] for r in rows],
                      'speedup': report['speedup_reuse_over_fresh']}))


if __name__ == '__main__':
    main()
