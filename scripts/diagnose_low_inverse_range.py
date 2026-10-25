#!/usr/bin/env python3
"""Compare final and distributed normalization on an unchanged FP16 C2R input.

This CPU radix-two trace diagnoses range, not bitwise equivalence to every GPU
lowering. It leaves the original row, input generator and error artifact intact.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import run_semantic_campaign as runner


def simulate(source, distributed):
    batch, n = source.shape
    log_n = n.bit_length() - 1
    index = np.arange(n, dtype=np.uint64)
    reverse = np.zeros(n, dtype=np.uint64)
    for _ in range(log_n):
        reverse = (reverse << 1) | (index & 1)
        index >>= 1
    real = source.real.astype(np.float16)[:, reverse].copy()
    imag = source.imag.astype(np.float16)[:, reverse].copy()
    stages = []
    with np.errstate(over='ignore', invalid='ignore'):
        for stage in range(log_n):
            half = 1 << stage
            angle = np.pi * np.arange(half, dtype=np.float64) / half
            wr = np.cos(angle).astype(np.float32).astype(np.float16)
            wi = np.sin(angle).astype(np.float32).astype(np.float16)
            r, i = real.reshape(-1, 2 * half), imag.reshape(-1, 2 * half)
            ar, ai = r[:, :half].copy(), i[:, :half].copy()
            br, bi = r[:, half:].copy(), i[:, half:].copy()
            if distributed:
                ar *= np.float16(.5)
                ai *= np.float16(.5)
                br *= np.float16(.5)
                bi *= np.float16(.5)
            vr, vi = br * wr - bi * wi, br * wi + bi * wr
            r[:, :half], r[:, half:] = ar + vr, ar - vr
            i[:, :half], i[:, half:] = ai + vi, ai - vi
            finite = real[np.isfinite(real)]
            stages.append(dict(stage_zero_based=stage,
                nonfinite_real=int((~np.isfinite(real)).sum()),
                nonfinite_imag=int((~np.isfinite(imag)).sum()),
                finite_real_peak=float(np.abs(finite).max()) if finite.size else None))
        if not distributed:
            real = (real.astype(np.float32) / n).astype(np.float16)
    return real.reshape(-1), stages


def diagnose(row_path):
    row_path = Path(row_path).resolve()
    row = json.loads(row_path.read_text())['row']
    s = row['descriptor']['semantics']
    if not (s['operator'] == 0 and s['fft_type'] == 2 and s['storage'] == 0
            and s['compute'] == 0 and s['inverse'] and s['normalize_inverse']
            and len(s['shape']) == 1 and s['fft_output_scale'] == 1):
        raise ValueError('trace requires native FP16 normalized rank-one C2R, scale=1')
    n = s['shape'][0]
    if n < 2 or n & (n - 1):
        raise ValueError('trace requires power-of-two N')
    compact = runner.make_source(s).reshape(s['batch'], -1)
    source = np.concatenate((compact, np.conj(compact[:, -2:0:-1])), axis=-1)
    artifact = Path(row['error_artifact']['path'])
    if runner.sha256_file(artifact) != row['error_artifact']['sha256']:
        raise ValueError('original error artifact changed')
    with np.load(artifact, allow_pickle=False) as arrays:
        actual, reference = arrays['actual'].copy(), arrays['reference'].copy()
    summaries = {}
    for name, distributed in (('final-normalization', False), ('distributed-normalization', True)):
        result, stages = simulate(source, distributed)
        finite = bool(np.isfinite(result).all())
        summaries[name] = dict(stages=stages,
            first_nonfinite_stage=next((r['stage_zero_based'] for r in stages
                if r['nonfinite_real'] or r['nonfinite_imag']), None),
            nonfinite_output=int((~np.isfinite(result)).sum()),
            same_nonfinite_positions_as_recorded_gpu=bool(np.array_equal(
                ~np.isfinite(result), ~np.isfinite(actual))),
            relative_l2=float(np.linalg.norm(result.astype(np.float64) - reference) /
                              np.linalg.norm(reference)) if finite and np.any(reference) else None)
    return dict(schema='cubutterfly-low-inverse-range-trace-v1', row=str(row_path),
        row_sha256=runner.sha256_file(row_path), error_artifact=row['error_artifact'],
        semantics=s, input_sha256=runner.hashlib.sha256(compact.tobytes()).hexdigest(),
        recorded_gpu_nonfinite=int((~np.isfinite(actual)).sum()),
        reference_peak=float(np.abs(reference).max()), traces=summaries,
        policy='unchanged quantized input, storage and arithmetic; equivalent 1/N normalization schedule',
        limitation='CPU half radix-two recurrence; GPU lowering rounding/order may differ; no accuracy acceptance')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--row', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be fresh')
    report = diagnose(args.row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k in ('recorded_gpu_nonfinite', 'reference_peak')}))
