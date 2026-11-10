#!/usr/bin/env python3
"""Run the frozen two-request profile supplement, resuming clean per-task releases.

CPU preparation is explicit and precedes this runner. Each GPU task uses the
existing exclusive-device guard. Profiling never supplies ordinary timing.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load(path):
    data = json.loads(path.read_text())
    if data['schema'] != 'cubutterfly-representative-profile-campaign-v1':
        raise ValueError('wrong campaign schema')
    for filename, expected in data['file_hashes'].items():
        if sha(filename) != expected:
            raise ValueError('prepared identity changed: ' + filename)
    ids = [task['id'] for task in data['tasks']]
    repair = data.get('capture_repair')
    expected = 8 if repair else 30
    if len(ids) != len(set(ids)) or len(ids) != expected:
        raise ValueError('expected 30 primary tasks or eight explicit capture repairs')
    if repair and (not repair.get('parent_campaign_sha256') or any(
            t['family'] != 'factor' or t['kind'] != 'nsys' for t in data['tasks'])):
        raise ValueError('repair scope must be the eight factor NSYS captures')
    return data


def command(data, task, output):
    if task['family'] == 'subgraph':
        argv = [data['subgraph_binary'], '--logN', '16', '--batch', '256',
                '--input-reference', data['subgraph_reference_prefix'],
                '--report-errors-only', '--warmup', '100', '--repeat', '100',
                '--trials', '1']
        if task['kind'] == 'ordinary':
            argv += ['--trial-offset', str(task['process']), '--csv']
        else:
            argv += ['--core', task['core'], '--organization', task['organization'],
                     '--profile-one-call']
    else:
        argv = [sys.executable, str(ROOT / 'scripts/run_factor_graph_experiment.py'),
                'measure' if task['kind'] == 'ordinary' else 'profile',
                '--prepared', data['factor_prepared'], '--binary', data['factor_binary'],
                '--output', str(output / 'application')]
        if task['kind'] == 'ordinary':
            argv += ['--trial-offset', str(task['process'])]
        else:
            argv += ['--case', task['case']]
    return argv


def run_task(path, key):
    data = load(path)
    task = next(task for task in data['tasks'] if task['id'] == key)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != data['gpu_uuid']:
        raise ValueError('guarded visible UUID must match campaign')
    out = Path(os.environ['XD_REPORT_DIR'])
    child = command(data, task, out)
    argv = child
    if task['kind'] == 'ncu':
        argv = ['ncu', '--target-processes', 'all', '--profile-from-start', 'off',
                '--replay-mode', 'kernel', '--graph-profiling', 'node',
                '--cache-control', 'none', '--clock-control', 'none',
                '--kernel-name-base', 'demangled', '--metrics', ','.join(data['metrics']),
                '--csv', '--page', 'raw', '--print-units', 'base',
                '--log-file', str(out / 'ncu.csv'), '--export', str(out / 'capture')] + child
    elif task['kind'] == 'nsys':
        argv = ['nsys', 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                '--cuda-graph-trace=node',
                '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                '--output', str(out / 'capture')] + child
    env = dict(os.environ, CUBUTTERFLY_COMPILE_MODE='research',
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'])
    write(out / 'task.json', dict(task=task, campaign_sha256=sha(path), argv=argv,
          gpu_uuid=data['gpu_uuid'], normal_timing_accepted=task['kind'] == 'ordinary'))
    with (out / 'stdout.txt').open('w') as stdout, (out / 'stderr.txt').open('w') as stderr:
        result = subprocess.run(argv, stdout=stdout, stderr=stderr, env=env, timeout=900)
    if result.returncode:
        raise RuntimeError(f'{key} failed with {result.returncode}; preserved logs in {out}')
    if task['family'] == 'subgraph':
        text = (out / 'stdout.txt').read_text()
        if task['kind'] == 'ordinary':
            start = text.index('device,')
            rows = list(csv.DictReader(io.StringIO(text[start:])))
            expected = {(core, org) for core in ('native', 'cufftdx-thread')
                        for org in ('stagewise-hbm', 'resident-prefix-2')}
            if len(rows) != 4 or {(r['core'], r['organization']) for r in rows} != expected:
                raise ValueError('ordinary subgraph rows incomplete')
        elif 'profile_time_not_performance=1 logical_calls=1' not in text:
            raise ValueError('single-call capture receipt missing')
    else:
        observed = json.loads((out / 'application/results.json').read_text())
        if task['kind'] == 'ordinary':
            if len(observed['results']) != 8 or not observed['all_outputs_finite']:
                raise ValueError('ordinary factor rows incomplete')
        elif observed.get('logical_calls') != 1 or observed.get('normal_timing_accepted') is not False:
            raise ValueError('factor capture receipt mismatch')
    if task['kind'] == 'ncu':
        text = (out / 'ncu.csv').read_text()
        start = text.index('"ID"')
        rows = list(csv.DictReader(io.StringIO(text[start:])))
        kernels = {r['ID'] for r in rows if r.get('Kernel Name')}
        if len(kernels) != task['expected_kernels']:
            raise ValueError(f'capture has {len(kernels)} kernels, expected {task["expected_kernels"]}')
        if 'Metric Name' in (rows[0] if rows else {}):
            missing = [r for r in rows if r.get('Kernel Name') and
                       r.get('Metric Value') in ('n/a', 'N/A', '', None)]
        else:
            missing = [dict(kernel_id=r['ID'], metric=metric, value=r.get(metric))
                       for r in rows if r.get('Kernel Name') and r.get('ID')
                       for metric in data['metrics'] if r.get(metric) in ('n/a', 'N/A', '', None)]
        write(out / 'capture_scope.json', dict(kernel_count=len(kernels),
              expected_kernels=task['expected_kernels'], unavailable_metrics=missing,
              timing_scope='profiler diagnostics only', cache_control='none', clock_control='none'))
    if task['kind'] == 'nsys' and not (out / 'capture.nsys-rep').is_file():
        raise ValueError('NSYS report missing')
    write(out / 'complete.json', dict(task=key, campaign_sha256=sha(path),
          gpu_uuid=data['gpu_uuid'], completed=datetime.now().astimezone().isoformat()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['run', 'task'])
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--task')
    parser.add_argument('--until', help='explicit bounded idle-wait deadline, ISO with timezone')
    args = parser.parse_args()
    path = args.campaign.resolve()
    if args.phase == 'task':
        if not args.task:
            parser.error('task phase requires --task')
        run_task(path, args.task)
        return
    if not args.until or datetime.fromisoformat(args.until).tzinfo is None:
        parser.error('run requires --until with timezone')
    data = load(path)
    root = path.parent
    for task in data['tasks']:
        report = root / 'jobs' / task['id']
        receipt = report / 'release/complete.json'
        if receipt.is_file():
            if json.loads(receipt.read_text())['campaign_sha256'] != sha(path):
                raise ValueError('completed task belongs to a different campaign')
            continue
        argv = [sys.executable, str(ROOT / 'scripts/run_isolated_gpu.py'),
                '--gpu-uuid', data['gpu_uuid'], '--report-dir', str(report),
                '--until', args.until, '--idle-seconds', '3', '--',
                sys.executable, str(Path(__file__).resolve()), 'task',
                '--campaign', str(path), '--task', task['id']]
        write(root / 'status.json', dict(state='running', task=task['id'], deadline=args.until))
        with (root / 'guard.log').open('a') as log:
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write(root / 'status.json', dict(state='stopped', task=task['id'],
                  returncode=result.returncode, deadline=args.until))
            raise SystemExit(result.returncode)
        print('complete', task['id'], flush=True)
    write(root / 'status.json', dict(state='complete', tasks=len(data['tasks']),
          gpu_uuid=data['gpu_uuid'], campaign_sha256=sha(path)))


if __name__ == '__main__':
    main()
