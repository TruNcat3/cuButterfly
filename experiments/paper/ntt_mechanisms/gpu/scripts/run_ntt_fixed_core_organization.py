#!/usr/bin/env python3
"""Prepare and execute the bounded A100 fixed-arithmetic organization control.

Preparation compiles and computes CPU references. GPU tasks run only inside the
exclusive-device guard. Ordinary CUDA-event samples and NCU diagnostics are kept
separate; failed or permission-blocked captures never become performance evidence.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'cubutterfly-ntt-fixed-core-campaign-v1'
METRICS = ['dram__bytes_read.sum', 'dram__bytes_write.sum',
           'lts__t_sectors_op_read.sum', 'lts__t_sectors_op_write.sum',
           'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
           'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
           'launch__registers_per_thread', 'launch__shared_mem_per_block',
           'sm__warps_active.avg.pct_of_peak_sustained_active',
           'smsp__warp_issue_stalled_barrier_per_warp_active.pct',
           'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def load(path):
    data = json.loads(path.read_text())
    if data['schema'] != SCHEMA: raise ValueError('wrong campaign schema')
    for file, digest in data['file_hashes'].items():
        if sha(file) != digest: raise ValueError('prepared file changed: ' + file)
    return data

def prepare(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'campaign.json').exists(): raise ValueError('campaign exists; use a fresh directory')
    source = ROOT / 'benchmarks/ntt_fixed_core_organization.cu'
    vendors = args.vendors.resolve()
    subprocess.run([sys.executable, str(ROOT / 'resolve_vendors.py'), '--verify', '--output', str(vendors)], check=True)
    includes = [ROOT / 'src', ROOT / 'include', vendors / 'GPU-NTT/src/include',
                vendors / 'PhantomFHE/include']
    binary = root / 'ntt_fixed_core_organization'
    argv = [args.nvcc, '-std=c++17', '-O3', '-arch=sm_80', '-lineinfo', '-Xptxas=-v',
            *['-I' + str(p) for p in includes], str(source), '-o', str(binary), '-lcudart']
    with (root / 'build.log').open('w') as log:
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
    prepared = root / 'prepared'
    cpu_argv = [str(binary), '--prepare', str(prepared), '--log-n', str(args.log_n),
                '--batch', str(args.batch), '--modulus', str(args.modulus)]
    with (root / 'cpu_prepare.log').open('w') as log:
        subprocess.run(cpu_argv, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not (prepared / 'metadata.json').is_file(): raise ValueError('CPU metadata missing')
    # Freeze included local headers and external sources as well as input/reference bytes.
    files = [source, binary, Path(__file__).resolve(), *prepared.rglob('*')]
    for folder in [ROOT / 'include', ROOT / 'src', *[p for p in includes if vendors in p.parents]]:
        files.extend(p for p in folder.rglob('*') if p.suffix in ('.h', '.hpp', '.cuh', '.cu'))
    configs = [dict(core=c, local_log=l, organization=o) for c in ('native', 'gpuntt', 'phantom')
               for l in (8, 10) for o in ('split-prefix', 'resident-prefix')]
    tasks = []
    for trial in range(3):
        # Rotate then reverse alternate trials to avoid a fixed organization order.
        ordered = configs[trial:] + configs[:trial]
        if trial % 2: ordered = list(reversed(ordered))
        for cfg in ordered:
            key = f'{cfg["core"]}-l{cfg["local_log"]}-{cfg["organization"]}'
            tasks.append(dict(id=f'ordinary-p{trial}-{key}', kind='ordinary', process=trial, **cfg))
    for cfg in configs:
        tasks.append(dict(id=f'ncu-{cfg["core"]}-l{cfg["local_log"]}-{cfg["organization"]}',
                          kind='ncu', **cfg))
    write(root / 'campaign.json', dict(schema=SCHEMA, status='cpu-prepared',
          platform='A100; Figure 11 only', gpu_uuid=args.gpu_uuid,
          core_scope='Unchanged radix2 arithmetic primitives, not complete vendor NTT kernels',
          controlled_pair='Unchanged arithmetic provider and suffix; common shared_iterative template, 128 threads and writer-aligned boundary; grouping changes tile/shared/grid',
          partitions={'8': {'split-prefix': [4, 4, 8], 'resident-prefix': [8, 8]},
                      '10': {'split-prefix': [5, 5, 6], 'resident-prefix': [10, 6]}},
          source=str(source), binary=str(binary), prepared=str(prepared),
          request=dict(log_n=args.log_n, batch=args.batch, modulus=args.modulus),
          build_command=argv, cpu_prepare_command=cpu_argv,
          file_hashes={str(p): sha(p) for p in sorted(set(files)) if p.is_file()},
          metrics=METRICS, tasks=tasks, protocol=dict(warmup=100, repeat=100, samples=7,
          independent_processes=3, timing='warm CUDA Graph complete call; CPU checking/setup excluded'),
          created=datetime.now().astimezone().isoformat()))
    print(root / 'campaign.json')

def task(path, key):
    data = load(path)
    cfg = next(t for t in data['tasks'] if t['id'] == key)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != data['gpu_uuid'] or not os.environ.get('XD_REPORT_DIR'):
        raise ValueError('task requires exclusive guard with matching UUID and XD_REPORT_DIR')
    out = Path(os.environ['XD_REPORT_DIR'])
    argv = [data['binary'], '--prepared', data['prepared'], '--core', cfg['core'],
            '--local-log', str(cfg['local_log']), '--organization', cfg['organization'],
            '--warmup', '100', '--repeat', '100', '--samples', '7', '--output', str(out / 'result.json')]
    if cfg['kind'] == 'ncu':
        argv.append('--profile-one-call')
        argv = ['ncu', '--target-processes', 'all', '--profile-from-start', 'off',
                '--replay-mode', 'kernel', '--graph-profiling', 'node', '--cache-control', 'none',
                '--clock-control', 'none', '--kernel-name-base', 'demangled', '--metrics',
                ','.join(data['metrics']), '--csv', '--page', 'raw', '--print-units', 'base',
                '--log-file', str(out / 'ncu.csv'), '--export', str(out / 'capture')] + argv
    gpu_info = subprocess.check_output(['nvidia-smi', '--id=' + data['gpu_uuid'],
          '--query-gpu=uuid,name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
    if 'A100' not in gpu_info: raise ValueError('Figure 11 campaign requires an A100 target')
    write(out / 'task.json', dict(task=cfg, argv=argv, campaign_sha256=sha(path), gpu=gpu_info,
          normal_timing_accepted=cfg['kind'] == 'ordinary'))
    with (out / 'stdout.txt').open('w') as stdout, (out / 'stderr.txt').open('w') as stderr:
        proc = subprocess.run(argv, stdout=stdout, stderr=stderr, timeout=900)
    log = '\n'.join(p.read_text(errors='replace') for p in (out / 'stdout.txt', out / 'stderr.txt', out / 'ncu.csv') if p.exists())
    if cfg['kind'] == 'ncu' and ('ERR_NVGPUCTRPERM' in log or 'permission' in log.lower() and 'counter' in log.lower()):
        write(out / 'blocked.json', dict(task=key, status='profile-blocked', reason='NCU counter permission',
              campaign_sha256=sha(path), normal_timing_accepted=False))
        return
    if proc.returncode: raise RuntimeError(f'task failed: {key}; returncode={proc.returncode}')
    result = json.loads((out / 'result.json').read_text())
    if result.get('correctness_exact_mismatches') != 0: raise ValueError('exact correctness missing or failed')
    if cfg['kind'] == 'ordinary':
        samples = result['samples_us']
        if len(samples) != 7 or any(not math.isfinite(x) or x <= 0 for x in samples):
            raise ValueError('expected seven positive finite ordinary samples')
    else:
        raw = (out / 'ncu.csv').read_text()
        if '"ID"' not in raw: raise ValueError('NCU metric rows missing')
        rows = list(csv.DictReader(io.StringIO(raw[raw.index('"ID"'):])))
        if not rows: raise ValueError('empty NCU capture')
        write(out / 'capture_scope.json', dict(timing='diagnostic only', kernel_ids=sorted({r['ID'] for r in rows}),
              requested_metrics=data['metrics'], unavailable_rows=[r for r in rows if r.get('Metric Value') in ('n/a', 'N/A', '')]))
    write(out / 'complete.json', dict(task=key, campaign_sha256=sha(path), gpu_uuid=data['gpu_uuid'],
          completed=datetime.now().astimezone().isoformat(), normal_timing_accepted=cfg['kind'] == 'ordinary'))

def run(args):
    path = args.campaign.resolve()
    data = load(path)
    if not args.until or datetime.fromisoformat(args.until).tzinfo is None:
        raise ValueError('run requires --until with timezone')
    root = path.parent
    for cfg in data['tasks']:
        report = root / 'jobs' / cfg['id']
        release = report / 'release'
        for filename in ('complete.json', 'blocked.json'):
            receipt = release / filename
            if receipt.exists() and json.loads(receipt.read_text())['campaign_sha256'] != sha(path):
                raise ValueError('receipt belongs to different campaign')
        if (release / 'complete.json').exists() or (release / 'blocked.json').exists(): continue
        argv = [sys.executable, str(ROOT / 'scripts/run_isolated_gpu.py'), '--gpu-uuid', data['gpu_uuid'],
                '--report-dir', str(report), '--until', args.until, '--idle-seconds', '3', '--',
                sys.executable, str(Path(__file__).resolve()), 'task', '--campaign', str(path), '--task', cfg['id']]
        write(root / 'status.json', dict(state='running', task=cfg['id'], deadline=args.until))
        with (root / 'guard.log').open('a') as log:
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write(root / 'status.json', dict(state='stopped', task=cfg['id'], returncode=result.returncode))
            return result.returncode
    rows, blocked = [], []
    for cfg in data['tasks']:
        folder = root / 'jobs' / cfg['id'] / 'release'
        if (folder / 'blocked.json').exists(): blocked.append(cfg['id']); continue
        result = json.loads((folder / 'result.json').read_text())
        rows.append(dict(task=cfg, result=result))
    write(root / 'summary.json', dict(status='timing-complete/profile-blocked' if blocked else 'complete',
          campaign_sha256=sha(path), results=rows, profile_blocked=blocked))
    write(root / 'status.json', dict(state='timing-complete/profile-blocked' if blocked else 'complete',
          tasks=len(data['tasks']), profile_blocked=blocked))
    return 0

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['prepare', 'run', 'task'])
    p.add_argument('--output', type=Path)
    p.add_argument('--vendors', type=Path, default=ROOT / 'vendors')
    p.add_argument('--gpu-uuid')
    p.add_argument('--nvcc', default='nvcc')
    p.add_argument('--log-n', type=int, choices=[16], default=16, help='fixed paper control uses N=2^16')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--modulus', type=int, default=576460756061519873)
    p.add_argument('--campaign', type=Path)
    p.add_argument('--task')
    p.add_argument('--until')
    a = p.parse_args()
    if a.phase == 'prepare':
        if not a.output or not a.gpu_uuid: p.error('prepare requires --output and --gpu-uuid')
        prepare(a)
    elif a.phase == 'task':
        if not a.campaign or not a.task: p.error('task requires --campaign and --task')
        task(a.campaign.resolve(), a.task)
    else:
        if not a.campaign: p.error('run requires --campaign')
        return run(a)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
