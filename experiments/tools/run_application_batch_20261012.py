#!/usr/bin/env python3
"""Finite application batches: untimed public tuning then cached default calls."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/paper_targeted_20261012/application_batch'
LIB = ROOT / 'results/search_budget_20261011/frozen_runtime/libcuntt.so'
PYTHON = '/home/wt/yes/envs/cubutterfly/bin/python'
TARGETS = {
    'fft_filter': ('GPU-0eae5f07-33de-321b-70a9-ffeae49309e6',
                   ROOT/'results/plan_batch_trace_20261010/recalibration_gpu2_finalized_bound_20261011/calibration_manifest.json'),
    'quarot': ('GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02',
               ROOT/'results/parallel_resume_20261011_gpu1/r2/calibration/calibration_manifest.json'),
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, document):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(document, indent=2)+'\n')
    temporary.replace(path)


def fingerprint(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def input_identity(lane):
    if lane != 'fft_filter':
        return None
    manifest_path = OUT/'fft_filter/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    artifacts = {}
    for contract in manifest['contracts']:
        path = (manifest_path.parent/contract['artifact']).resolve()
        if not path.is_relative_to(manifest_path.parent.resolve()):
            raise RuntimeError('FFT artifact escapes prepared input directory')
        actual = digest(path)
        if actual != contract['sha256']:
            raise RuntimeError(f'FFT prepared artifact hash mismatch: {path}')
        artifacts[str(path)] = actual
    return dict(manifest_path=str(manifest_path), manifest_sha256=digest(manifest_path), artifacts=artifacts)


def effective_environment(calibration, uuid, xd_tune_max_bytes):
    env = dict(os.environ, CUBUTTERFLY_COMPILE_MODE='research',
               CUBUTTERFLY_REGISTRY=str(Path(calibration['registry_path']).resolve()),
               CUBUTTERFLY_XD_TUNE_MAX_BYTES=str(xd_tune_max_bytes),
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
               CUBUTTERFLY_SEARCH_BUDGET='512', CUDA_VISIBLE_DEVICES=uuid)
    env.pop('XD_REPORT_DIR', None)
    env.pop('CUBUTTERFLY_PLAN_CACHE', None)
    env.pop('CUBUTTERFLY_E13_ALGORITHM_POLICY', None)
    jit = calibration['measurement_identity']['jit_runtime']
    env.update(LD_LIBRARY_PATH=str(LIB.parent)+':'+jit['ld_library_path'],
               CUBUTTERFLY_TEMPLATE_ROOT=jit['config']['root'], CUBUTTERFLY_JIT_CACHE=jit['config']['cache'],
               CUBUTTERFLY_MATHDX_ROOT=jit['config']['mathdx'], CUBUTTERFLY_NVCC=jit['config']['nvcc'])
    names = {'PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'CUDA_HOME',
             'CUDA_PATH', 'CUDA_MODULE_LOADING', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'}
    identity = {k: v for k, v in env.items() if k in names or k.startswith('CUBUTTERFLY_')}
    identity['CUBUTTERFLY_PLAN_CACHE'] = '<guard-attempt>/plan_cache.plans'
    identity['CUBUTTERFLY_E13_ALGORITHM_POLICY'] = '<job-policy>'
    return env, identity


def cache_child(argv):
    """Each guard retry gets a fresh cache, never a failed attempt's writes."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-seed', type=Path)
    parser.add_argument('--policy', choices=('measure', 'default'), required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    attempt = Path(os.environ['XD_REPORT_DIR']).resolve()
    cache = attempt/'plan_cache.plans'
    if cache.exists():
        raise RuntimeError('guard attempt cache already exists')
    if args.cache_seed:
        shutil.copyfile(args.cache_seed, cache)
    elif args.policy == 'default':
        raise RuntimeError('DEFAULT requires a successfully promoted tuning cache')
    before = digest(cache) if cache.exists() else None
    result = subprocess.run(command, cwd=ROOT, env=dict(os.environ,
                            CUBUTTERFLY_PLAN_CACHE=str(cache), CUBUTTERFLY_E13_ALGORITHM_POLICY=args.policy))
    after = digest(cache) if cache.is_file() else None
    if result.returncode:
        return result.returncode
    if after is None or cache.stat().st_size == 0:
        raise RuntimeError('successful runner did not produce a persistent plan cache')
    if args.policy == 'default' and before != after:
        raise RuntimeError('DEFAULT changed its frozen cache copy; attempt is not publishable')
    save(attempt/'cache_receipt.json', dict(policy=args.policy, seed_sha256=before, cache_sha256=after))
    return 0


def release_hashes(release):
    return {str(p.relative_to(release)): digest(p) for p in sorted(release.rglob('*')) if p.is_file()}


def accept_release(record, spec_hash, seed):
    """Only a clean guard release can create a promoted cache and receipt."""
    report_dir = Path(record['report_dir'])
    release = report_dir/'release'
    status = json.loads((report_dir/'status.json').read_text())
    if status.get('state') != 'complete' or Path(status.get('release', '')).resolve() != release.resolve():
        raise RuntimeError('guard did not leave a successful release')
    child = json.loads((release/'cache_receipt.json').read_text())
    seed_hash = digest(seed) if seed else None
    if child['policy'] != record['policy'] or child['seed_sha256'] != seed_hash:
        raise RuntimeError('released cache seed or policy mismatch')
    cache = release/'plan_cache.plans'
    if digest(cache) != child['cache_sha256']:
        raise RuntimeError('released cache identity mismatch')
    if record['policy'] == 'default' and child['cache_sha256'] != seed_hash:
        raise RuntimeError('measurement rewrote its frozen cache')
    promoted = report_dir/'promoted.plans' if record['policy'] == 'measure' else seed
    if record['policy'] == 'measure':
        if promoted.exists():
            raise RuntimeError('refusing to overwrite an existing promoted cache')
        with tempfile.NamedTemporaryFile(dir=report_dir, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        shutil.copyfile(cache, temporary_path)
        temporary_path.replace(promoted)
    receipt = dict(status='success', spec_sha256=spec_hash, job_sha256=fingerprint(record),
                   seed_sha256=seed_hash, cache_sha256=child['cache_sha256'],
                   promoted_path=str(promoted), release_hashes=release_hashes(release))
    save(report_dir/'receipt.json', receipt)
    return promoted


def reuse_release(record, spec_hash, seed):
    report_dir = Path(record['report_dir'])
    receipt = json.loads((report_dir/'receipt.json').read_text())
    expected_seed = digest(seed) if seed else None
    if (receipt.get('status') != 'success' or receipt['spec_sha256'] != spec_hash or
        receipt['job_sha256'] != fingerprint(record) or receipt['seed_sha256'] != expected_seed or
        receipt['release_hashes'] != release_hashes(report_dir/'release')):
        raise RuntimeError('existing release receipt does not match this immutable cohort')
    status = json.loads((report_dir/'status.json').read_text())
    if status.get('state') != 'complete':
        raise RuntimeError('existing release lacks successful guard status')
    promoted = Path(receipt['promoted_path'])
    if digest(promoted) != receipt['cache_sha256']:
        raise RuntimeError('existing promoted/frozen cache changed')
    return promoted


def check_fixed_inputs(spec):
    files = {ROOT/relative: expected for relative, expected in spec['source_hashes'].items()}
    files.update({LIB: spec['library_sha256'], Path(spec['registry_path']): spec['registry_sha256'],
                  Path(spec['calibration_path']): spec['calibration_sha256']})
    if spec['input_identity']:
        files[Path(spec['input_identity']['manifest_path'])] = spec['input_identity']['manifest_sha256']
    for path, expected in files.items():
        if digest(path) != expected:
            raise RuntimeError(f'frozen cohort input changed: {path}')


def jobs(lane):
    if lane == 'fft_filter':
        common = [PYTHON, str(ROOT/'experiments/workloads/e13/run_fft_filter_end_to_end.py'),
                  '--run', '--output', str(OUT/'fft_filter'), '--library', str(LIB),
                  '--calibration-identity', str(TARGETS[lane][1]), '--torch-control',
                  '--warmup', '20', '--repeats', '20', '--samples', '7']
        yield 'fft_filter/tune_guard', 'measure', common + ['--plan-only', '--run-id', 'tune']
        for number in (1,2,3):
            yield f'fft_filter/process{number}_guard', 'default', common + [
                '--run-id', f'process{number}', '--process-index', str(number), '--require-cache']
    else:
        for batch in (1,4,8):
            common = [PYTHON, str(ROOT/'experiments/workloads/e13/run_quarot_random_end_to_end.py'),
                      '--library', str(LIB), '--attention-backend', 'torch_sdpa',
                      '--batch-size', str(batch), '--prefill', '512', '--decode', '32',
                      '--compute-contract', 'fp32_composed', '--atol', '0', '--rtol', '0',
                      '--warmup', '5', '--calls', '10']
            # Relative output is resolved under the guard's attempt directory.
            yield f'quarot_b{batch}_tune', 'measure', common + ['--validation-only', '--output', 'result.json']
            for number in (1,2):
                yield f'quarot_b{batch}_p{number}', 'default', common + [
                    '--variant-order', 'baseline-first' if number == 1 else 'treatment-first',
                    '--output', 'result.json', '--require-cache']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane', choices=tuple(TARGETS), required=True)
    parser.add_argument('--revision', required=True, help='New immutable cohort below application_batch; v1 files stay in place')
    parser.add_argument('--xd-tune-max-bytes', type=int, default=512*1024**2,
                        help='Explicit xD selection allocation budget, recorded in the immutable plan')
    parser.add_argument('--until', required=True, help='finite ISO deadline with timezone')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if datetime.fromisoformat(args.until).tzinfo is None:
        parser.error('--until needs timezone')
    if args.xd_tune_max_bytes < 0:
        parser.error('--xd-tune-max-bytes must be nonnegative')
    if not args.revision or args.revision in ('.', '..', 'fft_filter') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.revision):
        parser.error('--revision must be a simple new cohort name')
    output = OUT/args.revision
    output.mkdir(parents=True, exist_ok=True)
    uuid, calibration_path = TARGETS[args.lane]
    calibration = json.loads(calibration_path.read_text())
    if calibration['measurement_identity']['library_sha256'] != digest(LIB):
        raise RuntimeError('library/calibration identity mismatch')
    env, environment = effective_environment(calibration, uuid, args.xd_tune_max_bytes)
    registry = Path(env['CUBUTTERFLY_REGISTRY'])
    records = []
    for name, policy, command in jobs(args.lane):
        records.append(dict(id=name, policy=policy, command=command,
                            report_dir=str(output/name), phase='tuning' if policy=='measure' else 'measurement'))
    spec = dict(schema='cubutterfly-application-batch-v2', lane=args.lane, revision=args.revision,
                gpu_uuid=uuid, cache_policy='attempt-local; promote successful tuning only; hash-frozen measurement copy',
                xd_tune_max_bytes=args.xd_tune_max_bytes, library_sha256=digest(LIB),
                calibration_path=str(calibration_path), calibration_sha256=digest(calibration_path),
                registry_path=str(registry), registry_sha256=digest(registry),
                input_identity=input_identity(args.lane), effective_environment=environment,
                effective_environment_sha256=fingerprint(environment),
                calibration_usage='rank1 registry for compatible keys; xD measured whole plans use explicit persistent cache',
                jobs=records, source_hashes={str(p.relative_to(ROOT)):digest(p) for p in [
                    Path(__file__), ROOT/'scripts/run_isolated_gpu.py', ROOT/'experiments/workloads/e13/run_fft_filter_end_to_end.py',
                    ROOT/'results/e13_assets/cubutterfly_real_fft.py',
                    ROOT/'experiments/workloads/e13/quarot_adapter.py',
                    ROOT/'experiments/workloads/e13/run_quarot_random_end_to_end.py']})
    spec_hash = fingerprint(spec)
    spec_path = output/f'{args.lane}_plan.json'
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise RuntimeError('prepared plan identity changed; preserve prior cohort and choose a new revision')
    save(spec_path, spec)
    if not args.run:
        print(json.dumps(dict(status='prepared', lane=args.lane, revision=args.revision,
                              plan=str(spec_path), spec_sha256=spec_hash, jobs=len(records))))
        return 0
    state_path = output/f'{args.lane}_state.json'
    state = dict(status='running', gpu_uuid=uuid, plan=str(spec_path), steps=[])
    seed = None
    for record in records:
        check_fixed_inputs(spec)
        report_dir = Path(record['report_dir'])
        if (report_dir/'release').exists():
            seed = reuse_release(record, spec_hash, seed)
            state['steps'].append(dict(id=record['id'], status='existing-release'))
            continue
        state.update(current=record['id'], status='running')
        save(state_path, state)
        child = [PYTHON, str(Path(__file__).resolve()), '--cache-child', '--policy', record['policy']]
        if seed:
            child += ['--cache-seed', str(seed)]
        command = [PYTHON, str(ROOT/'scripts/run_isolated_gpu.py'), '--gpu-uuid', uuid,
                   '--report-dir', str(report_dir), '--until', args.until, '--idle-seconds', '3', '--']+child+['--']+record['command']
        with (output/f'{args.lane}_queue.log').open('a') as log:
            result = subprocess.run(command, cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        state['steps'].append(dict(id=record['id'], returncode=result.returncode))
        if result.returncode:
            state['status']='stopped-on-failure-or-deadline'
            save(state_path,state)
            return result.returncode
        check_fixed_inputs(spec)
        seed = accept_release(record, spec_hash, seed)
    state['status']='complete'
    save(state_path,state)
    return 0


if __name__ == '__main__':
    raise SystemExit(cache_child(sys.argv[2:]) if sys.argv[1:2] == ['--cache-child'] else main())
