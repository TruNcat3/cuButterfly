#!/usr/bin/env python3
"""Per-variant full-process diagnostics, separate from ordinary timing."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_controlled_experiment as controlled
from run_comprehensive_suite import require_exclusive_gpu

METRICS = ','.join([
    'gpu__time_duration.sum', 'dram__bytes_read.sum', 'dram__bytes_write.sum',
    'lts__t_sector_hit_rate.pct',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
    'smsp__warp_issue_stalled_barrier_per_warp_active.pct',
    'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
    'launch__registers_per_thread', 'launch__shared_mem_per_block',
])


def _safe(path):
    path = Path(path).absolute()
    resolved = path.resolve()
    if resolved == Path('/sdb') or Path('/sdb') in resolved.parents:
        raise ValueError(f'forbidden /sdb path: {path}')
    return path


def _sha256(path):
    digest = hashlib.sha256()
    with _safe(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    path = _safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _safe(path.with_suffix(path.suffix + '.tmp'))
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def _command(variant, case=None, protocol=None):
    command = (controlled._timed_command(case, variant, protocol) if case is not None
               else list(variant['timed_command']))
    for flag, value in (('--warmup', '0'), ('--repeat', '1')):
        index = command.index(flag)
        command[index + 1] = value
    return command


def _selected_cases(journal, requested):
    cases = journal['cases']
    ids = [case['id'] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate journal case IDs')
    unknown = set(requested or []) - set(ids)
    if unknown:
        raise ValueError('unknown --case: ' + ', '.join(sorted(unknown)))
    selected = [case for case in cases if not requested or case['id'] in requested]
    if not selected:
        raise ValueError('empty case selection')
    for case in selected:
        names = [variant['id'] for variant in case['variants']]
        if not names or len(names) != len(set(names)):
            raise ValueError(f'empty or duplicate variants: {case["id"]}')
        for name in [case['id'], *names]:
            if Path(name).name != name or name in ('.', '..'):
                raise ValueError('case/variant ID must be one path component')
    return selected


def _validate_measured(journal, cases, gpu_uuid):
    if journal.get('status') != 'complete' or journal.get('completed') is not True:
        raise ValueError('profiling requires a complete measured journal')
    identity = journal['identity']
    if str(identity.get('gpu_uuid', '')).lower() != gpu_uuid.lower():
        raise ValueError('journal GPU UUID differs from selected GPU')
    protocol = journal['protocol']
    if identity.get('protocol') != protocol:
        raise ValueError('journal identity/protocol mismatch')
    for case in cases:
        if case.get('status') != 'complete':
            raise ValueError(f'incomplete case: {case["id"]}')
        for variant in case['variants']:
            preflight = variant.get('preflight', {})
            if preflight.get('status') != 'accepted':
                raise ValueError(f'missing accepted preflight: {case["id"]}/{variant["id"]}')
            expected = controlled._expected_semantics(case['workload'])
            mapping, _ = controlled._validate_sample(preflight.get('sample'), case['workload'], expected,
                                                     identity['device'], require_correctness=True)
            fingerprint = preflight.get('runtime_fingerprint')
            if mapping != preflight.get('resolved_mapping') or preflight['sample']['runtime_fingerprint'] != fingerprint:
                raise ValueError('preflight mapping/fingerprint mismatch')
            trials = variant.get('trials', [])
            if len(trials) != protocol['trials'] or {row['trial'] for row in trials} != set(range(protocol['trials'])):
                raise ValueError('incomplete measured trials')
            for trial in trials:
                if trial.get('status') != 'measured' or trial.get('performance_accepted') is not True:
                    raise ValueError('unsuccessful measured trial')
                trial_mapping, _ = controlled._validate_sample(trial.get('sample'), case['workload'], expected,
                                                               identity['device'], require_correctness=False)
                if trial_mapping != mapping or trial.get('resolved_mapping') != mapping or trial['sample']['runtime_fingerprint'] != fingerprint:
                    raise ValueError('measured mapping/fingerprint mismatch')
                if not controlled._finite_kernel(trial['sample']):
                    raise ValueError('invalid ordinary trial timing')


def _binary_identity(journal, case, require_measured):
    binary = _safe(case['binary'])
    registered = journal['identity']['binaries'].get(binary.name, {})
    if _safe(registered.get('path', '')) != binary:
        raise ValueError('journal binary path mismatch')
    actual = _sha256(binary)
    recorded = registered.get('sha256')
    if (recorded is not None and recorded != actual) or (require_measured and not recorded):
        raise ValueError(f'journal binary hash mismatch: {binary}')
    return {'path': str(binary), 'actual_sha256': actual, 'journal_sha256': recorded,
            'matches_measured_identity': recorded == actual}


def _profiler_commands(command, directory):
    return {
        'nsys': ['nsys', 'profile', '--trace=cuda', '--cuda-memory-usage=true',
                 '--force-overwrite=true', '-o', str(directory / 'nsys')] + command,
        'ncu': ['ncu', '--target-processes', 'all', '--replay-mode', 'kernel',
                '--cache-control', 'none', '--clock-control', 'base', '--metrics', METRICS,
                '--page', 'raw', '--csv', '--force-overwrite', '--log-file',
                str(directory / 'ncu.csv')] + command,
    }


def _tool_identity(tool, run):
    path = shutil.which(tool)
    value = {'path': str(_safe(path)) if path else None, 'sha256': _sha256(path) if path else None,
             'version': None, 'version_status': 'not_queried_cpu_preparation'}
    if run:
        try:
            result = subprocess.run([path or tool, '--version'], capture_output=True, text=True)
            value.update(version=result.stdout.strip(), version_stderr=result.stderr.strip(),
                         version_status='queried', version_returncode=result.returncode)
        except OSError as error:
            value.update(version_status='unavailable', reason=str(error))
    return value


def _guard(gpu_uuid):
    original = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    try:
        require_exclusive_gpu()
    finally:
        if original is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = original


def _execute(command, gpu_uuid, artifact):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid)
    row = {}
    try:
        _guard(gpu_uuid)
        result = subprocess.run(command, env=env, text=True, capture_output=True, cwd=ROOT)
        row = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
        _guard(gpu_uuid)
        row['status'] = 'succeeded' if result.returncode == 0 and artifact.is_file() and artifact.stat().st_size else 'failed'
        if row['status'] == 'failed' and not result.returncode:
            row['reason'] = 'profiler returned success without a nonempty expected artifact'
        return row
    except Exception as error:
        return dict(row, status='failed', reason=str(error))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--case', action='append', help='Select existing case IDs; repeat for a subset')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    if args.run and not args.gpu_uuid:
        parser.error('--run requires --gpu-uuid')
    journal = json.loads(_safe(args.journal).read_text())
    cases = _selected_cases(journal, args.case)
    if args.run:
        _validate_measured(journal, cases, args.gpu_uuid)
    output = _safe(args.output_dir)
    binaries = {case['binary']: _binary_identity(journal, case, args.run) for case in cases}
    manifest = {
        'schema': 'cubutterfly-mechanism-profile-v2', 'status': 'running' if args.run else 'prepared',
        'source_journal': str(args.journal.resolve()), 'source_journal_sha256': _sha256(args.journal),
        'source_journal_status': journal.get('status'), 'source_identity': journal['identity'],
        'source_protocol': journal['protocol'], 'gpu_uuid': args.gpu_uuid, 'gpu_queried': bool(args.run),
        'metrics': METRICS.split(','), 'profiler_timing_is_diagnostic': True,
        'ordinary_timing_source': str(args.journal.resolve()),
        'capture_scope': 'complete process; setup and teardown kernels may be included; no operator-only range',
        'aggregation': 'none; retain individual kernel metrics; no performance inference from profiler replay',
        'runner_sha256': _sha256(__file__), 'controlled_runner_sha256': _sha256(controlled.__file__),
        'python': platform.python_version(),
        'profilers': {tool: _tool_identity(tool, args.run) for tool in ('nsys', 'ncu')},
        'selected_case_count': len(cases), 'population_case_count': len(journal['cases']),
        'full_population': len(cases) == len(journal['cases']), 'variants': [], 'failures': [],
    }
    for case in cases:
        for variant in case['variants']:
            directory = _safe(output / case['id'] / variant['id'])
            command = _command(variant, case, journal['protocol'])
            command[0] = binaries[case['binary']]['path']
            manifest['variants'].append({
                'case': case['id'], 'id': variant['id'], 'status': 'pending' if args.run else 'prepared',
                'command': command, 'commands': _profiler_commands(command, directory),
                'binary': binaries[case['binary']], 'output_directory': str(directory),
                'requires_complete_measured_journal': True,
            })
    _write(output / 'manifest.json', manifest)
    try:
        for row in manifest['variants']:
            directory = Path(row['output_directory'])
            directory.mkdir(parents=True, exist_ok=True)
            _write(directory / 'manifest.json', row)
            if not args.run:
                continue
            # Recheck the exact executable before each independent profiler process.
            for tool, artifact in (('nsys', directory / 'nsys.nsys-rep'), ('ncu', directory / 'ncu.csv')):
                if _sha256(row['binary']['path']) != row['binary']['actual_sha256']:
                    result = {'status': 'failed', 'reason': 'binary changed after measured identity validation'}
                else:
                    result = _execute(row['commands'][tool], args.gpu_uuid, artifact)
                row[tool] = result
                if result['status'] != 'succeeded':
                    manifest['failures'].append({'case': row['case'], 'variant': row['id'], 'profiler': tool,
                                                 'reason': result.get('reason', 'profiler command failed')})
                _write(directory / 'manifest.json', row)
                _write(output / 'manifest.json', manifest)
            row['status'] = 'complete' if all(row[tool]['status'] == 'succeeded' for tool in ('nsys', 'ncu')) else 'failed'
            _write(directory / 'manifest.json', row)
            _write(output / 'manifest.json', manifest)
    except BaseException:
        manifest['status'] = 'interrupted'
        _write(output / 'manifest.json', manifest)
        raise
    if args.run:
        manifest['status'] = 'failed' if manifest['failures'] else 'complete'
    _write(output / 'manifest.json', manifest)
    print(json.dumps({'status': manifest['status'], 'cases': len(cases), 'variants': len(manifest['variants']), 'run': args.run}))
    return int(manifest['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
