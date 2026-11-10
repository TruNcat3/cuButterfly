#!/usr/bin/env python3
"""CPU external libraries for an unchanged semantic comparison population.

Floating outputs require accuracy review, without a pass threshold. CPU and
GPU measurements occupy separate columns, with no pooled speedup statistic.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import run_semantic_campaign as semantic
import semantic_preparation_queue as preparation
from semantic_campaign_manifest import validate_manifest_hash

REVISION = 'cpu-external-v2-error-report'
NAMES = {0: 'fft', 1: 'ntt', 2: 'fwht', 3: 'subset-zeta', 4: 'superset-zeta', 5: 'structured-2x2'}
STORAGE = {2: 'fp32', 3: 'fp64', 6: 'complex64', 7: 'complex128', 8: 'uint32', 9: 'uint64'}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def same_precision_reference(s, source, integer_reference=None):
    """Original precision CPU oracle; historic wide references remain diagnostics."""
    if s['storage'] in (8, 9):
        expected = semantic.reference(s, source) if integer_reference is None else integer_reference
        return np.asarray(expected), {'kind': 'exact-integer', 'threshold': None}
    if s['storage'] not in (2, 3, 6, 7):
        raise ValueError('no same-precision CPU reference for requested storage')
    dtype = semantic.side_dtype(s, 'input')
    compute = 2 if s['storage'] in (3, 7) else 1
    if s['compute'] not in (0, compute) or source.dtype != dtype:
        raise ValueError('storage/compute must match; no implicit widening')
    if s['operator'] == 0:
        if s['fft_type'] != 0 or s['length_mode']:
            raise ValueError('CPU comparison currently requires standard C2C FFT')
        import scipy
        from scipy import fft
        axes = tuple(range(1, source.ndim))
        work = (fft.ifftn(source, axes=axes, norm='backward' if s['normalize_inverse'] else 'forward', workers=1)
                if s['inverse'] else fft.fftn(source, axes=axes, workers=1))
        np.multiply(work, np.asarray(s['fft_output_scale'], dtype=work.real.dtype), out=work)
        if work.dtype != dtype:
            raise ValueError('SciPy FFT changed requested precision')
        return work, {'kind': 'scipy-pocketfft', 'version': scipy.__version__, 'dtype': str(dtype), 'threshold': None}
    if s['operator'] not in (2, 5):
        raise ValueError('unsupported floating CPU reference operator')
    work = source.copy()
    for axis, n in enumerate(s['shape'], 1):
        moved = np.moveaxis(work, axis, -1).copy()
        rows = moved.reshape(-1, n)
        half, stage = 1, 0
        while half < n:
            blocks = rows.reshape(-1, 2 * half)
            left, right = blocks[:, :half].copy(), blocks[:, half:].copy()
            if s['operator'] == 2:
                blocks[:, :half], blocks[:, half:] = left + right, left - right
            else:
                matrices = s['matrices'][axis - 1]
                matrix = np.asarray(matrices[0 if len(matrices) == 1 else stage], dtype=np.float64).reshape(2, 2)
                if s['inverse']:
                    a, b, c, d = matrix.ravel()
                    matrix = np.asarray([[d, -b], [-c, a]]) / (a*d - b*c)
                matrix = matrix.astype(dtype)
                blocks[:, :half] = matrix[0, 0] * left + matrix[0, 1] * right
                blocks[:, half:] = matrix[1, 0] * left + matrix[1, 1] * right
            half *= 2
            stage += 1
        if s['operator'] == 2 and s['inverse'] and s['normalize_inverse']:
            np.multiply(rows, dtype.type(1.0 / n), out=rows)
        work = np.moveaxis(rows.reshape(moved.shape), -1, axis)
    return work, {'kind': 'independent-same-precision-stage-reference', 'dtype': str(dtype),
                  'numpy': np.__version__, 'threshold': None}


def measure(core, source, expected, s, warmup, trials, calls):
    saved = source.copy()
    def verify():
        if not np.array_equal(source.view(np.uint8), saved.view(np.uint8)):
            raise AssertionError('out-of-place input modified')
        if core.output.dtype != semantic.side_dtype(s, 'output'):
            raise AssertionError('output storage changed')
        if core.output.shape != source.shape:
            raise AssertionError('output shape changed')
        return semantic.check_full(core.output, expected, s['storage'], 'error-report', False)
    core.execute()
    verify()
    for _ in range(warmup):
        core.execute()
    samples = []
    for _ in range(trials):
        core.output.view(np.uint8).fill(0xA5)
        start = time.perf_counter_ns()
        for _ in range(calls):
            core.execute()
        samples.append((time.perf_counter_ns() - start) / 1e6 / calls)
        verify()
    return {'median_ms': statistics.median(samples), 'samples_ms': samples,
            'protocol': f'{trials} wall-clock samples x {calls} complete calls; {warmup} warmup',
            'includes': 'CPU dispatch, full transform, required copies/conversions/normalization and allocations inside execute',
            'excludes': 'planning, compilation, setup allocation, input preparation and reference validation'}


def backend_specs(s, threads):
    if s['operator'] == 0:
        return [('fftw-cpu-t' + str(t), 'cpu_fftw_baseline', 'FFTWCore', t) for t in threads]
    if s['operator'] in (2, 3, 4):
        return [('fxt-cpu-t1', 'cpu_fxt_baseline', 'FXTCore', 1)]
    if s['operator'] == 1:
        return [('hexl-cpu-t1', 'cpu_ntt_baseline', 'NTTCore', 1)]
    return [('butterfly-author-cpu-t1', 'cpu_structured_baseline', 'StructuredCore', 1)]


def construct(spec, s, source, args):
    _, module_name, cls_name, threads = spec
    cls = getattr(importlib.import_module(module_name), cls_name)
    if module_name == 'cpu_fxt_baseline':
        return cls(s, source, args.fxt_library, threads=threads)
    if module_name == 'cpu_fftw_baseline':
        return cls(s, source, threads=threads)
    return cls(s, source, root=args.hexl_root if module_name == 'cpu_ntt_baseline' else args.butterfly_root)


def host_identity():
    return {'hostname': platform.node(), 'cpu': next((line.split(':', 1)[1].strip()
            for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), ''),
            'affinity': sorted(os.sched_getaffinity(0)), 'python': sys.version, 'numpy': np.__version__,
            'cpu_exclusive': False, 'limitation': 'shared host; CPU timings provisional pending uncontended repeat'}


def comparison_source(manifest, source_manifest=None):
    """Permit explicit reuse only when build fingerprints are the sole change."""
    if source_manifest is None:
        return manifest
    for value in (manifest, source_manifest):
        if not validate_manifest_hash(value):
            raise ValueError('comparison manifest content checksum mismatch')
    def content(value):
        return {key: item for key, item in value.items()
                if key not in ('fingerprints', 'manifest_sha256')}
    if content(manifest) != content(source_manifest):
        raise ValueError('comparison source manifest semantic content mismatch')
    return source_manifest


def comparison_preparation(manifest, directory, source_manifest=None):
    source = comparison_source(manifest, source_manifest)
    config, checksum = preparation._config(directory)
    if config['manifest_sha256'] != semantic.digest(source):
        raise ValueError('preparation manifest identity mismatch')
    if config['source_sha256'] != preparation._sources():
        raise ValueError('preparation source identity mismatch')
    return {'path': str(Path(directory).resolve()), 'config_sha256': checksum,
            'manifest_sha256': config['manifest_sha256']}


def export_table(manifest, report, output, gpu_path=None, source_manifest=None):
    """Join exact IDs without modifying frozen GPU rows or mixing platform means."""
    source = comparison_source(manifest, source_manifest)
    if report['identity']['manifest_sha256'] != semantic.digest(source):
        raise ValueError('CPU results belong to another semantic manifest')
    cpu = {(r['case_id'], r['backend']): r for r in report['rows']}
    if len(cpu) != len(report['rows']):
        raise ValueError('duplicate CPU case/backend rows')
    if source_manifest is not None:
        cases = {case['id']: case for case in source['cases']}
        for observation in report['rows']:
            case = cases.get(observation['case_id'])
            if case is None or observation.get('semantics') != case['descriptor']['semantics']:
                raise ValueError('CPU row semantic identity mismatch')
    names = sorted({r['backend'] for r in report['rows']})
    selected = report['identity'].get('selected_case_ids')
    if selected is not None:
        known = {case['id'] for case in manifest['cases']}
        if len(selected) != len(set(selected)) or not set(selected) <= known:
            raise ValueError('invalid CPU request selection')
        if any(row['case_id'] not in selected for row in report['rows']):
            raise ValueError('CPU row outside selected requests')
    gpu = {}
    if gpu_path and Path(gpu_path).exists():
        from summarize_semantic_campaign import load_checkpoint
        gpu_report, rows = load_checkpoint(gpu_path)
        if gpu_report['identity']['manifest_sha256'] != semantic.digest(manifest):
            raise ValueError('GPU results belong to another semantic manifest')
        gpu = {r['id']: r for r in rows}
    gpu_names = ['cubutterfly', 'cufft', 'vkfft', 'daofht', 'gpu-ntt']
    table = []
    for case in manifest['cases']:
        if selected is not None and case['id'] not in selected:
            continue
        s = case['descriptor']['semantics']
        row = dict(case_id=case['id'], population=';'.join(case['populations']), operator=NAMES[s['operator']],
                   precision=STORAGE.get(s['storage'], str(s['storage'])), shape='x'.join(map(str, s['shape'])),
                   batch=s['batch'], direction='inverse' if s['inverse'] else 'forward',
                   input_order=json.dumps(s['input_orders']), output_order=json.dumps(s['output_orders']),
                   cpu_environment='shared-host; provisional', floating_acceptance='error-report-only',
                   comparison_manifest_sha256=semantic.digest(manifest),
                   cpu_source_manifest_sha256=semantic.digest(source))
        native = gpu.get(case['id'])
        observations = {'cubutterfly': native} if native else {}
        observations.update({r['name']: r for r in (native or {}).get('baselines', [])})
        valid = ('pass', 'accuracy-review-required')
        own_ms = (native or {}).get('timing', {}).get('median_ms') if (native or {}).get('status') in valid else None
        for name in gpu_names + names:
            obs = observations.get(name) if name in gpu_names else cpu.get((case['id'], name))
            row[name + '_status'] = obs['status'] if obs else ('pending' if name in gpu_names else 'not-applicable')
            value = (obs or {}).get('timing', {}).get('median_ms') if (obs or {}).get('status') in valid else None
            row[name + '_ms'] = value
            if name in names:
                metadata = (obs or {}).get('metadata', {})
                row[name + '_implementation'] = metadata.get('precision_status', metadata.get('timing_scope', ''))
            if name != 'cubutterfly':
                row[name + ('_gpu_speedup' if name in gpu_names else '_cross_platform_speedup')] = value / own_ms if value and own_ms else None
        table.append(row)
    target = Path(output) / 'unified_comparison.csv'
    tmp = target.with_suffix('.tmp')
    with tmp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    tmp.replace(target)
    write(Path(output) / 'comparison_provenance.json', {
        'comparison_manifest_sha256': semantic.digest(manifest),
        'cpu_source_manifest_sha256': semantic.digest(source),
        'cpu_report_sha256': semantic.digest(report),
        'cpu_identity': report['identity'],
        'cpu_measurements_reused': source_manifest is not None,
        'gpu_manifest_sha256': semantic.digest(manifest) if gpu else None})


def runtime_identity(args, manifest, cases=None):
    """Bind actual libraries and prepared input bytes before resuming rows."""
    import ctypes
    import ctypes.util
    import scipy
    from scipy.fft._pocketfft import pypocketfft
    from cpu_fftw_baseline import _identity
    artifacts = {}
    for stem, prefix in (('fftw3', 'fftw'), ('fftw3f', 'fftwf')):
        for suffix in ('', '_threads'):
            lib = ctypes.CDLL(ctypes.util.find_library(stem + suffix) or ('lib' + stem + suffix + '.so.3'), mode=ctypes.RTLD_GLOBAL)
            symbol = prefix + ('_init_threads' if suffix else '_execute')
            info = _identity(lib, symbol, None)
            artifacts[info['path']] = info['sha256']
    artifacts[str(Path(pypocketfft.__file__).resolve())] = semantic.sha256_file(pypocketfft.__file__)
    for root in (args.hexl_root, args.butterfly_root):
        if root:
            root = Path(root).resolve()
            for pattern in ('*.so', '*.so.*', 'bridge.cpp'):
                for path in root.rglob(pattern):
                    if path.is_file():
                        artifacts[str(path.resolve())] = semantic.sha256_file(path)
    inputs = {}
    for case in manifest['cases'] if cases is None else cases:
        prepared = preparation.load_case(args.preparation, case)
        inputs[case['id']] = hashlib.sha256(prepared['source'].tobytes() + prepared['expected'].tobytes()).hexdigest()
    return {'artifacts': artifacts, 'scipy': scipy.__version__, 'prepared_arrays': inputs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fxt-root', type=Path)
    parser.add_argument('--fxt-library', type=Path)
    parser.add_argument('--hexl-root', type=Path)
    parser.add_argument('--butterfly-root', type=Path)
    parser.add_argument('--fftw-threads', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--trials', type=int, default=7)
    parser.add_argument('--calls', type=int, default=20)
    parser.add_argument('--gpu-results', type=Path)
    parser.add_argument('--cpu-source-manifest', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--join-only', action='store_true')
    parser.add_argument('--case-id', action='append', help='Measure only these exact IDs; retain the full manifest identity')
    args = parser.parse_args(argv)
    if args.warmup < 0 or min(args.trials, args.calls) < 1 or any(t not in (1, 8) for t in args.fftw_threads) or len(set(args.fftw_threads)) != len(args.fftw_threads):
        parser.error('invalid timing/thread protocol')
    manifest = json.loads(args.manifest.read_text())
    if not validate_manifest_hash(manifest):
        raise ValueError('semantic manifest checksum mismatch')
    selected = set(args.case_id or [])
    if args.case_id and (len(selected) != len(args.case_id) or
                         not selected <= {case['id'] for case in manifest['cases']}):
        parser.error('--case-id must name unique requests in the manifest')
    cases = [case for case in manifest['cases'] if not selected or case['id'] in selected]
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / 'results.json'
    if args.join_only:
        source = json.loads(args.cpu_source_manifest.read_text()) if args.cpu_source_manifest else None
        comparison_preparation(manifest, args.preparation, source)
        export_table(manifest, json.loads(report_path.read_text()), args.output, args.gpu_results, source)
        return 0
    if args.cpu_source_manifest:
        parser.error('--cpu-source-manifest requires --join-only; reused measurements cannot be resumed')
    if args.fxt_root:
        from cpu_fxt_baseline import build_library
        args.fxt_library = build_library(args.fxt_root, args.fxt_library or args.output / 'libfxt.so')
    if args.fxt_library:
        args.fxt_library = args.fxt_library.resolve()
    if args.hexl_root is None:
        from cpu_ntt_baseline import ROOT
        args.hexl_root = ROOT
    if args.butterfly_root is None:
        from cpu_structured_baseline import ROOT
        args.butterfly_root = ROOT
    source_names = ('cpu_external_baselines.py', 'cpu_fftw_baseline.py', 'cpu_fxt_baseline.py',
                    'cpu_external_fxt.cpp', 'cpu_ntt_baseline.py', 'cpu_structured_baseline.py',
                    'cpu_external_hexl_bridge.cpp', 'cpu_external_structured_bridge.cpp',
                    'run_semantic_campaign.py', 'semantic_preparation_queue.py', 'semantic_ntt_reference.py')
    identity = {'revision': REVISION, 'manifest_sha256': semantic.digest(manifest), 'host': host_identity(),
                'protocol': dict(warmup=args.warmup, trials=args.trials, calls=args.calls,
                                 fftw_threads=args.fftw_threads, float_verdict='error-report'),
                'sources': {name: semantic.sha256_file(Path(__file__).with_name(name)) for name in source_names
                            if Path(__file__).with_name(name).exists()},
                'fxt_library': str(args.fxt_library),
                'fxt_sha256': semantic.sha256_file(args.fxt_library) if args.fxt_library else None,
                'runtime': runtime_identity(args, manifest, cases)}
    if selected:
        identity['selected_case_ids'] = sorted(selected)
    report = {'schema': 'cubutterfly-cpu-external-v2', 'identity': identity, 'status': 'running', 'rows': [],
              'expected_rows': sum(len(backend_specs(c['descriptor']['semantics'], args.fftw_threads)) for c in cases)}
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if not args.resume or previous['identity'] != identity:
            raise ValueError('CPU output exists; use matching --resume or a fresh directory')
        report = previous
    completed = {(r['case_id'], r['backend']) for r in report['rows']}
    write(report_path, report)
    for case in cases:
        s = case['descriptor']['semantics']
        todo = [spec for spec in backend_specs(s, args.fftw_threads) if (case['id'], spec[0]) not in completed]
        if not todo:
            continue
        prepared = preparation.load_case(args.preparation, case)
        source = prepared['source']
        for spec in todo:
            core = None
            row = {'case_id': case['id'], 'backend': spec[0], 'platform': 'cpu', 'semantics': s,
                   'input_sha256': hashlib.sha256(source.tobytes()).hexdigest(),
                   'status': 'pending', 'loadavg_start': os.getloadavg()}
            try:
                expected, reference = same_precision_reference(s, source, prepared['expected'])
                row['reference'] = reference
                core = construct(spec, s, source, args)
                if hashlib.sha256(source.tobytes()).hexdigest() != row['input_sha256']:
                    raise AssertionError('core construction modified input')
                info = core.metadata
                row['metadata'] = info() if callable(info) else info
                timing = measure(core, source, expected, s, args.warmup, args.trials, args.calls)
                row['error_artifact'] = semantic.retain_error_artifact(args.output / 'errors', case['id'], spec[0], core.output, expected)
                row['correctness'] = semantic.check_full(core.output, expected, s['storage'], 'error-report', True)
                if s['storage'] not in (8, 9):
                    row['historic_high_precision_diagnostic'] = semantic.check_full(core.output, prepared['expected'], s['storage'], 'error-report', False)
                row.update(timing=timing, status='pass' if s['storage'] in (8, 9) else 'accuracy-review-required')
            except (ImportError, OSError) as error:
                row.update(status='missing-external-dependency', reason=str(error))
            except ValueError as error:
                row.update(status='unsupported-semantic', reason=str(error))
            except AssertionError as error:
                row.update(status='correctness-failure', error=str(error))
                if core is not None:
                    row['error_artifact'] = semantic.retain_error_artifact(args.output / 'errors', case['id'], spec[0], core.output, expected)
            except Exception as error:
                row.update(status='baseline-error', error=str(error))
            finally:
                if core is not None:
                    core.close()
            row['loadavg_end'] = os.getloadavg()
            report['rows'].append(row)
            report['updated_unix'] = time.time()
            write(report_path, report)
            export_table(manifest, report, args.output, args.gpu_results)
            print(json.dumps({'backend': spec[0], 'case': case['id'], 'status': row['status'],
                              'completed': len(report['rows']), 'total': report['expected_rows']}), flush=True)
    errors = [r for r in report['rows'] if r['status'] in ('baseline-error', 'correctness-failure')]
    gaps = [r for r in report['rows'] if r['status'] in ('missing-external-dependency', 'unsupported-semantic')]
    report.update(status='complete-with-errors' if errors else 'complete-with-gaps' if gaps else 'complete-pending-floating-review',
                  error_count=len(errors), gap_count=len(gaps), measured_count=sum('timing' in r for r in report['rows']))
    write(report_path, report)
    export_table(manifest, report, args.output, args.gpu_results)
    return 2 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
