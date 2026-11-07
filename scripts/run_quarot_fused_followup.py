#!/usr/bin/env python3
"""Prepare or execute four isolated full-model Graph jobs with one deadline."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'paper/experiments/e13/run_quarot_random_end_to_end.py'
DEFAULT_CONFIG = ROOT / 'configs/experiments/quarot_fused_followup_20261107.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    config = json.loads(Path(path).read_text())
    expected = json.loads(DEFAULT_CONFIG.read_text())
    if config != expected:
        raise ValueError('followup requires the fixed four-job contract; no thresholds or precision changes')
    return config


# hardware_model.hpp resolves the model; rank_static_candidates.py then reads
# these siblings directly (registry records do not dereference evidence paths).
PROFILE_FILES = ('hardware_model.json', 'calibration_device.json', 'hardware_profile.json', 'cost_model.json')
RANKER_INPUTS = ('rank_static_candidates.py', 'stage_cost_model.py', 'static_hardware_profile.py',
                 'stage_schedule_native.py', 'stage_schedule_native.cpp', 'stage_service_model.py',
                 'schedule_cost_model.py', 'hardware_registry.py', 'shared_work_geometry.py', 'resident_mapping.py')


def static_file(path):
    path = Path(path).expanduser().resolve()
    return {'path': str(path), 'sha256': digest(path) if path.is_file() else None}


def profile_tree(path):
    path = Path(path).expanduser().resolve()
    # Preserve absence too: installing a previously absent cost model changes
    # selection just as overwriting an existing model does.
    directories = {path}
    if path.is_dir():
        directories.update(item.parent for item in path.rglob('hardware_model.json'))
    return {'path': str(path), 'present': path.is_dir(), 'files': {
        str((directory / name).relative_to(path)): static_file(directory / name)
        for directory in sorted(directories) for name in PROFILE_FILES}}


def selection_environment():
    """Bind selection inputs, while refusing mutable custom plan-cache state."""
    for variable in ('CUBUTTERFLY_COST_MODEL', 'CUBUTTERFLY_HARDWARE_PROFILE'):
        if variable in os.environ:
            raise ValueError(variable + ' is not a supported runtime override; use CUBUTTERFLY_HARDWARE_MODEL/PROFILE_ROOT')
    if 'CUBUTTERFLY_PLAN_CACHE' in os.environ:
        raise ValueError('custom CUBUTTERFLY_PLAN_CACHE is unsupported for isolated followups; unset it')
    # Include every project override, including future selection switches. Do not
    # hash JIT cache contents: compilation populates it without changing policy.
    names = {key for key in os.environ if key.startswith('CUBUTTERFLY_')}
    names.update(('HOME', 'XDG_CACHE_HOME', 'PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH',
                  'CUDA_HOME', 'CUDA_VISIBLE_DEVICES'))
    values = {key: os.environ.get(key) for key in sorted(names)}
    files = {}
    for variable in ('CUBUTTERFLY_REGISTRY', 'CUBUTTERFLY_HARDWARE_MODEL',
                     'CUBUTTERFLY_PLATFORM_TUNING', 'CUBUTTERFLY_STATIC_RANKER'):
        if variable in os.environ:
            value = os.environ[variable]
            path = Path(value).expanduser().resolve()
            if not value or not path.is_file():
                raise ValueError(variable + ' must name an existing static file')
            files[variable] = static_file(path)
            if variable == 'CUBUTTERFLY_HARDWARE_MODEL':
                files['hardware_model_profile_siblings'] = {name: static_file(path.parent / name) for name in PROFILE_FILES}
    if 'CUBUTTERFLY_REGISTRY' not in os.environ:
        cache = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache')))
        path = (cache / 'cubutterfly/registry.json').expanduser().resolve()
        files['default_registry'] = {'path': str(path), 'sha256': digest(path) if path.is_file() else None}
    if 'CUBUTTERFLY_PROFILE_ROOT' in os.environ:
        path = Path(os.environ['CUBUTTERFLY_PROFILE_ROOT']).expanduser().resolve()
        if not os.environ['CUBUTTERFLY_PROFILE_ROOT'] or not path.is_dir():
            raise ValueError('CUBUTTERFLY_PROFILE_ROOT must name an existing static profile directory')
        files['CUBUTTERFLY_PROFILE_ROOT'] = profile_tree(path)
    files['bundled_profiles'] = profile_tree(ROOT / 'config/hardware')
    files['default_installed_profiles'] = profile_tree(Path('/usr/local/share/cuButterfly/hardware'))
    files['default_ranker_inputs'] = {name: static_file(ROOT / 'scripts' / name) for name in RANKER_INPUTS}
    return {'schema': 'quarot-selection-environment-v2', 'values': values,
            'static_files': files, 'plan_cache_policy': 'disabled; custom mutable plan cache rejected',
            'jit_cache_policy': 'path and environment policy bound; mutable compiled cache contents excluded'}


def prepare(config, library, extension, python, gpu_uuid=None):
    library, extension = Path(library).resolve(), Path(extension).resolve()
    receipt = extension.with_name('build_receipt.json')
    value = json.loads(receipt.read_text())
    if value.get('protocol') != 'quarot_current_stream_only_v1' or value.get('extension_sha256') != digest(extension):
        raise ValueError('extension receipt mismatch')
    identity = {'library_sha256': digest(library), 'extension_sha256': digest(extension),
                'extension_receipt_sha256': digest(receipt), 'runner_sha256': digest(RUNNER),
                'adapter_sha256': digest(RUNNER.with_name('quarot_adapter.py')),
                'checks_sha256': digest(RUNNER.with_name('quarot_random_checks.py')),
                'queue_sha256': digest(__file__), 'python': str(python), 'gpu_uuid': gpu_uuid,
                'contract': config['contract'], 'selection_environment': selection_environment()}
    jobs = []
    for job in config['jobs']:
        values = dict(config['contract'], **{k: v for k, v in job.items() if k != 'id'})
        command = [str(python), str(RUNNER), '--library', str(library),
                   '--quarot-extension', str(extension), '--output', 'report.json']
        for key, value in values.items():
            if key != 'precision_contract':
                command.extend(['--' + key.replace('_', '-'), str(value)])
        jobs.append({'id': job['id'], 'identity': dict(identity, job=job), 'command': command})
    return jobs


def release_matches(directory, identity):
    directory = Path(directory)
    if not (directory / 'release').exists():
        return False
    recorded = json.loads((directory / 'identity.json').read_text())
    status = json.loads((directory / 'status.json').read_text())
    report = json.loads((directory / 'release/report.json').read_text())
    if (recorded != identity or status.get('state') != 'complete'
            or status.get('gpu_uuid') != identity['gpu_uuid']
            or report.get('status') not in ('pass', 'pass_timing', 'pass_timing_numerical_unassessed')
            or report.get('compute_contract') != 'public_fp32_final'
            or report.get('execution_mode') != 'graph'
            or report.get('library_sha256') != identity['library_sha256']
            or report.get('runner_sha256') != identity['runner_sha256']
            or report.get('adapter_sha256') != identity['adapter_sha256']
            or report.get('checks_sha256') != identity['checks_sha256']
            or report.get('quarot_extension_sha256') != identity['extension_sha256']
            or report.get('gpu_uuid') != identity['gpu_uuid']
            or report.get('payload') != {'batch_size': identity['job']['batch_size'],
                                        'prefill': 128, 'decode': 8}
            or report.get('variant_order') != identity['job']['variant_order']
            or report.get('seed') != identity['contract']['seed']
            or report.get('validation_warmup_sequences') != identity['contract']['validation_warmup']
            or report.get('attention_backend') != identity['contract']['attention_backend']
            or report.get('timing_calls') != identity['contract']['calls']
            or report.get('timing_warmup_sequences') != identity['contract']['warmup']):
        raise ValueError('prior release identity or clean completion mismatch')
    return True


def execute(jobs, output, gpu_uuid, until, python, now=None, run=subprocess.run):
    cutoff = datetime.fromisoformat(until)
    if cutoff.tzinfo is None:
        raise ValueError('deadline requires timezone')
    now = now or (lambda: datetime.now().astimezone())
    for job in jobs:
        recorded_environment = job['identity'].get('selection_environment')
        if recorded_environment is None:
            raise ValueError('new execution requires a prepared selection environment; historical identity is read-only evidence')
        if recorded_environment != selection_environment():
            raise ValueError('selection environment changed after prepare; use a fresh prepared identity/output root')
        if job['identity'].get('gpu_uuid') != gpu_uuid:
            raise ValueError('execution GPU differs from prepared identity')
        directory = Path(output).resolve() / job['id']
        if release_matches(directory, job['identity']):
            continue
        if now() >= cutoff:
            return 2
        # A failed/interrupted job retains its guard attempts; never overwrite it.
        if directory.exists():
            raise ValueError('unfinished job directory exists; select a fresh output root')
        directory.mkdir(parents=True)
        (directory / 'identity.json').write_text(json.dumps(job['identity'], indent=2) + '\n')
        result = run([str(python), str(ROOT / 'scripts/run_isolated_gpu.py'),
                      '--gpu-uuid', gpu_uuid, '--until', until, '--report-dir', str(directory),
                      '--', *job['command']], cwd=ROOT)
        if result.returncode:
            return result.returncode
        if not release_matches(directory, job['identity']):
            raise ValueError('guard did not produce a release')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'run'], default='prepare')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--quarot-extension', type=Path, required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--until')
    args = parser.parse_args(argv)
    if args.mode == 'run' and (not args.gpu_uuid or not args.until):
        parser.error('run requires explicit --gpu-uuid and --until')
    jobs = prepare(load_config(args.config), args.library, args.quarot_extension, args.python, args.gpu_uuid)
    if args.mode == 'prepare':
        print(json.dumps({'mode': 'CPU prepare; no model or GPU initialization', 'output': str(args.output), 'jobs': jobs}, indent=2))
        return 0
    return execute(jobs, args.output, args.gpu_uuid, args.until, args.python)


if __name__ == '__main__':
    raise SystemExit(main())
