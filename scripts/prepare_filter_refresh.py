#!/usr/bin/env python3
"""Prepare the two frozen matched-padding filter cases without CUDA."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'paper/experiments/e13/run_fft_filter_end_to_end.py'
ADAPTER = ROOT / 'results/e13_assets/cubutterfly_real_fft.py'
DEFAULT_CONFIG = ROOT / 'configs/experiments/filter_refresh_20261025.json'


def safe_path(path):
    path = Path(path).absolute()
    # Resolve each ancestor first, so no file read follows an /sdb asset link.
    for parent in reversed((path,) + tuple(path.parents)):
        resolved = parent.resolve()
        if resolved == Path('/sdb') or Path('/sdb') in resolved.parents:
            raise RuntimeError(f'forbidden /sdb asset: {path}')
    return path


def digest(path):
    value = hashlib.sha256()
    with safe_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(safe_path(path).read_text())


def runner_module():
    spec = importlib.util.spec_from_file_location('filter_refresh_runner', safe_path(RUNNER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_identity():
    import torch
    return {'torch': torch.__version__, 'numpy': np.__version__,
            'compile_mode': os.environ.get('CUBUTTERFLY_COMPILE_MODE', 'research'),
            'algorithm_policy': os.environ.get('CUBUTTERFLY_E13_ALGORITHM_POLICY', 'default')}


def code_identity():
    return {str(path.relative_to(ROOT)): digest(path) for path in (RUNNER, ADAPTER, Path(__file__))}


def prepare(config_path, output, library):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError("prepare-only requires CUDA_VISIBLE_DEVICES=''")
    config = read_json(config_path)
    api = runner_module()
    output = safe_path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {'library': str(safe_path(library)), 'library_sha256': digest(library),
                'config': str(safe_path(config_path)), 'protocol_sha256': object_digest(config),
                'code': code_identity(), 'runtime': runtime_identity()}
    binding_path = output / 'prepared.json'
    if binding_path.exists():
        binding = validate_prepared(binding_path, library)
        if binding['identity'] != identity:
            raise RuntimeError('existing preparation identity changed; use a new output')
        print('reused complete filter preparation', flush=True)
        return binding
    cases = []
    for case in config['cases']:
        source = safe_path(ROOT / case['source'])
        receipt_path = source / case['receipt']
        receipt = read_json(receipt_path)
        manifest_path = source / 'manifest.json'
        manifest = read_json(manifest_path)
        contract, = [item for item in manifest['contracts'] if item['id'] == case['id']]
        request = receipt['requested']
        expected = dict(warmup=case['warmup'], repeats=case['repeats'], samples=case['samples'],
                        execution_mode='graph', padding_policy='fast', cache_filter_spectrum=True,
                        torch_control=True, atol=None, rtol=None, process_index=1,
                        require_cache=False, plan_only=False)
        if request != expected or receipt['actual_cubutterfly_padding'] != case['padding'] or receipt['cupy_native_fft_padding'] != case['padding']:
            raise RuntimeError(f'historical protocol mismatch: {case["id"]}')
        if contract['batch'] != 16 or contract['filter_shape'] != [31, 31]:
            raise RuntimeError('frozen filter population changed')
        archive = source / contract['artifact']
        if digest(archive) != contract['sha256']:
            raise RuntimeError(f'historical input hash mismatch: {archive}')
        case_dir = safe_path(output / case['id'])
        case_dir.mkdir(parents=True, exist_ok=True)
        artifact = case_dir / contract['artifact']
        provenance = {'archive': str(archive), 'archive_sha256': contract['sha256'],
                      'receipt': str(receipt_path), 'receipt_sha256': digest(receipt_path),
                      'manifest': str(manifest_path), 'manifest_sha256': digest(manifest_path)}
        case_binding_path = case_dir / 'cpu_prepared.json'
        resume_key = {'identity': identity, 'source': provenance, 'protocol': case}
        if case_binding_path.exists():
            prepared = read_json(case_binding_path)
            if prepared['resume_key'] != resume_key or digest(artifact) != prepared['contract']['sha256']:
                raise RuntimeError(f'incompatible case checkpoint: {case_dir}')
        else:
            with np.load(safe_path(archive), allow_pickle=False) as data:
                image, kernel = data['image'], data['kernel']
            if image.dtype != np.float32 or image.shape != (16, *contract['image_shape']) or kernel.dtype != np.float32 or kernel.shape != (31, 31):
                raise RuntimeError('historical input storage mismatch')
            reference = api.cpu_fp32_convolve(image, kernel, case['padding'])
            changed = np.roll(image, shift=(13, -7), axis=(-2, -1)).copy()
            changed_reference = api.cpu_fp32_convolve(changed, kernel, case['padding'])
            if not np.isfinite(reference).all() or not np.isfinite(changed_reference).all():
                raise RuntimeError('nonfinite CPU reference')
            temporary = safe_path(artifact.with_suffix('.tmp.npz'))
            np.savez(temporary, image=image, kernel=kernel, reference_fp32=reference,
                     changed_reference_fp32=changed_reference)
            temporary.replace(artifact)
            prepared_contract = dict(contract, sha256=digest(artifact),
                                     cpu_reference='persisted torch.fft CPU float32/complex64',
                                     reference_padding=case['padding'])
            prepared = {'resume_key': resume_key, 'contract': prepared_contract}
            api.write_json(case_binding_path, prepared)
            del image, kernel, reference, changed, changed_reference
        api.write_json(case_dir / 'manifest.json', dict(manifest, contracts=[prepared['contract']]))
        cases.append({'id': case['id'], 'directory': str(case_dir), 'protocol': expected,
                      'padding': case['padding'], 'source': provenance,
                      'artifact': str(artifact), 'artifact_sha256': prepared['contract']['sha256'],
                      'manifest_sha256': digest(case_dir / 'manifest.json')})
        print(f'prepared {case["id"]}: CPU FP32 original and changed-input references', flush=True)
    binding = {'schema': 'filter-refresh-prepared-v1', 'status': 'cpu-prepared-gpu-pending',
               'identity': identity, 'cases': cases, 'full_population': True,
               'preparation_sha256': object_digest({'identity': identity, 'cases': cases})}
    api.write_json(binding_path, binding)
    return binding


def validate_prepared(path, library, *, allow_helper_change=False):
    binding = read_json(path)
    identity = binding['identity']
    if digest(library) != identity['library_sha256'] or str(safe_path(library)) != identity['library']:
        raise RuntimeError('prepared library identity mismatch')
    if object_digest(read_json(identity['config'])) != identity['protocol_sha256']:
        raise RuntimeError('prepared protocol hash mismatch')
    current_code = code_identity()
    if allow_helper_change:
        helper = str(Path(__file__).relative_to(ROOT))
        current_code[helper] = identity['code'][helper]
    if current_code != identity['code'] or runtime_identity() != identity['runtime']:
        raise RuntimeError('prepared code/runtime identity mismatch')
    if binding['preparation_sha256'] != object_digest({'identity': identity, 'cases': binding['cases']}):
        raise RuntimeError('preparation binding hash mismatch')
    for case in binding['cases']:
        for key in ('archive', 'receipt', 'manifest'):
            if digest(case['source'][key]) != case['source'][key + '_sha256']:
                raise RuntimeError(f'prepared source hash mismatch: {key}')
        if digest(case['artifact']) != case['artifact_sha256'] or digest(Path(case['directory']) / 'manifest.json') != case['manifest_sha256']:
            raise RuntimeError('prepared input/reference/manifest hash mismatch')
    return binding


def rebind_helper_code(binding_path, library):
    """Explicitly update only this wrapper's code hash after verifying CPU assets."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError("code-only rebind requires CUDA_VISIBLE_DEVICES=''")
    binding = validate_prepared(binding_path, library, allow_helper_change=True)
    old_identity = binding['identity']
    new_identity = dict(old_identity, code=code_identity())
    if new_identity == old_identity:
        print('helper code identity already current; CPU references retained')
        return binding
    checkpoints = []
    config = read_json(old_identity['config'])
    for case in binding['cases']:
        directory = safe_path(case['directory'])
        if (directory / f'{case["id"]}.current_graph_fast.json').exists() or (directory / 'current_graph_fast.intent.json').exists():
            raise RuntimeError('code-only rebind cannot replace a binding with GPU run evidence')
        checkpoint = read_json(directory / 'cpu_prepared.json')
        frozen = next(item for item in config['cases'] if item['id'] == case['id'])
        if checkpoint['resume_key'] != {'identity': old_identity, 'source': case['source'], 'protocol': frozen}:
            raise RuntimeError('CPU checkpoint resume identity mismatch')
        contract, = read_json(directory / 'manifest.json')['contracts']
        if checkpoint['contract'] != contract:
            raise RuntimeError('CPU checkpoint contract mismatch')
        checkpoints.append((directory / 'cpu_prepared.json', checkpoint))
    history_root = safe_path(Path(binding_path).parent / 'code_rebind_history')
    history_root.mkdir(exist_ok=True)
    history = Path(tempfile.mkdtemp(prefix='rebind-', dir=history_root))
    shutil.copy2(safe_path(binding_path), history / 'prepared.json')
    api = runner_module()
    for path, checkpoint in checkpoints:
        shutil.copy2(safe_path(path), history / f'{path.parent.name}.cpu_prepared.json')
        checkpoint['resume_key']['identity'] = new_identity
        api.write_json(path, checkpoint)
    binding['identity'] = new_identity
    binding['preparation_sha256'] = object_digest({'identity': new_identity, 'cases': binding['cases']})
    api.write_json(binding_path, binding)
    validate_prepared(binding_path, library)
    print('rebound wrapper code only; checked NPZ inputs/references retained; prior binding at ' + str(history))
    return binding


def validate_runner_binding(args):
    binding = validate_prepared(args.prepared_binding, args.library)
    matches = [case for case in binding['cases'] if safe_path(case['directory']) == safe_path(args.output)]
    if len(matches) != 1 or args.contract != [matches[0]['id']]:
        raise RuntimeError('prepared case selection mismatch')
    case = matches[0]
    if any(getattr(args, key) != value for key, value in case['protocol'].items()):
        raise RuntimeError('prepared runner timing/protocol mismatch')


def _report_identity(binding, case, binding_path, gpu_uuid):
    return {'prepared_binding_sha256': digest(binding_path), 'requested': case['protocol'],
            'library': binding['identity']['library'],
            'library_sha256': binding['identity']['library_sha256'],
            'runner_sha256': binding['identity']['code'][str(RUNNER.relative_to(ROOT))],
            'adapter_sha256': binding['identity']['code'][str(ADAPTER.relative_to(ROOT))],
            'gpu_uuid': gpu_uuid}


def _known_report(report, expected, binding, case):
    if not isinstance(report, dict) or not isinstance(report.get('identity'), dict):
        return False
    identity = report.get('identity', {})
    for key in ('library', 'library_sha256', 'runner_sha256', 'adapter_sha256'):
        if identity.get(key) != expected[key]:
            return False
    if str(identity.get('gpu_uuid', '')).lower() != expected['gpu_uuid'].lower():
        return False
    for key in ('torch', 'compile_mode', 'algorithm_policy'):
        if identity.get(key) != binding['identity']['runtime'][key]:
            return False
    manifest = read_json(Path(case['directory']) / 'manifest.json')
    return (report.get('prepared_binding_sha256') == expected['prepared_binding_sha256']
            and report.get('requested') == expected['requested']
            and report.get('preparation_contract') == manifest['contracts'][0])


def _finite_error(error):
    if not isinstance(error, dict) or error.get('finite') is not True:
        return False
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    for key in ('rms_error', 'relative_l2', 'reference_rms', 'reference_abs_max'):
        if not finite(error.get(key)):
            return False
    for key in ('absolute_quantiles', 'relative_quantiles'):
        values = error.get(key)
        if not isinstance(values, dict) or set(values) != {'0.5', '0.9', '0.99', '0.999', '1.0'} or not all(finite(value) for value in values.values()):
            return False
    return True


def _complete_report(report, case):
    try:
        return _report_contents_complete(report, case)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def _report_contents_complete(report, case):
    api = runner_module()
    shape = case['padding']
    protocol = case['protocol']
    preparation, = read_json(Path(case['directory']) / 'manifest.json')['contracts']
    if (report.get('schema') != 'fft-filter-complete-call-v2'
            or report.get('contract') != api.runtime_contract(preparation, shape, cached_filter=True, execution_mode='graph')
            or report.get('complete_filter_call') is not True or report.get('execution_mode') != 'graph'
            or report.get('graph_capture_outside_timing') is not True or report.get('filter_spectrum_cached') is not True
            or report.get('actual_cubutterfly_padding') != shape or report.get('cupy_native_fft_padding') != shape):
        return False
    graph_contract = report.get('graph_call_contract') or {}
    if any(graph_contract.get(key) is not True for key in ('input_device_copy_included', 'independent_output_copy_included')):
        return False
    reference = report.get('validation_reference') or {}
    if reference.get('dtype') != 'float32' or reference.get('fft_dtype') != 'complex64' or reference.get('padding') != shape or reference.get('precision_promotion') is not False:
        return False
    methods = {'cubutterfly', 'cupy_fftconvolve_native_padding', 'torch_cufft_same_padding_diagnostic'}
    rows = report.get('rows')
    if not isinstance(rows, list) or len(rows) != 3 or {row.get('method') for row in rows} != methods:
        return False
    if report.get('method_order') != [row['method'] for row in rows]:
        return False
    for row in rows:
        if row.get('batch') != 16 or row.get('accepted') is not None or row.get('timing_status') != 'measured_complete_call' or not _finite_error(row.get('error')):
            return False
        if row.get('call_contract') != api.method_call_contract(shape, cached_filter=True, execution_mode='graph'):
            return False
        timing = row.get('timing') or {}
        samples = timing.get('samples_us')
        if (timing.get('protocol') != 'cuda_events_cuda_graph_replay'
                or timing.get('warmup_calls') != protocol['warmup'] or timing.get('calls_per_sample') != protocol['repeats']
                or not isinstance(samples, list) or len(samples) != protocol['samples']
                or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0 for value in samples)):
            return False
        median = timing.get('median_us')
        if not isinstance(median, (int, float)) or not math.isfinite(median) or not math.isclose(median, float(np.median(samples)), rel_tol=1e-12):
            return False
        graph = row.get('graph_changing_input_validation') or {}
        if any(graph.get(key) is not True for key in ('distinct_input_outputs', 'previous_output_preserved', 'restored_input_exact')):
            return False
        errors = graph.get('same_precision_cpu_errors')
        if not isinstance(errors, list) or len(errors) != 2 or not all(_finite_error(error) for error in errors):
            return False
    return True


def _existing_report(report_path, intent_path, expected, binding, case):
    temporary_path = safe_path(report_path.with_suffix(report_path.suffix + '.tmp'))
    if not report_path.exists():
        if intent_path.exists() and read_json(intent_path) != expected:
            raise RuntimeError('unknown prior run intent identity; refusing overwrite')
        if not temporary_path.exists():
            return False
        if not intent_path.exists():
            raise RuntimeError('temporary report without matching run intent; refusing overwrite')
        report = None
    else:
        try:
            report = read_json(report_path)
        except json.JSONDecodeError:
            if not intent_path.exists() or read_json(intent_path) != expected:
                raise RuntimeError('unreadable report without matching run intent; refusing overwrite')
            report = None
    if report is not None and not _known_report(report, expected, binding, case):
        raise RuntimeError('unknown prior report identity; refusing overwrite')
    if report is not None and _complete_report(report, case):
        return True
    archive_root = safe_path(report_path.parent / 'incomplete_reports')
    archive_root.mkdir(exist_ok=True)
    archive = Path(tempfile.mkdtemp(prefix='current_graph_fast-', dir=archive_root))
    if report_path.exists():
        report_path.replace(archive / report_path.name)
    if temporary_path.exists():
        if not intent_path.exists() or read_json(intent_path) != expected:
            raise RuntimeError('temporary report has no matching run intent; retained without overwrite')
        temporary_path.replace(archive / temporary_path.name)
    if intent_path.exists():
        intent_path.replace(archive / intent_path.name)
    runner_module().write_json(archive / 'reason.json', {'reason': 'incomplete same-binding report; retained before rerun', 'expected_identity': expected})
    print(f'archived incomplete {case["id"]} report: {archive}', flush=True)
    return False


def run_prepared(binding_path, library, gpu_uuid):
    binding = validate_prepared(binding_path, library)
    for case in binding['cases']:
        protocol = case['protocol']
        report_path = safe_path(Path(case['directory']) / f'{case["id"]}.current_graph_fast.json')
        intent_path = safe_path(Path(case['directory']) / 'current_graph_fast.intent.json')
        expected = _report_identity(binding, case, binding_path, gpu_uuid)
        if _existing_report(report_path, intent_path, expected, binding, case):
            print(f'reused complete {case["id"]} report', flush=True)
            continue
        runner_module().write_json(intent_path, expected)
        command = [sys.executable, str(RUNNER), '--run', '--output', case['directory'],
                   '--library', str(safe_path(library)), '--prepared-binding', str(safe_path(binding_path)),
                   '--gpu-uuid', gpu_uuid, '--contract', case['id'], '--run-id', 'current_graph_fast',
                   '--execution-mode', 'graph', '--padding-policy', 'fast',
                   '--cache-filter-spectrum', '--torch-control']
        for key in ('warmup', 'repeats', 'samples'):
            command.extend(['--' + key, str(protocol[key])])
        env = dict(os.environ)
        env.pop('XD_REPORT_DIR', None)
        subprocess.run(command, check=True, cwd=ROOT, env=env)
        if not report_path.exists():
            raise RuntimeError('filter runner exited successfully without the expected report')
        report = read_json(report_path)
        if not _known_report(report, expected, binding, case) or not _complete_report(report, case):
            raise RuntimeError('filter runner exited successfully without a complete valid report; evidence retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--run-prepared', action='store_true')
    mode.add_argument('--validate-only', action='store_true')
    mode.add_argument('--rebind-code-only', action='store_true',
                      help='CPU only: verify existing arrays and update only wrapper code identity')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/full_current_20261025/fft_filters')
    parser.add_argument('--library', type=Path, default=ROOT / 'build-e13-cufftdx/libcuntt.so')
    parser.add_argument('--gpu-uuid')
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.config, args.output_dir, args.library)
    elif args.rebind_code_only:
        rebind_helper_code(args.output_dir / 'prepared.json', args.library)
    elif args.validate_only:
        validate_prepared(args.output_dir / 'prepared.json', args.library)
        print('validated complete frozen filter preparation; no CUDA initialized')
    else:
        if not args.gpu_uuid:
            parser.error('--run-prepared requires --gpu-uuid')
        run_prepared(args.output_dir / 'prepared.json', args.library, args.gpu_uuid)


if __name__ == '__main__':
    main()
