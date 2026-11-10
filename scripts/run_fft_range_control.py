#!/usr/bin/env python3
"""Independent FP16 FFT/FWHT input-range control; exact mapping replay, no timing.

The historical CLI filename and report schema remain compatible. Each operator
has its own input profile identity; requested storage and compute are replayed.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import numpy as np

import run_semantic_campaign as r
import semantic_input_profiles as profiles


def input_profile_id(semantic_id, metadata):
    return r.digest({'semantic_id': semantic_id, 'input_profile': metadata})


def load_source_row(path):
    envelope = json.loads(Path(path).read_text())
    row = envelope['row']
    if envelope.get('sha256') != r.digest(row):
        raise ValueError('source row checksum mismatch')
    if row['id'] != r.digest(row['descriptor']['semantics']):
        raise ValueError('source semantic ID mismatch')
    if not isinstance(row.get('mapping'), dict):
        raise ValueError('source row lacks replay mapping')
    artifact = row.get('error_artifact')
    if not artifact:
        raise ValueError('source row lacks original error artifact')
    if r.sha256_file(artifact['path']) != artifact['sha256']:
        raise ValueError('original error artifact checksum mismatch')
    return row, envelope['sha256']


def bind_source_measurement(args, row, checksum):
    path = args.row.parent.parent / 'results.json'
    measurement = json.loads(path.read_text())
    if measurement.get('schema') != 'cubutterfly-semantic-results-v1':
        raise ValueError('source measurement report schema mismatch')
    indices = measurement['rows']
    if measurement.get('rows_sha256') != r.digest(indices):
        raise ValueError('source measurement row index checksum mismatch')
    matching = [index for index in indices if index['id'] == row['id']]
    if len(matching) != 1 or matching[0].get('sha256') != checksum:
        raise ValueError('source measurement row checksum mismatch')
    if (path.parent / matching[0]['path']).resolve() != args.row.resolve():
        raise ValueError('source measurement row path mismatch')
    identity = measurement['identity']
    if identity.get('library_sha256') != r.sha256_file(args.library):
        raise ValueError('source measurement library identity mismatch')
    if not r.same_gpu_uuid(args.gpu_uuid, identity.get('gpu_uuid', '')):
        raise ValueError('source measurement GPU identity mismatch')
    return {'original_report_path': str(path.resolve()),
            'original_report_sha256': r.sha256_file(path),
            'original_measurement_identity': copy.deepcopy(identity),
            'original_row_index': copy.deepcopy(matching[0])}


def prepare(args):
    if args.output.exists():
        raise FileExistsError('output already exists: ' + str(args.output))
    row, checksum = load_source_row(args.row)
    measurement_evidence = bind_source_measurement(args, row, checksum)
    descriptor = copy.deepcopy(row['descriptor'])
    s = descriptor['semantics']
    points = s['batch'] * max(np.prod(r.side_extents(s, side)) for side in ('input', 'output'))
    if points > 1048576:
        raise ValueError('complete reference exceeds fixed 1048576-element bound')
    source, metadata = profiles.apply_input_profile(
        s, r.make_source(s), profiles.scaled_profile(s),
        encode_storage=r.encode_storage, decode_storage=r.decode_storage)
    expected = r.reference(s, source)
    calibration = json.loads(args.calibration_manifest.read_text())
    identity = {'library': str(args.library.resolve()), 'library_sha256': r.sha256_file(args.library),
                'runner_sha256': r.sha256_file(__file__),
                'frozen_runner_sha256': r.sha256_file(r.__file__),
                'profile_script_sha256': r.sha256_file(profiles.__file__),
                'python': sys.version, 'numpy': np.__version__, 'gpu_uuid': args.gpu_uuid,
                'calibration_manifest': str(args.calibration_manifest.resolve()),
                'calibration_sha256': r.digest(calibration),
                'calibration_file_sha256': r.sha256_file(args.calibration_manifest),
                'timing': False}
    identity['baseline_adapter_sha256'] = {
        name: r.sha256_file(Path(r.__file__).with_name(name))
        for name in ('diagnose_xd_fft.py', 'vkfft_xd.py', 'semantic_dao_baseline.py',
                     'semantic_gpuntt_baseline.py')}
    identity.update(r.bind_calibration_environment(calibration))
    if calibration.get('measurement_identity', {}).get('library_sha256') != identity['library_sha256']:
        raise ValueError('calibration library identity mismatch')
    for name in ('vkfft_library', 'gpuntt_library'):
        path = getattr(args, name)
        identity[name] = {'path': str(path.resolve()), 'sha256': r.sha256_file(path)} if path else None
    report = {'schema': 'cubutterfly-fft-range-control-v1', 'status': 'prepared',
              'id': input_profile_id(row['id'], metadata), 'original_semantic_id': row['id'],
              'descriptor': descriptor, 'mapping': copy.deepcopy(row['mapping']),
              'input_profile': metadata, 'identity': identity,
              'source_evidence': {'row_path': str(args.row.resolve()),
                                  'row_file_sha256': r.sha256_file(args.row),
                                  'row_checksum': checksum, 'original_status': row['status'],
                                  'original_error_artifact': copy.deepcopy(row['error_artifact']),
                                  **measurement_evidence},
              'protocol': {'float_verdict': 'error-report', 'reference_max_elements': 1048576,
                           'selection': 'explicit original mapping replay',
                           'reference_input': 'final quantized scaled input', 'timing': False,
                           'error_distribution': copy.deepcopy(r.PROTOCOL['error_distribution'])},
              'baseline_coverage_limitations': [
                  'Current cuFFT/VkFFT bridges lack exact FP16 C2C storage binding; '
                  'these remain explicit baseline gaps with no precision promotion.'],
              'baselines': []}
    if s['operator'] == 2:
        report['baseline_coverage_limitations'] = [
            'DaoFHT executes only when its audited FP32 arithmetic matches the requested compute contract; '
            'native FP16 compute remains an explicit gap with no precision promotion.']
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'source.npz'
    np.savez_compressed(path, source=source, reference=expected,
                        storage=r.encode_storage(source, s['storage']))
    report['source_artifact'] = {'path': str(path.resolve()), 'sha256': r.sha256_file(path),
                                 'arrays': ['source', 'reference', 'storage']}
    save_report(args.output, report)
    return report, source, expected, calibration


def save_report(directory, report):
    path = directory / 'results.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def guarded(uuid, operation):
    r.exclusive_gpu(uuid)
    try:
        return operation()
    finally:
        r.exclusive_gpu(uuid)


def execute_control(args, report, source, expected, calibration):
    uuid = args.gpu_uuid
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    # Importing torch and initializing CUDA are deliberately absent from prepare.
    def initialize():
        import torch
        import cubutterfly as cb
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        properties = torch.cuda.get_device_properties(0)
        if not r.same_gpu_uuid(uuid, str(getattr(properties, 'uuid', ''))):
            raise ValueError('CUDA GPU UUID mismatch')
        report['identity'].update(gpu_name=properties.name, gpu_total_memory=properties.total_memory,
                                  compute_capability=f'{properties.major}.{properties.minor}',
                                  torch=torch.__version__, torch_cuda=torch.version.cuda,
                                  frontend_sha256=r.sha256_file(cb.__file__))
        report['calibration_binding'] = r.validate_calibration(calibration, report['identity'])
        report['calibration_binding']['case_selection'] = 'explicit original mapping; no timing or search'
        report['identity']['runtime_environment'] = {
            key: value for key, value in sorted(os.environ.items())
            if key.startswith('CUBUTTERFLY_') or key in ('CUDA_VISIBLE_DEVICES', 'CUDA_MODULE_LOADING')}
        return torch, cb, cb.Library(args.library), torch.cuda.Stream()

    torch, cb, library, stream = guarded(uuid, initialize)
    descriptor, s = report['descriptor'], report['descriptor']['semantics']
    plan = None
    safe_for_baselines = False
    try:
        def create_plan():
            chosen = r.selection_descriptor(descriptor, mapping=report['mapping'])
            with cb.Descriptor.from_json(library, chosen) as imported:
                imported.validate()
                r.assert_semantics(descriptor, json.loads(imported.json()))
                return cb.Plan(library, imported, stream=stream.cuda_stream)

        plan = guarded(uuid, create_plan)
        r.assert_semantics(descriptor, json.loads(plan.descriptor_json()))
        if json.loads(plan.mapping_json()) != report['mapping']:
            raise AssertionError('explicit replay changed original mapping')
        report.update(plan_descriptor=json.loads(plan.descriptor_json()),
                      algorithm=plan.algorithm_name(), input_bytes=plan.input_size(),
                      output_bytes=plan.output_size(), workspace_bytes=plan.workspace_size())
        in_place = bool(s['placement'])
        raw = r.physical_bytes(s, 'input', source, plan.input_size())
        initial = np.full(max(plan.input_size(), plan.output_size()) if in_place else plan.input_size(),
                          0xA5, dtype=np.uint8)
        initial[:len(raw)] = raw
        output_initial = initial.copy() if in_place else np.full(plan.output_size(), 0xA5, np.uint8)

        def allocate():
            workspace = torch.empty(plan.workspace_size(), dtype=torch.uint8, device='cuda') if plan.workspace_size() else None
            if workspace is not None:
                plan.set_workspace(workspace, plan.workspace_size())
            gpu_input = torch.from_numpy(initial).to('cuda')
            saved = gpu_input.clone()
            output = gpu_input if in_place else torch.from_numpy(output_initial).to('cuda')
            return workspace, gpu_input, saved, output

        with torch.cuda.stream(stream):
            workspace, gpu_input, saved, output = guarded(uuid, allocate)

            def run():
                plan.execute(gpu_input, output)
                stream.synchronize()
                return output.cpu().numpy().copy(), gpu_input.cpu().numpy().copy()

            actual_raw, input_after = guarded(uuid, run)
            safe_for_baselines = True
            actual = r.output_from_bytes(s, actual_raw)
            report['numeric_diagnostics'] = r.floating_range_report(actual, expected, s['storage'])
            report['error_artifact'] = r.retain_error_artifact(args.output / 'errors', report['id'],
                                                            'cubutterfly', actual, expected)
            errors = r.error_arrays(actual, expected)
            report['error_distributions'] = {
                name: r.distribution(errors[name]) for name in
                ('abs_error', 'relative_error_nonzero_reference', 'abs_error_zero_reference')}
            # Check memory boundaries even when numerical validation fails.
            failures = []
            for label, check in (
                ('padding', lambda: r.check_padding(s, actual_raw, output_initial)),
                ('input', lambda: check_input_unchanged(in_place, input_after, initial)),
                ('numerics', lambda: r.check_full(actual, expected, s['storage'], float_verdict='error-report'))):
                try:
                    result = check()
                    if label == 'numerics':
                        report['correctness'] = result
                except AssertionError as exc:
                    failures.append({'check': label, 'error': str(exc)})
            report['validation_failures'] = failures
            report['status'] = 'correctness-failure' if failures else 'accuracy-review-required'
    except Exception as exc:
        report.update(status='execution-error', error=str(exc), exception_type=type(exc).__name__)
    finally:
        if plan is not None:
            guarded(uuid, plan.close)
    if safe_for_baselines:
        case = {'id': report['id'], 'descriptor': descriptor}
        try:
            with torch.cuda.stream(stream):
                report['baselines'] = guarded(uuid, lambda: r.run_baselines(
                    case, torch, stream, source, expected, saved, output_initial, False,
                    args.vkfft_library, float_verdict='error-report', error_dir=args.output / 'errors',
                    gpuntt_library=args.gpuntt_library))
        except r.BaselineCorrectnessFailure as exc:
            report['baselines'] = exc.rows
            report['baseline_stop'] = {'reason': str(exc), 'remaining': 'not run after baseline correctness failure'}
        except Exception as exc:
            report['baseline_stop'] = {'reason': str(exc), 'exception_type': type(exc).__name__}
    else:
        report['baseline_stop'] = {'reason': 'execution did not complete and synchronize safely'}


def check_input_unchanged(in_place, actual, initial):
    if not in_place and not np.array_equal(actual, initial):
        raise AssertionError('out-of-place input or padding modified')


def finalize_status(report):
    report['cubutterfly_status'] = report['status']
    failures = [baseline for baseline in report.get('baselines', [])
                if baseline.get('status') in ('baseline-error', 'runtime-error',
                                               'correctness-failure', 'resource-limit',
                                               'runner-error', 'execution-error')
                or baseline.get('support_status') == 'external-binding-error']
    if failures or 'baseline_stop' in report:
        report['status'] = 'incomplete-error'
        report['overall_error'] = 'baseline execution failed or stopped; see baselines and baseline_stop'
    return 0 if report['status'] in ('prepared', 'accuracy-review-required') else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('row', 'library', 'output', 'calibration-manifest'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--vkfft-library', type=Path)
    parser.add_argument('--gpuntt-library', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    report, source, expected, calibration = prepare(args)
    if not args.prepare_only:
        try:
            execute_control(args, report, source, expected, calibration)
        except Exception as exc:
            report.update(status='execution-error', error=str(exc), exception_type=type(exc).__name__)
        finally:
            finalize_status(report)
            save_report(args.output, report)
    print(str((args.output / 'results.json').resolve()))
    return 0 if report['status'] in ('prepared', 'accuracy-review-required') else 1


if __name__ == '__main__':
    raise SystemExit(main())
