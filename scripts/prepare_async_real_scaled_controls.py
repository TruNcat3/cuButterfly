#!/usr/bin/env python3
"""Prepare finite FP16 real FFT controls and complete CPU oracles without CUDA."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import pickle
import types
import uuid

import numpy as np

import run_fft_range_control as control
import semantic_preparation_queue as queue
import semantic_real_input_profiles as profiles

runner = control.r


def _fingerprint(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': runner.sha256_file(path)}


def _save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare(args):
    """Bind historical failures and prepare their final quantized scaled input."""
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('output already exists: ' + str(output))
    if not args.rows:
        raise ValueError('at least one source row is required')
    target = getattr(args, 'diagnostic_gpu_uuid', None) or args.gpu_uuid
    replay = runner.same_gpu_uuid(target, args.gpu_uuid)
    selection = ('explicit original mapping replay' if replay else
                 'fresh-selection diagnostic on a different GPU')
    calibration = json.loads(Path(args.calibration_manifest).read_text())
    library = _fingerprint(args.library)
    if calibration.get('measurement_identity', {}).get('library_sha256') != library['sha256']:
        raise ValueError('calibration library identity mismatch')
    calibration_environment = runner.bind_calibration_environment(calibration)
    calibration_sha = runner.digest(calibration)
    cases, prepared_values, seen = [], [], set()
    for path in args.rows:
        path = Path(path).resolve(strict=True)
        row, checksum = control.load_source_row(path)
        if row['status'] != 'correctness-failure':
            raise ValueError('real scaled controls require source correctness-failure rows')
        if row['id'] in seen:
            raise ValueError('duplicate source semantic ID: ' + row['id'])
        seen.add(row['id'])
        s = row['descriptor']['semantics']
        if s.get('operator') != 0 or s.get('storage') != 0 or s.get('fft_type') not in (1, 2):
            raise ValueError('real scaled controls require FP16 R2C or C2R requests')
        points = s['batch'] * max(math.prod(runner.side_extents(s, side))
                                  for side in ('input', 'output'))
        if points > runner.PROTOCOL['reference_max_elements']:
            raise ValueError('real scaled control exceeds complete-reference element bound')
        binding = types.SimpleNamespace(row=path, library=Path(args.library), gpu_uuid=args.gpu_uuid)
        evidence = control.bind_source_measurement(binding, row, checksum)
        identity = evidence['original_measurement_identity']
        if identity.get('calibration_sha256', calibration_sha) != calibration_sha:
            raise ValueError('source measurement calibration identity mismatch')
        source, metadata = profiles.prepare_input(s)
        expected = runner.reference(s, source)
        if not np.isfinite(source).all() or not np.isfinite(expected).all():
            raise ValueError('scaled source and complete reference must be finite')
        source_evidence = {
            'row_path': str(path), 'row_file_sha256': runner.sha256_file(path),
            'row_checksum': checksum, 'original_status': row['status'],
            'original_error_artifact': copy.deepcopy(row['error_artifact']), **evidence,
            'original_gpu_uuid': args.gpu_uuid, 'target_gpu_uuid': target,
            'original_mapping_replayed': replay, 'selection': selection, 'timing': False}
        case = {'id': row['id'], 'descriptor': copy.deepcopy(row['descriptor']),
                'populations': ['real-scaled-control'], 'requested_variants': [],
                'input_profile': metadata, 'source_evidence': source_evidence}
        if replay:
            case['execution_mapping'] = copy.deepcopy(row['mapping'])
        if 'capabilities' in row:
            case['capabilities'] = copy.deepcopy(row['capabilities'])
        cases.append(case)
        prepared_values.append({
            'case_id': case['id'], 's': copy.deepcopy(s), 'source': source, 'expected': expected,
            'encoded_source': runner.encode_storage(source, runner.side_storage(s, 'input')),
            'input_indices': runner.layout_indices(s, 'input'),
            'output_indices': runner.layout_indices(s, 'output')})
    directory = Path(__file__).resolve().parent
    source_paths = [Path(__file__), Path(control.__file__), Path(queue.__file__),
                    Path(queue.cpu.__file__), Path(runner.__file__), Path(profiles.__file__),
                    Path(control.profiles.__file__), directory / 'semantic_ntt_reference.py',
                    Path(args.calibration_manifest), Path(calibration['registry_path']),
                    Path(calibration['hardware_rates_source']), Path(calibration['cost_model'])]
    source_paths += [Path(path) for path in (args.vkfft_library, args.gpuntt_library) if path]
    manifest = {
        'schema': 'cubutterfly-semantic-campaign-v1', 'cases': cases, 'invalid_requests': [],
        'metrics': {'cases': len(cases), 'real_scaled_controls': len(cases), 'not_applicable': 0},
        'fingerprints': {'library': library, 'sources': [_fingerprint(path) for path in source_paths]},
        'real_scaled_control_provenance': {
            'gpu_uuid': target, 'original_gpu_uuid': args.gpu_uuid, 'target_gpu_uuid': target,
            'source_gpu_uuid': args.gpu_uuid,
            'original_mapping_replayed': replay, 'calibration_sha256': calibration_sha,
            'calibration_gpu_uuid': args.gpu_uuid, 'calibration_environment': calibration_environment,
            'selection': selection, 'timing': False, 'float_verdict': 'error-report',
            'reference_input': 'final quantized scaled input'}}
    manifest['manifest_sha256'] = runner.digest(manifest)
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / 'manifest.json'
    _save_json(manifest_path, manifest)
    cache = output / 'prepared'
    queue.initialize(cache, manifest)
    _, config_sha = queue._config(cache)
    (cache / 'artifacts').mkdir()
    (cache / 'records').mkdir()
    for case, prepared in zip(cases, prepared_values):
        relative = 'artifacts/' + case['id'] + '.pickle'
        path = cache / relative
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('xb') as handle:
                pickle.dump(prepared, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        record = {'case_id': case['id'], 'case_sha256': runner.digest(case), 'config_sha256': config_sha,
                  'status': 'ready', 'artifact': {'path': relative, 'sha256': runner.sha256_file(path),
                                                'bytes': path.stat().st_size}}
        queue._write_json(cache / 'records' / (case['id'] + '.json'), record)
    result = {'manifest_path': str(manifest_path), 'cache_path': str(cache),
              'eligible': [case['id'] for case in cases], 'gaps': [],
              'summary_path': str(output / 'preparation.json')}
    _save_json(output / 'preparation.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True, help='GPU UUID of the original measurement')
    parser.add_argument('--diagnostic-gpu-uuid', help='Different GPU for fresh-selection untimed diagnostics')
    parser.add_argument('--calibration-manifest', type=Path, required=True)
    parser.add_argument('--vkfft-library', type=Path)
    parser.add_argument('--gpuntt-library', type=Path)
    args = parser.parse_args(argv)
    result = prepare(args)
    print(json.dumps({'cases': len(result['eligible']), 'manifest_path': result['manifest_path'],
                      'cache_path': result['cache_path']}, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
