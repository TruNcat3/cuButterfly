#!/usr/bin/env python3
"""Prepare exact scaled FFT/FWHT controls and CPU oracle cache without CUDA."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import pickle
import types
import uuid

import run_fft_range_control as control
import run_semantic_range_queue as range_queue
import semantic_preparation_queue as queue

runner = control.r
profiles = control.profiles


def _fingerprint(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': runner.sha256_file(path)}


def _save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare(args):
    """Validate all sources before writing a standalone manifest and ready cache."""
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('output already exists: ' + str(output))
    calibration = json.loads(Path(args.calibration_manifest).read_text())
    library = _fingerprint(args.library)
    if calibration.get('measurement_identity', {}).get('library_sha256') != library['sha256']:
        raise ValueError('calibration library identity mismatch')
    calibration_environment = runner.bind_calibration_environment(calibration)
    calibration_sha = runner.digest(calibration)
    cases, prepared_values, gaps = [], [], []
    seen = set()
    for path in args.rows:
        path = Path(path).resolve(strict=True)
        envelope = json.loads(path.read_text())
        row = envelope['row']
        if envelope.get('sha256') != runner.digest(row):
            raise ValueError('source row checksum mismatch')
        if row['id'] != runner.digest(row['descriptor']['semantics']):
            raise ValueError('source semantic ID mismatch')
        if row['status'] != 'correctness-failure':
            raise ValueError('scaled controls require source correctness-failure rows')
        if row['id'] in seen:
            raise ValueError('duplicate source semantic ID: ' + row['id'])
        seen.add(row['id'])
        reason = range_queue.scope_reason(row)
        if reason is not None:
            gaps.append({'id': row['id'], 'row_path': str(path),
                         'status': 'not-applicable', 'reason': reason})
            continue
        row, checksum = control.load_source_row(path)
        binding = types.SimpleNamespace(row=path, library=Path(args.library), gpu_uuid=args.gpu_uuid)
        evidence = control.bind_source_measurement(binding, row, checksum)
        identity = evidence['original_measurement_identity']
        if identity.get('calibration_sha256', calibration_sha) != calibration_sha:
            raise ValueError('source measurement calibration identity mismatch')
        source, metadata = profiles.apply_input_profile(
            row['descriptor']['semantics'], runner.make_source(row['descriptor']['semantics']),
            profiles.scaled_profile(row['descriptor']['semantics']),
            encode_storage=runner.encode_storage, decode_storage=runner.decode_storage)
        s = row['descriptor']['semantics']
        expected = runner.reference(s, source)
        source_evidence = {'row_path': str(path), 'row_file_sha256': runner.sha256_file(path),
                           'row_checksum': checksum, 'original_status': row['status'],
                           'original_error_artifact': copy.deepcopy(row['error_artifact']), **evidence}
        case = {'id': row['id'], 'descriptor': copy.deepcopy(row['descriptor']),
                'populations': ['scaled-control'], 'requested_variants': [],
                'input_profile': metadata, 'execution_mapping': copy.deepcopy(row['mapping']),
                'source_evidence': source_evidence}
        if 'capabilities' in row:
            case['capabilities'] = copy.deepcopy(row['capabilities'])
        cases.append(case)
        prepared_values.append({'case_id': case['id'], 's': copy.deepcopy(s),
                                'source': source, 'expected': expected,
                                'encoded_source': runner.encode_storage(source, runner.side_storage(s, 'input')),
                                'input_indices': runner.layout_indices(s, 'input'),
                                'output_indices': runner.layout_indices(s, 'output')})
    directory = Path(__file__).resolve().parent
    source_paths = [Path(__file__), Path(control.__file__), Path(range_queue.__file__),
                    Path(queue.__file__), Path(queue.cpu.__file__), Path(runner.__file__),
                    Path(profiles.__file__), directory / 'semantic_ntt_reference.py']
    source_paths += [Path(args.calibration_manifest), Path(calibration['registry_path']),
                     Path(calibration['hardware_rates_source']), Path(calibration['cost_model'])]
    source_paths += [Path(path) for path in (args.vkfft_library, args.gpuntt_library) if path]
    manifest = {'schema': 'cubutterfly-semantic-campaign-v1', 'cases': cases,
                'invalid_requests': [], 'metrics': {'cases': len(cases), 'scaled_controls': len(cases),
                                                   'not_applicable': len(gaps)},
                'fingerprints': {'library': library, 'sources': [_fingerprint(path) for path in source_paths]},
                'scaled_control_provenance': {'gpu_uuid': args.gpu_uuid,
                    'calibration_sha256': calibration_sha,
                    'calibration_environment': calibration_environment,
                    'selection': 'explicit original mapping replay', 'timing': False,
                    'float_verdict': 'error-report', 'reference_input': 'final quantized scaled input'}}
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
              'eligible': [case['id'] for case in cases], 'gaps': gaps,
              'summary_path': str(output / 'preparation.json')}
    _save_json(output / 'preparation.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--calibration-manifest', type=Path, required=True)
    parser.add_argument('--vkfft-library', type=Path)
    parser.add_argument('--gpuntt-library', type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args), allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
