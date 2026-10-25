#!/usr/bin/env python3
"""CPU-only supplemental review of retained GPU outputs against native precision."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

import cpu_external_baselines as cpu
import run_semantic_campaign as semantic
import semantic_preparation_queue as preparation
from semantic_campaign_manifest import validate_manifest_hash
from summarize_semantic_campaign import load_checkpoint


def _actual(artifact, checkpoint):
    path = Path(artifact['path'])
    if not path.is_absolute():
        path = checkpoint.parent / path
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != artifact['sha256']:
        raise ValueError('error artifact checksum mismatch: ' + str(path))
    # Decode the same bytes whose checksum was verified, with pickle disabled.
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        return arrays['actual'].copy()


def review(manifest_path, preparation_path, gpu_results, output, cpu_source_manifest=None):
    manifest_path, gpu_results, output = map(lambda p: Path(p).resolve(),
                                             (manifest_path, gpu_results, output))
    if output in (manifest_path, gpu_results) or output.suffix != '.json':
        raise ValueError('output must be a separate .json review file')
    artifact_dir = output.parent / (output.stem + '_errors')
    if artifact_dir == gpu_results.parent / 'errors':
        raise ValueError('review artifacts must not overwrite measured error artifacts')
    manifest = json.loads(manifest_path.read_text())
    if not validate_manifest_hash(manifest):
        raise ValueError('manifest content checksum mismatch')
    source = json.loads(Path(cpu_source_manifest).read_text()) if cpu_source_manifest is not None else None
    cpu.comparison_source(manifest, source)
    report, rows = load_checkpoint(gpu_results)
    protected = {manifest_path, gpu_results, Path(preparation_path).resolve() / 'config.json'}
    if cpu_source_manifest is not None:
        protected.add(Path(cpu_source_manifest).resolve())
    for index in report['rows']:
        protected.add((gpu_results.parent / index['path']).resolve())
    for measured in rows:
        for implementation in [measured] + measured.get('baselines', []):
            artifact = implementation.get('error_artifact')
            if artifact is None:
                continue
            path = Path(artifact['path'])
            path = (path if path.is_absolute() else gpu_results.parent / path).resolve()
            protected.add(path)
            if artifact_dir == path.parent:
                raise ValueError('review artifacts must not overwrite measured error artifacts')
    if output in protected:
        raise ValueError('output must not overwrite existing measurement or preparation files')
    checksum = semantic.digest(manifest)
    if report['identity']['manifest_sha256'] != checksum:
        raise ValueError('GPU checkpoint manifest identity mismatch')
    config, _ = preparation._config(preparation_path)
    source_checksum = semantic.digest(source) if source is not None else checksum
    if config['manifest_sha256'] != source_checksum:
        raise ValueError('preparation manifest identity mismatch')
    preparation_identity = (cpu.comparison_preparation(manifest, preparation_path, source)
                            if source is not None else None)
    cases = {case['id']: case for case in manifest['cases']}
    if len(cases) != len(manifest['cases']):
        raise ValueError('duplicate manifest case id')
    reviewed = []
    for measured in rows:
        case_id = measured['id']
        if case_id not in cases or measured['descriptor'] != cases[case_id]['descriptor']:
            raise ValueError('GPU row descriptor does not match manifest: ' + case_id)
        s = cases[case_id]['descriptor']['semantics']
        implementations = [('cubutterfly', measured)] + [(b['name'], b) for b in measured.get('baselines', [])]
        expected = reference = prepared = None
        reference_error = None
        for backend, implementation in implementations:
            row = dict(case_id=case_id, backend=backend, recorded_status=implementation['status'],
                       recorded_correctness=implementation.get('correctness'), semantics=s,
                       status='validation-unavailable', reference_comparison='same-precision CPU')
            artifact = implementation.get('error_artifact')
            if artifact is None:
                row['reason'] = 'no retained actual-output artifact for this implementation'
                reviewed.append(row)
                continue
            row['measured_error_artifact'] = artifact
            actual = _actual(artifact, gpu_results)
            if expected is None and reference_error is None:
                try:
                    prepared = preparation.load_case(preparation_path, cases[case_id])
                    expected, reference = cpu.same_precision_reference(s, prepared['source'], prepared['expected'])
                except (semantic.Gap, FileNotFoundError, RuntimeError, ValueError, ImportError) as error:
                    reference_error = str(error)
            if reference_error is not None:
                row['reason'] = 'same-precision reference unavailable: ' + reference_error
                reviewed.append(row)
                continue
            row['reference'] = reference
            row['input_sha256'] = hashlib.sha256(prepared['source'].tobytes()).hexdigest()
            if actual.size != expected.size or actual.dtype != expected.dtype:
                row.update(status='artifact-contract-failure', reason='actual/reference element count or exact storage dtype differs',
                           actual_dtype=str(actual.dtype), reference_dtype=str(expected.dtype),
                           actual_elements=int(actual.size), reference_elements=int(expected.size))
                reviewed.append(row)
                continue
            backend_token = hashlib.sha256(backend.encode()).hexdigest()[:16]
            row['error_artifact'] = semantic.retain_error_artifact(artifact_dir, case_id, backend_token, actual, expected)
            try:
                row['correctness'] = semantic.check_full(actual, expected, s['storage'], 'error-report', True)
                row['status'] = 'pass' if s['storage'] in (8, 9) else 'accuracy-review-required'
            except AssertionError as error:
                row.update(status='correctness-failure' if s['storage'] in (8, 9) else 'numerical-validation-failure',
                           reason=str(error))
                if s['storage'] in (8, 9):
                    row['correctness'] = {'exact': False, 'elements': int(actual.size),
                                          'mismatches': int(np.count_nonzero(actual.reshape(-1) != expected.reshape(-1)))}
            reviewed.append(row)
    result = dict(schema='cubutterfly-gpu-same-precision-review-v1',
                  identity={'manifest_sha256': checksum, 'gpu_results': str(gpu_results),
                            'gpu_results_sha256': semantic.sha256_file(gpu_results),
                            'preparation': str(Path(preparation_path).resolve()),
                            'cpu_source_manifest_sha256': source_checksum,
                            'cpu_source_manifest': str(Path(cpu_source_manifest).resolve()) if source is not None else None,
                            'preparation_identity': preparation_identity},
                  protocol={'gpu_calls': 0, 'floating_verdict': 'error-report; no acceptance threshold',
                            'integer_verdict': 'exact unsigned element equality',
                            'timing': 'original checkpoint measurements are not changed'},
                  expected_cases=len(cases), completed_checkpoint_cases=len(rows), rows=reviewed)
    result['rows_sha256'] = semantic.digest(reviewed)
    cpu.write(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--gpu-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-source-manifest', type=Path)
    args = parser.parse_args(argv)
    result = review(args.manifest, args.preparation, args.gpu_results, args.output, args.cpu_source_manifest)
    print(json.dumps({'output': str(args.output.resolve()), 'reviewed_implementations': len(result['rows'])}))


if __name__ == '__main__':
    main()
