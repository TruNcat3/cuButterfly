#!/usr/bin/env python3
"""Audit interrupted asynchronous timing evidence into fresh remaining manifests."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import continue_semantic_campaign as continuation
import prepare_async_semantic_handoff as handoff
import run_semantic_campaign as runner
from semantic_campaign_manifest import validate_manifest_hash


HERE = Path(__file__).resolve().parent


def verify_identity(identity, calibration, asynchronous, snapshot=None):
    if identity.get('protocol') != dict(runner.PROTOCOL, float_verdict='error-report'):
        raise ValueError('verification protocol mismatch')
    if identity.get('compile_mode') != 'research':
        raise ValueError('compile policy mismatch')
    if identity.get('calibration_sha256') != runner.digest(calibration):
        raise ValueError('calibration identity mismatch')
    runner.validate_calibration(calibration, identity)
    for key, calibration_key in (('registry', 'registry_path'),
                                 ('hardware_profile', 'hardware_rates_source'),
                                 ('cost_model', 'cost_model')):
        path = Path(calibration[calibration_key]).resolve(strict=True)
        if (Path(identity[key]).resolve(strict=True) != path
                or identity[key + '_sha256'] != runner.sha256_file(path)):
            raise ValueError('calibration component mismatch: ' + key)
    if asynchronous:
        for name, checksum in identity['sources'].items():
            path = snapshot / name if snapshot and (snapshot / name).is_file() else HERE / name
            if Path(name).name != name or runner.sha256_file(path) != checksum:
                raise ValueError('asynchronous source fingerprint mismatch: ' + name)
        frontend = HERE.parent / 'python/cubutterfly/__init__.py'
        if runner.sha256_file(frontend) != identity['frontend_sha256']:
            raise ValueError('frontend fingerprint mismatch')
    elif identity['runner_sha256'] != runner.sha256_file(runner.__file__):
        raise ValueError('historical runner fingerprint mismatch')


def check_compiler_failure_boundary(path):
    first_failure, unsafe = None, []
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if first_failure is None and 'could not start specialization compiler' in line:
                first_failure = number
            if first_failure is not None:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if value.get('status') in ('pass', 'accuracy-review-required') and 'completed' in value:
                    unsafe.append({'line': number, 'case': value.get('case')})
    if unsafe:
        raise ValueError('successful commit after compiler failure; adoption unsafe: ' + runner.canonical(unsafe))
    return {'log': continuation.evidence(path), 'first_compiler_failure_line': first_failure,
            'successful_commits_after_failure': 0, 'adoption_safe': True}


def orphan_evidence(directory, indexed, cases):
    records = []
    for path in sorted((directory / 'rows').glob('*.json')):
        if path.stem in indexed:
            continue
        envelope = continuation.read(path)
        row = envelope['row']
        if (row.get('id') != path.stem or runner.digest(row) != envelope.get('sha256')
                or row.get('descriptor') != cases[path.stem]['descriptor']):
            raise ValueError('orphan checksum/descriptor mismatch: ' + str(path))
        artifacts = []
        for implementation in [row, *row.get('baselines', [])]:
            if implementation.get('error_artifact'):
                artifact = implementation['error_artifact']
                continuation.verify_evidence(artifact)
                artifacts.append(artifact)
        records.append({'id': row['id'], 'row': continuation.evidence(path),
                        'status': row['status'], 'error': row.get('error'),
                        'artifacts': artifacts, 'adopted': False,
                        'action': 'rerun; row was not committed to the result index'})
    return records


def prepare_slot(source, slot, prior, snapshot=None):
    phase = 'timing_gpu' + str(slot)
    parent_path = source / 'handoff' / (phase + '.json')
    parent = continuation.read(parent_path)
    if not validate_manifest_hash(parent):
        raise ValueError('parent manifest checksum mismatch')
    cases = {case['id']: case for case in parent['cases']}
    if len(cases) != len(parent['cases']) or any(
            case['id'] != runner.digest(case['descriptor']['semantics']) for case in parent['cases']):
        raise ValueError('parent case identity mismatch')
    library = Path(parent['fingerprints']['library']['path'])
    runner.validate_fingerprints(parent, library)
    previous = prior['phases'][phase]
    evidence = [continuation.evidence(parent_path), continuation.evidence(library)]
    for item in previous['evidence']:
        continuation.verify_evidence(item)
        evidence.append(item)
    calibration_path = Path(prior['source_root']) / f'gpu{slot}/calibration/calibration_manifest.json'
    calibration = continuation.read(calibration_path)
    uuid = previous['gpu_uuid']
    boundary = check_compiler_failure_boundary(source / f'gpu{slot}/supervision/queue.log')
    evidence.append(boundary['log'])
    if snapshot:
        evidence.extend(continuation.evidence(path) for path in sorted(snapshot.glob('*.py')))
    historical_counts = Counter()
    for record in previous['historical_reports']:
        path = Path(record['results']['path'])
        shard = continuation.read(path.parent.parent / 'manifest.json')
        report, rows, files = continuation.checked_report(path, shard, library, uuid, True)
        verify_identity(report['identity'], calibration, False)
        historical_counts.update(row['status'] for row in rows)
        del rows
        evidence.extend(files)
    processed, counts, reports, orphans = set(), Counter(), [], []
    for path in sorted((source / f'gpu{slot}/timing').glob('shard-*/measurement/results.json')):
        shard_path = path.parent.parent / 'manifest.json'
        shard = continuation.read(shard_path)
        if not validate_manifest_hash(shard):
            raise ValueError('shard manifest checksum mismatch')
        if any(case != cases.get(case['id']) for case in shard['cases']):
            raise ValueError('shard changed the parent case object')
        report, rows, files = continuation.checked_report(path, shard, library, uuid, True)
        verify_identity(report['identity'], calibration, True, snapshot)
        if report.get('status') not in ('running', 'interrupted', 'tranche-complete') and not report.get('status', '').startswith('complete'):
            raise ValueError('report requires investigation: ' + report.get('status', ''))
        for row in rows:
            if row['id'] in processed or row['status'] not in continuation.PROCESSED or continuation.fatal_baseline(row):
                raise ValueError('duplicate or unadoptable indexed row: ' + row['id'])
            processed.add(row['id'])
            counts[row['status']] += 1
        del rows
        extras = orphan_evidence(path.parent, processed, cases)
        orphans.extend(extras)
        evidence.extend([continuation.evidence(shard_path), *files])
        for extra in extras:
            evidence.extend([extra['row'], *extra['artifacts']])
        reports.append({'results': continuation.evidence(path), 'indexed_rows': len(report['rows']),
                        'status': report['status'], 'identity': report['identity']})
        print(runner.canonical({'slot': slot, 'audited_indexed_rows': len(processed),
                                'unadopted_orphans': len(orphans)}), flush=True)
    remaining = [case for case in parent['cases'] if case['id'] not in processed]
    manifest = continuation.submanifest(parent, parent_path, remaining,
                                         'bounded-interrupted-' + phase, processed, [])
    metadata = {'slot': slot, 'gpu_uuid': uuid, 'timing': True,
                'parent_cases': len(parent['cases']), 'processed_cases': len(processed),
                'remaining_cases': len(remaining), 'quarantined_cases': 0,
                'quarantine': [], 'adopted_status_counts': dict(counts),
                'historical_status_counts': dict(historical_counts),
                'historical_processed_cases': sum(historical_counts.values()),
                'total_processed_cases': len(processed) + sum(historical_counts.values()),
                'missing_lowering_cases': counts['missing-lowering'] + historical_counts['missing-lowering'],
                'reports': reports, 'unadopted_orphans': orphans,
                'compiler_failure_boundary': boundary,
                'evidence': list({item['path']: item for item in evidence}.values())}
    handoff.freeze_evidence_stat_signatures(metadata)
    return manifest, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-snapshot', type=Path)
    args = parser.parse_args(argv)
    source, output = args.source_root.resolve(strict=True), args.output.resolve()
    existing = set(path.name for path in output.iterdir()) if output.exists() else set()
    if existing - {'source_snapshot'} or output == source or source in output.parents:
        raise ValueError('output must be a new directory outside the frozen source root')
    prior_path = source / 'handoff/handoff.json'
    prior = continuation.read(prior_path)
    snapshot = args.source_snapshot.resolve(strict=True) if args.source_snapshot else None
    phases, manifests = {}, {}
    for slot in (1, 2):
        name = f'timing_gpu{slot}'
        manifest, metadata = prepare_slot(source, slot, prior, snapshot)
        handoff.atomic_save(output / (name + '.json'), manifest)
        manifests[name + '.json'] = continuation.evidence(output / (name + '.json'))
        phases[name] = metadata
    document = {'schema': 'cubutterfly-interrupted-async-handoff-v1', 'status': 'prepared',
                'source_root': str(source), 'prior_handoff': continuation.evidence(prior_path),
                'script': continuation.evidence(__file__), 'phases': phases, 'manifests': manifests,
                'processed_cases': sum(p['total_processed_cases'] for p in phases.values()),
                'missing_lowering_cases': sum(p['missing_lowering_cases'] for p in phases.values()),
                'remaining_cases': sum(p['remaining_cases'] for p in phases.values()),
                'scope': 'Timing only. Completed recovery and prepared scaled controls are retained in their original campaign; orphans remain unadopted.'}
    handoff.atomic_save(output / 'handoff.json', document)
    print(runner.canonical({key: document[key] for key in ('processed_cases', 'missing_lowering_cases', 'remaining_cases')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
