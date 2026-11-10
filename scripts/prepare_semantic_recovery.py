#!/usr/bin/env python3
"""Prepare fresh cross-build populations from historical metadata, without CUDA.

Old rows establish selection and historical progress only. Their numerical
artifacts are not read and their correctness or timings are never imported.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import continue_semantic_campaign as campaign


def checked_manifest(path):
    manifest = campaign.read(path)
    if not campaign.validate_manifest_hash(manifest):
        raise ValueError('manifest checksum mismatch: ' + str(path))
    if manifest.get('schema') != 'cubutterfly-semantic-campaign-v1':
        raise ValueError('unsupported manifest schema: ' + str(path))
    cases = {}
    for case in manifest['cases']:
        identifier = case['id']
        if identifier in cases:
            raise ValueError('duplicate manifest case ID: ' + identifier)
        if identifier != campaign.runner.digest(case['descriptor']['semantics']):
            raise ValueError('descriptor semantic ID mismatch: ' + identifier)
        cases[identifier] = case
    return manifest, cases


def affected(case):
    semantics = case['descriptor']['semantics']
    if semantics['operator'] == 1:
        return 'ntt-required-recheck'
    if (semantics['operator'] in (0, 2) and semantics['storage'] in (0, 1, 4, 5)
            and semantics['inverse'] and semantics['normalize_inverse']):
        return 'normalized-low-precision-inverse-required-recheck'
    return None


def checked_report(path, manifest, old_cases, seen):
    report = campaign.read(path)
    if report.get('schema') != 'cubutterfly-semantic-results-v1':
        raise ValueError('unsupported result schema: ' + str(path))
    if report.get('rows_sha256') != campaign.runner.digest(report['rows']):
        raise ValueError('result row index checksum mismatch: ' + str(path))
    identity = report['identity']
    if (identity.get('manifest_sha256') != campaign.runner.digest(manifest)
            or identity.get('timing') is not False
            or identity.get('library_sha256') != manifest['fingerprints']['library']['sha256']):
        raise ValueError('historical result manifest/library/phase identity mismatch')
    allowed = [case['id'] for case in manifest['cases']
               if identity.get('population') is None or identity['population'] in case['populations']]
    rows, metadata = [], [campaign.evidence(path)]
    identifiers = set()
    for index in report['rows']:
        identifier = index['id']
        if identifier in identifiers or identifier in seen:
            raise ValueError('duplicate historical row ID: ' + identifier)
        identifiers.add(identifier)
        if index['path'] != 'rows/' + identifier + '.json':
            raise ValueError('invalid historical row path: ' + identifier)
        row_path = path.parent / index['path']
        envelope = campaign.read(row_path)
        row = envelope['row']
        checksum = campaign.runner.digest(row)
        if envelope.get('sha256') != checksum or index.get('sha256') != checksum:
            raise ValueError('row envelope checksum mismatch: ' + identifier)
        if row.get('id') != identifier or row.get('status') != index.get('status'):
            raise ValueError('row index identity/status mismatch: ' + identifier)
        if (identifier not in old_cases
                or identifier != campaign.runner.digest(row['descriptor']['semantics'])
                or row['descriptor'] != old_cases[identifier]['descriptor']):
            raise ValueError('historical row descriptor semantic mismatch: ' + identifier)
        rows.append(row)
        metadata.append(campaign.evidence(row_path))
    if [row['id'] for row in rows] != allowed[:len(rows)]:
        raise ValueError('historical row execution order mismatch')
    if report.get('remaining_case_ids') != allowed[len(rows):]:
        raise ValueError('historical remaining-case index mismatch')
    seen.update(identifiers)
    return report, rows, metadata


def prepare(old_manifest_path, old_controller_path, new_manifest_path, output_dir, timing_shards=1):
    if type(timing_shards) is not int or timing_shards < 1:
        raise ValueError('timing-shards must be a positive integer')
    old_manifest_path, old_controller_path, new_manifest_path, output_dir = (
        Path(path).resolve() for path in
        (old_manifest_path, old_controller_path, new_manifest_path, output_dir))
    if output_dir.exists():
        raise ValueError('use a fresh output directory: ' + str(output_dir))
    old_manifest, old_cases = checked_manifest(old_manifest_path)
    new_manifest, new_cases = checked_manifest(new_manifest_path)
    if old_cases.keys() != new_cases.keys():
        raise ValueError('old/new semantic ID sets differ: removed=%s added=%s' % (
            sorted(old_cases.keys() - new_cases.keys()), sorted(new_cases.keys() - old_cases.keys())))
    for identifier, case in old_cases.items():
        if case['descriptor'] != new_cases[identifier]['descriptor']:
            raise ValueError('old/new descriptor differs: ' + identifier)
    state = campaign.load_state(old_controller_path)
    if state['identity'].get('phase') != 'conformance':
        raise ValueError('old controller must describe conformance')
    old_evidence = campaign.evidence(old_manifest_path)
    if state['identity']['inputs'][0] != old_evidence:
        raise ValueError('old controller parent manifest binding mismatch')
    metadata = [old_evidence, campaign.evidence(old_controller_path), campaign.evidence(new_manifest_path)]
    reports, historical, seen = [], {}, set()
    sources = [(Path(item['path']), old_manifest)
               for item in state['identity'].get('prior_evidence', [])
               if Path(item['path']).name == 'results.json']
    for shard in state['shards']:
        manifest_path = Path(shard['manifest']['path'])
        if campaign.evidence(manifest_path) != shard['manifest']:
            raise ValueError('shard manifest evidence mismatch: ' + str(manifest_path))
        manifest, cases = checked_manifest(manifest_path)
        if any(identifier not in old_cases or case != old_cases[identifier]
               for identifier, case in cases.items()):
            raise ValueError('historical shard cases differ from old parent')
        metadata.append(campaign.evidence(manifest_path))
        sources.append((Path(shard['output']) / 'results.json', manifest))
    if not sources:
        raise ValueError('old controller has no historical reports')
    for path, manifest in sources:
        report, rows, files = checked_report(path, manifest, old_cases, seen)
        reports.append({'file': files[0], 'status': report['status'],
                        'row_count': len(rows), 'remaining_case_ids': report['remaining_case_ids']})
        metadata.extend(files)
        historical.update((row['id'], row['status']) for row in rows)
    checked_files = {item['path']: item['sha256'] for item in metadata}
    frozen = list(state['identity'].get('prior_evidence', []))
    for shard in state['shards']:
        frozen.extend(shard.get('evidence', []))
    for item in frozen:
        # Only compare files already checked above, never load numerical arrays.
        path = str(Path(item['path']).resolve())
        if path in checked_files and checked_files[path] != item['sha256']:
            raise ValueError('historical metadata evidence checksum mismatch: ' + path)
    remaining = reports[-1]['remaining_case_ids']
    if seen & set(remaining) or seen | set(remaining) != set(old_cases):
        raise ValueError('historical rows and latest remaining do not partition old population')
    for item in state.get('quarantine', []):
        if historical.get(item['id']) != 'correctness-failure':
            raise ValueError('controller quarantine has no checked failure row: ' + item['id'])
    reasons = {'latest-unvisited': remaining,
               'historical-correctness-failure': [key for key, status in historical.items()
                                                   if status == 'correctness-failure'],
               'historical-runtime-failure': [key for key, status in historical.items()
                                              if status in ('runtime-error', 'runner-error')]}
    for case in new_manifest['cases']:
        reason = affected(case)
        if reason:
            reasons.setdefault(reason, []).append(case['id'])
    selected = {identifier for identifiers in reasons.values() for identifier in identifiers}
    recovery_cases = [case for case in new_manifest['cases'] if case['id'] in selected]
    recovery = campaign.submanifest(new_manifest, new_manifest_path, recovery_cases,
                                    'conformance', set(), [])
    full_timing = campaign.submanifest(new_manifest, new_manifest_path, new_manifest['cases'],
                                       'validated-timing', set(), [])
    assignments = [new_manifest['cases'][number::timing_shards] for number in range(timing_shards)]
    assigned_ids = [case['id'] for cases in assignments for case in cases]
    assert len(assigned_ids) == len(set(assigned_ids))
    assert set(assigned_ids) == set(new_cases)
    timing_manifests = [campaign.submanifest(new_manifest, new_manifest_path, cases,
                                             'validated-timing', set(), [])
                        for cases in assignments]
    selection = {
        'schema': 'cubutterfly-semantic-recovery-selection-v1',
        'status': 'prepared-not-executed',
        'validation_scope': 'metadata/index/envelope checks only; no acceptance of old measurements or error-artifact reads',
        'old_measurements_imported': False, 'old_timings_imported': False,
        'evidence': metadata, 'reports': reports,
        'selection_reasons': {reason: {'count': len(ids), 'case_ids': ids}
                              for reason, ids in reasons.items()},
        'recovery_case_ids': [case['id'] for case in recovery_cases],
        'recovery_count': len(recovery_cases), 'full_timing_count': len(new_cases),
        'timing_shards': timing_shards,
        'timing_assignment': [dict(shard=number, manifest='full-timing-shard-%d/manifest.json' % number,
                                   count=len(cases), case_ids=[case['id'] for case in cases])
                              for number, cases in enumerate(assignments, start=1)],
        'timing_shard_policy': 'round-robin original case order; each GPU requires its own calibration, UUID, baselines and measurement identity',
        'historical_status_by_case_id': historical,
        'historically_completed_required_recheck_case_ids': [case['id'] for case in recovery_cases
            if historical.get(case['id']) in campaign.PROCESSED and affected(case)],
        'full_timing_policy': 'retain every valid request including boundary negatives; exclude boundary rows only in performance statistics',
        'new_validation_policy': 'fresh per-case complete reference and mapping replay before timing; original inputs and error-report verdict retained',
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    campaign.save(output_dir / 'recovery/manifest.json', recovery)
    campaign.save(output_dir / 'full-timing/manifest.json', full_timing)
    for number, manifest in enumerate(timing_manifests, start=1):
        campaign.save(output_dir / ('full-timing-shard-%d/manifest.json' % number), manifest)
    campaign.save(output_dir / 'recovery_selection.json', selection)
    return selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('old-manifest', 'old-controller', 'new-manifest', 'output-dir'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--timing-shards', type=int, default=1,
                        help='positive number of round-robin timing populations; each GPU needs local calibration')
    args = parser.parse_args(argv)
    if args.timing_shards < 1:
        parser.error('--timing-shards must be a positive integer')
    selection = prepare(args.old_manifest, args.old_controller, args.new_manifest, args.output_dir,
                        timing_shards=args.timing_shards)
    print('prepared recovery=%d full-timing=%d; no measurements imported' % (
        selection['recovery_count'], selection['full_timing_count']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
