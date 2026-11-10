#!/usr/bin/env python3
"""Validate immutable historical evidence and emit fresh asynchronous queues."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import continue_semantic_campaign as continuation
import run_semantic_campaign as runner
from semantic_campaign_manifest import validate_manifest_hash


PHASES = (('recovery_gpu1', 1, False, 'prepared/recovery/manifest.json', 'recovery/conformance'),
          ('timing_gpu1', 1, True, 'prepared/full-timing-shard-1/manifest.json', 'full_timing/validated-timing'),
          ('timing_gpu2', 2, True, 'prepared/full-timing-shard-2/manifest.json', 'full_timing/validated-timing'))


def atomic_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(runner.canonical(value) + '\n')
    temporary.replace(path)


def freeze_evidence_stat_signatures(metadata):
    """Bind same-host startup checks to the files whose hashes were verified."""
    signatures = {}
    for item in metadata['evidence']:
        before = continuation.file_signature(item['path'])
        continuation.verify_evidence(item)
        after = continuation.file_signature(item['path'])
        if before != after:
            raise ValueError('evidence changed between verification and stat capture: ' + item['path'])
        signatures[item['path']] = list(after)
    metadata['evidence_stat_signatures'] = signatures


def pilot_cases(cases, count):
    """Keep parent order while including each operator when the budget allows."""
    if count < 1:
        raise ValueError('pilot count must be positive')
    first = {}
    for index, case in enumerate(cases):
        first.setdefault(case['descriptor']['semantics']['operator'], index)
    selected = set(list(first.values())[:count])
    for index in range(len(cases)):
        if len(selected) >= min(count, len(cases)):
            break
        selected.add(index)
    return [case for index, case in enumerate(cases) if index in selected]


def prepare_phase(source, spec, pilot_only=False):
    name, slot, timing, relative_parent, relative_history = spec
    parent_path = source / relative_parent
    parent = continuation.read(parent_path)
    if not validate_manifest_hash(parent):
        raise ValueError('parent manifest checksum mismatch: ' + str(parent_path))
    parent_cases = {case['id']: case for case in parent['cases']}
    if len(parent_cases) != len(parent['cases']) or any(
            case['id'] != runner.digest(case['descriptor']['semantics']) for case in parent['cases']):
        raise ValueError('parent case checksum/uniqueness mismatch')
    library = Path(parent['fingerprints']['library']['path'])
    runner.validate_fingerprints(parent, library)
    calibration_path = source / f'gpu{slot}/calibration/calibration_manifest.json'
    calibration = continuation.read(calibration_path)
    profile_path = Path(calibration['hardware_rates_source'])
    profile = continuation.read(profile_path)
    gpu_uuid = profile['provenance']['cuda_visible_devices']
    # Parse UUID now, including pilot-only mode which has no report identity.
    runner.same_gpu_uuid(gpu_uuid, gpu_uuid)
    library_sha = runner.sha256_file(library)
    if calibration.get('measurement_identity', {}).get('library_sha256') != library_sha:
        raise ValueError('slot calibration library mismatch')
    evidence = [continuation.evidence(parent_path), continuation.evidence(library),
                continuation.evidence(calibration_path), continuation.evidence(profile_path)]
    for item in parent['fingerprints'].get('sources', []):
        evidence.append(continuation.evidence(item['path']))
    if parent['fingerprints'].get('config', {}).get('path'):
        evidence.append(continuation.evidence(parent['fingerprints']['config']['path']))
    for field in ('registry_path', 'cost_model'):
        if calibration.get(field):
            evidence.append(continuation.evidence(calibration[field]))
    processed, quarantine, adopted, reports = set(), {}, Counter(), []
    if not pilot_only:
        history = source / f'gpu{slot}' / relative_history
        for result_path in sorted(history.glob('shard-*/measurement/results.json')):
            shard_path = result_path.parent.parent / 'manifest.json'
            shard = continuation.read(shard_path)
            if not validate_manifest_hash(shard):
                raise ValueError('shard manifest checksum mismatch')
            for case in shard['cases']:
                if case != parent_cases.get(case['id']):
                    raise ValueError('shard case changed from parent: ' + case['id'])
            report, rows, files = continuation.checked_report(result_path, shard, library, gpu_uuid, timing)
            report_status = report.get('status', '')
            if not (report_status.startswith('complete') or report_status in
                    ('running', 'tranche-complete', 'interrupted', 'stopped-correctness-failure')):
                raise ValueError('historical report stop requires investigation: ' + report_status)
            identity = report['identity']
            if identity.get('protocol') != dict(runner.PROTOCOL, float_verdict='error-report'):
                raise ValueError('historical verification protocol mismatch')
            if identity.get('runner_sha256') != runner.sha256_file(runner.__file__):
                raise ValueError('historical frozen runner build mismatch')
            if identity.get('calibration_sha256') != runner.digest(calibration):
                raise ValueError('historical slot calibration mismatch')
            if identity.get('compile_mode') != 'research':
                raise ValueError('historical build compile mode mismatch')
            runner.validate_calibration(calibration, identity)
            for path_key, hash_key, calibration_key in (
                    ('registry', 'registry_sha256', 'registry_path'),
                    ('hardware_profile', 'hardware_profile_sha256', 'hardware_rates_source'),
                    ('cost_model', 'cost_model_sha256', 'cost_model')):
                target = Path(calibration[calibration_key]).resolve(strict=True)
                if Path(identity[path_key]).resolve(strict=True) != target or identity[hash_key] != runner.sha256_file(target):
                    raise ValueError('historical calibration component mismatch: ' + path_key)
            failures = [row for row in rows if row['status'] == 'correctness-failure']
            if failures and (len(failures) != 1 or rows[-1] is not failures[0]):
                raise ValueError('ambiguous historical correctness stop')
            if report_status == 'stopped-correctness-failure' and not failures:
                raise ValueError('correctness stop has no failure evidence')
            if [row['id'] for row in rows] != [case['id'] for case in shard['cases']][:len(rows)]:
                raise ValueError('historical shard execution order mismatch')
            for row in rows:
                if continuation.fatal_baseline(row):
                    raise ValueError('historical baseline runtime error: ' + row['id'])
                status = row['status']
                if status not in continuation.PROCESSED | {'correctness-failure'}:
                    raise ValueError('historical runtime/runner error: ' + row['id'])
                adopted[status] += 1
                if status == 'correctness-failure':
                    quarantine[row['id']] = continuation.quarantine_entry(row, result_path)
                else:
                    processed.add(row['id'])
            evidence.extend([continuation.evidence(shard_path), *files])
            reports.append({'results': continuation.evidence(result_path), 'status': report['status'],
                            'rows': len(rows), 'remaining': len(report['remaining_case_ids'])})
    if processed & set(quarantine):
        raise ValueError('case has conflicting processed and quarantine evidence')
    remaining = [case for case in parent['cases'] if case['id'] not in processed | set(quarantine)]
    manifest = continuation.submanifest(parent, parent_path, remaining,
        'async-' + name, processed, [quarantine[key] for key in sorted(quarantine)])
    metadata = {'slot': slot, 'timing': timing, 'gpu_uuid': gpu_uuid,
        'parent_cases': len(parent['cases']), 'processed_cases': len(processed),
        'quarantined_cases': len(quarantine), 'remaining_cases': len(remaining),
        'adopted_status_counts': dict(sorted(adopted.items())), 'historical_reports': reports,
        'quarantine': [quarantine[key] for key in sorted(quarantine)],
        'evidence': list({item['path']: item for item in evidence}.values())}
    return manifest, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot-cases', type=int)
    parser.add_argument('--pilot-only', action='store_true')
    args = parser.parse_args(argv)
    if args.pilot_cases is not None and args.pilot_cases < 1:
        parser.error('--pilot-cases must be positive')
    if args.pilot_only and args.pilot_cases is None:
        parser.error('--pilot-only requires --pilot-cases')
    source, output = args.source_root.resolve(strict=True), args.output.resolve()
    if output.exists():
        raise ValueError('handoff output must be a new directory')
    if output == source or source in output.parents:
        raise ValueError('handoff output must be outside historical source root')
    documents, phases = {}, {}
    for spec in (PHASES[2:] if args.pilot_only else PHASES):
        manifest, metadata = prepare_phase(source, spec, args.pilot_only)
        phases[spec[0]] = metadata
        if not args.pilot_only:
            documents[spec[0] + '.json'] = manifest
        if spec[0] == 'timing_gpu2' and args.pilot_cases is not None:
            parent_path = source / spec[3]
            parent = continuation.read(parent_path)
            pilot = continuation.submanifest(parent, parent_path, pilot_cases(manifest['cases'], args.pilot_cases),
                'async-timing-gpu2-pilot', set(), [])
            documents['pilot_timing_gpu2.json'] = pilot
    # Recheck every adopted file immediately before publishing the handoff.
    for metadata in phases.values():
        freeze_evidence_stat_signatures(metadata)
    for name, document in documents.items():
        atomic_save(output / name, document)
    handoff = {'schema': 'cubutterfly-async-handoff-v1', 'source_root': str(source),
        'mode': 'pilot-only-parent' if args.pilot_only else 'validated-historical-adoption',
        'status': 'prepared', 'phases': phases,
        'manifests': {name: continuation.evidence(output / name) for name in documents},
        'pilot_excludes_from_remaining': False,
        'scope': 'Scaled controls and unresolved quarantines remain in the campaign scope; no GPU phase launched.'}
    atomic_save(output / 'handoff.json', handoff)
    print(runner.canonical({'output': str(output), 'mode': handoff['mode'],
                           'remaining': {name: value['remaining_cases'] for name, value in phases.items()}}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
