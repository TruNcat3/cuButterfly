#!/usr/bin/env python3
"""Prepare a timing-only continuation from stopped bounded workers.

Usage: prepare_bounded_queue_handoff.py --source-root CAMPAIGN
       --output FRESH_EXTERNAL_DIRECTORY --source-snapshot FROZEN_PYTHON_SOURCES
Stop workers and freeze their sources before calling. This utility never stops
processes or launches work. Controls and uncommitted orphans remain evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import time

import continue_semantic_campaign as continuation
import prepare_async_semantic_handoff as handoff
from prepare_interrupted_async_handoff import (check_compiler_failure_boundary,
                                              orphan_evidence, verify_identity)
import run_bounded_semantic_queue as bounded
import run_semantic_campaign as runner
from semantic_campaign_manifest import validate_manifest_hash


def process_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError('missing or invalid worker PID')
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path('/proc') / str(pid) / 'stat'
    return not (stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z')


def case_index(manifest):
    if not validate_manifest_hash(manifest):
        raise ValueError('manifest checksum mismatch')
    cases = {case['id']: case for case in manifest['cases']}
    if len(cases) != len(manifest['cases']) or any(
            case['id'] != runner.digest(case['descriptor']['semantics'])
            for case in manifest['cases']):
        raise ValueError('duplicate or invalid manifest case identity')
    return cases


def freeze_metadata(metadata, verified_signatures=None):
    signatures = {}
    for item in metadata['evidence']:
        signature = list(continuation.file_signature(item['path']))
        prior = (verified_signatures or {}).get(item['path'])
        recent = max(signature[-2:]) >= time.time_ns() - 1_000_000_000
        if prior != signature or recent:
            continuation.verify_evidence(item)
        if signature != list(continuation.file_signature(item['path'])):
            raise ValueError('evidence changed during audit: ' + item['path'])
        signatures[item['path']] = signature
    metadata['evidence_stat_signatures'] = signatures


def audit_phase(source, slot, spec, plan, state, snapshot):
    parent_path = Path(spec['manifest']).resolve(strict=True)
    parent = continuation.read(parent_path)
    cases = case_index(parent)
    library = Path(plan['library']).resolve(strict=True)
    runner.validate_fingerprints(parent, library)
    calibration_path = Path(plan['calibration_manifest']).resolve(strict=True)
    calibration = continuation.read(calibration_path)
    phase = state['phases'][spec['name']]
    directory = source / f'gpu{slot}' / spec['name']
    listed = {str(Path(shard['report']).resolve()) for shard in phase['shards']}
    found = {str(path.resolve()) for path in directory.glob('shard-*/measurement/results.json')}
    manifests = {str(path.parent / 'measurement/results.json') for path in directory.glob('shard-*/manifest.json')}
    if listed != found or listed != manifests or len(listed) != len(phase['shards']):
        raise ValueError('unindexed, missing, or duplicate shard')
    files = [continuation.evidence(parent_path), continuation.evidence(library),
             continuation.evidence(calibration_path)]
    processed, counts, reports, orphans, boundaries = set(), Counter(), [], [], []
    for shard_record in phase['shards']:
        if process_alive(shard_record.get('pid')):
            raise ValueError('worker is still alive')
        path = Path(shard_record['report']).resolve(strict=True)
        if not path.is_relative_to(directory.resolve()):
            raise ValueError('shard outside assigned GPU phase')
        shard_path = path.parent.parent / 'manifest.json'
        if Path(shard_record['manifest']).resolve(strict=True) != shard_path:
            raise ValueError('shard manifest path mismatch')
        shard = continuation.read(shard_path)
        shard_cases = case_index(shard)
        if any(case != cases.get(identifier) for identifier, case in shard_cases.items()):
            raise ValueError('shard changed original case object')
        report, rows, artifacts = continuation.checked_report(
            path, shard, library, plan['gpu_uuid'], bool(spec['timing']))
        status = report.get('status', '')
        if status not in ('interrupted', 'worker-recycle-required') and not status.startswith('complete'):
            raise ValueError('report requires investigation: ' + status)
        if status.startswith('complete') and report['remaining_case_ids']:
            raise ValueError('complete report has remaining cases')
        if report['identity'].get('sources') != state['identity']['sources']:
            raise ValueError('worker source identity mismatch')
        for name in report['identity']['sources']:
            if Path(name).name != name or not (snapshot / name).is_file():
                raise ValueError('missing frozen source: ' + name)
        verify_identity(report['identity'], calibration, True, snapshot)
        boundary = check_compiler_failure_boundary(path.parent.parent / 'worker.log')
        boundaries.append(boundary)
        for row in rows:
            identifier = row['id']
            if identifier in processed:
                raise ValueError('duplicate indexed case: ' + identifier)
            if row['status'] not in continuation.PROCESSED or continuation.fatal_baseline(row):
                raise ValueError('unadoptable indexed row: ' + identifier)
            case = cases[identifier]
            if any(row.get(field) != case.get(field) for field in ('input_profile', 'source_evidence')):
                raise ValueError('row input provenance mismatch')
            if 'execution_mapping' in case and row.get('mapping') != case['execution_mapping']:
                raise ValueError('row control mapping mismatch')
            processed.add(identifier)
            counts[row['status']] += 1
        extras = orphan_evidence(path.parent, {row['id'] for row in rows}, shard_cases)
        orphans.extend(extras)
        files.extend([continuation.evidence(shard_path), *artifacts, boundary['log']])
        for extra in extras:
            files.extend([extra['row'], *extra['artifacts']])
        reports.append({'results': continuation.evidence(path), 'indexed_rows': len(rows),
                        'status': status, 'identity': report['identity']})
    completed = phase.get('completed', {})
    if any(identifier not in processed for identifier in completed):
        raise ValueError('controller completed case missing from verified index')
    remaining = [case for case in parent['cases'] if case['id'] not in processed]
    metadata = {'slot': slot, 'gpu_uuid': plan['gpu_uuid'], 'timing': bool(spec['timing']),
                'parent_manifest': continuation.evidence(parent_path),
                'parent_manifest_sha256': parent['manifest_sha256'],
                'parent_cases': len(cases), 'processed_cases': len(processed),
                'remaining_cases': len(remaining), 'adopted_status_counts': dict(counts),
                'reports': reports, 'unadopted_orphans': orphans,
                'compiler_failure_boundaries': boundaries, 'evidence': files}
    return parent, parent_path, remaining, processed, metadata


def prepare(source, output, snapshot):
    source, output, snapshot = source.resolve(strict=True), output.resolve(), snapshot.resolve(strict=True)
    if output.exists() or output == source or source in output.parents:
        raise ValueError('output must be a fresh directory outside source root')
    prior_path = source / 'evidence/handoff.json'
    prior = bounded.verify_handoff(prior_path)
    phases, manifests, controls = {}, {}, {}
    all_ids = set()
    for slot in (1, 2):
        plan_path, state_path = source / f'gpu{slot}_plan.json', source / f'gpu{slot}/queue_status.json'
        plan, state = continuation.read(plan_path), continuation.read(state_path)
        continuation.verify_evidence(state['identity']['plan'])
        if Path(state['identity']['plan']['path']).resolve() != plan_path:
            raise ValueError('controller plan path mismatch')
        if Path(plan['handoff']).resolve() != prior_path:
            raise ValueError('prior handoff path mismatch')
        if state['identity']['gpu_uuid'] != plan['gpu_uuid']:
            raise ValueError('controller GPU identity mismatch')
        for filename, checksum in state['identity']['sources'].items():
            if Path(filename).name != filename or runner.sha256_file(snapshot / filename) != checksum:
                raise ValueError('controller source fingerprint mismatch')
        queue_source = state['identity'].get('queue_source')
        if queue_source and runner.sha256_file(snapshot / Path(queue_source['path']).name) != queue_source['sha256']:
            raise ValueError('controller queue source fingerprint mismatch')
        if len({spec['name'] for spec in plan['phases']}) != len(plan['phases']):
            raise ValueError('duplicate plan phase')
        name = f'timing_gpu{slot}'
        previous = prior['phases'][name]
        if previous['gpu_uuid'] != plan['gpu_uuid']:
            raise ValueError('GPU assignment changed')
        for spec in plan['phases']:
            parent, parent_path, remaining, processed, metadata = audit_phase(
                source, slot, spec, plan, state, snapshot)
            metadata['evidence'].extend([continuation.evidence(plan_path), continuation.evidence(state_path)])
            if not spec['timing']:
                if remaining:
                    raise ValueError('scaled controls are incomplete')
                freeze_metadata(metadata)
                controls[f'gpu{slot}_{spec["name"]}'] = metadata
                continue
            if spec['name'] != 'timing' or parent_path != Path(prior['manifests'][name + '.json']['path']):
                raise ValueError('timing parent differs from audited handoff')
            continuation.verify_evidence(prior['manifests'][name + '.json'])
            ids = {case['id'] for case in parent['cases']}
            if all_ids & ids:
                raise ValueError('duplicate case across GPUs')
            all_ids.update(ids)
            metadata.update(historical_processed_cases=previous['total_processed_cases'],
                            total_processed_cases=previous['total_processed_cases'] + len(processed),
                            missing_lowering_cases=previous['missing_lowering_cases'] + metadata['adopted_status_counts'].get('missing-lowering', 0),
                            prior_phase=previous, quarantine=[], quarantined_cases=0)
            metadata['evidence'].extend(previous['evidence'])
            metadata['evidence'].extend(continuation.evidence(path) for path in sorted(snapshot.glob('*.py')))
            metadata['evidence'] = list({item['path']: item for item in metadata['evidence']}.values())
            freeze_metadata(metadata, previous.get('evidence_stat_signatures'))
            phases[name] = metadata
            manifests[name + '.json'] = continuation.submanifest(
                parent, parent_path, remaining, 'bounded-continuation-' + name, processed, [])
    if set(phases) != {'timing_gpu1', 'timing_gpu2'}:
        raise ValueError('missing timing phase')
    if sum(p['historical_processed_cases'] for p in phases.values()) != prior['processed_cases']:
        raise ValueError('historical processed count mismatch')
    for name, metadata in phases.items():
        signatures = dict(metadata['evidence_stat_signatures'])
        metadata['evidence'].append(continuation.evidence(prior_path))
        for control in controls.values():
            if control['slot'] == metadata['slot']:
                metadata['evidence'].extend(control['evidence'])
                signatures.update(control['evidence_stat_signatures'])
        metadata['evidence'] = list({item['path']: item for item in metadata['evidence']}.values())
        freeze_metadata(metadata, signatures)
    # Publish only after every phase and artifact has passed the audit.
    for name, manifest in manifests.items():
        handoff.atomic_save(output / name, manifest)
    document = {'schema': 'cubutterfly-bounded-queue-handoff-v1', 'status': 'prepared',
                'source_root': str(source), 'prior_handoff': continuation.evidence(prior_path),
                'script': continuation.evidence(__file__), 'phases': phases,
                'scaled_controls': controls,
                'scaled_control_cases': sum(p['processed_cases'] for p in controls.values()),
                'manifests': {name: continuation.evidence(output / name) for name in manifests},
                'original_manifests': prior['manifests'],
                'processed_cases': sum(p['total_processed_cases'] for p in phases.values()),
                'remaining_cases': sum(p['remaining_cases'] for p in phases.values()),
                'missing_lowering_cases': sum(p['missing_lowering_cases'] for p in phases.values()),
                'scope': 'Same GPU timing population; completed controls retained separately; unindexed orphans rerun.'}
    handoff.atomic_save(output / 'handoff.json', document)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-snapshot', type=Path, required=True)
    args = parser.parse_args(argv)
    document = prepare(args.source_root, args.output, args.source_snapshot)
    print(runner.canonical({key: document[key] for key in
                           ('processed_cases', 'remaining_cases', 'missing_lowering_cases', 'scaled_control_cases')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
