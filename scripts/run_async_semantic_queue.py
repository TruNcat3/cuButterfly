#!/usr/bin/env python3
"""Run remaining asynchronous GPU phases from an audited handoff.

Only confirmed correctness failures are quarantined before advancing. Runtime,
compiler, identity and baseline failures stop this queue for investigation.
Scaled-input controls remain separate from original stress measurements.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import continue_semantic_campaign as continuation
import run_semantic_campaign as old
from run_async_semantic_campaign import source_identity

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def save(path, data):
    continuation.save(path, data)


def run_phase(name, manifest_path, cache, args, state, common):
    parent = continuation.read(manifest_path)
    cases = {case['id']: case for case in parent['cases']}
    phase = state.setdefault('phases', {}).setdefault(name, {'status': 'pending', 'shards': [], 'quarantine': []})
    if phase['status'].startswith('complete'):
        return
    processed, quarantine = set(), {}
    for shard in phase['shards']:
        report, rows, _ = continuation.checked_report(shard['report'], continuation.read(shard['manifest']),
            args.library, args.gpu_uuid, name.startswith('timing'))
        for row in rows:
            if row['descriptor'] != cases[row['id']]['descriptor']:
                raise ValueError('phase row descriptor mismatch')
            if continuation.fatal_baseline(row):
                raise ValueError('prior baseline error requires investigation')
            for field in ('input_profile', 'source_evidence'):
                if row.get(field) != cases[row['id']].get(field):
                    raise ValueError('phase input provenance mismatch: ' + field)
            if cases[row['id']].get('execution_mapping') is not None and row.get('mapping') != cases[row['id']]['execution_mapping']:
                raise ValueError('control mapping provenance mismatch')
            if row['status'] == 'correctness-failure':
                quarantine[row['id']] = continuation.quarantine_entry(row, shard['report'])
            elif row['status'] in continuation.PROCESSED:
                processed.add(row['id'])
            else:
                raise ValueError('cannot continue an unexplained prior error')
        if report['remaining_case_ids'] and not report['status'].startswith('stopped'):
            raise ValueError('interrupted shard needs explicit runner resume before queue continuation')
    while remaining := [case for case in parent['cases'] if case['id'] not in processed | set(quarantine)]:
        directory = args.output / name / ('shard-%04d' % len(phase['shards']))
        directory.mkdir(parents=True, exist_ok=False)
        selected = continuation.submanifest(parent, manifest_path, remaining, name, processed, list(quarantine.values()))
        selected_path = directory / 'manifest.json'
        save(selected_path, selected)
        result_path = directory / 'measurement/results.json'
        command = [sys.executable, '-u', str(HERE / 'run_async_semantic_campaign.py'),
            '--manifest', str(selected_path), '--output', str(result_path.parent),
            '--preparation', str(cache), '--compile-broker', str(args.compile_broker),
            '--cpu-workers', str(args.cpu_workers), '--in-flight', str(args.in_flight), *common]
        if name.startswith('timing'):
            command.append('--timing')
        shard = {'manifest': str(selected_path), 'report': str(result_path), 'command': command, 'status': 'running'}
        phase['shards'].append(shard)
        phase['status'] = 'running'
        state.update(status='running', current_phase=name, updated_unix=time.time())
        save(args.output / 'queue_status.json', state)
        completed = subprocess.run(command)
        shard['returncode'] = completed.returncode
        report, rows, _ = continuation.checked_report(result_path, selected, args.library,
            args.gpu_uuid, name.startswith('timing'))
        failures = []
        for row in rows:
            if continuation.fatal_baseline(row):
                raise RuntimeError('baseline error requires investigation')
            if row['status'] == 'correctness-failure':
                failures.append(row)
                quarantine[row['id']] = continuation.quarantine_entry(row, result_path)
            elif row['status'] in continuation.PROCESSED:
                processed.add(row['id'])
            else:
                raise RuntimeError('unexpected measurement error: ' + row.get('error', row['status']))
        if completed.returncode and (len(failures) != 1 or not report['status'].startswith('stopped')):
            raise RuntimeError('worker stopped without one verified correctness failure: ' + report.get('stop_reason', ''))
        if not completed.returncode and report['remaining_case_ids']:
            raise RuntimeError('worker returned success with incomplete population')
        if not rows:
            raise RuntimeError('worker made no verified progress')
        shard['status'] = 'closed-with-quarantine' if failures else 'complete'
        phase.update(processed_cases=len(processed), quarantine=list(quarantine.values()))
        save(args.output / 'queue_status.json', state)
    phase.update(status='complete-with-unresolved-evidence', processed_cases=len(processed),
                 quarantine=list(quarantine.values()))
    save(args.output / 'queue_status.json', state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('handoff', 'output', 'preparation', 'compile-broker'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--slot', type=int, choices=(1, 2), required=True)
    parser.add_argument('--cpu-workers', type=int, default=4)
    parser.add_argument('--in-flight', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    for name in ('handoff', 'output', 'preparation', 'compile_broker'):
        setattr(args, name, getattr(args, name).resolve())
    handoff = continuation.read(args.handoff)
    if handoff['mode'] != 'validated-historical-adoption':
        raise ValueError('a diagnostic pilot is not a production handoff')
    source = Path(handoff['source_root'])
    args.library = ROOT / 'build-e13-cufftdx/libcuntt.so'
    args.gpu_uuid = handoff['phases'][f'timing_gpu{args.slot}']['gpu_uuid']
    calibration = source / f'gpu{args.slot}/calibration/calibration_manifest.json'
    common = ['--library', str(args.library), '--gpu-uuid', args.gpu_uuid,
        '--calibration-manifest', str(calibration),
        '--vkfft-library', str(ROOT / 'build-a100-vkfft/libvkfft_xd_bridge.so'),
        '--gpuntt-library', str(ROOT / 'build-e13-cufftdx/libgpuntt_semantic_bridge.so')]
    identity = {'handoff': continuation.evidence(args.handoff), 'slot': args.slot,
        'sources': source_identity(), 'queue_sha256': old.sha256_file(__file__),
        'library': continuation.evidence(args.library), 'calibration': continuation.evidence(calibration),
        'prepared': str(args.preparation), 'compile_broker': str(args.compile_broker),
        'gpu_uuid': args.gpu_uuid, 'cpu_workers': args.cpu_workers, 'in_flight': args.in_flight}
    for metadata in handoff['phases'].values():
        if metadata['slot'] != args.slot:
            continue
        for item in metadata['evidence']:
            signature = metadata.get('evidence_stat_signatures', {}).get(item['path'])
            if signature is None:
                continuation.verify_evidence(item)
            elif list(continuation.file_signature(item['path'])) != signature:
                raise ValueError('audited handoff evidence changed: ' + item['path'])
    for item in handoff['manifests'].values():
        continuation.verify_evidence(item)
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / 'queue_status.json'
    state = {'schema': 'cubutterfly-async-queue-v1', 'identity': identity,
        'status': 'starting', 'pid': os.getpid(), 'phases': {}, 'historical_adoption': handoff['phases']}
    if args.resume:
        state = continuation.read(status_path)
        if state['identity'] != identity:
            raise ValueError('queue resume identity mismatch')
    elif status_path.exists():
        raise ValueError('queue already exists; use explicit resume after checking its state')
    save(status_path, state)
    scaled_process = None
    scaled_ready = False
    try:
        if args.slot == 1:
            run_phase('recovery', args.handoff.parent / 'recovery_gpu1.json', args.preparation, args, state, common)
            quarantines = (handoff['phases']['recovery_gpu1']['quarantine'] +
                           state['phases']['recovery']['quarantine'])
            rows = sorted({entry['row']['path'] for entry in quarantines})
            if rows:
                scaled = args.output / 'scaled_preparation'
                if scaled.exists():
                    import semantic_preparation_queue as preparation
                    scaled_manifest = continuation.read(scaled / 'manifest.json')
                    old.validate_fingerprints(scaled_manifest, args.library)
                    preparation.initialize(scaled / 'prepared', scaled_manifest)
                    for case in scaled_manifest['cases']:
                        preparation.load_case(scaled / 'prepared', case)
                    scaled_ready = True
                else:
                    scaled_command = [sys.executable, '-u', str(HERE / 'prepare_async_scaled_controls.py'),
                        '--rows', *rows, '--output', str(scaled), *common]
                    state['scaled_preparation'] = {'status': 'running', 'command': scaled_command}
                    # The CPU-only producer overlaps the complete GPU timing phase.
                    scaled_process = subprocess.Popen(scaled_command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
                    state['scaled_preparation']['pid'] = scaled_process.pid
                save(status_path, state)
        run_phase('timing', args.handoff.parent / f'timing_gpu{args.slot}.json', args.preparation, args, state, common)
        if scaled_process is not None or scaled_ready:
            returncode = scaled_process.wait() if scaled_process is not None else 0
            if returncode:
                raise RuntimeError('scaled CPU preparation failed')
            state['scaled_preparation']['status'] = 'complete'
            scaled = args.output / 'scaled_preparation'
            manifest_path = scaled / 'manifest.json'
            if continuation.read(manifest_path)['cases']:
                run_phase('scaled_controls', manifest_path, scaled / 'prepared', args, state, common)
        state.update(status='execution-finished-evidence-review-required', current_phase=None)
    except BaseException as exc:
        state.update(status='stopped-investigation-required', error=str(exc))
        raise
    finally:
        state['updated_unix'] = time.time()
        save(status_path, state)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
