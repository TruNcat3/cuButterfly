#!/usr/bin/env python3
"""Run immutable prepared phases in bounded GPU worker processes.

Complete verified rows survive process recycling. Partial searches never do.
CPU-only compilation can proceed while no CUDA worker is resident.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import continue_semantic_campaign as evidence
import run_semantic_campaign as runner
from run_async_semantic_campaign import source_identity, request_state
from semantic_campaign_manifest import validate_manifest_hash

REVISION = '0.9.3.4-async-experiments.3'
HERE = Path(__file__).resolve().parent


def module_family(case):
    semantics = case['descriptor']['semantics']
    key = (semantics['operator'], semantics['storage'], semantics['compute'],
           tuple(semantics['shape']), semantics.get('word_bits'),
           semantics.get('fft_type', 0), semantics.get('length_mode', 0))
    if semantics['operator'] == 1:
        key += (tuple(semantics.get('output_orders', [])),
                semantics.get('ntt_output_layout', 0))
    return key


def ready_cases(cases, completed, waiting, broker, count, group_by_module_family=False):
    ready = []
    family = None
    for case in cases:
        if case['id'] in completed:
            continue
        candidate_family = module_family(case) if group_by_module_family else None
        if ready and candidate_family != family:
            continue
        requests = waiting.get(case['id'], [])
        try:
            available = not requests or request_state(broker, requests)
        except Exception as error:
            # Keep the bounded queue's stable diagnostic and case identity
            # while preserving the strict fatal handling in request_state.
            if requests:
                raise RuntimeError('compile request failed: ' + requests[0]) from error
            raise
        if available:
            family = candidate_family
            ready.append(case)
            if len(ready) == count:
                break
    return ready


def validate_worker_exit(returncode, report, rows):
    failures = [row for row in rows if row['status'] == 'correctness-failure']
    if any(evidence.fatal_baseline(row) for row in rows):
        raise RuntimeError('baseline error requires investigation')
    if any(row['status'] not in evidence.PROCESSED | {'correctness-failure'} for row in rows):
        raise RuntimeError('runtime or runner error requires investigation')
    if returncode == 75:
        if report['status'] != 'worker-recycle-required' or failures:
            raise RuntimeError('unverified resource-recycle exit')
    elif returncode == 0:
        if not report['status'].startswith('complete') or report['remaining_case_ids']:
            raise RuntimeError('incomplete successful worker')
    elif not (len(failures) == 1 and report['status'] == 'stopped-investigation-required'):
        raise RuntimeError('worker failed without a verified correctness failure')
    return failures


def verify_handoff(path):
    handoff = evidence.read(path)
    for phase in handoff['phases'].values():
        signatures = phase.get('evidence_stat_signatures', {})
        for item in phase.get('evidence', []):
            signature = signatures.get(item['path'])
            current = list(evidence.file_signature(item['path']))
            recent = max(current[-2:]) >= time.time_ns() - 1_000_000_000
            if signature is None or current != signature or recent:
                evidence.verify_evidence(item)
    return handoff


def run_phase(spec, args, plan, state):
    manifest_path = Path(spec['manifest'])
    parent = evidence.read(manifest_path)
    if not validate_manifest_hash(parent):
        raise ValueError('phase manifest checksum mismatch')
    runner.validate_fingerprints(parent, Path(plan['library']))
    cases = {case['id']: case for case in parent['cases']}
    phase = state['phases'].setdefault(spec['name'], {
        'status': 'pending', 'total': len(cases), 'shards': [], 'completed': {},
        'waiting_compilation': {}, 'attempts': {}, 'quarantine': [], 'zero_progress_recycles': {}})
    if phase['status'].startswith('complete'):
        return
    if phase['shards']:
        raise ValueError('existing shards require an audited handoff, not an implicit retry')
    status_path = args.output / 'queue_status.json'
    deadline = time.monotonic() + args.cpu_wait_seconds
    while len(phase['completed']) < len(cases):
        selected = ready_cases(parent['cases'], phase['completed'], phase['waiting_compilation'],
                               plan['compile_broker'], args.shard_cases, args.group_by_module_family)
        if not selected:
            if time.monotonic() > deadline:
                raise RuntimeError('CPU compilation wait budget expired; prepared work retained')
            phase['status'] = 'waiting-cpu-compilation'
            state.update(current_phase=spec['name'], updated_unix=time.time())
            evidence.save(status_path, state)
            time.sleep(15)
            continue
        # No waiting for other GPU users. A conflict stops this queue explicitly.
        runner.exclusive_gpu(plan['gpu_uuid'])
        if state['identity']['sources'] != source_identity():
            raise ValueError('worker sources changed during queue execution')
        runner.validate_fingerprints(parent, Path(plan['library']))
        directory = args.output / spec['name'] / ('shard-%04d' % len(phase['shards']))
        directory.mkdir(parents=True, exist_ok=False)
        manifest = evidence.submanifest(parent, manifest_path, selected, spec['name'],
                                       set(phase['completed']), phase['quarantine'])
        selected_path = directory / 'manifest.json'
        evidence.save(selected_path, manifest)
        output = directory / 'measurement'
        command = [sys.executable, '-u', str(HERE / 'run_async_semantic_campaign.py'),
                   '--manifest', str(selected_path), '--preparation', spec['preparation'],
                   '--compile-broker', plan['compile_broker'], '--output', str(output),
                   '--cpu-workers', str(args.cpu_workers), '--in-flight', str(args.in_flight),
                   '--worker-max-cases', str(args.shard_cases)]
        for key in ('library', 'gpu_uuid', 'calibration_manifest', 'vkfft_library', 'gpuntt_library',
                    'compile_client', 'compile_socket'):
            if plan.get(key):
                command += ['--' + key.replace('_', '-'), plan[key]]
        if spec.get('timing'):
            command.append('--timing')
        shard = {'manifest': str(selected_path), 'report': str(output / 'results.json'),
                 'command': command, 'status': 'running'}
        phase['shards'].append(shard)
        phase['status'] = 'running'
        state.update(current_phase=spec['name'], updated_unix=time.time())
        evidence.save(status_path, state)
        with (directory / 'worker.log').open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            shard['pid'] = child.pid
            evidence.save(status_path, state)
            code = child.wait()
        shard['returncode'] = code
        report, rows, files = evidence.checked_report(
            output / 'results.json', manifest, Path(plan['library']), plan['gpu_uuid'], bool(spec.get('timing')))
        if report['identity']['sources'] != state['identity']['sources']:
            raise ValueError('worker source identity mismatch')
        if report['identity']['protocol'] != dict(runner.PROTOCOL, float_verdict='error-report'):
            raise ValueError('worker protocol mismatch')
        calibration = evidence.read(plan['calibration_manifest'])
        if report['identity'].get('calibration_sha256') != runner.digest(calibration):
            raise ValueError('worker calibration identity mismatch')
        runner.validate_calibration(calibration, report['identity'])
        failures = validate_worker_exit(code, report, rows)
        for row in rows:
            case = cases[row['id']]
            if row['descriptor'] != case['descriptor']:
                raise ValueError('worker changed semantic descriptor')
            for field in ('input_profile', 'source_evidence'):
                if row.get(field) != case.get(field):
                    raise ValueError('worker changed input provenance')
            if 'execution_mapping' in case and case['execution_mapping'] != row.get('mapping'):
                raise ValueError('worker changed original control mapping')
            if row['id'] in phase['completed']:
                raise ValueError('duplicate completed request')
            phase['completed'][row['id']] = row['status']
            phase['waiting_compilation'].pop(row['id'], None)
        phase['quarantine'] += [evidence.quarantine_entry(row, output / 'results.json') for row in failures]
        for key, value in report.get('preparation_attempts', {}).items():
            phase['attempts'][key] = phase['attempts'].get(key, 0) + value
            if phase['attempts'][key] > 64:
                raise RuntimeError('cross-worker preparation attempt budget exhausted: ' + key)
        phase['waiting_compilation'].update(report.get('waiting_compilation', {}))
        if code == 75 and not rows:
            signature = runner.digest([case['id'] for case in selected])
            count = phase['zero_progress_recycles'].get(signature, 0) + 1
            phase['zero_progress_recycles'][signature] = count
            if count > 2:
                raise RuntimeError('same fresh-worker population repeatedly exceeds resources; investigate without pruning')
        shard.update(status=report['status'], evidence=files,
                     processed=len(rows), worker_resources=report.get('worker_resources'),
                     recycle_reason=report.get('recycle_reason'))
        phase.update(processed=len(phase['completed']),
                     status_counts=dict(Counter(phase['completed'].values())))
        state['updated_unix'] = time.time()
        evidence.save(status_path, state)
        print(json.dumps({'phase': spec['name'], 'processed': phase['processed'],
                          'total': phase['total'], 'worker_exit': code}), flush=True)
        if rows:
            deadline = time.monotonic() + args.cpu_wait_seconds
    phase['status'] = 'complete-with-unresolved-evidence'
    evidence.save(status_path, state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard-cases', type=int, default=128)
    parser.add_argument('--cpu-workers', type=int, default=2)
    parser.add_argument('--in-flight', type=int, default=2)
    parser.add_argument('--group-by-module-family', action='store_true')
    parser.add_argument('--cpu-wait-seconds', type=float, default=43200)
    args = parser.parse_args(argv)
    if min(args.shard_cases, args.cpu_workers, args.in_flight, args.cpu_wait_seconds) <= 0:
        parser.error('queue budgets must be positive')
    args.output = args.output.resolve()
    plan = evidence.read(args.plan)
    if bool(plan.get('compile_client')) != bool(plan.get('compile_socket')):
        raise ValueError('compile_client and compile_socket must be paired')
    handoff = verify_handoff(plan['handoff']) if plan.get('handoff') else None
    if handoff and plan['gpu_uuid'] not in {p['gpu_uuid'] for p in handoff['phases'].values()}:
        raise ValueError('GPU outside audited handoff')
    if len({phase['name'] for phase in plan['phases']}) != len(plan['phases']):
        raise ValueError('duplicate phase names')
    args.output.mkdir(parents=True, exist_ok=False)
    state = {'schema': 'cubutterfly-bounded-queue-v2', 'revision': REVISION,
             'status': 'running', 'pid': os.getpid(), 'phases': {},
             'identity': {'sources': source_identity(), 'plan': evidence.evidence(args.plan),
                          'queue_source': evidence.evidence(__file__), 'gpu_uuid': plan['gpu_uuid'],
                          'shard_cases': args.shard_cases, 'cpu_workers': args.cpu_workers,
                          'group_by_module_family': args.group_by_module_family,
                          'in_flight': args.in_flight, 'cpu_wait_seconds': args.cpu_wait_seconds}}
    try:
        for spec in plan['phases']:
            run_phase(spec, args, plan, state)
        state.update(status='execution-finished-evidence-review-required', current_phase=None)
    except BaseException as exc:
        state.update(status='stopped-investigation-required', error=str(exc))
        raise
    finally:
        state['updated_unix'] = time.time()
        evidence.save(args.output / 'queue_status.json', state)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
