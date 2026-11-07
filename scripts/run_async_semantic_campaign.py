#!/usr/bin/env python3
"""Consume prepared semantic cases, yielding CPU checks between GPU steps.

Input preparation and compilation have independent CPU producers. GPU work is
serialized, even when several cases await CPU verification. An incomplete
MEASURE attempt is discarded in full and retried with a fresh Plan once every
requested module is ready. This does not change the legal search population.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing
import os
from pathlib import Path
import re
import signal
import sys
import time
import uuid
from types import SimpleNamespace

import numpy as np

import run_semantic_campaign as old
import semantic_compile_broker as broker
from semantic_cpu_pipeline import validate_snapshot
from semantic_gpu_pipeline import (execution_steps, PreparationPending, PreparationFailed,
                                   _candidate_rejection, _transient_compile_failure)
import semantic_worker_resources as resources
from semantic_worker_resources import WorkerRecycle
import semantic_preparation_queue as preparation

REVISION = '0.9.3.4-async-experiments.4'
HERE = Path(__file__).resolve().parent
GPU_BUSY_EXIT = 76


class ForeignGPUWork(RuntimeError):
    """An explicit exclusive-GPU guard rejection, never a measurement error."""


class CampaignInterrupted(Exception):
    pass


def exclusive_gpu(uuid):
    try:
        old.exclusive_gpu(uuid)
    except RuntimeError as exc:
        pattern = (r'GPU ' + re.escape(uuid) +
                   r' has foreign compute processes: \[[1-9][0-9]*(?:, [1-9][0-9]*)*\]; stopping without killing')
        if type(exc) is RuntimeError and re.fullmatch(pattern, str(exc)):
            raise ForeignGPUWork(str(exc)) from exc
        raise


def record_gpu_interruption(args, exc):
    receipt = dict(status='interrupted-by-foreign-gpu-work', gpu_uuid=args.gpu_uuid,
                   pid=os.getpid(), observed_unix=time.time(), reason=str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'gpu_interruption.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    temporary.replace(path)
    return receipt


def cpu_only():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def validate_payload(payload):
    if 'batch_snapshots' in payload:
        return {'batch_results': [validate_snapshot(snapshot) for snapshot in payload['batch_snapshots']]}
    return validate_snapshot(payload)


def source_identity():
    names = ('run_async_semantic_campaign.py', 'run_semantic_campaign.py',
             'semantic_cpu_pipeline.py', 'semantic_gpu_pipeline.py',
             'semantic_compile_broker.py', 'semantic_preparation_queue.py',
             'semantic_ntt_reference.py', 'semantic_input_profiles.py',
             'semantic_dao_baseline.py', 'semantic_gpuntt_baseline.py', 'semantic_fft_baseline.py',
             'diagnose_xd_fft.py', 'vkfft_xd.py', 'semantic_worker_resources.py',
             'semantic_resource_compile_proxy.py', 'semantic_compile_service.py',
             'semantic_compile_client.cpp', 'jit_calibration_identity.py')
    return {name: old.sha256_file(HERE / name) for name in names}


def configure_compile_transport(client=None, endpoint=None):
    """Choose query transport explicitly; compilation and search stay unchanged."""
    if (client is None) != (endpoint is None):
        raise ValueError('compile-client and compile-socket must be provided together')
    if client is None:
        os.environ['CUBUTTERFLY_JIT_PYTHON'] = str(HERE / 'semantic_resource_compile_proxy.py')
        os.environ.pop('CUBUTTERFLY_COMPILE_SOCKET', None)
        return {'mode': 'per-query-python'}
    client, endpoint = Path(client).resolve(strict=True), Path(endpoint).resolve(strict=True)
    if not client.is_file() or not os.access(client, os.X_OK):
        raise ValueError('compile-client must be an executable file')
    if not endpoint.is_socket():
        raise ValueError('compile-socket must be an existing Unix socket')
    os.environ['CUBUTTERFLY_JIT_PYTHON'] = str(client)
    os.environ['CUBUTTERFLY_COMPILE_SOCKET'] = str(endpoint)
    return {'mode': 'persistent-service', 'client': str(client),
            'client_sha256': old.sha256_file(client), 'socket': str(endpoint)}


def request_state(directory, identifiers):
    states = []
    for identifier in identifiers:
        path = Path(directory) / 'records' / (identifier + '.json')
        states.append(json.loads(path.read_text()) if path.exists() else {'status': 'pending'})
    failures = [state for state in states if state['status'] == 'failed'
                and not _candidate_rejection(state)
                and not _transient_compile_failure(state)]
    if failures:
        raise PreparationFailed(failures)
    # An explicitly unsupported candidate is terminal and will be rejected by
    # a fresh C++ Plan. A pending candidate still invalidates that entire Plan.
    return bool(states) and all(state['status'] == 'ready' or
                                _candidate_rejection(state) for state in states)


def gap_row(case, record):
    return {'id': case['id'], 'descriptor': case['descriptor'],
            'populations': case['populations'],
            'requested_variants': case.get('requested_variants', []),
            'status': record.get('gap_status', 'runner-error'),
            'error': record.get('reason', record.get('error', 'CPU preparation failed')),
            'capabilities': case.get('capabilities', {}), 'baselines': []}


def ranking_cancellation_receipt(row, item):
    """Only a fresh, signalled ranking attempt explains this one C API error."""
    if (not item or row.get('status') != 'runtime-error' or row.get('runtime_status') != 6
            or row.get('error') != 'backend error: static ranking helper failed; inspect its diagnostic above'):
        return None
    try:
        records = [json.loads(line) for line in item['events'].read_text().splitlines() if line.strip()]
    except (OSError, ValueError, KeyError):
        return None
    if any(record.get('kind') in ('compile_failed', 'broker_error', 'broker_runtime_error')
           or record.get('status') in ('failed', 'error') for record in records):
        return None
    helpers = [record for record in records if record.get('kind', '').startswith('cpu_helper_')]
    failures = [record for record in helpers if record.get('kind') == 'cpu_helper_finished'
                and record.get('returncode') != 0]
    if len(failures) != 1:
        return None
    receipt = failures[0]
    signum = receipt.get('termination_signal')
    if (receipt.get('attempt_token') != item['attempt_token']
            or receipt.get('worker_pid') != os.getpid()
            or receipt.get('script') != str(HERE / 'rank_static_candidates.py')
            or signum not in (signal.SIGTERM, signal.SIGINT)
            or receipt.get('returncode') != -signum):
        return None
    started, ended = receipt.get('started_unix'), receipt.get('ended_unix')
    if (not isinstance(started, (int, float)) or not isinstance(ended, (int, float))
            or not item['attempt_started_unix'] <= started <= ended <= time.time()
            or not any(entry.get('signal') == signum and
                       started <= entry.get('received_unix', 0) <= ended
                       for entry in receipt.get('received_signals', []))):
        return None
    identity = ('worker_pid', 'proxy_pid', 'helper_pid', 'script', 'started_unix', 'attempt_token')
    starts = [record for record in helpers if record.get('kind') == 'cpu_helper_started'
              and all(record.get(key) == receipt.get(key) for key in identity)]
    if (len(starts) != 1 or any(not isinstance(receipt.get(key), int) or receipt[key] <= 0
                               for key in ('proxy_pid', 'helper_pid'))):
        return None
    return receipt


def drive(cases, report, args, library, torch, stream, executor):
    """Bounded cooperative scheduler. Only this process submits CUDA work."""
    output = args.output / 'results.json'
    known = {index['id'] for index in report['rows']}
    pending = deque(case for case in cases if case['id'] not in known)
    active, blocked = {}, {}
    attempts = dict(report.get('preparation_attempts', {}))
    stopping = False
    stopping_signals = set()
    last_checkpoint = 0.0
    previous_signals = {}
    initial_completed = len(report['rows'])
    worker_attempts = 0
    report.pop('interruption_diagnostics', None)
    budget = getattr(args, 'resource_budget', resources.limits())

    def resource_guard():
        sample = resources.snapshot()
        reason = resources.exceeded(sample, budget, len(report['rows']) - initial_completed, worker_attempts)
        report['worker_resources'] = dict(sample, completed_cases=len(report['rows']) - initial_completed,
                                          plan_attempts=worker_attempts, budget=budget)
        peaks = report.setdefault('worker_resource_peaks', {})
        for key in ('vma_count', 'rss_bytes', 'virtual_bytes', 'fd_count', 'module_mappings'):
            peaks[key] = max(peaks.get(key, 0), sample[key])
        if reason:
            raise WorkerRecycle(reason, sample)

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        stopping_signals.add(signum)

    def checkpoint():
        nonlocal last_checkpoint
        report.update(preparation_attempts=attempts,
                      waiting_compilation={key: value['requests'] for key, value in blocked.items()},
                      async_progress={'cpu_waiting': len(pending), 'in_flight': len(active),
                                      'compile_waiting': len(blocked), 'updated_unix': time.time()})
        old.write_report(output, report)
        last_checkpoint = time.monotonic()

    def interrupted(status):
        diagnostics = report.setdefault('interruption_diagnostics', {})
        diagnostics['validation_results'] = {key: item['validation_result']
            for key, item in active.items() if 'validation_result' in item}

        def failed(value):
            if isinstance(value, dict):
                return (value.get('status') in ('baseline-error', 'runtime-error', 'runner-error', 'correctness-failure')
                        or any(failed(item) for item in value.values()))
            return isinstance(value, list) and any(failed(item) for item in value)

        if failed(diagnostics):
            reason = 'investigation required for unaccepted errors during interruption'
            report.update(status='stopped-investigation-required', stop_reason=reason)
            raise RuntimeError(reason)
        report['status'] = status

    def commit(case, row):
        if stopping:
            receipt = ranking_cancellation_receipt(row, active.get(case['id']))
            if receipt and receipt['termination_signal'] not in stopping_signals:
                receipt = None
            if receipt:
                row = dict(row, status='interrupted-helper-cancellation')
            report['interruption_diagnostics'] = {'unaccepted_row': row}
            if receipt:
                report['interruption_diagnostics']['ranking_helper_cancellation'] = receipt
            raise CampaignInterrupted()
        if any(b.get('status') in ('baseline-error', 'runtime-error', 'runner-error', 'correctness-failure')
               for b in row.get('baselines', [])) and row['status'] in ('pass', 'accuracy-review-required'):
            row.update(status='runner-error', error='baseline investigation required')
        try:
            exclusive_gpu(args.gpu_uuid)
        except ForeignGPUWork:
            report['interruption_diagnostics'] = {'unaccepted_row': row}
            if row['status'] in ('correctness-failure', 'runtime-error', 'runner-error'):
                raise RuntimeError('investigation required for unaccepted error row during GPU interruption')
            raise
        if stopping:
            report['interruption_diagnostics'] = {'unaccepted_row': row}
            raise CampaignInterrupted()
        report['rows'].append(old.write_case_checkpoint(args.output, row))
        report['remaining_case_ids'].remove(case['id'])
        checkpoint()
        print(json.dumps({'case': case['id'], 'status': row['status'],
                          'completed': len(report['rows']),
                          'remaining': len(report['remaining_case_ids'])}), flush=True)
        if row['status'] in ('correctness-failure', 'runtime-error', 'runner-error'):
            raise RuntimeError('investigation required: ' + row['status'] + ': ' + row.get('error', ''))
        if any(b.get('status') in ('baseline-error', 'runtime-error', 'runner-error', 'correctness-failure')
               for b in row.get('baselines', [])):
            raise RuntimeError('baseline investigation required: ' + case['id'])

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_signals[sig] = signal.signal(sig, stop)
        while pending or active or blocked:
            resource_guard()
            if stopping:
                interrupted('interrupted')
                break
            progress = False
            # A ready compile never grants acceptance to the previous selection.
            # Reconstruct the entire Plan from the original descriptor.
            for identifier, item in list(blocked.items()):
                if request_state(args.compile_broker, item['requests']):
                    pending.appendleft(item['case'])
                    del blocked[identifier]
                    progress = True
            for _ in range(min(len(pending), 256)):
                if len(active) >= args.in_flight:
                    break
                case = pending.popleft()
                record = preparation.record_for(args.preparation, case['id'])
                if record is None:
                    pending.append(case)
                    continue
                if record['status'] != 'ready':
                    # load_case validates identity even for terminal CPU gaps.
                    try:
                        preparation.load_case(args.preparation, case)
                    except old.Gap:
                        commit(case, gap_row(case, record))
                    else:
                        raise RuntimeError('unexpected preparation state: ' + record['status'])
                    progress = True
                    continue
                active[case['id']] = {'case': case, 'phase': 'load',
                    'attempt_token': uuid.uuid4().hex, 'attempt_started_unix': time.time(),
                    'future': executor.submit(preparation.load_case, args.preparation, case)}
                progress = True
            for identifier, item in list(active.items()):
                resource_guard()
                future = item['future']
                if not future.done():
                    continue
                value = future.result()
                if item['phase'] == 'verify':
                    item['validation_result'] = value
                if stopping:
                    raise CampaignInterrupted()
                case = item['case']
                if item['phase'] == 'load':
                    resource_guard()
                    worker_attempts += 1
                    attempt = attempts.get(identifier, 0) + 1
                    attempts[identifier] = attempt
                    if attempt > args.max_prepare_attempts:
                        raise RuntimeError('preparation closure budget exhausted for ' + identifier)
                    events = args.output / 'compile_events' / (identifier + '-' + str(attempt) + '.jsonl')
                    if events.exists():
                        raise ValueError('attempt event file already exists')
                    options = SimpleNamespace(timing=args.timing, float_verdict='error-report',
                        error_dir=args.output / 'errors', vkfft_library=args.vkfft_library,
                        gpuntt_library=args.gpuntt_library, events_path=events)
                    item['events'] = events
                    item['generator'] = execution_steps(case, value, library, torch, stream, options)
                    checkpoint()  # Persist the attempt number before a subprocess can create its event file.
                os.environ['CUBUTTERFLY_PREPARE_EVENTS'] = str(item['events'])
                os.environ['CUBUTTERFLY_HELPER_ATTEMPT'] = item['attempt_token']
                exclusive_gpu(args.gpu_uuid)
                try:
                    with torch.cuda.stream(stream):
                        snapshot = (next(item['generator']) if item['phase'] == 'load'
                                    else item['generator'].send(value))
                    item.update(phase='verify', future=executor.submit(validate_payload, snapshot))
                except StopIteration as done:
                    commit(case, done.value)
                    del active[identifier]
                except PreparationPending as exc:
                    item['generator'].close()
                    blocked[identifier] = {'case': case, 'requests': exc.request_ids}
                    del active[identifier]
                    checkpoint()
                progress = True
            if time.monotonic() - last_checkpoint > 5:
                checkpoint()
            if not progress:
                time.sleep(.05)
        else:
            report['status'] = old.completion_status(report)
    except CampaignInterrupted:
        interrupted('interrupted')
    except ForeignGPUWork as exc:
        report.update(stop_reason=str(exc), gpu_interruption=record_gpu_interruption(args, exc))
        interrupted('interrupted-by-foreign-gpu-work')
    except WorkerRecycle as exc:
        report.update(status='worker-recycle-required', recycle_reason=exc.reason,
                      recycle_resources=exc.snapshot)
    except BaseException as exc:
        report.update(status='stopped-investigation-required', stop_reason=str(exc))
        raise
    finally:
        for item in active.values():
            if item.get('generator') is not None:
                item['generator'].close()
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
        checkpoint()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'library', 'output', 'preparation', 'compile-broker'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--compile-client', type=Path)
    parser.add_argument('--compile-socket', type=Path)
    parser.add_argument('--calibration-manifest', type=Path)
    parser.add_argument('--preparation-source-manifest', type=Path,
                        help='Source manifest bound to a semantically identical reused preparation package')
    parser.add_argument('--vkfft-library', type=Path)
    parser.add_argument('--gpuntt-library', type=Path)
    parser.add_argument('--timing', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cpu-workers', type=int, default=4)
    parser.add_argument('--in-flight', type=int, default=4)
    parser.add_argument('--max-prepare-attempts', type=int, default=64)
    parser.add_argument('--worker-max-vmas', type=int, default=45000)
    parser.add_argument('--worker-max-rss-bytes', type=int, default=8 * 1024**3)
    parser.add_argument('--worker-max-cases', type=int, default=128)
    parser.add_argument('--worker-max-plan-attempts', type=int, default=64)
    args = parser.parse_args(argv)
    args.resource_budget = resources.limits(args.worker_max_vmas, args.worker_max_rss_bytes,
                                           args.worker_max_cases, args.worker_max_plan_attempts)
    if min(args.cpu_workers, args.in_flight, args.max_prepare_attempts) < 1:
        parser.error('worker and attempt limits must be positive')
    if args.timing and not args.calibration_manifest:
        parser.error('timing requires per-device calibration')
    for name in ('manifest', 'library', 'output', 'preparation', 'compile_broker'):
        setattr(args, name, getattr(args, name).resolve())
    from semantic_campaign_manifest import validate_manifest_hash
    manifest = json.loads(args.manifest.read_text())
    if not validate_manifest_hash(manifest):
        raise ValueError('manifest checksum mismatch')
    preparation_manifest = manifest
    if args.preparation_source_manifest:
        import cpu_external_baselines as cpu_comparison
        preparation_manifest = json.loads(args.preparation_source_manifest.read_text())
        cpu_comparison.comparison_source(manifest, preparation_manifest)
    old.validate_fingerprints(manifest, args.library)
    cases = manifest['cases']
    if not cases or len({case['id'] for case in cases}) != len(cases):
        raise ValueError('empty or duplicate case population')
    for case in cases:
        if case['id'] != old.digest(case['descriptor']['semantics']):
            raise ValueError('case checksum mismatch')
    config = broker._document(args.compile_broker)
    # Reject incompatible formal timing before importing Torch or touching CUDA.
    if args.timing:
        from jit_calibration_identity import validate_timing
        validate_timing(json.loads(args.calibration_manifest.read_text())
                        if args.calibration_manifest else {}, config)
    prepared_config, prepared_sha = preparation._config(args.preparation)
    if (prepared_config['reference_max_elements'] != old.PROTOCOL['reference_max_elements'] or
            prepared_config['ntt_max_operations'] != old.PROTOCOL['ntt_max_operations']):
        raise ValueError('CPU preparation bounds differ from the measurement protocol')
    if prepared_config['source_sha256'] != preparation._sources():
        raise ValueError('CPU preparation source identity changed')
    compile_transport = configure_compile_transport(args.compile_client, args.compile_socket)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,
        CUBUTTERFLY_WORKER_MAX_VMAS=str(args.worker_max_vmas),
        CUBUTTERFLY_WORKER_MAX_RSS_BYTES=str(args.worker_max_rss_bytes),
        CUBUTTERFLY_PREPARE_BROKER=str(args.compile_broker),
        CUBUTTERFLY_TEMPLATE_ROOT=config['config']['root'],
        CUBUTTERFLY_MATHDX_ROOT=config['config']['mathdx'],
        CUBUTTERFLY_NVCC=config['config']['nvcc'],
        CUBUTTERFLY_JIT_CACHE=config['config']['cache'])
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    os.environ.pop('CUBUTTERFLY_PREPARE_EVENTS', None)
    try:
        exclusive_gpu(args.gpu_uuid)
    except ForeignGPUWork as exc:
        record_gpu_interruption(args, exc)
        print(str(exc), file=sys.stderr, flush=True)
        return GPU_BUSY_EXIT
    import torch
    import cubutterfly as cb
    properties = torch.cuda.get_device_properties(0)
    if not old.same_gpu_uuid(args.gpu_uuid, str(properties.uuid)):
        raise ValueError('CUDA UUID mismatch')
    identity = {'manifest_sha256': old.digest(manifest), 'library_sha256': old.sha256_file(args.library),
        'library': str(args.library), 'gpu_uuid': args.gpu_uuid, 'gpu_name': properties.name,
        'gpu_total_memory': properties.total_memory,
        'compute_capability': f'{properties.major}.{properties.minor}',
        'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'python': sys.version,
        'numpy': np.__version__, 'runner_revision': REVISION, 'sources': source_identity(),
        'frontend_sha256': old.sha256_file(cb.__file__), 'timing': args.timing, 'population': None,
        'protocol': dict(old.PROTOCOL, float_verdict='error-report'),
        'preparation': str(args.preparation), 'preparation_config_sha256': prepared_sha,
        'preparation_source_manifest': str(args.preparation_source_manifest.resolve())
            if args.preparation_source_manifest else None,
        'preparation_source_manifest_sha256': old.digest(preparation_manifest),
        'broker_identity': config,
        'compile_transport': compile_transport,
        'worker_resource_budget': args.resource_budget,
        'scheduling': {'gpu_serial': True, 'cpu_workers': args.cpu_workers, 'in_flight': args.in_flight,
                       'timing_group': 'seven contiguous GPU samples; all snapshots verified before acceptance'},
        'vkfft_sha256': old.sha256_file(args.vkfft_library) if args.vkfft_library else None,
        'gpuntt_sha256': old.sha256_file(args.gpuntt_library) if args.gpuntt_library else None}
    limitations = []
    if args.calibration_manifest:
        calibration = json.loads(args.calibration_manifest.read_text())
        identity.update(old.bind_calibration_environment(calibration))
        limitations = old.validate_calibration(calibration, identity)
        identity['calibration_sha256'] = old.digest(calibration)
    else:
        identity['compile_mode'] = os.environ.get('CUBUTTERFLY_COMPILE_MODE', 'research')
        os.environ['CUBUTTERFLY_COMPILE_MODE'] = identity['compile_mode']
    identity['runtime_environment'] = {key: val for key, val in sorted(os.environ.items())
        if key.startswith('CUBUTTERFLY_') or key in ('CUDA_VISIBLE_DEVICES', 'CUDA_MODULE_LOADING')}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'results.json'
    report = {'schema': 'cubutterfly-semantic-results-v1', 'identity': identity,
        'calibration_limitations': limitations, 'manifest_metrics': manifest.get('metrics'),
        'invalid_requests_source': str(args.manifest),
        'invalid_request_count': len(manifest.get('invalid_requests', [])),
        'rows': [], 'status': 'running', 'remaining_case_ids': [case['id'] for case in cases]}
    if args.resume:
        report = json.loads(path.read_text())
        old.validate_resume(report, identity, args.output, {case['id'] for case in cases})
        if report['status'].startswith('stopped'):
            raise ValueError('investigate stopped run before preparing a separate recovery')
        report['status'] = 'running'
    elif path.exists():
        raise ValueError('output exists; use --resume or a fresh directory')
    old.write_report(path, report)
    library = cb.Library(args.library)
    stream = torch.cuda.Stream()
    with ProcessPoolExecutor(max_workers=args.cpu_workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=cpu_only) as executor:
        drive(cases, report, args, library, torch, stream, executor)
    if report['status'] == 'interrupted-by-foreign-gpu-work':
        return GPU_BUSY_EXIT
    return 75 if report['status'] == 'worker-recycle-required' else (0 if report['status'].startswith('complete') else 2)


if __name__ == '__main__':
    raise SystemExit(main())
