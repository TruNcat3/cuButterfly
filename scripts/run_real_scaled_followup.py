#!/usr/bin/env python3
"""Run prepared real-FFT controls after one identified, owned queue finishes.

The wait is bounded and is only for the named queue. A foreign GPU workload,
failed dependency or changed queue identity stops this follow-up without retry.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_semantic_campaign as runner
from run_async_semantic_queue import save
from semantic_campaign_manifest import validate_manifest_hash

FINISHED = 'execution-finished-evidence-review-required'


def process_token(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def dependency_state(current, identity, pid, token, live_token):
    if current['pid'] != pid or runner.digest(current['identity']) != identity:
        raise ValueError('dependency queue identity changed')
    if live_token is not None and live_token != token:
        raise ValueError('dependency PID was reused')
    if current['status'].startswith('stopped'):
        raise RuntimeError('dependency requires investigation')
    if current['status'] == FINISHED and live_token is None:
        return 'ready'
    if live_token is None:
        raise RuntimeError('dependency exited without completed state')
    return 'waiting-for-owned-queue'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'preparation', 'compile-broker', 'library', 'calibration-manifest',
                 'dependency-queue', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--vkfft-library', type=Path)
    parser.add_argument('--gpuntt-library', type=Path)
    parser.add_argument('--dependency-timeout-seconds', type=float, required=True)
    args = parser.parse_args(argv)
    if not 0 < args.dependency_timeout_seconds <= 86400:
        parser.error('dependency timeout must be positive and at most one day')
    manifest = json.loads(args.manifest.read_text())
    if not validate_manifest_hash(manifest):
        raise ValueError('manifest checksum mismatch')
    runner.validate_fingerprints(manifest, args.library)
    provenance = manifest['real_scaled_control_provenance']
    uuid = provenance['target_gpu_uuid']
    if (provenance['source_gpu_uuid'] != uuid or provenance['timing'] or
            not provenance['original_mapping_replayed'] or
            any('execution_mapping' not in case for case in manifest['cases'])):
        raise ValueError('follow-up requires untimed same-GPU original mapping replay')
    current = json.loads(args.dependency_queue.read_text())
    if current['identity']['gpu_uuid'] != uuid:
        raise ValueError('dependency GPU differs from the replay target')
    pid, identity = current['pid'], runner.digest(current['identity'])
    token = process_token(pid)
    command = [sys.executable, '-u', str(Path(__file__).with_name('run_async_semantic_campaign.py'))]
    for name in ('manifest', 'preparation', 'compile_broker', 'library', 'calibration_manifest',
                 'vkfft_library', 'gpuntt_library'):
        value = getattr(args, name)
        if value is not None:
            command += ['--' + name.replace('_', '-'), str(value.resolve())]
    command += ['--gpu-uuid', uuid, '--output', str(args.output.resolve() / 'measurement'),
                '--cpu-workers', '2', '--in-flight', '2']
    args.output.mkdir(parents=True, exist_ok=False)
    status_path = args.output / 'status.json'
    state = {'schema': 'cubutterfly-real-scaled-followup-v1', 'pid': os.getpid(),
             'dependency_queue': str(args.dependency_queue.resolve()), 'dependency_pid': pid,
             'dependency_identity': identity, 'dependency_start_token': token,
             'gpu_uuid': uuid, 'command': command,
             'deadline_unix': time.time() + args.dependency_timeout_seconds,
             'foreign_gpu_wait': False, 'timing': False}

    def update(status, **details):
        state.update(status=status, updated_unix=time.time(), **details)
        save(status_path, state)

    try:
        while True:
            current = json.loads(args.dependency_queue.read_text())
            status = dependency_state(current, identity, pid, token, process_token(pid))
            if status == 'ready':
                break
            if time.time() >= state['deadline_unix']:
                update('dependency-timeout-prepared-not-run')
                return 2
            update(status)
            time.sleep(min(15, max(.01, state['deadline_unix'] - time.time())))
        runner.validate_fingerprints(manifest, args.library)
        runner.exclusive_gpu(uuid)
        update('running')
        result = subprocess.run(command, check=False)
        report = json.loads((args.output / 'measurement/results.json').read_text())
        if result.returncode or report['remaining_case_ids'] or not report['status'].startswith('complete'):
            raise RuntimeError('real scaled control requires investigation; original evidence retained')
        update('execution-finished-accuracy-review-required', returncode=result.returncode)
        return 0
    except BaseException as exc:
        update('stopped-investigation-required', error=str(exc))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
