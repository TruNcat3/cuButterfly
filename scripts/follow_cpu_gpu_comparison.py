#!/usr/bin/env python3
"""Observe an existing finite driver and finalize CPU/GPU comparison artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import cpu_external_baselines as cpu
import review_gpu_same_precision as gpu_review
from semantic_campaign_manifest import validate_manifest_hash


def parse_process_stat(text):
    # comm is parenthesized and may itself contain spaces or closing parentheses.
    fields = text.rsplit(')', 1)[1].split()
    return {'state': fields[0], 'starttime': int(fields[19])}


def process_identity(pid):
    try:
        return parse_process_stat(Path(f'/proc/{pid}/stat').read_text())
    except FileNotFoundError:
        return None


def driver_observation(bound, current):
    if current is None:
        return 'exited'
    if bound is None or current['starttime'] != bound['starttime']:
        return 'pid-reused'
    if current['state'] in ('Z', 'X', 'x'):
        return 'exited-unreaped'
    return 'running'


def follow(manifest_path, preparation_path, cpu_output, gpu_results, driver_pid, cpu_source_manifest=None):
    manifest_path, cpu_output, gpu_results = map(lambda p: Path(p).resolve(),
                                                (manifest_path, cpu_output, gpu_results))
    status_path = cpu_output / 'join_status.json'
    cpu_results = cpu_output / 'results.json'
    state = dict(schema='cubutterfly-comparison-follower-v1', status='initializing',
                 driver_pid=driver_pid, poll_seconds=30, gpu_calls=0,
                 manifest=str(manifest_path), cpu_results=str(cpu_results),
                 gpu_results=str(gpu_results), started_unix=time.time())

    def record(**updates):
        state.update(updates, updated_unix=time.time())
        cpu.write(status_path, state)

    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if not validate_manifest_hash(manifest):
            raise ValueError('manifest content checksum mismatch')
        source_bytes = Path(cpu_source_manifest).read_bytes() if cpu_source_manifest is not None else None
        source = json.loads(source_bytes) if source_bytes is not None else None
        source = cpu.comparison_source(manifest, source)
        preparation_identity = (cpu.comparison_preparation(manifest, preparation_path, source)
                                if cpu_source_manifest is not None else None)
        cpu_bytes = cpu_results.read_bytes()
        cpu_sha = hashlib.sha256(cpu_bytes).hexdigest()
        report = json.loads(cpu_bytes)
        if report['identity']['manifest_sha256'] != cpu.semantic.digest(source):
            raise ValueError('CPU results belong to another semantic manifest')
        bound = process_identity(driver_pid)
        record(status='following', cpu_results_sha256=cpu_sha,
               comparison_manifest_sha256=cpu.semantic.digest(manifest),
               cpu_source_manifest_sha256=cpu.semantic.digest(source),
               cpu_source_manifest=str(Path(cpu_source_manifest).resolve()) if cpu_source_manifest is not None else None,
               preparation_identity=preparation_identity,
               manifest_file_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
               driver_starttime=None if bound is None else bound['starttime'])
        current = bound
        while True:
            observation = driver_observation(bound, current)
            record(driver_observation=observation)
            if observation != 'running':
                break
            time.sleep(30)
            current = process_identity(driver_pid)
        if hashlib.sha256(cpu_results.read_bytes()).hexdigest() != cpu_sha:
            raise ValueError('CPU results changed while following driver; refusing final merge')
        if manifest_path.read_bytes() != manifest_bytes:
            raise ValueError('manifest changed while following driver; refusing final merge')
        if cpu_source_manifest is not None and Path(cpu_source_manifest).read_bytes() != source_bytes:
            raise ValueError('CPU source manifest changed while following driver; refusing final merge')
        if not gpu_results.is_file():
            record(status='driver-ended-without-checkpoint', reason='GPU checkpoint is absent; no merge or review produced')
            return 2
        record(status='finalizing')
        if cpu_source_manifest is None:
            cpu.export_table(manifest, report, cpu_output, gpu_results)
        else:
            cpu.export_table(manifest, report, cpu_output, gpu_results, source)
        review_path = cpu_output / 'gpu_same_precision_review.json'
        if cpu_source_manifest is None:
            reviewed = gpu_review.review(manifest_path, preparation_path, gpu_results, review_path)
        else:
            reviewed = gpu_review.review(manifest_path, preparation_path, gpu_results, review_path, cpu_source_manifest)
        checkpoint = json.loads(gpu_results.read_text())
        complete = reviewed['completed_checkpoint_cases'] == reviewed['expected_cases']
        record(status='joined-and-reviewed' if complete else 'joined-partial-checkpoint-and-reviewed',
               review=str(review_path), comparison=str(cpu_output / 'unified_comparison.csv'),
               checkpoint_status=checkpoint.get('status', 'unreported'),
               gpu_results_sha256=cpu.semantic.sha256_file(gpu_results),
               completed_checkpoint_cases=reviewed['completed_checkpoint_cases'],
               expected_cases=reviewed['expected_cases'],
               reviewed_implementations=len(reviewed['rows']),
               review_status_counts={name: sum(row['status'] == name for row in reviewed['rows'])
                                     for name in sorted({row['status'] for row in reviewed['rows']})})
        return 0
    except Exception as error:
        record(status='failed', error_type=type(error).__name__, reason=str(error))
        return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--cpu-output', type=Path, required=True)
    parser.add_argument('--gpu-results', type=Path, required=True)
    parser.add_argument('--driver-pid', type=int, required=True)
    parser.add_argument('--cpu-source-manifest', type=Path)
    args = parser.parse_args(argv)
    if args.driver_pid <= 0:
        parser.error('--driver-pid must be positive')
    result = follow(args.manifest, args.preparation, args.cpu_output, args.gpu_results, args.driver_pid, args.cpu_source_manifest)
    print((args.cpu_output / 'join_status.json').read_text(), end='')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
