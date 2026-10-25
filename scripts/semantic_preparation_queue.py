#!/usr/bin/env python3
"""Persistent CPU-only preparation spool for an immutable campaign subset.

Artifacts are trusted-local pickle files, never an import format for externally
supplied data. Checksums detect corruption; they do not authenticate a pickle.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import signal
import sys
import uuid

import semantic_cpu_pipeline as cpu

runner = cpu.runner
SCHEMA = 'cubutterfly-semantic-preparation-v1'
_CONFIG_CACHE = {}


def _sources():
    directory = Path(__file__).resolve().parent
    return {name: runner.sha256_file(directory / name) for name in (
        'run_semantic_campaign.py', 'semantic_cpu_pipeline.py',
        'semantic_ntt_reference.py', 'semantic_input_profiles.py')}


def _check_manifest(manifest):
    if not isinstance(manifest, dict) or not isinstance(manifest.get('cases'), list):
        raise ValueError('preparation manifest requires a cases list')
    ids = set()
    for case in manifest['cases']:
        case_id = case['id']
        if case_id != runner.digest(case['descriptor']['semantics']):
            raise ValueError('case semantic checksum mismatch: ' + case_id)
        if case_id in ids:
            raise ValueError('duplicate preparation case id: ' + case_id)
        ids.add(case_id)


def _envelope(value):
    return {'sha256': runner.digest(value), 'value': value}


def _read_json(path):
    envelope = json.loads(Path(path).read_text())
    value = envelope['value']
    if envelope.get('sha256') != runner.digest(value):
        raise ValueError('preparation metadata checksum mismatch: ' + str(path))
    return value, envelope['sha256']


def _write_json(path, value, immutable=False):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x') as handle:
            json.dump(_envelope(value), handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        if immutable:
            os.link(temporary, path)
        else:
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def initialize(directory, manifest, reference_max_elements=1048576,
               ntt_max_operations=16777216):
    """Bind an exact manifest subset, bounds and frozen CPU sources once."""
    _check_manifest(manifest)
    if any(type(n) is not int or n < 1 for n in (reference_max_elements, ntt_max_operations)):
        raise ValueError('preparation bounds must be positive integers')
    config = {'schema': SCHEMA, 'manifest': manifest,
              'manifest_sha256': runner.digest(manifest), 'source_sha256': _sources(),
              'reference_max_elements': reference_max_elements,
              'ntt_max_operations': ntt_max_operations}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'config.json'
    if not path.exists():
        try:
            _write_json(path, config, immutable=True)
        except FileExistsError:
            pass
    existing, _ = _read_json(path)
    if existing != config:
        raise ValueError('preparation identity mismatch (manifest/bounds/sources)')
    return existing


def _config(directory):
    path = (Path(directory) / 'config.json').resolve()
    stat = path.stat()
    signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    cached = _CONFIG_CACHE.get(path)
    if cached is not None and cached[0] == signature:
        return cached[1], cached[2]
    config, checksum = _read_json(path)
    if config.get('schema') != SCHEMA or runner.digest(config['manifest']) != config['manifest_sha256']:
        raise ValueError('invalid preparation configuration')
    _CONFIG_CACHE[path] = (signature, config, checksum,
                           {case['id']: case for case in config['manifest']['cases']})
    return config, checksum


def _record_path(directory, case_id):
    if len(case_id) != 64 or any(c not in '0123456789abcdef' for c in case_id):
        raise ValueError('invalid preparation case id')
    return Path(directory) / 'records' / (case_id + '.json')


def record_for(directory, case_id):
    """Poll only small metadata; artifact hashing is deferred to actual use."""
    path = _record_path(directory, case_id)
    if not path.exists():
        return None
    config, config_sha = _config(directory)
    record, _ = _read_json(path)
    cases = _CONFIG_CACHE[(Path(directory) / 'config.json').resolve()][3]
    if (case_id not in cases or record.get('case_id') != case_id or
            record.get('case_sha256') != runner.digest(cases[case_id]) or
            record.get('config_sha256') != config_sha):
        raise ValueError('preparation record identity mismatch')
    if record.get('status') not in ('ready', 'known-gap', 'failed'):
        raise ValueError('unknown preparation record status')
    if record['status'] == 'ready' and record['artifact'].get('path') != 'artifacts/' + case_id + '.pickle':
        raise ValueError('invalid preparation artifact path')
    return record


def load_case(directory, case):
    """Check immutable identity and artifact bytes before trusted-local unpickle."""
    config, _ = _config(directory)
    if config['source_sha256'] != _sources():
        raise ValueError('preparation source identity mismatch')
    record = record_for(directory, case['id'])
    if record is None:
        raise FileNotFoundError('case preparation is pending: ' + case['id'])
    if runner.digest(case) != record['case_sha256']:
        raise ValueError('requested preparation case identity mismatch')
    if record['status'] == 'known-gap':
        raise runner.Gap(record['gap_status'], record['reason'])
    if record['status'] == 'failed':
        raise RuntimeError('CPU preparation failed: ' + record['error'])
    path = Path(directory) / record['artifact']['path']
    if not path.is_file() or path.stat().st_size != record['artifact']['bytes']:
        raise ValueError('missing or corrupted preparation artifact: ' + str(path))
    # Verify the same bytes that are decoded, avoiding a second file read.
    data = path.read_bytes()
    import hashlib
    if hashlib.sha256(data).hexdigest() != record['artifact']['sha256']:
        raise ValueError('corrupted preparation artifact: ' + str(path))
    prepared = pickle.loads(data)
    if prepared.get('case_id') != case['id'] or prepared.get('s') != case['descriptor']['semantics']:
        raise ValueError('preparation payload identity mismatch')
    return prepared


def _worker_init():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def _prepare_one(directory, case, config_sha, max_elements, max_operations):
    _worker_init()
    record = {'case_id': case['id'], 'case_sha256': runner.digest(case),
              'config_sha256': config_sha}
    try:
        prepared = cpu.prepare_case(case, max_elements, max_operations)
        relative = 'artifacts/' + case['id'] + '.pickle'
        path = Path(directory) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('xb') as handle:
                pickle.dump(prepared, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        record.update(status='ready', artifact={'path': relative,
                      'sha256': runner.sha256_file(path), 'bytes': path.stat().st_size})
    except runner.Gap as error:
        record.update(status='known-gap', gap_status=error.status, reason=str(error))
    except Exception as error:
        record.update(status='failed', error=str(error), exception_type=type(error).__name__)
    path = _record_path(directory, case['id'])
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, record)
    return record


def _executor(jobs):
    return ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context('spawn'),
                               initializer=_worker_init)


def prepare_manifest(manifest, directory, jobs=4):
    """Prepare in manifest priority order, with at most jobs*2 submitted jobs."""
    if type(jobs) is not int or jobs < 1:
        raise ValueError('preparation jobs must be a positive integer')
    directory = Path(directory).resolve()
    if (directory / 'config.json').exists():
        config, _ = _config(directory)
        initialize(directory, manifest, config['reference_max_elements'], config['ntt_max_operations'])
    else:
        initialize(directory, manifest)
    config, config_sha = _config(directory)
    pending = []
    for case in manifest['cases']:
        record = record_for(directory, case['id'])
        if record is None:
            pending.append(case)
        elif record['status'] == 'ready':
            load_case(directory, case)
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    old_handler = signal.signal(signal.SIGTERM, request_stop)
    try:
        if pending:
            with _executor(jobs) as pool:
                futures = {}
                next_case = 0
                while futures or (next_case < len(pending) and not stop):
                    while not stop and next_case < len(pending) and len(futures) < jobs * 2:
                        case = pending[next_case]
                        future = pool.submit(_prepare_one, str(directory), case, config_sha,
                                             config['reference_max_elements'], config['ntt_max_operations'])
                        futures[future] = case
                        next_case += 1
                    if not futures:
                        break
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        case = futures.pop(future)
                        try:
                            future.result()
                        except Exception as error:
                            record = {'case_id': case['id'], 'case_sha256': runner.digest(case),
                                      'config_sha256': config_sha, 'status': 'failed',
                                      'error': str(error), 'exception_type': type(error).__name__}
                            path = _record_path(directory, case['id'])
                            path.parent.mkdir(parents=True, exist_ok=True)
                            _write_json(path, record)
    finally:
        signal.signal(signal.SIGTERM, old_handler)
    counts = {'ready': 0, 'known-gap': 0, 'failed': 0, 'pending': 0}
    for case in manifest['cases']:
        record = record_for(directory, case['id'])
        counts[record['status'] if record else 'pending'] += 1
    return {'status': 'stopped' if stop else 'failed' if counts['failed'] else 'complete',
            'counts': counts, 'manifest_sha256': config['manifest_sha256']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--only-init', action='store_true')
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    if args.only_init:
        config = initialize(args.directory, manifest)
        result = {'status': 'initialized', 'manifest_sha256': config['manifest_sha256']}
    else:
        result = prepare_manifest(manifest, args.directory, args.jobs)
    print(json.dumps(result, allow_nan=False))
    return 1 if result['status'] in ('failed', 'stopped') else 0


if __name__ == '__main__':
    sys.exit(main())
