#!/usr/bin/env python3
"""Continue immutable campaign cases in fresh, separately identified shards.

Historical measurements remain evidence, never rows of a new measurement.
Only a checked correctness failure permits advancing to another shard.
Validated timing checks each new measurement before timing without claiming a
separate completed conformance campaign.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
try:
    import importlib.metadata as importlib_metadata
except ImportError:  # Python 3.7 compatibility on the research environment.
    import importlib_metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_semantic_campaign as runner
from semantic_campaign_manifest import manifest_hash, validate_manifest_hash


PROCESSED = {'pass', 'accuracy-review-required', 'unsupported', 'missing-lowering',
             'runner-gap', 'validation-unavailable', 'resource-limit'}
SCHEMA = 'cubutterfly-semantic-continuation-v1'
REVISION = '0.9.3-bounded-reference'
TIMING_PHASES = {'comparison', 'validated-timing'}
_FINGERPRINT_CACHE = {}


def read(path):
    return json.loads(Path(path).read_text())


def evidence(path):
    path = Path(path).resolve(strict=True)
    def signature():
        value = path.stat()
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    before = signature()
    cached = _FINGERPRINT_CACHE.get(str(path))
    # Rapid same-size writes can share filesystem timestamps. Rehash recent
    # files; only stable historical evidence benefits from the stat cache.
    recent = max(before[-2:]) >= time.time_ns() - 1_000_000_000
    if cached is None or cached[0] != before or recent:
        checksum = runner.sha256_file(path)
        if signature() != before:
            raise ValueError('evidence changed while hashing: ' + str(path))
        cached = (before, checksum)
        _FINGERPRINT_CACHE[str(path)] = cached
    return {'path': str(path), 'sha256': cached[1]}


def verify_evidence(item):
    if evidence(item['path']) != item:
        raise ValueError('evidence changed: ' + item['path'])


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_state(path, state):
    save(path, {'sha256': runner.digest(state), 'state': state})


def load_state(path):
    envelope = read(path)
    if runner.digest(envelope['state']) != envelope['sha256']:
        raise ValueError('controller state checksum mismatch')
    return envelope['state']


def checked_report(path, manifest, library, gpu_uuid, timing):
    """Validate the index, every row envelope, and every referenced artifact."""
    path = Path(path).resolve(strict=True)
    report = read(path)
    if report.get('schema') != 'cubutterfly-semantic-results-v1':
        raise ValueError('unsupported result schema')
    identity = report['identity']
    if (identity.get('manifest_sha256') != runner.digest(manifest)
            or identity.get('library_sha256') != runner.sha256_file(library)
            or identity.get('timing') is not timing
            or not runner.same_gpu_uuid(gpu_uuid, identity.get('gpu_uuid'))):
        raise ValueError('result manifest/library/GPU/phase identity mismatch')
    cases = {case['id']: case for case in manifest['cases']}
    runner.validate_resume(report, identity, path.parent, set(cases))
    rows, files = [], [evidence(path)]
    for index in report['rows']:
        row_path = path.parent / index['path']
        row = read(row_path)['row']
        if row.get('descriptor') != cases[index['id']]['descriptor']:
            raise ValueError('result descriptor mismatch: ' + index['id'])
        rows.append(row)
        files.append(evidence(row_path))
        for implementation in [row, *row.get('baselines', [])]:
            if implementation.get('error_artifact'):
                files.append(evidence(implementation['error_artifact']['path']))
    population = identity.get('population')
    allowed = [case['id'] for case in manifest['cases']
               if population is None or population in case['populations']]
    allowed_ids = set(allowed)
    row_ids = {row['id'] for row in rows}
    if any(row['id'] not in allowed_ids for row in rows):
        raise ValueError('result row outside recorded population')
    expected_remaining = [identifier for identifier in allowed
                          if identifier not in row_ids]
    if report.get('remaining_case_ids') != expected_remaining:
        raise ValueError('result remaining-case index mismatch')
    return report, rows, files


def quarantine_entry(row, result_path):
    artifacts = [evidence(item['error_artifact']['path'])
                 for item in [row, *row.get('baselines', [])]
                 if item.get('error_artifact')]
    return {'id': row['id'], 'status': 'correctness-failure',
            'case_sha256': runner.digest(row['descriptor']),
            'row': evidence(Path(result_path).parent / 'rows' / (row['id'] + '.json')),
            'results': evidence(result_path), 'artifacts': artifacts,
            'artifact_status': 'retained' if artifacts else 'not-produced-by-runner',
            'error': row.get('error')}


def subset_metrics(cases):
    populations = {}
    for case in cases:
        for name in case['populations']:
            counts = populations.setdefault(name, {'requested_variants': 0, 'valid_variants': 0,
                'invalid_variants': 0, 'gap_variants': 0, 'unique_cases': 0, 'unique_gap_cases': 0})
            variants = sum(request['population'] == name for request in case.get('requested_variants', []))
            gap = case.get('capabilities', {}).get('status') == 'missing-lowering'
            counts['requested_variants'] += variants
            counts['valid_variants'] += variants
            counts['gap_variants'] += variants * gap
            counts['unique_cases'] += 1
            counts['unique_gap_cases'] += gap
    return {'populations': populations, 'unique_cases': len(cases),
            'unique_gap_cases': sum(c.get('capabilities', {}).get('status') == 'missing-lowering' for c in cases),
            **{key: sum(counts[key] for counts in populations.values())
               for key in ('requested_variants', 'valid_variants', 'gap_variants', 'invalid_variants')}}


def submanifest(parent, parent_path, cases, phase, completed, quarantine):
    value = copy.deepcopy({key: item for key, item in parent.items() if key != 'cases'})
    value.pop('manifest_sha256', None)
    value['cases'] = copy.deepcopy(cases)
    value['invalid_requests'] = []
    value['metrics'] = subset_metrics(cases)
    value['continuation_provenance'] = {
        'parent': evidence(parent_path), 'parent_manifest_sha256': parent['manifest_sha256'],
        'parent_metrics': parent.get('metrics'), 'phase': phase,
        'selected_case_ids': [case['id'] for case in cases],
        'omitted_processed_case_ids': sorted(completed),
        'quarantine': quarantine,
        'invalid_requests': {'source': evidence(parent_path),
                             'count': len(parent.get('invalid_requests', []))},
        'scope': 'selected subset; parent invalid requests and quarantines remain unresolved'}
    value['manifest_sha256'] = manifest_hash(value)
    return value


def controller_identity(args, prior_files, conformance):
    paths = [args.manifest, args.library, Path(__file__),
             Path(runner.__file__), Path(__file__).with_name('semantic_campaign_manifest.py'),
             Path(__file__).with_name('semantic_dao_baseline.py'),
             Path(__file__).with_name('semantic_gpuntt_baseline.py'),
             Path(__file__).with_name('semantic_ntt_reference.py'),
             Path(__file__).resolve().parents[1] / 'python/cubutterfly/__init__.py']
    paths += [path for path in (args.calibration_manifest, args.vkfft_library, args.gpuntt_library) if path]
    # Calibration's external model and registry are also frozen between shards.
    if args.calibration_manifest:
        calibration = read(args.calibration_manifest)
        paths += [Path(calibration[key]) for key in ('registry_path', 'hardware_rates_source', 'cost_model')
                  if calibration.get(key)]
    parent = read(args.manifest)
    paths += [Path(item['path']) for item in parent['fingerprints']['sources']]
    if parent['fingerprints'].get('config', {}).get('path'):
        paths.append(Path(parent['fingerprints']['config']['path']))
    return {'schema': SCHEMA, 'phase': args.phase, 'output': str(args.output.resolve()),
            'gpu_uuid': args.gpu_uuid, 'inputs': [evidence(path) for path in paths],
            'prior_evidence': prior_files, 'conformance_evidence': conformance,
            'python': sys.version, 'executable': sys.executable, 'numpy': runner.np.__version__,
            'torch_distribution': importlib_metadata.version('torch'),
            'environment': {key: value for key, value in sorted(os.environ.items())
                            if key.startswith('CUBUTTERFLY_') or key in ('CUDA_VISIBLE_DEVICES', 'CUDA_MODULE_LOADING')},
            'protocol': dict(runner.PROTOCOL, float_verdict='error-report')}


def runner_identity_base(identity):
    return {key: value for key, value in identity.items()
            if key not in ('manifest_sha256', 'error_directory', 'population')}


def file_signature(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def check_closed_files(closed):
    for path, signature in closed.items():
        try:
            unchanged = file_signature(path) == signature
        except FileNotFoundError:
            unchanged = False
        if not unchanged:
            raise ValueError('evidence changed or missing: ' + path)


def fatal_baseline(row):
    return any(item.get('status') in ('runtime-error', 'runner-error', 'baseline-error')
               or item.get('support_status') == 'external-binding-error'
               for item in row.get('baselines', []))


def check_shard(shard, state, args, ids, parent_cases, completed, quarantined,
                verified_quarantine, closed):
    started = time.monotonic()
    print('controller: verifying shard ' + shard['output'], flush=True)
    for item in shard.get('evidence', []):
        verify_evidence(item)
    manifest_path = Path(shard['manifest']['path'])
    verify_evidence(shard['manifest'])
    manifest = read(manifest_path)
    if not validate_manifest_hash(manifest):
        raise ValueError('shard manifest checksum mismatch')
    excluded = completed | quarantined
    expected = [identifier for identifier in ids if identifier not in excluded]
    if [case['id'] for case in manifest['cases']] != expected:
        raise ValueError('shard selection mismatch')
    for case in manifest['cases']:
        if case != parent_cases[case['id']]:
            raise ValueError('shard case changed')
    path = Path(shard['output']) / 'results.json'
    if not path.exists():
        raise ValueError('interrupted shard has no result; investigate rather than retry')
    report, rows, files = checked_report(path, manifest, args.library, args.gpu_uuid,
                                       state['identity']['phase'] in TIMING_PHASES)
    if (report['identity'].get('runner_sha256') != runner.sha256_file(runner.__file__)
            or report['identity'].get('protocol') != state['identity']['protocol']):
        raise ValueError('shard runner source/protocol identity mismatch')
    base = runner_identity_base(report['identity'])
    if state['measurement_identity'] is not None and state['measurement_identity'] != base:
        raise ValueError('runner measurement identity changed between shards')
    state['measurement_identity'] = base
    failures = [row for row in rows if row['status'] == 'correctness-failure']
    expected_code = 1 if report['status'] == 'stopped-correctness-failure' else 0
    shard['evidence'] = [shard['manifest'], *files]
    if (shard.get('returncode') != expected_code
            or any(fatal_baseline(row) for row in rows)):
        return False
    if report['status'] == 'stopped-correctness-failure':
        if len(failures) != 1 or not rows or rows[-1]['id'] != failures[0]['id']:
            raise ValueError('ambiguous correctness stop')
    elif not report['status'].startswith('complete') or report['remaining_case_ids'] or failures:
        return False
    if any(row['status'] not in PROCESSED | {'correctness-failure'} for row in rows):
        raise ValueError('runtime/runner error cannot be continued')
    if [row['id'] for row in rows] != expected[:len(rows)]:
        raise ValueError('shard result execution order mismatch')
    completed.update(row['id'] for row in rows if row['status'] in PROCESSED)
    for row in failures:
        verified_quarantine.append(quarantine_entry(row, path))
        quarantined.add(row['id'])
    for item in shard['evidence']:
        closed[item['path']] = file_signature(item['path'])
    print('controller: verified shard (%d rows, %.3fs)' %
          (len(rows), time.monotonic() - started), flush=True)
    return True


def validate_upgrade(old_identity, identity, old_source):
    old_hash = runner.sha256_file(old_source)
    source_path = str(Path(__file__).resolve())
    previous = copy.deepcopy(old_identity)
    old_inputs = [item for item in previous['inputs'] if item['path'] == source_path]
    new_inputs = [item for item in identity['inputs'] if item['path'] == source_path]
    if len(old_inputs) != 1 or len(new_inputs) != 1 or old_inputs[0]['sha256'] != old_hash:
        raise ValueError('controller upgrade old source hash mismatch')
    old_inputs[0]['sha256'] = new_inputs[0]['sha256']
    if previous != identity:
        raise ValueError('controller upgrade permits only controller source identity change')
    return old_hash, new_inputs[0]['sha256']


def adopt_upgrade(state_path, state, identity, hashes):
    old_envelope = read(state_path)
    directory = state_path.parent / ('controller-upgrade-%d' % time.time_ns())
    directory.mkdir(exist_ok=False)
    save(directory / 'controller.before.json', old_envelope)
    state['identity'] = identity
    state['controller_revision'] = REVISION
    save(directory / 'migration.json', {
        'reason': 'explicit source-only upgrade to incremental evidence validation',
        'old_source_sha256': hashes[0], 'new_source_sha256': hashes[1],
        'old_state_sha256': old_envelope['sha256'], 'new_state_sha256': runner.digest(state)})
    save_state(state_path, state)


def continue_campaign(args, launch=subprocess.run):
    if args.phase == 'validated-timing' and args.prior_results:
        raise ValueError('validated-timing forbids --prior-results; every case needs a fresh validated measurement')
    if args.phase in TIMING_PHASES and args.calibration_manifest is None:
        raise ValueError(args.phase + ' requires --calibration-manifest')
    # Each invocation is a fresh trust boundary, including repeated calls in tests.
    _FINGERPRINT_CACHE.clear()
    parent = read(args.manifest)
    if not validate_manifest_hash(parent):
        raise ValueError('parent manifest checksum mismatch')
    runner.validate_fingerprints(parent, args.library)
    ids = [case['id'] for case in parent['cases']]
    parent_cases = {case['id']: case for case in parent['cases']}
    if len(ids) != len(set(ids)) or any(case['id'] != runner.digest(case['descriptor']['semantics']) for case in parent['cases']):
        raise ValueError('parent semantic case checksum/uniqueness mismatch')
    processed, quarantine, prior_files = set(), [], []
    for path in args.prior_results:
        started = time.monotonic()
        print('controller: verifying prior ' + str(path), flush=True)
        report, rows, files = checked_report(path, parent, args.library, args.gpu_uuid, False)
        if not (report['status'].startswith('complete') or report['status'] == 'stopped-correctness-failure'):
            raise ValueError('prior stopped/incomplete run requires investigation')
        if report['status'] == 'stopped-correctness-failure' and (
                not rows or rows[-1]['status'] != 'correctness-failure'
                or sum(row['status'] == 'correctness-failure' for row in rows) != 1):
            raise ValueError('ambiguous prior correctness stop')
        prior_files.extend(files)
        for row in rows:
            if fatal_baseline(row):
                raise ValueError('prior baseline runtime error requires investigation: ' + row['id'])
            if row['status'] in PROCESSED:
                processed.add(row['id'])
            elif row['status'] == 'correctness-failure':
                quarantine.append(quarantine_entry(row, path))
            else:
                raise ValueError('prior runtime/runner error requires investigation: ' + row['id'])
        print('controller: verified prior (%d rows, %.3fs)' %
              (len(rows), time.monotonic() - started), flush=True)
    conformance_evidence = []
    if args.phase == 'comparison':
        path = args.output / 'conformance/controller.json'
        conformance = load_state(path)
        if conformance['status'] != 'complete-with-unresolved-evidence':
            raise ValueError('conformance phase must finish before comparison')
        # Verify all frozen evidence, including the completed conformance shards.
        for item in conformance['identity']['inputs'] + conformance['identity']['prior_evidence']:
            verify_evidence(item)
        for shard in conformance['shards']:
            for item in shard.get('evidence', []):
                verify_evidence(item)
        if (conformance['identity']['inputs'][0] != evidence(args.manifest)
                or conformance['identity']['inputs'][1] != evidence(args.library)
                or conformance['identity']['gpu_uuid'] != args.gpu_uuid
                or conformance['identity']['prior_evidence'] != prior_files):
            raise ValueError('comparison does not match conformance parent/prior/GPU identity')
        conformance_completed = set(processed)
        conformance_quarantined = {item['id'] for item in quarantine}
        conformance_quarantine = list(quarantine)
        for shard in conformance['shards']:
            if not check_shard(shard, conformance, args, ids, parent_cases,
                               conformance_completed, conformance_quarantined,
                               conformance_quarantine, {}):
                raise ValueError('conformance shard requires investigation')
        if (set(ids) != conformance_completed | conformance_quarantined
                or conformance['processed_case_ids'] != sorted(conformance_completed)
                or conformance['quarantine'] != conformance_quarantine):
            raise ValueError('conformance completion/progress chain mismatch')
        quarantine = conformance['quarantine']
        conformance_evidence = [evidence(path)]
        processed = set()  # Every nonquarantined parent case still needs timing.
    identity = controller_identity(args, prior_files, conformance_evidence)
    if args.phase == 'comparison':
        comparable = lambda value: {key: item for key, item in value.items()
                                    if key not in ('phase', 'output', 'conformance_evidence')}
        if comparable(conformance['identity']) != comparable(identity):
            raise ValueError('comparison execution identity differs from conformance')
    state_path = args.output / args.phase / 'controller.json'
    upgrade = None
    if state_path.exists():
        state = load_state(state_path)
        if state['identity'] != identity:
            if getattr(args, 'upgrade_controller_from', None) is None:
                raise ValueError('controller resume identity mismatch')
            upgrade = validate_upgrade(state['identity'], identity, args.upgrade_controller_from)
    else:
        state = {'identity': identity, 'status': 'running', 'shards': [],
                 'processed_case_ids': sorted(processed), 'quarantine': quarantine,
                 'measurement_identity': None}
        save_state(state_path, state)
    completed = set(processed)
    quarantined = {item['id'] for item in quarantine}
    verified_quarantine = list(quarantine)
    closed = {}
    checked_count = 0
    while True:
        for item in identity['inputs'] + identity['prior_evidence'] + identity['conformance_evidence']:
            verify_evidence(item)
        check_closed_files(closed)
        for shard in state['shards'][checked_count:]:
            if not check_shard(shard, state, args, ids, parent_cases, completed,
                               quarantined, verified_quarantine, closed):
                if upgrade:
                    raise ValueError('controller upgrade chain requires investigation')
                state['status'] = 'stopped-investigation-required'
                save_state(state_path, state)
                return 1
            checked_count += 1
        if state['status'] == 'complete-with-unresolved-evidence' and (
                set(ids) != completed | quarantined
                or state['processed_case_ids'] != sorted(completed)
                or state['quarantine'] != verified_quarantine):
            raise ValueError('controller completion/progress chain mismatch')
        state['controller_revision'] = REVISION
        state.update(processed_case_ids=sorted(completed), quarantine=verified_quarantine)
        if upgrade:
            if state['status'] == 'stopped-investigation-required':
                raise ValueError('controller upgrade stopped state requires investigation')
            adopt_upgrade(state_path, state, identity, upgrade)
            upgrade = None
        excluded = completed | quarantined
        remaining = [case for case in parent['cases'] if case['id'] not in excluded]
        if not remaining:
            state.update(status='complete-with-unresolved-evidence',
                         excluded_quarantine_case_ids=sorted(quarantined),
                         parent_invalid_request_count=len(parent.get('invalid_requests', [])),
                         verdict='no aggregate accuracy pass; retained gaps and accuracy review remain')
            save_state(state_path, state)
            return 0
        if state['status'] == 'stopped-investigation-required':
            raise ValueError('stopped runtime/runner shard requires investigation; no retry')
        if getattr(args, 'verify_only', False):
            save_state(state_path, state)
            return 0
        directory = args.output / args.phase / ('shard-%04d' % len(state['shards']))
        manifest_path = directory / 'manifest.json'
        if directory.exists():
            raise ValueError('untracked shard directory exists; refusing overwrite')
        manifest = submanifest(parent, args.manifest, remaining, args.phase, completed, verified_quarantine)
        save(manifest_path, manifest)
        output = directory / 'measurement'
        command = [sys.executable, str(Path(runner.__file__).resolve()), '--manifest', str(manifest_path.resolve()),
                   '--library', str(args.library.resolve()), '--output', str(output.resolve()),
                   '--gpu-uuid', args.gpu_uuid, '--max-cases', str(len(remaining)),
                   '--float-verdict', 'error-report']
        for option, value in (('--calibration-manifest', args.calibration_manifest),
                              ('--vkfft-library', args.vkfft_library), ('--gpuntt-library', args.gpuntt_library)):
            if value:
                command.extend([option, str(value.resolve())])
        if args.phase in TIMING_PHASES:
            command.append('--timing')
        shard = {'manifest': evidence(manifest_path), 'output': str(output.resolve()), 'command': command}
        state['shards'].append(shard)
        save_state(state_path, state)
        print('controller: launching shard %d (%d cases)' %
              (len(state['shards']) - 1, len(remaining)), flush=True)
        result = launch(command, check=False)
        shard['returncode'] = result.returncode
        save_state(state_path, state)
        # A nonzero exit only permits continuation if a complete correctness-stop
        # checkpoint proves which case failed; all other exits stop immediately.
        path = output / 'results.json'
        if result.returncode not in (0, 1) or not path.exists():
            state.update(status='stopped-investigation-required', returncode=result.returncode)
            save_state(state_path, state)
            return result.returncode or 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior-results', type=Path, action='append', default=[])
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--phase', choices=('conformance', 'comparison', 'validated-timing'), required=True,
                        help='validated-timing validates each fresh case before timing; comparison requires full conformance')
    parser.add_argument('--calibration-manifest', type=Path)
    parser.add_argument('--vkfft-library', type=Path)
    parser.add_argument('--gpuntt-library', type=Path)
    parser.add_argument('--upgrade-controller-from', type=Path,
                        help='explicitly accept only the controller source hash change from this old source')
    parser.add_argument('--verify-only', action='store_true',
                        help='fully verify and optionally upgrade recorded evidence without launching measurements')
    args = parser.parse_args(argv)
    if args.phase in TIMING_PHASES and args.calibration_manifest is None:
        parser.error(args.phase + ' requires --calibration-manifest')
    if args.phase == 'validated-timing' and args.prior_results:
        parser.error('validated-timing forbids --prior-results')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'controller.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return continue_campaign(args)


if __name__ == '__main__':
    raise SystemExit(main())
