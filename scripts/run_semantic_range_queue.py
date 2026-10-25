#!/usr/bin/env python3
"""Separate, resumable range diagnostics for a completed conformance quarantine."""
from __future__ import annotations

import argparse
import fcntl
try:
    import importlib.metadata as importlib_metadata
except ImportError:  # Python 3.7 compatibility on the research environment.
    import importlib_metadata
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import continue_semantic_campaign as campaign
import run_fft_range_control as control
import run_semantic_campaign as runner
import semantic_input_profiles as profiles


TERMINAL = {'accuracy-review-required', 'correctness-failure'}
ERRORS = {'baseline-error', 'runtime-error', 'runner-error', 'execution-error',
          'resource-limit', 'incomplete-error'}


def artifact_evidence(item):
    value = campaign.evidence(item['path'])
    if value['sha256'] != item['sha256']:
        raise ValueError('artifact checksum mismatch: ' + item['path'])
    return value


def measurement_rows(path, args):
    """Check complete historical row envelopes using the stable-file hash cache."""
    path = Path(path)
    report = campaign.read(path)
    if report.get('schema') != 'cubutterfly-semantic-results-v1':
        raise ValueError('measurement schema mismatch')
    identity = report['identity']
    if (identity['library_sha256'] != campaign.evidence(args.library)['sha256']
            or not runner.same_gpu_uuid(identity['gpu_uuid'], args.gpu_uuid)
            or identity.get('timing') is not False):
        raise ValueError('source measurement identity mismatch')
    indices = report['rows']
    if report.get('rows_sha256') != runner.digest(indices):
        raise ValueError('measurement index checksum mismatch')
    rows = {}
    files = [campaign.evidence(path)]
    for index in indices:
        identifier = index['id']
        if identifier in rows or index['path'] != 'rows/' + identifier + '.json':
            raise ValueError('duplicate or invalid measurement row path')
        row_path = path.parent / index['path']
        envelope = campaign.read(row_path)
        row = envelope['row']
        if (envelope.get('sha256') != runner.digest(row)
                or envelope['sha256'] != index['sha256']
                or row['id'] != identifier or row['status'] != index['status']
                or identifier != runner.digest(row['descriptor']['semantics'])):
            raise ValueError('measurement row checksum/identity mismatch')
        if any(b.get('status') in ERRORS or b.get('support_status') == 'external-binding-error'
               for b in row.get('baselines', [])):
            raise ValueError('source baseline execution error')
        files.append(campaign.evidence(row_path))
        for item in [row, *row.get('baselines', [])]:
            if item.get('error_artifact'):
                files.append(artifact_evidence(item['error_artifact']))
        rows[identifier] = row
    return report, rows, files


def checked_controller(args):
    state = campaign.load_state(args.controller)
    if state.get('status') != 'complete-with-unresolved-evidence':
        raise ValueError('conformance controller must finish before range controls')
    identity = state['identity']
    if identity.get('phase') != 'conformance' or not runner.same_gpu_uuid(identity['gpu_uuid'], args.gpu_uuid):
        raise ValueError('conformance controller phase/GPU mismatch')
    files = [campaign.evidence(args.controller)]
    frozen = identity['inputs'] + identity['prior_evidence'] + identity.get('conformance_evidence', [])
    for item in frozen:
        campaign.verify_evidence(item)
    files.extend(frozen)
    library = campaign.evidence(args.library)
    if library not in identity['inputs']:
        raise ValueError('controller library mismatch')
    for path in (args.calibration_manifest, args.vkfft_library, args.gpuntt_library):
        if path and campaign.evidence(path) not in identity['inputs']:
            raise ValueError('controller calibration/baseline binding mismatch')
    parent_path = Path(identity['inputs'][0]['path'])
    parent = campaign.read(parent_path)
    if not campaign.validate_manifest_hash(parent):
        raise ValueError('parent manifest checksum mismatch')
    cases = {case['id']: case for case in parent['cases']}
    if len(cases) != len(parent['cases']):
        raise ValueError('duplicate parent cases')
    reports = {}
    # Prior evidence enumerates reports, rows and NPZs; only reports are decoded.
    for item in identity['prior_evidence']:
        if Path(item['path']).name == 'results.json':
            reports[item['path']] = measurement_rows(item['path'], args)
    for shard in state['shards']:
        for item in shard.get('evidence', []):
            campaign.verify_evidence(item)
        campaign.verify_evidence(shard['manifest'])
        manifest = campaign.read(shard['manifest']['path'])
        if not campaign.validate_manifest_hash(manifest):
            raise ValueError('shard manifest checksum mismatch')
        provenance = manifest['continuation_provenance']
        if (provenance['parent'] != campaign.evidence(parent_path)
                or provenance['parent_manifest_sha256'] != parent['manifest_sha256']
                or provenance['phase'] != 'conformance'):
            raise ValueError('shard parent provenance mismatch')
        if any(case != cases.get(case['id']) for case in manifest['cases']):
            raise ValueError('shard descriptor mismatch')
        path = str((Path(shard['output']) / 'results.json').resolve())
        checked = measurement_rows(path, args)
        report, rows, evidence = checked
        expected_code = 1 if report['status'] == 'stopped-correctness-failure' else 0
        if (shard.get('returncode') != expected_code
                or report['identity']['manifest_sha256'] != runner.digest(manifest)
                or list(rows) != [case['id'] for case in manifest['cases']][:len(rows)]
                or (expected_code == 0 and (not report['status'].startswith('complete')
                                           or report['remaining_case_ids']))):
            raise ValueError('incomplete or incompatible conformance shard')
        reports[path] = checked
        files.extend([shard['manifest'], *evidence])
    quarantine = []
    seen = set()
    for item in state['quarantine']:
        for value in [item['row'], item['results'], *item.get('artifacts', [])]:
            campaign.verify_evidence(value)
            files.append(value)
        path = item['results']['path']
        if path not in reports:
            reports[path] = measurement_rows(path, args)
        row = reports[path][1][item['id']]
        row_path = Path(path).parent / 'rows' / (row['id'] + '.json')
        if (row['id'] in seen or row['status'] != 'correctness-failure'
                or item['row'] != campaign.evidence(row_path)
                or item['case_sha256'] != runner.digest(row['descriptor'])
                or row['descriptor'] != cases[row['id']]['descriptor']):
            raise ValueError('quarantine provenance mismatch')
        seen.add(row['id'])
        quarantine.append((item, row))
    observed = {}
    for _, rows, evidence in reports.values():
        observed.update(rows)
        files.extend(evidence)
    if (set(state['processed_case_ids']) | seen != set(cases)
            or set(state['processed_case_ids']) & seen
            or {identifier for identifier, row in observed.items()
                if row['status'] == 'correctness-failure'} != seen):
        raise ValueError('controller coverage/quarantine mismatch')
    for identifier in state['processed_case_ids']:
        if observed.get(identifier, {}).get('status') not in campaign.PROCESSED:
            raise ValueError('controller processed evidence mismatch')
    return quarantine, unique_evidence(files)


def unique_evidence(files):
    by_path = {}
    for item in files:
        if item['path'] in by_path and by_path[item['path']] != item:
            raise ValueError('conflicting evidence hashes')
        by_path[item['path']] = item
    return [by_path[path] for path in sorted(by_path)]


def scope_reason(row):
    s = row['descriptor']['semantics']
    if (s.get('operator'), s.get('storage')) not in ((0, 4), (2, 0)):
        return 'control supports FP16 C2C FFT and FP16 FWHT only'
    if s.get('fft_type') != 0 or s.get('length_mode') != 0:
        return 'control requires standard exact-length contracts'
    shape = s.get('shape', [])
    if not shape or any(type(n) is not int or n < 1 or n & (n - 1) for n in shape):
        return 'control requires power-of-two extents'
    scale = s.get('fft_output_scale')
    if type(scale) not in (int, float) or not math.isfinite(scale) or abs(scale) > 1:
        return 'control requires absolute output scale at most one'
    if s['operator'] == 2 and scale != 1:
        return 'FWHT control requires canonical fft_output_scale=1'
    if s['batch'] * max(np.prod(runner.side_extents(s, side)) for side in ('input', 'output')) > 1048576:
        return 'complete control reference exceeds the fixed element bound'
    if not isinstance(row.get('mapping'), dict) or not row.get('error_artifact'):
        return 'original failure lacks mapping replay or error artifact'
    return None


def profile_source(row):
    s = row['descriptor']['semantics']
    return profiles.apply_input_profile(s, runner.make_source(s), profiles.scaled_profile(s),
                                        encode_storage=runner.encode_storage,
                                        decode_storage=runner.decode_storage)


def checked_control(path, row, args, historical=False):
    report = campaign.read(path)
    if report.get('schema') != 'cubutterfly-fft-range-control-v1':
        raise ValueError('range control schema mismatch')
    identity = report['identity']
    if (report['original_semantic_id'] != row['id'] or report['descriptor'] != row['descriptor']
            or report['mapping'] != row['mapping']
            or identity['library_sha256'] != campaign.evidence(args.library)['sha256']
            or not runner.same_gpu_uuid(identity['gpu_uuid'], args.gpu_uuid)
            or identity.get('timing') is not False or report['protocol'].get('timing') is not False
            or identity['calibration_file_sha256'] != campaign.evidence(args.calibration_manifest)['sha256']):
        raise ValueError('range control semantic/mapping/measurement identity mismatch')
    for name in ('vkfft_library', 'gpuntt_library'):
        library = getattr(args, name)
        if identity.get(name) != (campaign.evidence(library) if library else None):
            raise ValueError('range control baseline identity mismatch')
    if not historical:
        for key, path_value in (('runner_sha256', control.__file__),
                                ('frozen_runner_sha256', runner.__file__),
                                ('profile_script_sha256', profiles.__file__)):
            if identity.get(key) != campaign.evidence(path_value)['sha256']:
                raise ValueError('new control source identity mismatch')
    evidence = report['source_evidence']
    if (evidence['row_checksum'] != runner.digest(row)
            or evidence['row_file_sha256'] != campaign.evidence(evidence['row_path'])['sha256']
            or campaign.read(evidence['row_path'])['row'] != row
            or evidence['original_report_sha256'] != campaign.evidence(evidence['original_report_path'])['sha256']):
        raise ValueError('control original evidence mismatch')
    original = campaign.read(evidence['original_report_path'])
    matching = [index for index in original['rows'] if index['id'] == row['id']]
    if (len(matching) != 1 or matching[0] != evidence['original_row_index']
            or original['identity'] != evidence['original_measurement_identity']
            or original.get('rows_sha256') != runner.digest(original['rows'])
            or matching[0]['sha256'] != evidence['row_checksum']
            or (Path(evidence['original_report_path']).parent / matching[0]['path']).resolve()
               != Path(evidence['row_path']).resolve()
            or evidence['original_error_artifact'] != row.get('error_artifact')):
        raise ValueError('control original report binding mismatch')
    files = [campaign.evidence(path), campaign.evidence(evidence['row_path']),
             campaign.evidence(evidence['original_report_path']),
             artifact_evidence(evidence['original_error_artifact']),
             artifact_evidence(report['source_artifact'])]
    source, metadata = profile_source(row)
    if (report['input_profile'] != metadata
            or report['id'] != control.input_profile_id(row['id'], metadata)):
        raise ValueError('range control profile algorithm mismatch')
    with np.load(report['source_artifact']['path'], allow_pickle=False) as arrays:
        s = row['descriptor']['semantics']
        for name, expected in (('source', source), ('storage', runner.encode_storage(source, s['storage'])),
                               ('reference', runner.reference(s, source))):
            actual = arrays[name]
            if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
                raise ValueError('range control source/reference array mismatch')
    for item in [report, *report.get('baselines', [])]:
        if item.get('error_artifact'):
            files.append(artifact_evidence(item['error_artifact']))
    if report['status'] not in TERMINAL:
        raise ValueError('control did not finish: ' + report['status'])
    if not report.get('error_artifact') or 'baseline_stop' in report or any(
            item.get('status') in ERRORS or item.get('support_status') == 'external-binding-error'
            for item in report.get('baselines', [])):
        raise ValueError('control contains execution/baseline errors')
    if any(item.get('status') == 'correctness-failure' for item in report.get('baselines', [])):
        report = dict(report, status='correctness-failure')
    return report, unique_evidence(files)


def queue_identity(args, frozen):
    paths = [Path(__file__), Path(control.__file__), Path(runner.__file__), Path(profiles.__file__),
             Path(campaign.__file__), args.library, args.calibration_manifest]
    paths += [Path(runner.__file__).with_name(name) for name in
              ('diagnose_xd_fft.py', 'vkfft_xd.py', 'semantic_dao_baseline.py', 'semantic_gpuntt_baseline.py')]
    paths.append(Path(__file__).resolve().parents[1] / 'python/cubutterfly/__init__.py')
    paths.extend(path for path in (args.vkfft_library, args.gpuntt_library) if path)
    calibration = campaign.read(args.calibration_manifest)
    paths += [Path(calibration[key]) for key in ('registry_path', 'hardware_rates_source', 'cost_model')]
    from semantic_dao_baseline import backend_inventory as dao_inventory
    from semantic_gpuntt_baseline import DEFAULT_LIBRARY, backend_inventory as ntt_inventory
    return {'schema': 'cubutterfly-semantic-range-queue-v1', 'gpu_uuid': args.gpu_uuid,
            'controller': campaign.evidence(args.controller),
            'output': str(args.output.resolve()), 'inputs': unique_evidence([campaign.evidence(p) for p in paths]),
            'conformance_evidence': frozen,
            'existing_controls': [campaign.evidence(p) for p in args.existing_control],
            'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__,
            'torch_distribution': importlib_metadata.version('torch'),
            'baseline_backends': {'daofht': dao_inventory(),
                                  'gpu-ntt': ntt_inventory(args.gpuntt_library or DEFAULT_LIBRARY)},
            'environment': {k: v for k, v in sorted(os.environ.items())
                            if k.startswith('CUBUTTERFLY_') or k in ('CUDA_VISIBLE_DEVICES', 'CUDA_MODULE_LOADING')},
            'timing': False, 'scope': 'separate range diagnostics; original pressure tests remain unresolved'}


def run_queue(args, launch=subprocess.run):
    campaign._FINGERPRINT_CACHE.clear()
    quarantine, frozen = checked_controller(args)
    identity = queue_identity(args, frozen)
    signatures = {item['path']: file_signature(item['path']) for item in frozen}
    state_path = args.output / 'state.json'
    if state_path.exists():
        if not args.resume:
            raise ValueError('queue exists; use --resume')
        state = campaign.load_state(state_path)
        if state['identity'] != identity:
            raise ValueError('range queue resume identity mismatch')
    else:
        if args.resume:
            raise ValueError('no range queue to resume')
        state = {'identity': identity, 'status': 'running', 'cases': []}
        campaign.save_state(state_path, state)
    existing = {}
    for path in args.existing_control:
        identifier = campaign.read(path)['original_semantic_id']
        if identifier in existing or identifier not in {row['id'] for _, row in quarantine}:
            raise ValueError('duplicate or foreign existing control')
        existing[identifier] = path
    by_id = {entry['original_semantic_id']: entry for entry in state['cases']}
    if len(by_id) != len(state['cases']) or set(by_id) - {row['id'] for _, row in quarantine}:
        raise ValueError('invalid queue case index')
    for item, row in quarantine:
        if any(file_signature(path) != signature for path, signature in signatures.items()):
            raise ValueError('frozen conformance evidence changed during range queue')
        for evidence in identity['inputs'] + identity['existing_controls'] + [identity['controller']]:
            campaign.verify_evidence(evidence)
        for entry in state['cases']:
            for evidence in entry.get('evidence', []):
                campaign.verify_evidence(evidence)
        identifier = row['id']
        reason = scope_reason(row)
        if identifier in by_id:
            entry = by_id[identifier]
            for evidence in entry.get('evidence', []):
                campaign.verify_evidence(evidence)
            if entry['status'] == 'not-applicable-control' and reason == entry.get('reason'):
                continue
            if entry['status'] not in TERMINAL:
                raise ValueError('partial/unknown control requires investigation; no automatic retry')
            report, files = checked_control(entry['output'], row, args, entry['origin'] == 'historical-control-reference')
            if report['status'] != entry['status'] or files != entry['evidence']:
                raise ValueError('completed control index mismatch')
            continue
        if reason:
            entry = {'original_semantic_id': identifier, 'status': 'not-applicable-control',
                     'reason': reason, 'evidence': [item['row'], item['results']]}
            state['cases'].append(entry)
            campaign.save_state(state_path, state)
            continue
        entry = {'original_semantic_id': identifier, 'status': 'running', 'evidence': []}
        state['cases'].append(entry)
        if identifier in existing:
            path = existing[identifier]
            entry.update(origin='historical-control-reference', output=str(path.resolve()))
            campaign.save_state(state_path, state)
            try:
                report, files = checked_control(path, row, args, historical=True)
            except Exception as exc:
                state.update(status='stopped-investigation-required', error=str(exc))
                campaign.save_state(state_path, state)
                raise
        else:
            directory = args.output / 'controls' / identifier
            command = [sys.executable, str(Path(control.__file__).resolve()), '--row', item['row']['path'],
                       '--library', str(args.library.resolve()), '--output', str(directory.resolve()),
                       '--gpu-uuid', args.gpu_uuid, '--calibration-manifest', str(args.calibration_manifest.resolve())]
            for name in ('vkfft_library', 'gpuntt_library'):
                if getattr(args, name):
                    command.extend(['--' + name.replace('_', '-'), str(getattr(args, name).resolve())])
            path = directory / 'results.json'
            entry.update(origin='new-control', command=command, output=str(path.resolve()))
            campaign.save_state(state_path, state)
            try:
                result = launch(command, check=False)
                entry['returncode'] = result.returncode
                campaign.save_state(state_path, state)
                report, files = checked_control(path, row, args)
                if result.returncode != (1 if report['status'] == 'correctness-failure' else 0):
                    raise ValueError('control exit/checkpoint mismatch')
            except Exception as exc:
                entry.update(status='investigation-required', error=str(exc))
                if path.is_file():
                    entry['evidence'] = [campaign.evidence(path)]
                    failed = campaign.read(path)
                    for artifact in [failed.get('source_artifact'), failed.get('error_artifact'),
                                     *(b.get('error_artifact') for b in failed.get('baselines', []))]:
                        if artifact and Path(artifact['path']).is_file():
                            entry['evidence'].append(campaign.evidence(artifact['path']))
                state.update(status='stopped-investigation-required', error=str(exc))
                campaign.save_state(state_path, state)
                return 1
        entry.update(status=report['status'], input_profile_id=report['id'], evidence=files,
                     measurement_identity=report['identity'])
        campaign.save_state(state_path, state)
    counts = {}
    for entry in state['cases']:
        counts[entry['status']] = counts.get(entry['status'], 0) + 1
    state.update(status='complete-with-unresolved-accuracy', case_counts=counts,
                 quarantined_cases=len(quarantine),
                 new_controls=sum(e['origin'] == 'new-control' for e in state['cases'] if 'origin' in e),
                 historical_control_references=sum(e.get('origin') == 'historical-control-reference' for e in state['cases']),
                 verdict='diagnostic controls only; no aggregate accuracy pass or timing result')
    campaign.save_state(state_path, state)
    return 0


def file_signature(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('controller', 'library', 'output', 'calibration-manifest'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--vkfft-library', type=Path)
    parser.add_argument('--gpuntt-library', type=Path)
    parser.add_argument('--existing-control', type=Path, action='append', default=[])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'queue.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_queue(args)


if __name__ == '__main__':
    raise SystemExit(main())
