#!/usr/bin/env python3
"""CPU audit and explicitly applied recovery of the stopped cyclic GPU-NTT followup."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

import run_async_semantic_campaign as asynchronous
import run_semantic_campaign as semantic
import semantic_campaign_manifest as manifests
import semantic_gpuntt_baseline as gpuntt

BASELINE = 'semantic_gpuntt_baseline.py'
BACKEND_SOURCES = {BASELINE, 'semantic_bridge.cu'}


def read(path):
    return json.loads(Path(path).read_text())


def affected(row):
    s = row['descriptor']['semantics']
    return s['word_bits'] == 64 and s['modulus'] >= 2**61


def exact(row):
    return all(row.get(key, {}).get('exact') is True and
               row[key].get('mismatches') == 0
               for key in ('correctness', 'mapping_replay_correctness'))


def backend(row):
    matches = [b for b in row.get('baselines', []) if b['name'] == 'gpu-ntt']
    if len(matches) != 1:
        raise ValueError('expected exactly one GPU-NTT baseline')
    return matches[0]


def audit_backend(previous, current, library=None):
    """Allow only the diagnosed support guard and its rebuilt bridge."""
    if not current.get('available'):
        raise ValueError('current GPU-NTT inventory unavailable')
    current = {k: v for k, v in current.items() if k != 'available'}
    old, new = copy.deepcopy(previous), copy.deepcopy(current)
    changes = {}
    for label in ('binary',):
        if (old[label]['path'] != new[label]['path'] and
            (library is None or new[label]['path'] != str(Path(library).resolve()))):
            raise ValueError('backend binary path changed')
        changes[label] = dict(before=old.pop(label), after=new.pop(label))
    old_sources = {item['path']: item for item in old.pop('sources')}
    new_sources = {item['path']: item for item in new.pop('sources')}
    if old_sources.keys() != new_sources.keys():
        raise ValueError('backend source population changed')
    for path, item in old_sources.items():
        if item != new_sources[path]:
            if Path(path).name not in BACKEND_SOURCES:
                raise ValueError('unapproved backend source change: ' + path)
            changes[path] = dict(before=item, after=new_sources[path])
    if old != new:
        raise ValueError('vendor/root helper/backend identity changed')
    return changes


def plan(report_path, manifest_path, limits_path, library, gpu_uuid, *, sources=None, inventory=None,
         support_check=None, gpuntt_library=None):
    report_path, manifest_path, limits_path, library = map(Path,
        (report_path, manifest_path, limits_path, library))
    report, manifest, limits = map(read, (report_path, manifest_path, limits_path))
    identity = report['identity']
    if not manifests.validate_manifest_hash(manifest):
        raise ValueError('manifest checksum mismatch')
    semantic.validate_fingerprints(manifest, library)
    if (identity['manifest_sha256'] != semantic.digest(manifest) or
        identity['library_sha256'] != semantic.sha256_file(library) or
        identity['gpu_uuid'] != gpu_uuid or identity.get('compile_mode') != 'research' or
        identity.get('timing') is not True or identity.get('runner_revision') != asynchronous.REVISION):
        raise ValueError('manifest/library/GPU/build identity mismatch')
    expected_protocol = dict(semantic.PROTOCOL, float_verdict='error-report',
        reference_max_elements=limits['reference_max_elements'],
        ntt_max_operations=limits['ntt_max_operations'])
    if identity['protocol'] != expected_protocol:
        raise ValueError('protocol differs')
    for field in ('registry', 'hardware_profile', 'cost_model'):
        if field in identity and semantic.sha256_file(Path(identity[field])) != identity[field + '_sha256']:
            raise ValueError('calibration identity changed: ' + field)
    if report['status'] != 'stopped-investigation-required' or report.get('invalid_request_count') != 0:
        raise ValueError('expected stopped valid cyclic campaign')
    cases = manifest['cases']
    semantic.validate_resume(report, identity, report_path.parent, {c['id'] for c in cases})
    indices = report['rows']
    if (len(indices) < 2 or [i['id'] for i in indices] != [c['id'] for c in cases[:len(indices)]] or
        report['remaining_case_ids'] != [c['id'] for c in cases[len(indices):]]):
        raise ValueError('row execution order/remaining index differs')
    rows = [read(report_path.parent / i['path'])['row'] for i in indices]
    for row, case in zip(rows, cases):
        if row['descriptor'] != case['descriptor'] or row['id'] != semantic.digest(case['descriptor']['semantics']):
            raise ValueError('row descriptor mismatch')
    failed = rows[-1]
    if (failed['status'] != 'correctness-failure' or not exact(failed) or not affected(failed) or
        backend(failed)['status'] != 'correctness-failure' or
        any(b['status'] not in ('pass', 'not-applicable', 'missing-baseline')
            for b in failed['baselines'] if b['name'] != 'gpu-ntt')):
        raise ValueError('failure is not exclusively corrected GPU-NTT baseline scope')
    if any(row['status'] != 'pass' or not exact(row) or affected(row) or
           any(semantic.status_bucket(b['status']) == 'error' for b in row['baselines'])
           for row in rows[:-1]):
        raise ValueError('retained row is not valid outside corrected baseline scope')
    if support_check is None:
        support_check = lambda s: gpuntt.support(s, gpuntt_library or gpuntt.DEFAULT_LIBRARY)
    for row in rows[:-1]:
        if backend(row)['status'] == 'pass' and not support_check(row['descriptor']['semantics'])['supported']:
            raise ValueError('retained baseline no longer satisfies corrected CPU guard')
    failed_support = support_check(failed['descriptor']['semantics'])
    if failed_support['supported'] or failed_support['status'] != 'unsupported-semantic':
        raise ValueError('failed request is not rejected by corrected CPU guard')
    sources = asynchronous.source_identity() if sources is None else sources
    if sources.keys() != identity['sources'].keys():
        raise ValueError('source population changed')
    changed = {k: dict(before=identity['sources'][k], after=sources[k])
               for k in sources if sources[k] != identity['sources'][k]}
    if set(changed) != {BASELINE}:
        raise ValueError('source changes differ from exact baseline allowlist')
    inventory = gpuntt.backend_inventory(gpuntt_library or gpuntt.DEFAULT_LIBRARY) if inventory is None else inventory
    previous = backend(failed)['support_metadata']['backend_identity']
    backend_changes = audit_backend(previous, inventory, gpuntt_library)
    for row in rows[:-1]:
        prior = backend(row).get('support_metadata', {}).get('backend_identity')
        if prior is not None and prior != previous:
            raise ValueError('retained baseline backend identity differs from original campaign')
    files = {str(report_path.resolve()): semantic.sha256_file(report_path)}
    for index, row in zip(indices, rows):
        path = (report_path.parent / index['path']).resolve()
        files[str(path)] = semantic.sha256_file(path)
        for implementation in [row, *row['baselines']]:
            artifact = implementation.get('error_artifact')
            if artifact:
                files[str(Path(artifact['path']).resolve())] = artifact['sha256']
    receipt = dict(schema='cubutterfly-cyclic-recovery-v1', status='audited-not-applied',
        original_identity=identity, accepted_source_updates=changed,
        original_backend_inventory=previous, current_backend_inventory=inventory,
        accepted_backend_updates=backend_changes, archived_files=files,
        retained_rows=[dict(index=i, original_measurement_identity=identity,
            original_backend_identity=backend(row).get('support_metadata', {}).get('backend_identity'))
            for i, row in zip(indices[:-1], rows[:-1])],
        remeasure_case_id=failed['id'])
    input_paths = [manifest_path, limits_path, library]
    input_paths.extend(Path(identity[field]) for field in ('registry', 'hardware_profile', 'cost_model')
                       if field in identity)
    receipt['verified_inputs'] = {str(path.resolve()): semantic.sha256_file(path) for path in input_paths}
    recovered = copy.deepcopy(report)
    recovered['identity']['sources'] = sources
    if gpuntt_library is not None:
        if identity.get('gpuntt_sha256') is not None:
            raise ValueError('explicit recovery requires original default baseline binding')
        actual_hash = semantic.sha256_file(Path(gpuntt_library))
        if inventory['binary']['sha256'] != actual_hash:
            raise ValueError('explicit baseline library inventory hash differs')
        recovered['identity']['gpuntt_sha256'] = actual_hash
    receipt['accepted_identity_updates'] = {
        key: dict(before=identity.get(key), after=value)
        for key, value in recovered['identity'].items() if value != identity.get(key)}
    recovered['rows'] = recovered['rows'][:-1]
    recovered['remaining_case_ids'] = [failed['id'], *report['remaining_case_ids']]
    recovered['status'] = 'running'
    recovered.pop('stop_reason', None)
    recovered['recovery'] = receipt
    recovered['rows_sha256'] = semantic.digest(recovered['rows'])
    recovered['status_summary'] = semantic.status_summary(recovered['rows'])
    semantic.validate_resume(recovered, recovered['identity'], report_path.parent, {c['id'] for c in cases})
    return recovered, receipt


def apply(report_path, recovered, receipt, archive, *, sources=None, inventory=None):
    report_path, archive = Path(report_path), Path(archive)
    if archive.exists():
        raise ValueError('archive must be a fresh directory')
    sources = asynchronous.source_identity() if sources is None else sources
    if inventory is None:
        gpuntt._identity.cache_clear()
        inventory = gpuntt.backend_inventory(receipt['current_backend_inventory']['binary']['path'])
    if sources != recovered['identity']['sources'] or inventory != receipt['current_backend_inventory']:
        raise ValueError('current source/backend identity changed after audit')
    for path, checksum in {**receipt['archived_files'], **receipt['verified_inputs']}.items():
        if semantic.sha256_file(Path(path)) != checksum:
            raise ValueError('evidence changed after audit: ' + path)
    archive.mkdir(parents=True)
    archived = []
    for number, (path, checksum) in enumerate(receipt['archived_files'].items()):
        target = archive / (str(number) + '-' + Path(path).name)
        shutil.copy2(path, target)
        if semantic.sha256_file(target) != checksum:
            raise ValueError('archive copy checksum mismatch')
        archived.append(dict(original=path, archive=str(target.resolve()), sha256=checksum))
    receipt.update(status='applied', archive=archived)
    (archive / 'recovery_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    semantic.write_report(report_path, recovered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'manifest', 'limits', 'library', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--gpuntt-library', type=Path,
                        help='explicit corrected bridge; forward the same path on subsequent resume')
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error('receipt must be a fresh path')
    recovered, receipt = plan(args.report, args.manifest, args.limits, args.library, args.gpu_uuid,
                              gpuntt_library=args.gpuntt_library)
    if args.apply:
        if args.archive is None:
            parser.error('--apply requires --archive')
        apply(args.report, recovered, receipt, args.archive)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status=receipt['status'], retained=len(recovered['rows']),
        remaining=len(recovered['remaining_case_ids']), source_updates=receipt['accepted_source_updates'],
        backend_updates=receipt['accepted_backend_updates']), indent=2))


if __name__ == '__main__':
    main()
