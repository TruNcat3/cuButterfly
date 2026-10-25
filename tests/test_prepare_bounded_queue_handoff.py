import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_bounded_queue_handoff as audit
from semantic_campaign_manifest import MANIFEST_SCHEMA, manifest_hash


def save(path, value):
    audit.continuation.save(path, value)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    source, snapshot = tmp_path / 'old', tmp_path / 'snapshot'
    source.mkdir()
    snapshot.mkdir()
    (snapshot / 'worker.py').write_text('frozen worker\n')
    library = tmp_path / 'library.so'
    library.write_bytes(b'library')
    component = tmp_path / 'calibration-component'
    component.write_text('calibration')
    calibration = {'registry_path': str(component), 'hardware_rates_source': str(component),
                   'cost_model': str(component)}
    calibration_path = tmp_path / 'calibration.json'
    save(calibration_path, calibration)
    sources = {'worker.py': audit.runner.sha256_file(snapshot / 'worker.py')}
    monkeypatch.setattr(audit.runner, 'validate_fingerprints', lambda *args: None)
    monkeypatch.setattr(audit.runner, 'validate_calibration', lambda *args: None)
    monkeypatch.setattr(audit, 'process_alive', lambda pid: False)
    prior = {'phases': {}, 'manifests': {}, 'processed_cases': 29170,
             'remaining_cases': 4, 'missing_lowering_cases': 232}
    paths = []
    for slot in (1, 2):
        uuid = 'GPU-00000000-0000-0000-0000-' + f'{slot:012d}'
        cases = []
        for index in range(2):
            descriptor = {'semantics': {'slot': slot, 'index': index}}
            cases.append({'id': audit.runner.digest(descriptor['semantics']),
                          'descriptor': descriptor, 'populations': ['timing']})
        parent = {'schema': MANIFEST_SCHEMA, 'cases': cases, 'fingerprints': {'library': {'path': str(library)}},
                  'invalid_requests': [], 'metrics': {}}
        parent['manifest_sha256'] = manifest_hash(parent)
        parent_path = source / f'evidence/timing_gpu{slot}.json'
        save(parent_path, parent)
        prior['manifests'][parent_path.name] = audit.continuation.evidence(parent_path)
        prior['phases'][f'timing_gpu{slot}'] = {'gpu_uuid': uuid, 'total_processed_cases': 14585,
                                             'missing_lowering_cases': 116,
                                             'evidence': [audit.continuation.evidence(parent_path)]}
        plan = {'handoff': str(source / 'evidence/handoff.json'), 'library': str(library),
                'calibration_manifest': str(calibration_path), 'gpu_uuid': uuid,
                'phases': [{'name': 'timing', 'manifest': str(parent_path), 'timing': True}]}
        plan_path = source / f'gpu{slot}_plan.json'
        save(plan_path, plan)
        shard_dir = source / f'gpu{slot}/timing/shard-0000'
        shard_path = shard_dir / 'manifest.json'
        save(shard_path, parent)
        (shard_dir / 'worker.log').write_text('worker stopped\n')
        report_path = shard_dir / 'measurement/results.json'
        row = dict(cases[0], status='pass', baselines=[])
        artifact = shard_dir / 'error.npz'
        artifact.write_bytes(b'error evidence')
        row['error_artifact'] = audit.continuation.evidence(artifact)
        checkpoint = audit.runner.write_case_checkpoint(report_path.parent, row)
        identity = {'manifest_sha256': audit.runner.digest(parent),
                    'library_sha256': audit.runner.sha256_file(library), 'gpu_uuid': uuid,
                    'timing': True, 'sources': sources,
                    'protocol': dict(audit.runner.PROTOCOL, float_verdict='error-report'),
                    'compile_mode': 'research', 'calibration_sha256': audit.runner.digest(calibration),
                    'frontend_sha256': audit.runner.sha256_file(Path(audit.runner.__file__).parents[1] / 'python/cubutterfly/__init__.py')}
        for key in ('registry', 'hardware_profile', 'cost_model'):
            identity[key] = str(component)
            identity[key + '_sha256'] = audit.runner.sha256_file(component)
        report = {'schema': 'cubutterfly-semantic-results-v1', 'identity': identity,
                  'status': 'interrupted', 'rows': [checkpoint],
                  'rows_sha256': audit.runner.digest([checkpoint]),
                  'remaining_case_ids': [cases[1]['id']]}
        save(report_path, report)
        state = {'identity': {'sources': sources, 'gpu_uuid': uuid,
                             'plan': audit.continuation.evidence(plan_path)},
                 'phases': {'timing': {'shards': [{'report': str(report_path),
                            'manifest': str(shard_path), 'pid': 123}], 'completed': {}}}}
        save(source / f'gpu{slot}/queue_status.json', state)
        paths.append(report_path)
    save(source / 'evidence/handoff.json', prior)
    return source, tmp_path / 'new/evidence', snapshot, paths


def test_transition_preserves_history_and_exact_remaining_cases(campaign):
    source, output, snapshot, _ = campaign
    result = audit.prepare(source, output, snapshot)
    assert (result['processed_cases'], result['remaining_cases'], result['missing_lowering_cases']) == (29172, 2, 232)
    assert audit.bounded.verify_handoff(output / 'handoff.json') == result
    for slot in (1, 2):
        old = audit.continuation.read(source / f'evidence/timing_gpu{slot}.json')
        new = audit.continuation.read(output / f'timing_gpu{slot}.json')
        assert new['cases'] == old['cases'][1:]
        assert new['continuation_provenance']['parent_manifest_sha256'] == old['manifest_sha256']


def test_empty_interrupted_index_preserves_orphan_for_rerun(campaign):
    source, output, snapshot, paths = campaign
    for path in paths:
        report = audit.continuation.read(path)
        manifest = audit.continuation.read(path.parent.parent / 'manifest.json')
        report.update(rows=[], rows_sha256=audit.runner.digest([]),
                      remaining_case_ids=[case['id'] for case in manifest['cases']])
        save(path, report)
    result = audit.prepare(source, output, snapshot)
    assert result['processed_cases'] == 29170
    assert result['remaining_cases'] == 4
    assert all(not p['unadopted_orphans'][0]['adopted'] for p in result['phases'].values())


@pytest.mark.parametrize('status', ['running', 'stopped-investigation-required', 'runtime-error'])
def test_unknown_or_active_report_rejected(campaign, status):
    source, output, snapshot, paths = campaign
    report = audit.continuation.read(paths[0])
    report['status'] = status
    save(paths[0], report)
    with pytest.raises(ValueError, match='requires investigation'):
        audit.prepare(source, output, snapshot)
    assert not output.exists()


def test_corrupted_artifact_rejected(campaign):
    source, output, snapshot, paths = campaign
    (paths[0].parent.parent / 'error.npz').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='corrupted error artifact'):
        audit.prepare(source, output, snapshot)


def test_changed_frozen_source_rejected(campaign):
    source, output, snapshot, _ = campaign
    (snapshot / 'worker.py').write_text('changed worker\n')
    with pytest.raises(ValueError, match='source fingerprint mismatch'):
        audit.prepare(source, output, snapshot)


def test_duplicate_commit_in_separate_shard_rejected(campaign):
    source, output, snapshot, paths = campaign
    path = paths[0]
    import shutil
    duplicate = path.parent.parent.with_name('shard-0001')
    shutil.copytree(path.parent.parent, duplicate)
    state_path = source / 'gpu1/queue_status.json'
    state = audit.continuation.read(state_path)
    state['phases']['timing']['shards'].append({'manifest': str(duplicate / 'manifest.json'),
        'report': str(duplicate / 'measurement/results.json'), 'pid': 124})
    save(state_path, state)
    with pytest.raises(ValueError, match='duplicate indexed case'):
        audit.prepare(source, output, snapshot)


def test_alive_worker_rejected(campaign, monkeypatch):
    source, output, snapshot, _ = campaign
    monkeypatch.setattr(audit, 'process_alive', lambda pid: True)
    with pytest.raises(ValueError, match='still alive'):
        audit.prepare(source, output, snapshot)


def test_commit_after_compiler_failure_rejected(campaign):
    source, output, snapshot, paths = campaign
    (paths[0].parent.parent / 'worker.log').write_text(
        'could not start specialization compiler\n' + json.dumps({'status': 'pass', 'completed': 1}) + '\n')
    with pytest.raises(ValueError, match='adoption unsafe'):
        audit.prepare(source, output, snapshot)


def test_scaled_controls_retained_separately(campaign):
    source, output, snapshot, paths = campaign
    import shutil
    for slot, path in enumerate(paths, 1):
        directory = source / f'gpu{slot}/scaled_controls/shard-0000'
        shutil.copytree(path.parent.parent, directory)
        manifest = audit.continuation.read(directory / 'manifest.json')
        report_path = directory / 'measurement/results.json'
        report = audit.continuation.read(report_path)
        second = dict(manifest['cases'][1], status='pass', baselines=[])
        report['rows'].append(audit.runner.write_case_checkpoint(report_path.parent, second))
        report.update(status='complete', remaining_case_ids=[], rows_sha256=audit.runner.digest(report['rows']))
        report['identity']['timing'] = False
        save(report_path, report)
        plan_path = source / f'gpu{slot}_plan.json'
        plan = audit.continuation.read(plan_path)
        plan['phases'].insert(0, {'name': 'scaled_controls', 'timing': False,
                                  'manifest': str(directory / 'manifest.json')})
        save(plan_path, plan)
        state_path = source / f'gpu{slot}/queue_status.json'
        state = audit.continuation.read(state_path)
        state['identity']['plan'] = audit.continuation.evidence(plan_path)
        state['phases']['scaled_controls'] = {'shards': [{'pid': 125,
            'manifest': str(directory / 'manifest.json'), 'report': str(report_path)}], 'completed': {}}
        save(state_path, state)
    result = audit.prepare(source, output, snapshot)
    assert result['scaled_control_cases'] == 4
    assert result['processed_cases'] == 29172
    assert result['remaining_cases'] == 2
    assert len(result['scaled_controls']) == 2
    assert audit.bounded.verify_handoff(output / 'handoff.json') == result
