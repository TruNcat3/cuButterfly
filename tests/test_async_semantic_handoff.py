"""Synthetic immutable reports exercise CPU-only historical adoption."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_async_semantic_handoff as handoff
import run_semantic_campaign as runner
from semantic_campaign_manifest import MANIFEST_SCHEMA, manifest_hash, validate_manifest_hash


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'historical'
    library = tmp_path / 'lib.so'
    library.write_bytes(b'synthetic library')
    cases = []
    for operator in (0, 0, 1, 2):
        semantics = {'operator': operator, 'marker': len(cases)}
        cases.append({'id': runner.digest(semantics), 'descriptor': {'semantics': semantics},
                      'populations': ['synthetic'], 'requested_variants': []})
    parent = {'schema': MANIFEST_SCHEMA, 'cases': cases, 'invalid_requests': [], 'metrics': {},
              'fingerprints': {'library': {'path': str(library), 'sha256': runner.sha256_file(library)}, 'sources': []}}
    parent['manifest_sha256'] = manifest_hash(parent)
    for spec in handoff.PHASES:
        save(root / spec[3], parent)
    uuids = {1: 'GPU-11111111-1111-1111-1111-111111111111',
             2: 'GPU-22222222-2222-2222-2222-222222222222'}
    for slot in (1, 2):
        directory = root / f'gpu{slot}/calibration'
        profile = directory / 'profile.json'
        registry, model = directory / 'registry.json', directory / 'model.json'
        save(profile, {'provenance': {'cuda_visible_devices': uuids[slot]}})
        save(registry, {})
        save(model, {})
        calibration = {'schema': 'cubutterfly-local-calibration-v1', 'status': 'ready-with-measured-mappings',
            'measurement_identity': {'library_sha256': runner.sha256_file(library), 'compile_mode': 'research', 'verify_batches': 0},
            'device': 'synthetic GPU', 'global_memory_bytes': 100, 'compute_capability': '8.0',
            'hardware_rates_remeasured': True, 'hardware_rates_source': str(profile),
            'registry_path': str(registry), 'cost_model': str(model)}
        save(directory / 'calibration_manifest.json', calibration)
    return root, parent, uuids


def historical_report(source, spec, statuses, baseline_error=False):
    root, parent, uuids = source
    name, slot, timing, relative_parent, relative_history = spec
    directory = root / f'gpu{slot}' / relative_history / 'shard-0001'
    manifest = copy.deepcopy(parent)
    save(directory / 'manifest.json', manifest)
    calibration_path = root / f'gpu{slot}/calibration/calibration_manifest.json'
    calibration = json.loads(calibration_path.read_text())
    identity = {'manifest_sha256': runner.digest(manifest),
        'library_sha256': parent['fingerprints']['library']['sha256'], 'gpu_uuid': uuids[slot],
        'timing': timing, 'population': None, 'protocol': dict(runner.PROTOCOL, float_verdict='error-report'),
        'runner_sha256': runner.sha256_file(runner.__file__), 'calibration_sha256': runner.digest(calibration),
        'compile_mode': 'research', 'gpu_name': 'synthetic GPU', 'gpu_total_memory': 100, 'compute_capability': '8.0'}
    for path_key, hash_key, calibration_key in (
            ('registry', 'registry_sha256', 'registry_path'),
            ('hardware_profile', 'hardware_profile_sha256', 'hardware_rates_source'),
            ('cost_model', 'cost_model_sha256', 'cost_model')):
        identity[path_key] = calibration[calibration_key]
        identity[hash_key] = runner.sha256_file(identity[path_key])
    output = directory / 'measurement'
    output.mkdir()
    indices = []
    for case, status in zip(parent['cases'], statuses):
        row = {'id': case['id'], 'descriptor': case['descriptor'], 'status': status,
               'baselines': [{'status': 'baseline-error'}] if baseline_error else []}
        indices.append(runner.write_case_checkpoint(output, row))
    report = {'schema': 'cubutterfly-semantic-results-v1', 'identity': identity,
        'status': 'stopped-correctness-failure' if 'correctness-failure' in statuses else 'running',
        'rows': indices, 'remaining_case_ids': [case['id'] for case in parent['cases'][len(statuses):]]}
    path = output / 'results.json'
    runner.write_report(path, report)
    return path


def test_adopts_partial_rows_and_quarantines_failure_without_claiming_pass(source):
    path = historical_report(source, handoff.PHASES[0], ['accuracy-review-required', 'correctness-failure'])
    before = runner.sha256_file(path)
    manifest, metadata = handoff.prepare_phase(source[0], handoff.PHASES[0])
    assert validate_manifest_hash(manifest)
    assert manifest['cases'] == source[1]['cases'][2:]
    assert metadata['processed_cases'] == 1 and metadata['quarantined_cases'] == 1
    assert metadata['adopted_status_counts'] == {'accuracy-review-required': 1, 'correctness-failure': 1}
    assert metadata['quarantine'][0]['status'] == 'correctness-failure'
    assert runner.sha256_file(path) == before
    assert str(path.resolve()) in {item['path'] for item in metadata['evidence']}


@pytest.mark.parametrize('status,baseline_error', [('runner-error', False), ('pass', True)])
def test_runtime_and_baseline_errors_abort_adoption(source, status, baseline_error):
    historical_report(source, handoff.PHASES[0], [status], baseline_error)
    with pytest.raises(ValueError, match='runtime error|runtime/runner error'):
        handoff.prepare_phase(source[0], handoff.PHASES[0])


@pytest.mark.parametrize('field,value', [('gpu_uuid', 'GPU-33333333-3333-3333-3333-333333333333'),
    ('timing', True), ('calibration_sha256', 'wrong'), ('protocol', {'float_verdict': 'legacy-fixed'})])
def test_report_slot_phase_calibration_and_protocol_are_bound(source, field, value):
    path = historical_report(source, handoff.PHASES[0], ['pass'])
    report = json.loads(path.read_text())
    report['identity'][field] = value
    save(path, report)
    with pytest.raises(ValueError):
        handoff.prepare_phase(source[0], handoff.PHASES[0])


def test_full_case_equality_rejects_changed_population_even_when_semantics_equal(source):
    path = historical_report(source, handoff.PHASES[0], ['pass'])
    manifest_path = path.parent.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['cases'][0]['populations'] = ['changed']
    manifest['manifest_sha256'] = manifest_hash(manifest)
    save(manifest_path, manifest)
    with pytest.raises(ValueError, match='case changed'):
        handoff.prepare_phase(source[0], handoff.PHASES[0])


def test_report_stop_reason_cannot_be_adopted_as_a_valid_partial_run(source):
    path = historical_report(source, handoff.PHASES[0], ['pass'])
    report = json.loads(path.read_text())
    report.update(status='stopped', stop_reason='GPU exclusivity violated')
    save(path, report)
    with pytest.raises(ValueError, match='stop requires investigation'):
        handoff.prepare_phase(source[0], handoff.PHASES[0])


def test_pilot_only_does_not_read_moving_historical_reports(source, tmp_path):
    root, parent, uuids = source
    moving = root / 'gpu2/full_timing/validated-timing/shard-0001/measurement/results.json'
    moving.parent.mkdir(parents=True)
    moving.write_text('incomplete moving report')
    output = tmp_path / 'pilot'
    assert handoff.main(['--source-root', str(root), '--output', str(output), '--pilot-cases', '3', '--pilot-only']) == 0
    pilot = json.loads((output / 'pilot_timing_gpu2.json').read_text())
    assert {case['descriptor']['semantics']['operator'] for case in pilot['cases']} == {0, 1, 2}
    assert validate_manifest_hash(pilot)
    assert not (output / 'timing_gpu2.json').exists()
    evidence = json.loads((output / 'handoff.json').read_text())
    assert evidence['mode'] == 'pilot-only-parent'
    assert evidence['phases']['timing_gpu2']['historical_reports'] == []
    assert evidence['pilot_excludes_from_remaining'] is False


def test_full_cli_writes_three_queues_and_pilot_without_excluding_pilot(source, tmp_path):
    root, parent, uuids = source
    historical_report(source, handoff.PHASES[2], ['pass'])
    output = tmp_path / 'handoff'
    handoff.main(['--source-root', str(root), '--output', str(output), '--pilot-cases', '2'])
    for name in ('recovery_gpu1.json', 'timing_gpu1.json', 'timing_gpu2.json', 'pilot_timing_gpu2.json', 'handoff.json'):
        assert (output / name).exists()
    remaining = json.loads((output / 'timing_gpu2.json').read_text())['cases']
    pilot = json.loads((output / 'pilot_timing_gpu2.json').read_text())['cases']
    assert remaining == parent['cases'][1:]
    assert all(case in remaining for case in pilot)
    evidence = json.loads((output / 'handoff.json').read_text())
    for metadata in evidence['phases'].values():
        assert metadata['evidence_stat_signatures'] == {
            item['path']: list(handoff.continuation.file_signature(item['path']))
            for item in metadata['evidence']}
    with pytest.raises(ValueError, match='new directory'):
        handoff.main(['--source-root', str(root), '--output', str(output)])


def test_evidence_stat_capture_rejects_mutation_during_final_verification(tmp_path, monkeypatch):
    path = tmp_path / 'evidence.json'
    path.write_text('initial evidence')
    metadata = {'evidence': [handoff.continuation.evidence(path)]}

    def mutate(item):
        path.write_text('changed evidence with a different file size')

    monkeypatch.setattr(handoff.continuation, 'verify_evidence', mutate)
    with pytest.raises(ValueError, match='changed between verification and stat capture'):
        handoff.freeze_evidence_stat_signatures(metadata)
    assert 'evidence_stat_signatures' not in metadata
