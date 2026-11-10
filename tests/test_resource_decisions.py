import importlib.util
import json
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('resource_decisions', ROOT / 'experiments/tools/export_resource_decisions.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_dependency_regions_are_not_ctas_or_compiler_smem():
    d = module.dependency_demands(20, 16, 16, [{'first_stage': 0, 'stage_count': 10},
                                             {'first_stage': 10, 'stage_count': 10}])
    assert [r['closed_regions'] for r in d['regions']] == [16384, 16384]
    assert d['regions'][0]['one_region_value_bytes'] == 16384
    assert d['group_interface_logical_bytes'] == 1073741824
    with pytest.raises(ValueError, match='cover'):
        module.dependency_demands(20, 16, 16, [{'first_stage': 0, 'stage_count': 8}])


def test_frozen_chain_recomputes_samples_and_preserves_missing_ntt():
    frozen = json.loads((ROOT / 'experiments/snapshots/resource_decisions_20261107.json').read_text())
    missing = [entry['path'] for entry in frozen['sources'] if not (ROOT / entry['path']).is_file()]
    if missing:
        pytest.skip('Raw-evidence recomputation unavailable: %s; frozen-artifact checks remain separate; this skip is not measurement completion' % ', '.join(missing[:3]))
    data = module.export()
    assert data['cohort']['arithmetic_families'] == ['fft', 'fwht']
    assert data['cohort']['independent_group_holdouts'] == 68
    assert data['cohort']['complete_plan_holdouts'] == 15
    assert data['cohort']['winner_matches'] == 3
    assert [r['confirmation']['median_ms'] for r in data['rows']] == [1.072701, 0.044872, 0.018391]
    assert all(r['prediction_temporality'].startswith('Final refit validation is retrospective') for r in data['rows'])
    assert data['extensions'][0]['status'] == 'missing-frozen-resource-selection-prediction-chain'
    for row in data['rows']:
        assert all(len(g['matching_curves']) == 1 for g in row['independent_service_curves'])
        assert row['prediction_parameter_audit']['unchanged_without_plan_fitted_parameters']
        assert row['prediction_parameter_audit']['whole_plan_fallback_used'] is False
    assert 'Retro.' in module.render(data)
    extension = data['prospective_measured_selection']
    assert extension['status'] == 'complete'
    assert [r['median_paired_ratio'] for r in extension['rows']] == [1.3176457765667575, 1.3651256579097268, 1.0, 1.0]
    assert all(r['cost_prediction_ms'] is None for r in extension['rows'])
    assert all(r['same_winner'] and r['choices']['extended']['id'] == 't256-o1' for r in extension['rows'][2:])
    assert len(extension['chronology']['guard_completion_receipts']) == 4
    assert 'same' in module.render_prospective(data)
    assert all(r['choices']['conventional']['mapping'] == r['choices']['extended']['mapping'] for r in extension['rows'][2:])
    assert (ROOT / 'experiments/snapshots/resource_measured_selection_20261107.txt').read_text() == module.render_prospective(data)
    assert json.loads((ROOT / 'experiments/snapshots/resource_decisions_20261107.json').read_text()) == data


def test_portable_frozen_artifact_numbers_semantics_and_tables():
    # No raw results, production model imports, CUDA or remote filesystem needed.
    data = json.loads((ROOT / 'experiments/snapshots/resource_decisions_20261107.json').read_text())
    assert data['schema'] == 'resource-decisions-audit-v1'
    cohort = data['cohort']
    assert (cohort['certified_keys'], cohort['clean_probes'], cohort['independent_group_holdouts'],
            cohort['complete_plan_holdouts'], cohort['winner_matches']) == (34, 117, 68, 15, 3)
    assert [r['request']['operator'] for r in data['rows']] == ['fft', 'fwht', 'fwht']
    assert [(r['shape'], r['request']['batch']) for r in data['rows']] == [([1048576], 16), ([65536], 16), ([32768], 32)]
    for row in data['rows']:
        confirmation = row['confirmation']
        assert len(confirmation['samples_ms']) == confirmation['trials'] == 3
        assert statistics.median(confirmation['samples_ms']) == confirmation['median_ms']
        predicted = row['retrospective_model_validation']
        assert predicted['observed_kernel_ms'] == confirmation['median_ms']
        assert module.relative_error(predicted['predicted_kernel_ms'], confirmation['median_ms']) == pytest.approx(predicted['relative_error'])
        assert row['prediction_parameter_audit']['whole_plan_fallback_used'] is False
        assert row['prediction_parameter_audit']['prediction_with_empty_fallback_parameters_ms'] == predicted['predicted_kernel_ms']
        assert 'retrospective' in row['prediction_temporality']
        w = row['request']
        expected = module.dependency_demands(w['logN'], w['batch'], row['dependency_derived']['value_bytes'], row['compiled_and_launch_resources'])
        assert expected == row['dependency_derived']
    prospective = data['prospective_measured_selection']
    assert prospective['status'] == 'complete'
    assert 'no numeric cost-model forecast' in prospective['interpretation']
    assert [r['median_paired_ratio'] for r in prospective['rows']] == pytest.approx([1.3176457765667575, 1.3651256579097268, 1, 1])
    for row in prospective['rows']:
        pairs = row['holdout_pairs']
        assert len(pairs) == 3 and row['cost_prediction_ms'] is None
        for pair in pairs:
            assert pair['ratio'] == pair['conventional_ms'] / pair['extended_ms']
        assert statistics.median(p['ratio'] for p in pairs) == row['median_paired_ratio']
        if row['workload']['operator'] == 'ntt':
            assert row['same_winner'] and row['choices']['extended']['id'] == 't256-o1'
    assert (ROOT / 'experiments/snapshots/resource_decisions_20261107.txt').read_text() == module.render(data)
    assert (ROOT / 'experiments/snapshots/resource_measured_selection_20261107.txt').read_text() == module.render_prospective(data)
    assert 'FFT64' in module.render(data) and 'SMEM' in module.render(data)
