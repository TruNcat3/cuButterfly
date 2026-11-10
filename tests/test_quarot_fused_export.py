"""CPU contract checks for four-process QuaRot evidence export."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('quarot_fused_export', ROOT / 'experiments/tools/export_quarot_fused_followup.py')
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


@pytest.fixture
def cohort(tmp_path):
    config = exporter.read(ROOT / 'configs/experiments/quarot_fused_followup_20261107.json')
    for job in config['jobs']:
        directory = tmp_path / job['id']
        (directory / 'release').mkdir(parents=True)
        identity = dict(gpu_uuid='stored-gpu', library_sha256='stored-library',
            runner_sha256='stored-runner', adapter_sha256='stored-adapter', checks_sha256='stored-checks',
            extension_sha256='stored-extension', contract=config['contract'], job=job)
        numerical = {key: dict(finite=True, actual_nonfinite_count=0, reference_nonfinite_count=0,
                                    max_abs_error=100, **{'pass': False})
                     for key in ['prefill'] + ['decode_' + str(i) for i in range(8)]}
        report = dict(status='pass_timing_numerical_unassessed', full_model_run=True, gpu_name='test-GPU',
            compute_contract='public_fp32_final', execution_mode='graph', gpu_uuid='stored-gpu',
            library_sha256=identity['library_sha256'], runner_sha256=identity['runner_sha256'],
            adapter_sha256=identity['adapter_sha256'], checks_sha256=identity['checks_sha256'],
            quarot_extension_sha256=identity['extension_sha256'],
            payload=dict(batch_size=job['batch_size'], prefill=128, decode=8), variant_order=job['variant_order'],
            seed=config['contract']['seed'], validation_warmup_sequences=4, attention_backend='torch_sdpa',
            timing_calls=7, timing_warmup_sequences=2, numerical_checks=numerical,
            graph_changed_token_validation={backend: {phase: numerical for phase in
                ('original', 'changed', 'restored_repeat', 'output_lifetime')}
                for backend in ('baseline', 'cubutterfly')},
            timing_contract=dict(graph_captured_full_prefill_decode=True, input_copy_and_output_clone_in_timing=True,
                finite_output_gate_before_timing=True, summary_statistic='median',
                summary_clock='host_wall_with_device_synchronization', precision_contract='sameFP16storageFP32compute'),
            baseline=dict(wall_ms=dict(end_to_end=[1, 2, 3, 4, 5, 6, 100]), cuda_event_ms=dict(end_to_end=[3]*7)),
            cubutterfly=dict(wall_ms=dict(end_to_end=[2]*7), cuda_event_ms=dict(end_to_end=[1]*7)),
            summary=dict(end_to_end=dict(baseline_median_ms=4, cubutterfly_median_ms=2, baseline_over_cubutterfly=2)),
            adapter_setup=dict(compute_contract='public_fp32_final', reuse_buffers=True, plans=[dict(algorithm='fused')]))
        for path, data in [('identity.json', identity), ('status.json', dict(state='complete', gpu_uuid='stored-gpu')),
                           ('release/report.json', report)]:
            (directory / path).write_text(json.dumps(data))
    return tmp_path


def test_incomplete_cohort_writes_no_formal_artifacts(cohort):
    (cohort / 'b4_treatment_first/release/report.json').unlink()
    output = cohort / 'quarot_fused_e2e.json'
    with pytest.raises(ValueError, match='Incomplete four-process'):
        exporter.main(['--source', str(cohort), '--output', str(output)])
    assert not output.exists()
    assert not (cohort / 'quarot_fused_rows.tex').exists()
    assert not (cohort / 'quarot_fused_numbers.tex').exists()


def test_stored_cross_process_identity_mismatch_rejected(cohort):
    directory = cohort / 'b4_treatment_first'
    for file in ('identity.json', 'release/report.json'):
        path = directory / file
        value = json.loads(path.read_text())
        value['library_sha256'] = 'different-stored-library'
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='Cross-process stored identity mismatch'):
        exporter.export(cohort)


def test_recompute_wall_event_medians_and_reject_stale_summary(cohort):
    data = exporter.export(cohort)
    assert len(data['records']) == 4
    assert data['aggregate_ranges'][0]['wall_ratio_range'] == [2, 2]
    assert data['aggregate_ranges'][0]['event_ratio_range'] == [3, 3]
    assert data['records'][0]['samples']['wall_ms']['baseline'][-1] == 100
    assert data['records'][0]['numerical_errors']['numerical_checks']['prefill']['pass'] is False
    assert len(exporter.tex_rows(data).splitlines()) == 2
    assert exporter.number_macros(data) == ('\\newcommand{\\CurrentQuaRotFusedBOne}{2.000}\n'
                                            '\\newcommand{\\CurrentQuaRotFusedBFour}{2.000}\n')
    path = cohort / 'b1_baseline_first/release/report.json'
    value = json.loads(path.read_text())
    value['summary']['end_to_end']['baseline_median_ms'] = 5
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='summary differs'):
        exporter.export(cohort)
