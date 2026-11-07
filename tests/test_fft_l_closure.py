"""Finite FFT-L runner contracts, with no CUDA initialization."""
import csv
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import run_fft_l_closure as run
from semantic_campaign_manifest import manifest_hash


def semantics(storage=4, compute=0, shape=None, batch=1, inverse=0):
    shape = shape or [4096]
    strides = [int(np.prod(shape[i+1:])) for i in range(len(shape))]
    return dict(operator=0, storage=storage, compute=compute, shape=shape, batch=batch,
        inverse=inverse, normalize_inverse=bool(inverse), placement=0, fft_type=0,
        length_mode=0, fft_output_scale=1., input_orders=[0]*len(shape),
        output_orders=[0]*len(shape), input_strides=strides, output_strides=strides,
        input_batch_stride=int(np.prod(shape)), output_batch_stride=int(np.prod(shape)),
        matrices=[[] for _ in shape], ntt_input_layout=0, ntt_output_layout=0)


@pytest.fixture
def prepared(tmp_path):
    cases, rows = [], []
    for storage, compute, shape, batch, inverse in itertools.product(
            [4, 5], [0, 1], [[4096], [64, 64]], [1, 4], [0, 1]):
        s = semantics(storage, compute, shape, batch, inverse)
        case_id = run.semantic.digest(s)
        cases.append(dict(id=case_id, descriptor=dict(semantics=s)))
        source = np.zeros((batch, *shape), dtype=np.complex64)
        artifact = tmp_path / (case_id + '.npz')
        np.savez_compressed(artifact, source=source, storage=run.semantic.encode_storage(source, storage),
                            reference=source.astype(np.complex128))
        rows.append(dict(semantic_id=case_id, semantics=s, input_profile=dict(profile='raw-stress'),
                         artifact=dict(path=artifact.name, sha256=run.semantic.sha256_file(artifact))))
    manifest = dict(schema='cubutterfly-semantic-campaign-v1', cases=cases)
    manifest['manifest_sha256'] = manifest_hash(manifest)
    (tmp_path / 'raw_manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'preparation.json').write_text(json.dumps(dict(
        manifest_sha256=manifest['manifest_sha256'], rows=rows)))
    return tmp_path


def test_source_bound_full_population_has_16_vendor_native_matches(prepared):
    manifest, arrays = run.load_prepared(prepared)
    assert len(manifest['cases']) == len(arrays) == 32
    contracts = [run.vendor_contract(case['descriptor']['semantics']) for case in manifest['cases']]
    assert sum(contract['supported'] for contract in contracts) == 16
    assert all(contract['strict_arithmetic_speedup'] is None for contract in contracts)
    assert all(not contract['strict_arithmetic_match'] for contract in contracts)


@pytest.mark.parametrize('storage', [4, 5])
def test_inverse_adapter_preserves_low_storage_contract(storage):
    value = run.vendor_contract(semantics(storage=storage, inverse=1, shape=[64, 64]))
    assert value['supported'] and value['complete_normalization'] == 'inverse'
    assert value['normalization_adapter_scale'] == 1/4096
    assert value['input_type'] == value['output_type'] == ('CUDA_C_16F' if storage==4 else 'CUDA_C_16BF')


def test_explicit_fp32_arithmetic_is_never_substituted():
    s = semantics(compute=1)
    row = run.run_vendor(dict(id='test'), dict(s=s), None, None, None, None)
    assert row['status'] == 'unsupported-semantic'
    assert 'FP32' in row['reason']


def test_prepared_artifact_corruption_and_population_changes_are_rejected(prepared):
    manifest = json.loads((prepared / 'raw_manifest.json').read_text())
    artifact = prepared / (manifest['cases'][0]['id'] + '.npz')
    artifact.write_bytes(artifact.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='artifact identity'):
        run.load_prepared(prepared)
    manifest['cases'] = manifest['cases'][:-1]
    manifest['manifest_sha256'] = manifest_hash(manifest)
    (prepared / 'raw_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='preparation manifest'):
        run.load_prepared(prepared)


def test_unchanged_cpu_snapshot_protocol_handles_single_and_batched(monkeypatch):
    monkeypatch.setattr(run.cpu, 'validate_snapshot', lambda value: dict(ok=True, value=value))
    def steps():
        first = yield 'first'
        assert first == dict(ok=True, value='first')
        second = yield dict(batch_snapshots=['a', 'b'])
        assert second == dict(batch_results=[dict(ok=True, value='a'), dict(ok=True, value='b')])
        return 'done'
    assert run.drive_steps(steps()) == 'done'


def test_strict_arithmetic_speedup_remains_empty_in_storage_level_table(tmp_path):
    row = dict(id='request', descriptor=dict(semantics=semantics()), status='accuracy-review-required',
        timing=dict(median_ms=2), low_storage_baseline=dict(status='accuracy-review-required', timing=dict(median_ms=3)))
    run.export_table(dict(rows=[row]), tmp_path)
    with (tmp_path / 'comparison.csv').open() as f:
        result = next(csv.DictReader(f))
    assert float(result['storage_semantic_speedup']) == 1.5
    assert result['strict_arithmetic_speedup'] == ''
    assert result['same_precision_cpu_oracle'] == 'available-declared-arithmetic'


def test_reference_metadata_declares_cpu_algorithm_without_acceptance():
    contract = run.reference_contract()
    assert contract['same_precision_cpu_oracle'] == 'available-declared-arithmetic'
    assert contract['acceptance_threshold'] is None
    assert 'diagnostic' in contract['secondary_reference']


def test_cpu_launch_preparation_freezes_the_main_study_search_budget(prepared, tmp_path, monkeypatch):
    monkeypatch.setattr(run.semantic, 'validate_fingerprints', lambda *args: None)
    calibration = tmp_path / 'calibration.json'
    calibration.write_text(json.dumps(dict(measurement_identity=dict(jit_runtime=dict(ld_library_path='recorded-loader')))))
    output = tmp_path / 'measurement'
    assert run.main(['--prepared', str(prepared), '--output', str(output),
                     '--calibration-manifest', str(calibration)]) == 0
    launch = json.loads((output / 'launch.json').read_text())
    assert launch['environment']['CUBUTTERFLY_SEARCH_BUDGET'] == '512'
    assert not launch['gpu_execution_started']


def terminal_row(own_status='accuracy-review-required', vendor_status='unsupported-semantic'):
    return dict(id='request', status=own_status, timing=dict(median_ms=1),
                low_storage_baseline=dict(status=vendor_status))


def test_final_status_distinguishes_expected_gaps_from_errors():
    report = dict(rows=[terminal_row()])
    assert run.finalize_report(report, ['request']) == 0
    assert report['status'] == 'complete-with-reported-gaps'
    assert report['summary']['vendor_support_gaps'] == 1
    assert report['summary']['execution_error_count'] == 0


@pytest.mark.parametrize('side,status', [('cubutterfly', 'correctness-failure'),
    ('cubutterfly', 'runtime-error'), ('vendor', 'baseline-error'), ('vendor', 'correctness-failure')])
def test_final_status_never_hides_execution_or_numerical_errors(side, status):
    row = terminal_row()
    if side == 'cubutterfly':
        row['status'] = status
    else:
        row['low_storage_baseline']['status'] = status
    report = dict(rows=[row])
    assert run.finalize_report(report, ['request']) == 2
    assert report['status'] == 'complete-with-errors'
    assert report['summary']['execution_error_count'] == 1
    assert report['errors'][0]['status'] == status


def test_pending_requests_do_not_hide_errors_or_falsely_complete():
    report = dict(rows=[])
    assert run.finalize_report(report, ['request']) == 75
    assert report['status'] == 'cpu-compilation-pending'
    report['rows'] = [terminal_row(own_status='runner-error')]
    assert run.finalize_report(report, ['request', 'pending']) == 2
    assert report['status'] == 'incomplete-with-errors'
    assert report['remaining_case_ids'] == ['pending']


def test_success_without_requested_timing_is_an_error():
    row = terminal_row()
    del row['timing']
    report = dict(rows=[row])
    assert run.finalize_report(report, ['request']) == 2
    assert report['summary']['execution_error_count'] == 1


def test_import_has_no_torch_or_gpu_initialization():
    subprocess.run([sys.executable, '-c',
        'import sys; import benchmark.run_fft_l_closure; assert "torch" not in sys.modules'], cwd=ROOT, check=True)

def test_cpu_finalization_sidecar_preserves_original_and_records_provenance(prepared, tmp_path):
    _, arrays = run.load_prepared(prepared)
    item = next(p for p in arrays.values() if p['s']['compute'] == 0 and len(p['s']['shape']) == 1)
    saved = tmp_path / 'saved.npz'
    np.savez_compressed(saved, actual=item['source'].reshape(-1))
    original = tmp_path / 'results.json'
    original.write_text(json.dumps(dict(rows=[dict(id=item['case_id'], error_artifact=dict(
        path=str(saved), sha256=run.semantic.sha256_file(saved)), low_storage_baseline=dict(status='unsupported-semantic'))])))
    before = original.read_bytes()
    sidecar = run.finalize_cpu_reference(prepared, original)
    result = json.loads(sidecar.read_text())
    assert original.read_bytes() == before
    assert result['summary']['references_computed'] == 1
    assert result['summary']['output_comparisons'] == 1
    assert result['summary']['same_precision_cpu_oracle'] == 'available-declared-arithmetic'
    assert result['reference_source_sha256']
    artifact = result['rows'][0]['backends']['cubutterfly']['error_artifact']
    assert Path(artifact['path']).exists()


def test_cpu_only_finalization_skips_gpu_library_fingerprints(prepared, tmp_path, monkeypatch):
    monkeypatch.setattr(run.semantic, 'validate_fingerprints', lambda *args: pytest.fail('GPU preparation invoked'))
    monkeypatch.setattr(run, 'finalize_cpu_reference', lambda p,r: tmp_path / 'sidecar.json')
    assert run.main(['--prepared',str(prepared),'--output',str(tmp_path),'--finalize-cpu']) == 0
