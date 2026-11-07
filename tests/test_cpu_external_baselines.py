"""CPU runner protocol checks with tiny arrays and no GPU initialization."""
import csv
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('cpu_external_protocol_test', SCRIPTS / 'cpu_external_baselines.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
from semantic_campaign_manifest import manifest_hash


def semantics(storage=6, **changes):
    result = dict(operator=0, storage=storage, compute=1 if storage == 6 else 2,
                  shape=[2, 4], batch=2, inverse=0, normalize_inverse=False,
                  fft_type=0, fft_output_scale=1.0, length_mode=0,
                  input_orders=[0, 0], output_orders=[0, 0])
    result.update(changes)
    return result


class Core:
    def __init__(self, source, result):
        self.source = source
        self.result = result
        self.output = np.empty_like(result)
        self.calls = 0

    def execute(self):
        self.calls += 1
        np.copyto(self.output, self.result)


@pytest.mark.parametrize('storage,dtype', [(6, np.complex64), (7, np.complex128)])
@pytest.mark.parametrize('inverse,normalize', [(0, False), (1, False), (1, True)])
def test_same_precision_fft_reference_complete_batch_and_scale(storage, dtype, inverse, normalize):
    s = semantics(storage, inverse=inverse, normalize_inverse=normalize, fft_output_scale=-2.0)
    source = np.zeros((2, 2, 4), dtype=dtype)
    source[:, 0, 0] = [1, 3]
    result, info = runner.same_precision_reference(s, source)
    scale = -2.0 / (8 if inverse and normalize else 1)
    expected = np.broadcast_to(np.array([1, 3], dtype=dtype).reshape(2, 1, 1) * scale, source.shape)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == source.dtype
    assert info['threshold'] is None


def test_reference_rejects_precision_substitution():
    with pytest.raises(ValueError, match='no implicit widening'):
        runner.same_precision_reference(semantics(compute=2), np.zeros((2, 2, 4), dtype=np.complex64))


def test_measure_float_error_report_has_no_acceptance_threshold():
    source = np.zeros((2, 2, 4), dtype=np.complex64)
    # Deliberately large but finite error must remain reportable.
    core = Core(source, np.full_like(source, 1000 + 2000j))
    result = runner.measure(core, source, source.copy(), semantics(), warmup=2, trials=3, calls=4)
    assert core.calls == 1 + 2 + 3 * 4
    assert len(result['samples_ms']) == 3
    assert all(value >= 0 for value in result['samples_ms'])


def test_measure_integer_mismatch_is_exact():
    source = np.zeros((2, 2, 4), dtype=np.uint64)
    expected = np.full_like(source, (1 << 55) + 1)
    result = expected.copy()
    result.flat[-1] += np.uint64(1)
    with pytest.raises(AssertionError, match='integer mismatch'):
        runner.measure(Core(source, result), source, expected, semantics(storage=9), 0, 1, 1)


def test_measure_checks_source_bytes_and_output_dtype():
    source = np.zeros((2, 2, 4), dtype=np.complex64)
    core = Core(source, source.copy())
    def modifying_execute():
        # Signed zero has equal numerical value but different storage bits.
        source.real.flat[0] = -0.0
        np.copyto(core.output, core.result)
    core.execute = modifying_execute
    with pytest.raises(AssertionError, match='input modified'):
        runner.measure(core, source, source.copy(), semantics(), 0, 1, 1)
    with pytest.raises(AssertionError, match='output storage changed'):
        runner.measure(Core(source, np.zeros(source.shape, dtype=np.complex128)), source, source.copy(), semantics(), 0, 1, 1)


def test_measure_rejects_wrong_logical_output_shape():
    source = np.zeros((2, 2, 4), dtype=np.complex64)
    with pytest.raises(AssertionError, match='shape'):
        runner.measure(Core(source, source.reshape(2, 4, 2)), source, source.copy(), semantics(), 0, 1, 1)


def manifest():
    return {'cases': [{'id': 'case-a', 'populations': ['cpu-protocol'],
                       'descriptor': {'semantics': semantics()}}]}


def report(value):
    return {'identity': {'manifest_sha256': runner.semantic.digest(value)},
            'rows': [{'case_id': 'case-a', 'backend': 'fftw-cpu-t1',
                      'status': 'accuracy-review-required', 'semantics': semantics(),
                      'timing': {'median_ms': 6.0}}]}


def test_join_rejects_another_cpu_manifest(tmp_path):
    value = manifest()
    other = report(value)
    other['identity']['manifest_sha256'] = 'different-manifest'
    with pytest.raises(ValueError, match='manifest'):
        runner.export_table(value, other, tmp_path)


def test_join_rejects_duplicate_cpu_backend_rows(tmp_path):
    value = manifest()
    duplicate = report(value)
    duplicate['rows'].append(duplicate['rows'][0].copy())
    with pytest.raises(ValueError, match='duplicate'):
        runner.export_table(value, duplicate, tmp_path)


def test_subset_join_preserves_full_manifest_and_excludes_unmeasured_requests(tmp_path):
    value = manifest()
    value['cases'].append(dict(value['cases'][0], id='case-b'))
    measured = report(value)
    measured['identity']['selected_case_ids'] = ['case-a']
    runner.export_table(value, measured, tmp_path)
    with (tmp_path / 'unified_comparison.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row['case_id'] for row in rows] == ['case-a']
    assert rows[0]['comparison_manifest_sha256'] == runner.semantic.digest(value)
    measured['identity']['selected_case_ids'] = ['case-b']
    with pytest.raises(ValueError, match='outside selected'):
        runner.export_table(value, measured, tmp_path)


def test_join_separates_cpu_gpu_timings_and_requires_gpu_manifest(tmp_path, monkeypatch):
    value = manifest()
    gpu = tmp_path / 'gpu.json'
    gpu.touch()
    gpu_identity = {'identity': {'manifest_sha256': runner.semantic.digest(value)}}
    gpu_rows = [{'id': 'case-a', 'status': 'accuracy-review-required',
                 'timing': {'median_ms': 2.0},
                 'baselines': [{'name': 'cufft', 'status': 'accuracy-review-required',
                                'timing': {'median_ms': 4.0}}]}]
    monkeypatch.setitem(sys.modules, 'summarize_semantic_campaign',
                        types.SimpleNamespace(load_checkpoint=lambda _: (gpu_identity, gpu_rows)))
    runner.export_table(value, report(value), tmp_path, gpu)
    with (tmp_path / 'unified_comparison.csv').open() as stream:
        row = next(csv.DictReader(stream))
    assert float(row['cubutterfly_ms']) == 2
    assert float(row['cufft_ms']) == 4
    assert float(row['fftw-cpu-t1_ms']) == 6
    assert 'fftw-cpu-t1_gpu_speedup' not in row
    assert row['floating_acceptance'] == 'error-report-only'
    gpu_identity['identity']['manifest_sha256'] = 'different-manifest'
    with pytest.raises(ValueError, match='manifest'):
        runner.export_table(value, report(value), tmp_path, gpu)


def build_manifests():
    source = dict(manifest(), schema='cubutterfly-semantic-campaign-v1',
                  fingerprints={'library': 'old'}, config={'seed': 17})
    source['manifest_sha256'] = manifest_hash(source)
    target = copy.deepcopy(source)
    target['fingerprints']['library'] = 'new'
    target['manifest_sha256'] = manifest_hash(target)
    return source, target


def test_explicit_build_reuse_preserves_cpu_identity_and_records_provenance(tmp_path):
    source, target = build_manifests()
    measured = report(source)
    frozen = copy.deepcopy(measured)
    with pytest.raises(ValueError, match='another semantic manifest'):
        runner.export_table(target, measured, tmp_path)
    runner.export_table(target, measured, tmp_path, source_manifest=source)
    assert measured == frozen
    provenance = json.loads((tmp_path / 'comparison_provenance.json').read_text())
    assert provenance['comparison_manifest_sha256'] == runner.semantic.digest(target)
    assert provenance['cpu_source_manifest_sha256'] == runner.semantic.digest(source)
    assert provenance['cpu_identity'] == measured['identity']
    assert provenance['cpu_measurements_reused'] is True


@pytest.mark.parametrize('field', ['cases', 'config', 'layout_recipe', 'invalid_requests', 'metrics'])
def test_build_reuse_rejects_any_nonfingerprint_change(field):
    source, target = build_manifests()
    target[field] = ['changed']
    target['manifest_sha256'] = manifest_hash(target)
    with pytest.raises(ValueError, match='semantic content mismatch'):
        runner.comparison_source(target, source)


@pytest.mark.parametrize('side', ['source', 'target'])
def test_build_reuse_checks_both_manifest_checksums(side):
    source, target = build_manifests()
    (source if side == 'source' else target)['fingerprints'] = {'corrupted': True}
    with pytest.raises(ValueError, match='checksum mismatch'):
        runner.comparison_source(target, source)


def test_build_reuse_requires_source_cpu_binding_and_row_semantics(tmp_path):
    source, target = build_manifests()
    with pytest.raises(ValueError, match='another semantic manifest'):
        runner.export_table(target, report(target), tmp_path, source_manifest=source)
    measured = report(source)
    measured['rows'][0]['semantics'] = {}
    with pytest.raises(ValueError, match='row semantic identity mismatch'):
        runner.export_table(target, measured, tmp_path, source_manifest=source)


def test_reused_preparation_requires_source_manifest_and_frozen_sources(tmp_path, monkeypatch):
    source, target = build_manifests()
    config = {'manifest_sha256': runner.semantic.digest(source), 'source_sha256': {'frozen': 'hash'}}
    monkeypatch.setattr(runner.preparation, '_config', lambda _: (config, 'config-hash'))
    monkeypatch.setattr(runner.preparation, '_sources', lambda: {'frozen': 'hash'})
    assert runner.comparison_preparation(target, tmp_path, source)['config_sha256'] == 'config-hash'
    config['manifest_sha256'] = runner.semantic.digest(target)
    with pytest.raises(ValueError, match='preparation manifest identity'):
        runner.comparison_preparation(target, tmp_path, source)
    config['manifest_sha256'] = runner.semantic.digest(source)
    config['source_sha256'] = {}
    with pytest.raises(ValueError, match='preparation source identity'):
        runner.comparison_preparation(target, tmp_path, source)
