"""CPU-only tests for supplemental review; measurements remain byte-identical."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('gpu_precision_review', ROOT / 'scripts/review_gpu_same_precision.py')
reviewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reviewer)
from semantic_campaign_manifest import manifest_hash


def setup_checkpoint(directory, monkeypatch, integer=False, artifact=True, actual_dtype=None):
    directory = Path(str(directory))
    gpu_dir = directory / 'gpu'
    (gpu_dir / 'rows').mkdir(parents=True)
    s = dict(operator=3 if integer else 2, storage=8 if integer else 2,
             compute=3 if integer else 1, shape=[4], batch=1, inverse=False,
             normalize_inverse=False, fft_type=0, length_mode=0, fft_output_scale=1.0,
             placement=0, input_orders=[0], output_orders=[0], input_strides=[1],
             output_strides=[1], input_batch_stride=4, output_batch_stride=4, matrices=[[]])
    case = {'id': 'a' * 64, 'descriptor': {'semantics': s}}
    manifest = {'schema': 'cubutterfly-semantic-campaign-v1', 'cases': [case]}
    manifest['manifest_sha256'] = manifest_hash(manifest)
    manifest_path = directory / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    source = np.array([[1, 2, 4, 8]], dtype=np.uint32 if integer else np.float32)
    expected = np.array([[1, 3, 5, 15]], dtype=np.uint32) if integer else np.array([[15, -5, -9, 3]], dtype=np.float32)
    actual = expected.copy()
    if not integer:
        actual[0, 0] += 1.0  # Deliberately violates the historic tolerance too.
    if actual_dtype:
        actual = actual.astype(actual_dtype)
    retained = reviewer.semantic.retain_error_artifact(gpu_dir / 'errors', case['id'], 'cubutterfly', actual, expected.astype(np.float64) if not integer else expected)
    measured = dict(id=case['id'], descriptor=case['descriptor'], status='pass',
                    correctness={'legacy': True}, timing={'median_ms': 1.25},
                    baselines=[{'name': 'external-fwht', 'status': 'pass', 'error_artifact': retained},
                               {'name': 'missing', 'status': 'missing-baseline'}])
    if artifact:
        measured['error_artifact'] = retained
    else:
        measured['baselines'] = []
    row_checksum = reviewer.semantic.digest(measured)
    relative = 'rows/' + case['id'] + '.json'
    row_path = gpu_dir / relative
    row_path.write_text(json.dumps({'row': measured, 'sha256': row_checksum}))
    indices = [{'id': case['id'], 'status': 'pass', 'path': relative, 'sha256': row_checksum}]
    gpu_path = gpu_dir / 'results.json'
    report = dict(schema='cubutterfly-semantic-results-v1', rows=indices,
                  rows_sha256=reviewer.semantic.digest(indices),
                  identity={'manifest_sha256': reviewer.semantic.digest(manifest)})
    gpu_path.write_text(json.dumps(report))
    prepared = {'source': source, 'expected': expected, 's': s}
    monkeypatch.setattr(reviewer.preparation, '_config', lambda path: ({'manifest_sha256': reviewer.semantic.digest(manifest)}, 'config-checksum'))
    monkeypatch.setattr(reviewer.preparation, 'load_case', lambda path, requested: prepared)
    return manifest_path, gpu_path, row_path, retained, directory / 'review.json'


def test_float_review_actuals_full_error_and_original_untouched(tmpdir, monkeypatch):
    manifest, gpu, row_path, artifact, output = setup_checkpoint(tmpdir, monkeypatch)
    before = {path: path.read_bytes() for path in [gpu, row_path, Path(artifact['path'])]}
    report = reviewer.review(manifest, tmpdir, gpu, output)
    assert report['protocol']['gpu_calls'] == 0
    assert len(report['rows']) == 3
    for row in report['rows'][:2]:
        assert row['status'] == 'accuracy-review-required'
        assert row['recorded_status'] == 'pass'
        assert row['correctness']['max_abs'] == 1.0
        assert row['correctness']['mismatches'] == 1
        assert row['reference']['dtype'] == 'float32'
        with np.load(row['error_artifact']['path']) as arrays:
            assert arrays['reference'].dtype == np.float32
            np.testing.assert_array_equal(arrays['abs_error'], [1, 0, 0, 0])
        assert reviewer.semantic.sha256_file(row['error_artifact']['path']) == row['error_artifact']['sha256']
    assert report['rows'][2]['status'] == 'validation-unavailable'
    assert all(path.read_bytes() == data for path, data in before.items())


def test_integer_exact_logic(tmpdir, monkeypatch):
    manifest, gpu, _, _, output = setup_checkpoint(tmpdir, monkeypatch, integer=True)
    report = reviewer.review(manifest, tmpdir, gpu, output)
    assert report['rows'][0]['status'] == 'pass'
    assert report['rows'][0]['correctness'] == {'elements': 4, 'exact': True, 'mismatches': 0}
    monkeypatch.setattr(reviewer.cpu, 'same_precision_reference', lambda s, source, expected: (expected + np.uint32(1), {'kind': 'exact-integer'}))
    report = reviewer.review(manifest, tmpdir, gpu, output)
    assert report['rows'][0]['status'] == 'correctness-failure'
    assert report['rows'][0]['correctness']['mismatches'] == 4


def test_missing_artifact_is_gap(tmpdir, monkeypatch):
    manifest, gpu, _, _, output = setup_checkpoint(tmpdir, monkeypatch, artifact=False)
    report = reviewer.review(manifest, tmpdir, gpu, output)
    assert report['rows'][0]['status'] == 'validation-unavailable'
    assert 'no retained' in report['rows'][0]['reason']


def test_rejects_widened_actual_without_casting(tmpdir, monkeypatch):
    manifest, gpu, _, _, output = setup_checkpoint(tmpdir, monkeypatch, actual_dtype=np.float64)
    report = reviewer.review(manifest, tmpdir, gpu, output)
    assert report['rows'][0]['status'] == 'artifact-contract-failure'


def test_artifact_checksum_required(tmpdir, monkeypatch):
    manifest, gpu, _, artifact, output = setup_checkpoint(tmpdir, monkeypatch)
    Path(artifact['path']).write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='corrupted error artifact'):
        reviewer.review(manifest, tmpdir, gpu, output)
    assert not output.exists()


def test_manifest_identity_required(tmpdir, monkeypatch):
    manifest, gpu, _, _, output = setup_checkpoint(tmpdir, monkeypatch)
    report = json.loads(gpu.read_text())
    report['identity']['manifest_sha256'] = 'wrong'
    gpu.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='manifest identity'):
        reviewer.review(manifest, tmpdir, gpu, output)
    assert not output.exists()


def test_no_overwriting_checkpoint(tmpdir, monkeypatch):
    manifest, gpu, _, _, _ = setup_checkpoint(tmpdir, monkeypatch)
    with pytest.raises(ValueError, match='separate'):
        reviewer.review(manifest, tmpdir, gpu, gpu)


def test_no_overwriting_row_checkpoint(tmpdir, monkeypatch):
    manifest, gpu, row_path, _, _ = setup_checkpoint(tmpdir, monkeypatch)
    before = row_path.read_bytes()
    with pytest.raises(ValueError, match='overwrite'):
        reviewer.review(manifest, tmpdir, gpu, row_path)


def test_explicit_source_build_review_preserves_measured_artifacts(tmp_path, monkeypatch):
    manifest, gpu, row_path, artifact, output = setup_checkpoint(tmp_path, monkeypatch)
    source = json.loads(manifest.read_text())
    source_path = tmp_path / 'source_manifest.json'
    source_path.write_text(json.dumps(source))
    target = dict(source, fingerprints={'library': 'new'})
    target['manifest_sha256'] = manifest_hash(target)
    manifest.write_text(json.dumps(target))
    checkpoint = json.loads(gpu.read_text())
    checkpoint['identity']['manifest_sha256'] = reviewer.semantic.digest(target)
    gpu.write_text(json.dumps(checkpoint))
    monkeypatch.setattr(reviewer.preparation, '_config', lambda _: ({
        'manifest_sha256': reviewer.semantic.digest(source),
        'source_sha256': reviewer.preparation._sources()}, 'source-config-sha'))
    before = {path: path.read_bytes() for path in (gpu, row_path, Path(artifact['path']), source_path)}
    with pytest.raises(ValueError, match='preparation manifest identity'):
        reviewer.review(manifest, tmp_path, gpu, output)
    result = reviewer.review(manifest, tmp_path, gpu, output, source_path)
    assert result['identity']['manifest_sha256'] == reviewer.semantic.digest(target)
    assert result['identity']['cpu_source_manifest_sha256'] == reviewer.semantic.digest(source)
    assert result['identity']['preparation_identity']['config_sha256'] == 'source-config-sha'
    assert all(path.read_bytes() == data for path, data in before.items())
    assert result['protocol']['gpu_calls'] == 0
