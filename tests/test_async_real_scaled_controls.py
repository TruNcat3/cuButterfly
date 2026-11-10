"""Real FFT controls bind raw evidence while preserving requested semantics."""
import copy
import importlib
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
scaled = importlib.import_module('prepare_async_real_scaled_controls')
from test_async_scaled_controls import fixture as original_fixture, GPU

OTHER_GPU = 'GPU-00000000-0000-0000-0000-000000000005'


def fixture(directory, fft_type=1, n=16):
    args, row = original_fixture(directory, storage=0, n=n)
    s = row['descriptor']['semantics']
    s.update(fft_type=fft_type, inverse=int(fft_type == 2),
             normalize_inverse=fft_type == 2,
             input_batch_stride=n // 2 + 1 if fft_type == 2 else n,
             output_batch_stride=n // 2 + 1 if fft_type == 1 else n)
    row['id'] = scaled.runner.digest(s)
    row['mapping']['contract'] = copy.deepcopy(s)
    path = args.rows[0].parent / (row['id'] + '.json')
    path.write_text(json.dumps({'row': row, 'sha256': scaled.runner.digest(row)}))
    args.rows = [path]
    index = {'id': row['id'], 'sha256': scaled.runner.digest(row),
             'status': row['status'], 'path': 'rows/' + path.name}
    report_path = directory / 'results.json'
    report = json.loads(report_path.read_text())
    report.update(rows=[index], rows_sha256=scaled.runner.digest([index]))
    report_path.write_text(json.dumps(report))
    args.diagnostic_gpu_uuid = None
    return args, row


@pytest.mark.parametrize('fft_type', [1, 2])
def test_exact_semantics_quantized_input_and_manifest_cache_binding(tmp_path, fft_type):
    args, row = fixture(tmp_path / 'source', fft_type)
    original = args.rows[0].read_bytes()
    result = scaled.prepare(args)
    manifest = json.loads(pathlib.Path(result['manifest_path']).read_text())
    assert manifest['manifest_sha256'] == scaled.runner.digest(
        {key: value for key, value in manifest.items() if key != 'manifest_sha256'})
    scaled.runner.validate_fingerprints(manifest, args.library)
    case = manifest['cases'][0]
    assert case['descriptor'] == row['descriptor'] and case['id'] == row['id']
    assert case['execution_mapping'] == row['mapping']
    assert case['populations'] == ['real-scaled-control']
    assert case['capabilities'] == row['capabilities']
    prepared = scaled.queue.load_case(result['cache_path'], case)
    source, metadata = scaled.profiles.prepare_input(row['descriptor']['semantics'])
    expected = scaled.runner.reference(row['descriptor']['semantics'], source)
    assert case['input_profile'] == metadata
    assert prepared['source'].tobytes() == source.tobytes()
    assert prepared['expected'].tobytes() == expected.tobytes()
    assert np.isfinite(source).all() and np.isfinite(expected).all()
    input_storage = scaled.runner.side_storage(row['descriptor']['semantics'], 'input')
    assert prepared['encoded_source'].tobytes() == scaled.runner.encode_storage(source, input_storage).tobytes()
    assert args.rows[0].read_bytes() == original
    assert case['source_evidence']['original_report_sha256'] == scaled.runner.sha256_file(
        args.rows[0].parent.parent / 'results.json')
    changed = copy.deepcopy(case)
    changed['populations'] = ['other']
    with pytest.raises(ValueError):
        scaled.queue.load_case(result['cache_path'], changed)
    sources = {pathlib.Path(item['path']).name for item in manifest['fingerprints']['sources']}
    assert {'prepare_async_real_scaled_controls.py', 'semantic_real_input_profiles.py'} <= sources


@pytest.mark.parametrize('target,replay', [(None, True), (GPU, True), (OTHER_GPU, False)])
def test_original_replay_and_different_gpu_fresh_selection(tmp_path, target, replay):
    args, row = fixture(tmp_path / 'source', fft_type=2)
    args.diagnostic_gpu_uuid = target
    result = scaled.prepare(args)
    manifest = json.loads(pathlib.Path(result['manifest_path']).read_text())
    case = manifest['cases'][0]
    assert ('execution_mapping' in case) == replay
    assert case['source_evidence']['original_mapping_replayed'] is replay
    provenance = manifest['real_scaled_control_provenance']
    assert provenance['source_gpu_uuid'] == GPU
    assert provenance['target_gpu_uuid'] == (target or GPU)
    assert provenance['timing'] is False
    assert provenance['float_verdict'] == 'error-report'
    assert ('original mapping replay' in provenance['selection']) == replay


@pytest.mark.parametrize('target', ['artifact', 'row', 'library', 'uuid', 'calibration', 'report'])
def test_tampered_source_fails_before_output(tmp_path, target):
    args, row = fixture(tmp_path / 'source')
    if target == 'artifact':
        pathlib.Path(row['error_artifact']['path']).write_bytes(b'corrupt')
    elif target == 'row':
        envelope = json.loads(args.rows[0].read_text())
        envelope['row']['status'] = 'pass'
        args.rows[0].write_text(json.dumps(envelope))
    elif target == 'library':
        args.library.write_bytes(b'changed')
    elif target == 'uuid':
        args.gpu_uuid = OTHER_GPU
    elif target == 'calibration':
        value = json.loads(args.calibration_manifest.read_text())
        value['measurement_identity']['library_sha256'] = 'invalid'
        args.calibration_manifest.write_text(json.dumps(value))
    else:
        path = args.rows[0].parent.parent / 'results.json'
        value = json.loads(path.read_text())
        value['rows_sha256'] = 'invalid'
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        scaled.prepare(args)
    assert not args.output.exists()


def test_duplicate_and_nonfailure_rejected(tmp_path):
    args, row = fixture(tmp_path / 'source')
    args.rows *= 2
    with pytest.raises(ValueError, match='duplicate'):
        scaled.prepare(args)
    args.rows = args.rows[:1]
    row['status'] = 'pass'
    args.rows[0].write_text(json.dumps({'row': row, 'sha256': scaled.runner.digest(row)}))
    with pytest.raises(ValueError, match='correctness-failure'):
        scaled.prepare(args)


def test_unsupported_request_is_rejected_without_silent_gap(tmp_path):
    args, row = fixture(tmp_path / 'source')
    row['descriptor']['semantics']['fft_type'] = 0
    row['id'] = scaled.runner.digest(row['descriptor']['semantics'])
    args.rows[0].write_text(json.dumps({'row': row, 'sha256': scaled.runner.digest(row)}))
    with pytest.raises(ValueError, match='FP16 R2C or C2R'):
        scaled.prepare(args)
    assert not args.output.exists()


def test_nonfinite_reference_is_rejected(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path / 'source')
    monkeypatch.setattr(scaled.runner, 'reference', lambda *a: np.array([np.inf]))
    with pytest.raises(ValueError, match='must be finite'):
        scaled.prepare(args)
    assert not args.output.exists()


def test_cli_is_cpu_only(tmp_path):
    args, _ = fixture(tmp_path / 'source', fft_type=2)
    cli = ['--rows', str(args.rows[0]), '--output', str(args.output), '--library', str(args.library),
           '--gpu-uuid', GPU, '--calibration-manifest', str(args.calibration_manifest),
           '--diagnostic-gpu-uuid', OTHER_GPU]
    code = ("import sys; import prepare_async_real_scaled_controls as s; "
            f"assert s.main({cli!r})==0; assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, '-c', code], cwd=SCRIPTS, check=True)
