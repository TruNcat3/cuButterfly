"""Scaled controls retain historical semantics and use final quantized inputs."""
import copy
import importlib
import json
import pathlib
import subprocess
import sys
import types

import numpy as np
import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
scaled = importlib.import_module('prepare_async_scaled_controls')
GPU = 'GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02'


def fixture(directory, operator=0, storage=4, n=16):
    directory.mkdir(parents=True)
    s = {'schema': 'cubutterfly-semantic-v1', 'operator': operator, 'storage': storage,
         'compute': 0, 'shape': [n], 'batch': 1, 'inverse': 0,
         'normalize_inverse': False, 'placement': 0, 'fft_type': 0,
         'fft_output_scale': 1.0, 'length_mode': 0, 'input_strides': [1],
         'output_strides': [1], 'input_batch_stride': n, 'output_batch_stride': n,
         'input_orders': [0], 'output_orders': [0], 'ntt_input_layout': 0,
         'ntt_output_layout': 0, 'modulus': None, 'word_bits': None, 'matrices': [[]]}
    artifact = directory / 'original.npz'
    np.savez(artifact, actual=np.array([np.inf]), reference=np.array([1.]))
    row = {'id': scaled.runner.digest(s), 'descriptor': {'schema': 'cubutterfly-descriptor-v1',
           'semantics': s, 'selection': {'algorithm_policy': 0, 'execution_mode': 0,
                                       'explicit_algorithm': ''}},
           'status': 'correctness-failure', 'mapping': {'kind': 'semantic-boundary', 'contract': s},
           'capabilities': {'status': 'eligible'},
           'error_artifact': {'path': str(artifact), 'sha256': scaled.runner.sha256_file(artifact)}}
    rows = directory / 'rows'
    rows.mkdir()
    path = rows / (row['id'] + '.json')
    path.write_text(json.dumps({'row': row, 'sha256': scaled.runner.digest(row)}))
    library = directory / 'library.so'
    library.write_bytes(b'cpu library')
    index = {'id': row['id'], 'sha256': scaled.runner.digest(row),
             'status': row['status'], 'path': 'rows/' + path.name}
    report = {'schema': 'cubutterfly-semantic-results-v1', 'rows': [index],
              'rows_sha256': scaled.runner.digest([index]),
              'identity': {'library_sha256': scaled.runner.sha256_file(library), 'gpu_uuid': GPU}}
    (directory / 'results.json').write_text(json.dumps(report))
    for name in ('registry.json', 'rates.json', 'model.json'):
        (directory / name).write_text('{}')
    calibration = {'measurement_identity': {'compile_mode': 'research',
                                            'library_sha256': scaled.runner.sha256_file(library)},
                   'registry_path': str(directory / 'registry.json'),
                   'hardware_rates_source': str(directory / 'rates.json'),
                   'cost_model': str(directory / 'model.json')}
    calibration_path = directory / 'calibration.json'
    calibration_path.write_text(json.dumps(calibration))
    args = types.SimpleNamespace(rows=[path], output=directory / 'controls', library=library,
                                 gpu_uuid=GPU, calibration_manifest=calibration_path,
                                 vkfft_library=None, gpuntt_library=None)
    return args, row


@pytest.mark.parametrize('operator,storage,n', [(0, 4, 32768), (2, 0, 16)])
def test_exact_scaled_profile_cache_and_manifest_identity(tmp_path, operator, storage, n):
    args, row = fixture(tmp_path / 'source', operator, storage, n)
    before = args.rows[0].read_bytes()
    result = scaled.prepare(args)
    manifest = json.loads(pathlib.Path(result['manifest_path']).read_text())
    assert manifest['manifest_sha256'] == scaled.runner.digest(
        {key: value for key, value in manifest.items() if key != 'manifest_sha256'})
    scaled.runner.validate_fingerprints(manifest, args.library)
    assert result['eligible'] == [row['id']] and result['gaps'] == []
    case = manifest['cases'][0]
    assert case['id'] == row['id'] and case['descriptor'] == row['descriptor']
    assert case['execution_mapping'] == row['mapping']
    assert case['capabilities'] == row['capabilities']
    assert case['populations'] == ['scaled-control'] and case['requested_variants'] == []
    prepared = scaled.queue.load_case(result['cache_path'], case)
    s = row['descriptor']['semantics']
    source, metadata = scaled.profiles.apply_input_profile(
        s, scaled.runner.make_source(s), scaled.profiles.scaled_profile(s),
        encode_storage=scaled.runner.encode_storage, decode_storage=scaled.runner.decode_storage)
    assert case['input_profile'] == metadata
    for field, expected in [('source', source), ('expected', scaled.runner.reference(s, source)),
                            ('encoded_source', scaled.runner.encode_storage(source, storage))]:
        assert prepared[field].dtype == expected.dtype
        assert prepared[field].tobytes() == expected.tobytes()
    assert args.rows[0].read_bytes() == before
    assert case['source_evidence']['original_report_sha256'] == scaled.runner.sha256_file(
        args.rows[0].parent.parent / 'results.json')


def test_scope_gap_is_explicit_without_case_or_prepared_record(tmp_path):
    args, row = fixture(tmp_path / 'source', storage=6)
    result = scaled.prepare(args)
    manifest = json.loads(pathlib.Path(result['manifest_path']).read_text())
    assert manifest['cases'] == [] and result['eligible'] == []
    assert result['gaps'][0]['status'] == 'not-applicable'
    assert scaled.queue.record_for(result['cache_path'], row['id']) is None


@pytest.mark.parametrize('target', ['artifact', 'row', 'library', 'uuid', 'calibration', 'report'])
def test_source_binding_failures_leave_no_output(tmp_path, target):
    args, row = fixture(tmp_path / 'source')
    if target == 'artifact':
        pathlib.Path(row['error_artifact']['path']).write_bytes(b'corrupt')
    elif target == 'row':
        value = json.loads(args.rows[0].read_text())
        value['row']['status'] = 'pass'
        args.rows[0].write_text(json.dumps(value))
    elif target == 'library':
        args.library.write_bytes(b'changed')
    elif target == 'uuid':
        args.gpu_uuid = 'GPU-0eae5f07-33de-321b-70a9-ffeae49309e6'
    elif target == 'calibration':
        calibration = json.loads(args.calibration_manifest.read_text())
        calibration['measurement_identity']['library_sha256'] = 'invalid'
        args.calibration_manifest.write_text(json.dumps(calibration))
    else:
        path = args.rows[0].parent.parent / 'results.json'
        report = json.loads(path.read_text())
        report['rows_sha256'] = 'invalid'
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        scaled.prepare(args)
    assert not args.output.exists()


def test_duplicate_and_nonfailure_rows_rejected(tmp_path):
    args, row = fixture(tmp_path / 'source')
    args.rows *= 2
    with pytest.raises(ValueError, match='duplicate'):
        scaled.prepare(args)
    args.rows = args.rows[:1]
    row['status'] = 'pass'
    args.rows[0].write_text(json.dumps({'row': row, 'sha256': scaled.runner.digest(row)}))
    with pytest.raises(ValueError, match='correctness-failure'):
        scaled.prepare(args)


def test_cli_preparation_never_imports_torch(tmp_path):
    args, row = fixture(tmp_path / 'source')
    cli = ['--rows', str(args.rows[0]), '--output', str(args.output), '--library', str(args.library),
           '--gpu-uuid', GPU, '--calibration-manifest', str(args.calibration_manifest)]
    code = ("import sys; import prepare_async_scaled_controls as s; "
            f"assert s.main({cli!r})==0; assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, '-c', code], cwd=SCRIPTS, check=True)
