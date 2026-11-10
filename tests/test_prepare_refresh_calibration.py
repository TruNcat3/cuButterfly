import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_refresh_calibration as refresh


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    root = tmp_path / 'campaign'
    root.mkdir()
    active = root / 'calibration'
    active.mkdir()
    build = tmp_path / 'build'
    build.mkdir()
    workloads = tmp_path / 'workloads.json'
    declared = [dict(precision='fp32', logN=10, batch=1),
                dict(operator='ntt', precision='uint64', logN=12, batch=2)]
    workloads.write_text(json.dumps({'schema': 'cubutterfly-install-search-v1', 'workloads': declared}))
    measurement = {'build': 'frozen', 'verify_batches': 3}
    documents = {'search_coverage.json': {'measurement_identity': measurement},
                 'calibration_device.json': {'hardware': 'frozen'},
                 'search_measurements.json': {'rows': [{'status': 'needs-compilation',
                     'configuration': dict(operator='fft', **declared[0]), 'errors': ['budget exhausted']}]},
                 'plan_descriptors.json': {'identity': {'measurement': measurement}, 'descriptors': {}}}
    for name, document in documents.items():
        (active / name).write_text(json.dumps(document))
    state = {'status': 'stopped', 'identity': {'environment': {
        'CUBUTTERFLY_COMPILE_MODE': 'research', 'CUBUTTERFLY_PARTITION_BUDGET': '9'},
        'gpu_uuid': 'GPU-frozen'}, 'phases': {'calibrate': {'command': [
            'python', 'calibrate_local_hardware.py', '--build-dir', str(build),
            '--output-dir', str(active), '--search-workloads', str(workloads),
            '--verify-batches', '3', '--search-stage-projection', 'plan']}}}
    (root / 'status.json').write_text(json.dumps(state))
    for name in (*state['identity']['environment'], 'CUBUTTERFLY_HARDWARE_MODEL', 'CUDA_VISIBLE_DEVICES'):
        monkeypatch.setenv(name, 'before-test')
    monkeypatch.setattr(refresh.calibration, 'measurement_identity', lambda args: measurement)
    monkeypatch.setattr(refresh.calibration, 'cpu_preparation_context',
                        lambda args, measured: {'measurement': measured, 'scope': args.search_stage_projection})
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: pytest.fail('unexpected external command'))
    return root, tmp_path / 'output', state, measurement, documents, declared


@pytest.mark.parametrize('rebound', [False, True])
def test_known_gaps_compile_before_capture_and_full_scoring(campaign, monkeypatch, rebound):
    root, output, state, measurement, documents, declared = campaign
    if rebound:
        command = state['phases']['calibrate']['command']
        (root / 'commands.json').write_text(json.dumps({'phases': [['calibrate', command]]}))
        state.update(status='prepared', phases={})
        (root / 'status.json').write_text(json.dumps(state))
    calls = []
    result = {'compile_ready': True, 'scoring_complete': True, 'gpu_verified': False}
    def prepare(path, destination, broker, **kwargs):
        assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
        assert os.environ['CUBUTTERFLY_COMPILE_MODE'] == 'research'
        assert os.environ['CUBUTTERFLY_PARTITION_BUDGET'] == '9'
        assert os.environ['CUBUTTERFLY_HARDWARE_MODEL'] == str(root / 'calibration/hardware/hardware_model.json')
        assert broker == root / 'compile'
        assert kwargs['jobs'] == 2
        if kwargs.get('compile_only'):
            calls.append('known-gaps')
            assert destination == output / 'compilation_recovery'
            manifest = json.loads(path.read_text())
            assert manifest['context'] == {'measurement': measurement, 'scope': 'plan'}
            assert [cell['workload'] for cell in manifest['cells']] == [{'operator': 'fft', **w} for w in declared]
            for key, filename in (('profile', 'calibration_device.json'),
                                  ('training', 'search_measurements.json'), ('descriptors', 'plan_descriptors.json')):
                reference = manifest[key]
                data = (path.parent / reference['path']).read_bytes()
                assert hashlib.sha256(data).hexdigest() == reference['sha256']
                assert json.loads(data) == documents[filename]
            for cell in manifest['cells']:
                assert not cell['complete']
                with gzip.open(path.parent / cell['inventory']['path'], 'rt') as stream:
                    assert stream.read() == ''
            return {'compile_ready': True, 'scoring_complete': False}
        calls.append('full-preparation')
        assert destination == output / 'prepared'
        assert kwargs == {'jobs': 2, 'compile_top': 7}
        assert path == output / 'capture/input.json'
        return result
    def capture(build, active, workloads, manifest, context_path):
        calls.append('capture')
        assert calls == ['known-gaps', 'capture']
        assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-frozen'
        assert active == root / 'calibration'
        assert json.loads(context_path.read_text())['measurement'] == measurement
    monkeypatch.setattr(refresh.preparation, 'prepare', prepare)
    monkeypatch.setattr(refresh.capture, 'capture', capture)
    original = {name: (root / 'calibration' / name).read_bytes() for name in documents}
    assert refresh.prepare(root, output, jobs=2, compile_top=7) is result
    assert calls == ['known-gaps', 'capture', 'full-preparation']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert {name: (root / 'calibration' / name).read_bytes() for name in documents} == original


@pytest.mark.parametrize('reason', ['identity', 'running-campaign', 'live-driver'])
def test_rejects_incompatible_or_active_campaign_before_preparation(campaign, monkeypatch, reason):
    root, output, state, _, _, _ = campaign
    if reason == 'identity':
        monkeypatch.setattr(refresh.calibration, 'measurement_identity', lambda args: {'build': 'changed'})
        message = 'contract differ'
    else:
        if reason == 'running-campaign':
            state['status'] = 'running'
        else:
            state['phases']['calibrate']['pid'] = os.getpid()
        (root / 'status.json').write_text(json.dumps(state))
        message = 'stop the owned measurement driver'
    monkeypatch.setattr(refresh.preparation, 'prepare', lambda *args, **kwargs: pytest.fail('preparation called'))
    monkeypatch.setattr(refresh.capture, 'capture', lambda *args, **kwargs: pytest.fail('capture called'))
    with pytest.raises(ValueError, match=message):
        refresh.prepare(root, output)
    assert not (output / 'capture/compile-input.json').exists()


def test_failed_known_gap_compilation_stops_before_capture(campaign, monkeypatch):
    root, output, _, _, _, _ = campaign
    def prepare(*args, **kwargs):
        assert kwargs['compile_only'] is True
        assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
        return {'compile_ready': False}
    monkeypatch.setattr(refresh.preparation, 'prepare', prepare)
    monkeypatch.setattr(refresh.capture, 'capture', lambda *args, **kwargs: pytest.fail('capture called'))
    with pytest.raises(RuntimeError, match='known compilation gaps remain'):
        refresh.prepare(root, output)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''


def test_rejects_sdb_paths_before_reading_campaign(tmp_path):
    for campaign, output in ((Path('/sdb/campaign'), tmp_path), (tmp_path, Path('/sdb/output'))):
        with pytest.raises(ValueError, match='must not use /sdb'):
            refresh.prepare(campaign, output)
