"""Bounded CPU preparation persistence and identity checks."""
from concurrent.futures import Future
import copy
import importlib
import json
import os
import pathlib
import signal
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
queue = importlib.import_module('semantic_preparation_queue')


def case(batch=1):
    s = {'schema': 'cubutterfly-semantic-v1', 'operator': 0, 'storage': 6,
         'compute': 1, 'shape': [4], 'batch': batch, 'inverse': 0,
         'normalize_inverse': False, 'placement': 0, 'fft_type': 0,
         'fft_output_scale': 1.0, 'length_mode': 0, 'input_strides': [1],
         'output_strides': [1], 'input_batch_stride': 4,
         'output_batch_stride': 4, 'input_orders': [0], 'output_orders': [0],
         'ntt_input_layout': 0, 'ntt_output_layout': 0, 'modulus': None,
         'word_bits': None, 'matrices': [[]]}
    return {'id': queue.runner.digest(s), 'descriptor': {'semantics': s}, 'populations': ['test']}


class ImmediatePool:
    def __init__(self, submitted, trigger_stop=False):
        self.submitted = submitted
        self.trigger_stop = trigger_stop

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, function, *args):
        self.submitted.append(args[1]['id'])
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as error:
            future.set_exception(error)
        if self.trigger_stop:
            signal.raise_signal(signal.SIGTERM)
        return future


def fake_pool(monkeypatch, submitted, trigger_stop=False):
    monkeypatch.setattr(queue, '_executor', lambda jobs: ImmediatePool(submitted, trigger_stop))
    monkeypatch.setattr(queue, '_worker_init', lambda: None)


def test_init_exact_identity_and_subset(tmp_path, monkeypatch):
    manifest = {'schema': 'test-manifest', 'cases': [case(), case(2)], 'metadata': {'a': 1}}
    config = queue.initialize(tmp_path, manifest, 32, 64)
    assert queue.initialize(tmp_path, manifest, 32, 64) == config
    assert set(config['source_sha256']) == {'run_semantic_campaign.py', 'semantic_cpu_pipeline.py',
                                          'semantic_ntt_reference.py', 'semantic_input_profiles.py'}
    before = (tmp_path / 'config.json').read_bytes()
    for changed in ({'cases': manifest['cases'][:1]}, dict(manifest, cases=list(reversed(manifest['cases'])))):
        with pytest.raises(ValueError, match='identity mismatch'):
            queue.initialize(tmp_path, changed, 32, 64)
    with pytest.raises(ValueError, match='identity mismatch'):
        queue.initialize(tmp_path, manifest, 31, 64)
    monkeypatch.setattr(queue, '_sources', lambda: {})
    with pytest.raises(ValueError, match='identity mismatch'):
        queue.initialize(tmp_path, manifest, 32, 64)
    assert (tmp_path / 'config.json').read_bytes() == before


def test_prepare_priority_resume_and_roundtrip(tmp_path, monkeypatch):
    manifest = {'cases': [case(3), case(1), case(2)]}
    submitted = []
    fake_pool(monkeypatch, submitted)
    result = queue.prepare_manifest(manifest, tmp_path, jobs=2)
    assert submitted == [value['id'] for value in manifest['cases']]
    assert result['counts']['ready'] == 3
    prepared = queue.load_case(tmp_path, manifest['cases'][0])
    expected = queue.cpu.prepare_case(manifest['cases'][0])
    for field in ('source', 'expected', 'encoded_source', 'input_indices', 'output_indices'):
        assert prepared[field].dtype == expected[field].dtype
        np.testing.assert_array_equal(prepared[field], expected[field])
    submitted.clear()
    assert queue.prepare_manifest(manifest, tmp_path, jobs=2)['status'] == 'complete'
    assert submitted == []
    assert queue.record_for(tmp_path, case(4)['id']) is None


def test_artifact_corruption_checked_on_load_not_poll(tmp_path, monkeypatch):
    manifest = {'cases': [case()]}
    fake_pool(monkeypatch, [])
    queue.prepare_manifest(manifest, tmp_path)
    record = queue.record_for(tmp_path, case()['id'])
    artifact = tmp_path / record['artifact']['path']
    data = bytearray(artifact.read_bytes())
    data[-1] ^= 1
    artifact.write_bytes(data)
    assert queue.record_for(tmp_path, case()['id']) == record
    with pytest.raises(ValueError, match='corrupted preparation artifact'):
        queue.load_case(tmp_path, case())
    with pytest.raises(ValueError, match='corrupted preparation artifact'):
        queue.prepare_manifest(manifest, tmp_path)


def test_known_gap_and_failure_are_explicit_and_not_retried(tmp_path, monkeypatch):
    manifest = {'cases': [case(), case(2)]}
    submitted = []
    fake_pool(monkeypatch, submitted)

    def prepare(value, *args):
        if value['descriptor']['semantics']['batch'] == 1:
            raise queue.runner.Gap('validation-unavailable', 'budget exceeded')
        raise RuntimeError('CPU failed')

    monkeypatch.setattr(queue.cpu, 'prepare_case', prepare)
    result = queue.prepare_manifest(manifest, tmp_path)
    assert result['status'] == 'failed'
    assert result['counts'] == {'ready': 0, 'known-gap': 1, 'failed': 1, 'pending': 0}
    gap = queue.record_for(tmp_path, case()['id'])
    assert gap['gap_status'] == 'validation-unavailable' and 'artifact' not in gap
    with pytest.raises(queue.runner.Gap, match='budget exceeded'):
        queue.load_case(tmp_path, case())
    with pytest.raises(RuntimeError, match='CPU preparation failed'):
        queue.load_case(tmp_path, case(2))
    submitted.clear()
    queue.prepare_manifest(manifest, tmp_path)
    assert submitted == []


def test_sigterm_stops_enqueue_and_finishes_submitted(tmp_path, monkeypatch):
    manifest = {'cases': [case(n) for n in range(1, 7)]}
    submitted = []
    fake_pool(monkeypatch, submitted, trigger_stop=True)
    result = queue.prepare_manifest(manifest, tmp_path, jobs=2)
    assert result['status'] == 'stopped'
    assert submitted == [case()['id']]
    assert result['counts']['ready'] == 1 and result['counts']['pending'] == 5


def test_submission_window_bounded_to_twice_jobs(tmp_path, monkeypatch):
    manifest = {'cases': [case(n) for n in range(1, 10)]}
    fake_pool(monkeypatch, [])
    wait_sizes = []
    original = queue.wait

    def bounded_wait(futures, **kwargs):
        wait_sizes.append(len(futures))
        return original(futures, **kwargs)

    monkeypatch.setattr(queue, 'wait', bounded_wait)
    assert queue.prepare_manifest(manifest, tmp_path, jobs=2)['counts']['ready'] == 9
    assert wait_sizes and max(wait_sizes) == 4


def test_record_poll_reuses_config_without_artifact_hashing(tmp_path, monkeypatch):
    fake_pool(monkeypatch, [])
    queue.prepare_manifest({'cases': [case()]}, tmp_path)
    original = queue._read_json
    reads = []

    def read(path):
        reads.append(pathlib.Path(path).name)
        return original(path)

    monkeypatch.setattr(queue, '_read_json', read)
    monkeypatch.setattr(queue.runner, 'sha256_file', lambda path: pytest.fail('poll hashed a file'))
    for _ in range(3):
        assert queue.record_for(tmp_path, case()['id'])['status'] == 'ready'
    assert 'config.json' not in reads


def test_case_metadata_and_record_corruption_rejected(tmp_path, monkeypatch):
    manifest = {'cases': [case()]}
    fake_pool(monkeypatch, [])
    queue.prepare_manifest(manifest, tmp_path)
    changed = copy.deepcopy(case())
    changed['populations'] = ['foreign']
    with pytest.raises(ValueError, match='case identity'):
        queue.load_case(tmp_path, changed)
    path = tmp_path / 'records' / (case()['id'] + '.json')
    data = json.loads(path.read_text())
    data['value']['status'] = 'failed'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='checksum'):
        queue.record_for(tmp_path, case()['id'])


def test_real_spawn_pool_cpu_only_and_cli(tmp_path):
    manifest = {'cases': [case(), case(2)]}
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    spool = tmp_path / 'spool'
    code = ("import sys; import semantic_preparation_queue as q; import json; "
            f"m=json.load(open({str(manifest_path)!r})); "
            f"r=q.prepare_manifest(m, {str(spool)!r}, jobs=2); "
            "assert r['counts']['ready']==2; assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, '-c', code], cwd=SCRIPTS, check=True,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-forbidden'))
    output = subprocess.check_output([sys.executable, str(SCRIPTS / 'semantic_preparation_queue.py'),
                                      '--manifest', str(manifest_path), '--directory', str(spool),
                                      '--only-init'], text=True)
    assert json.loads(output)['status'] == 'initialized'


@pytest.mark.parametrize('jobs', [0, -1, True, 1.5])
def test_jobs_rejected(tmp_path, jobs):
    with pytest.raises(ValueError, match='positive integer'):
        queue.prepare_manifest({'cases': [case()]}, tmp_path, jobs)
