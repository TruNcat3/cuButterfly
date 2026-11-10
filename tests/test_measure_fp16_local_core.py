"""Frozen FP16 worker contracts; no CUDA imports or execution."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/measure_fp16_local_core.py'
spec = importlib.util.spec_from_file_location('measure_fp16_local_core', SCRIPT)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


@pytest.fixture
def manifest(tmp_path):
    (tmp_path / 'cache').mkdir()
    (tmp_path / 'lib.so').write_bytes(b'frozen')
    np.save(tmp_path / 'input.npy', np.zeros((5, 16, 2), np.float16))
    np.save(tmp_path / 'reference.npy', np.zeros((5, 16), np.complex128))
    data = dict(gpu_uuid='GPU-test', library='lib.so', library_sha256=worker.sha256(tmp_path / 'lib.so'),
                vendor_library='lib.so', vendor_sha256=worker.sha256(tmp_path / 'lib.so'),
                jit_cache='cache', preparation='.', modules=[], tasks=[dict(id='test', n=16, batch=5,
                variant='local', mapping={'fft_core': 'cufftdx-block'}, input_file='input.npy',
                reference_file='reference.npy', input_sha256=worker.sha256(tmp_path / 'input.npy'),
                reference_sha256=worker.sha256(tmp_path / 'reference.npy'))])
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(data))
    return path, data


def save(path, data):
    path.write_text(json.dumps(data))


def test_import_and_help_do_not_import_cuda():
    command = ('import runpy,sys;runpy.run_path(sys.argv[1],run_name="worker_import");'
               'assert "torch" not in sys.modules;assert "cubutterfly" not in sys.modules')
    subprocess.run([sys.executable, '-c', command, str(SCRIPT)], check=True)
    subprocess.run([sys.executable, str(SCRIPT), '--help'], check=True, capture_output=True)


def test_complete_input_and_explicit_environment(manifest, monkeypatch):
    path, _ = manifest
    m, t, x = worker.load_task(path, 'test')
    assert x.shape == (5, 16, 2) and x.dtype == np.float16
    assert Path(t['reference_file']).is_absolute()
    for key in ('CUDA_VISIBLE_DEVICES', 'CUBUTTERFLY_COMPILE_MODE', 'CUBUTTERFLY_JIT_CACHE',
                'CUBUTTERFLY_TEMPLATE_ROOT', 'CUBUTTERFLY_JIT_CACHE_ONLY',
                'CUBUTTERFLY_JIT_PYTHON', 'CUBUTTERFLY_MATHDX_ROOT'):
        monkeypatch.setenv(key, 'foreign')
    worker.configure_environment(m)
    import os
    assert os.environ['CUBUTTERFLY_COMPILE_MODE'] == 'research'
    assert os.environ['CUBUTTERFLY_JIT_CACHE_ONLY'] == '1'
    assert os.environ['CUBUTTERFLY_JIT_PYTHON'] == sys.executable
    assert os.environ['CUBUTTERFLY_MATHDX_ROOT'] == str(worker.ROOT / 'external/mathdx/nvidia/mathdx')
    assert os.environ['CUBUTTERFLY_JIT_CACHE'] == str(path.parent / 'cache')


@pytest.mark.parametrize('problem', ['hash', 'duplicate', 'shape', 'dtype', 'mapping', 'module'])
def test_reject_invalid_artifacts_before_cuda(manifest, problem):
    path, m = manifest
    if problem == 'hash':
        m['library_sha256'] = 'wrong'
    elif problem == 'duplicate':
        m['tasks'] *= 2
    elif problem == 'shape':
        np.save(path.parent / 'input.npy', np.zeros((1, 16, 2), np.float16))
    elif problem == 'dtype':
        np.save(path.parent / 'input.npy', np.zeros((5, 16, 2), np.float32))
    elif problem == 'mapping':
        m['tasks'][0]['mapping'] = None
    else:
        m['modules'] = [dict(path='lib.so', sha256='wrong', point={})]
    save(path, m)
    with pytest.raises(ValueError):
        worker.load_task(path, 'test')


def test_existing_output_rejected_before_cuda(manifest, tmp_path):
    path, _ = manifest
    out = tmp_path / 'attempt'
    out.mkdir()
    (out / 'result.json').write_text('{}')
    from argparse import Namespace
    with pytest.raises(FileExistsError):
        worker.run(Namespace(manifest=path, task='test', output=out, mode='measure'))
    assert (out / 'result.json').read_text() == '{}'


def test_resource_classification_is_narrow():
    assert worker.resource_limit(RuntimeError('CUDA error: too many resources requested for launch'))
    assert worker.resource_limit(RuntimeError('cudaErrorInvalidConfiguration'))
    assert worker.resource_limit(RuntimeError('compiled specialization exceeds active-device resources'))
    for message in ('out of memory', 'output guard changed', 'RuntimeError', 'invalid configuration in manifest'):
        assert not worker.resource_limit(RuntimeError(message))


@pytest.mark.parametrize('artifact', ['input', 'reference'])
def test_reject_tampered_data_hash(manifest, artifact):
    path, _ = manifest
    target = path.parent / (artifact + '.npy')
    with target.open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match=artifact + '_file hash mismatch'):
        worker.load_task(path, 'test')
