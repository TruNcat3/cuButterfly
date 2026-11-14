import json, importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v',ROOT/'scripts/validate_hadacore_tile.py'); v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)

def test_quick_matrix_and_same_precision_reference(tmpdir):
    r=v.prepare(v.QUICK,Path(str(tmpdir)),device_uuid='GPU-test')
    assert len(r['cases'])==9
    z=np.load(Path(str(tmpdir))/'N256xB1.npz'); assert z['input'].dtype==np.float16 and z['reference'].dtype==np.float16
    assert r['cases'][0]['reference']=='radix2-stage-FP16'

def test_run_is_pending_without_execute(tmpdir):
    v.prepare([(256,1)],Path(str(tmpdir)),device_uuid='GPU-test')
    row=v.run_one(256,1,output=Path(str(tmpdir)),device_uuid='GPU-test')
    assert row['status']=='queued' and row['measurement'].startswith('pending')
    assert row['gpu_uuid']=='GPU-test'

def test_rejects_other_gpu_and_invalid_case():
    try: v.parse_case('N3xB1')
    except ValueError: pass
    else: assert False

def test_prepared_artifact_tampering_is_rejected_before_cuda(tmp_path):
    import pytest
    build=tmp_path/'build';build.mkdir()
    artifact=build/'dummy.bin';artifact.write_bytes(b'compiled')
    (build/'build.json').write_text(json.dumps({'artifacts':{'dummy.bin':v.sha(artifact)}}))
    output=tmp_path/'output'
    v.prepare([(256,1)],output,build=build,device_uuid='GPU-test')
    row,x,y,record=v.validate_inputs(256,1,output,build,'GPU-test')
    assert x.shape==y.shape==(1,256)
    with pytest.raises(ValueError,match='target UUID'):
        v.validate_inputs(256,1,output,build,'GPU-other')
    artifact.write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact hash mismatch'):
        v.validate_inputs(256,1,output,build,'GPU-test')


def test_prepare_marks_cpu_reference_different_from_mma_contract(tmp_path):
    result=v.prepare([(256,1)],tmp_path,device_uuid='GPU-test')
    assert result['contract']==v.CONTRACT
    assert result['cases'][0]['reference']=='radix2-stage-FP16'
    assert not result['gpu_execution']


def test_portable_build_and_required_target(tmp_path, monkeypatch):
    import pytest
    monkeypatch.delenv('CUBUTTERFLY_GPU_UUID', raising=False)
    assert v.BUILD == ROOT / 'build' / 'hadacore_tile'
    with pytest.raises(ValueError, match='target GPU UUID required'):
        v.prepare([(256,1)], tmp_path / 'uncreated')
    assert not (tmp_path / 'uncreated').exists()
    with pytest.raises(ValueError, match='target GPU UUID required'):
        v.run_one(256,1, output=tmp_path, execute=True)
    monkeypatch.setenv('CUBUTTERFLY_GPU_UUID', 'GPU-environment')
    assert v.prepare([(256,1)], tmp_path)['gpu_uuid'] == 'GPU-environment'
    assert v.target_uuid('GPU-explicit') == 'GPU-explicit'


def test_cli_requires_target_before_prepare_or_execution(tmp_path, monkeypatch):
    import pytest
    import sys
    monkeypatch.delenv('CUBUTTERFLY_GPU_UUID', raising=False)
    for mode in ([], ['--prepare'], ['--run', '--execute', '--case', 'N256xB1']):
        monkeypatch.setattr(sys, 'argv', ['validate_hadacore_tile.py', *mode,
                                         '--output', str(tmp_path / 'uncreated')])
        with pytest.raises(SystemExit) as exc:
            v.main()
        assert exc.value.code == 2
    assert not (tmp_path / 'uncreated').exists()
