import importlib.util
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import calibrate_hadacore_native as n
import validate_hadacore_tile as v


def fixture(tmp_path):
    prepared=tmp_path/'prepared';v.prepare([(256,1)],prepared,device_uuid='GPU-test')
    output=tmp_path/'native';lib=tmp_path/'lib.so';lib.write_bytes(b'compiled')
    r=n.identity(prepared,256,1,lib)
    err=tmp_path/'errors.npz';err.write_bytes(b'full-errors')
    r.update(status='measured-error-report',finite=True,exact_repeat=True,input_immutable=True,buffer_guards=True,
             graph_reproduced=True,guards={'before':True,'after':True},error_artifact=str(err),error_artifact_sha256=n.sha(err),
             mapping={'kind':'butterfly','tile_threads':128},gpu_identity={'uuid':'GPU-test','sm':80},
             timing={'median_ms':.01,'samples_ms':[.01]*7,'protocol':'20 complete calls per CUDA graph; 7 replays; 10 warmup'})
    n.write(n.result_path(output,256,1),r)
    return prepared,output,lib,r


def test_export_uses_native_evidence_and_protocol(tmp_path):
    prepared,out,lib,row=fixture(tmp_path)
    p=n.export(prepared,out,lib,tmp_path/'native-profile.json')
    assert p['native_rows'][0]['median_ms']==.01
    assert p['native_rows'][0]['timing_protocol']==row['timing']['protocol']
    assert p['native_rows'][0]['mapping']==row['mapping']
    assert not p['pending']


def test_changed_binary_does_not_reuse(tmp_path):
    prepared,out,lib,row=fixture(tmp_path);lib.write_bytes(b'changed')
    with pytest.raises(ValueError,match='library_sha256'):n.checked(prepared,out,256,1,lib)


def test_nonfinite_or_missing_guards_never_export(tmp_path):
    prepared,out,lib,row=fixture(tmp_path);row['guards']['after']=False
    n.write(n.result_path(out,256,1),row)
    with pytest.raises(ValueError,match='no current'):n.export(prepared,out,lib,tmp_path/'profile.json')
    assert not (tmp_path/'profile.json').exists()


def test_current_complete_rows_reused_without_gpu(tmp_path,monkeypatch):
    prepared,out,lib,row=fixture(tmp_path)
    def forbidden(*a,**kw):raise AssertionError('must reuse native row')
    monkeypatch.setattr(n.subprocess,'run',forbidden)
    state=n.execute(prepared,out,lib)
    assert state['status']=='complete' and state['completed']==1
