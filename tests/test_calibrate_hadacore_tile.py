import importlib.util
import json
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import calibrate_hadacore_tile as c


def fixture(tmp_path):
    build=tmp_path/'build';build.mkdir();lib=build/'libhadacore_tile.so';lib.write_bytes(b'compiled')
    c.write(build/'build.json',{'artifacts':{'libhadacore_tile.so':c.digest(lib)}})
    out=tmp_path/'out';prep=c.validation.prepare([(256,1)],out,build,device_uuid='GPU-test')
    task=prep['gpu_tasks'][0];path=c.result_path(out,task)
    errors=out/'errors.npz';errors.write_bytes(b'error-payload')
    row=dict(status='validated-exact-upstream',case=task['case'],groups_per_block=1,gpu_uuid='GPU-test',
        contract=c.validation.CONTRACT,build_sha256=c.digest(build/'build.json'),provider_sha256=c.digest(c.validation.PROVIDER),
        runner_sha256=c.digest(c.validation.__file__),payload_sha256=prep['cases'][0]['artifact_sha256'],
        library=str(lib.resolve()),library_sha256=c.digest(lib),guards={'before':True,'after':True},
        error_artifact=str(errors),error_artifact_sha256=c.digest(errors),
        gpu_identity=dict(uuid='GPU-test',sm=80,memory_bytes=80000000000,name='test A100'),
        resources={k:32 for k in c.RESOURCE_FIELDS},mapping=dict(log_n=8,tile_count=1,groups_per_block=1,chunks_per_warp=1),
        timing=dict(median_ms=.01,samples_ms=[.01]*7,protocol='test-graph-protocol'))
    for key in ('exact_repeat','input_immutable','buffer_guards','graph_reproduced','upstream_graph_reproduced','exact_upstream_match','finite'):row[key]=True
    c.write(path,row)
    return out,build,task,path,row


def test_partial_export_never_fabricates_pending_samples(tmp_path):
    out,build,task,path,row=fixture(tmp_path)
    profile=c.export_profile(out,build,out/'profile.json')
    assert len(profile['rows'])==1 and len(profile['pending'])==2
    assert profile['rows'][0]['median_ms']==.01
    assert not profile['complete']
    assert profile['rows'][0]['evidence_sha256']==c.digest(path)


def test_unsafe_or_stale_results_do_not_export(tmp_path):
    out,build,task,path,row=fixture(tmp_path)
    row['exact_upstream_match']=False;c.write(path,row)
    with pytest.raises(ValueError,match='no validated'):c.export_profile(out,build,out/'profile.json')
    assert not (out/'profile.json').exists()
    row['exact_upstream_match']=True;row['payload_sha256']='stale';c.write(path,row)
    with pytest.raises(ValueError,match='identity mismatch'):c.checked_row(out,build,task,'GPU-test')


def test_execution_stops_on_existing_invalid_without_launch(tmp_path,monkeypatch):
    out,build,task,path,row=fixture(tmp_path)
    row['status']='failed';c.write(path,row)
    def forbidden(*a,**kw):raise AssertionError('subprocess must not launch')
    monkeypatch.setattr(c.subprocess,'run',forbidden)
    r=c.execute_population(out,build,'unused')
    assert r['status']=='blocked-existing-result' and r['completed']==0
    assert c.read(path)['status']=='failed'


def test_execution_reuses_current_result_then_stops_on_subprocess_failure(tmp_path,monkeypatch):
    from types import SimpleNamespace
    out,build,task,path,row=fixture(tmp_path);calls=[]
    def fail(cmd,**kwargs):calls.append(cmd);return SimpleNamespace(returncode=1)
    monkeypatch.setattr(c.subprocess,'run',fail)
    r=c.execute_population(out,build,'unused')
    assert r['completed']==1 and r['status']=='stopped' and len(calls)==1
    assert c.read(path)==row


def test_native_profile_requires_matching_device_and_binary(tmp_path):
    out,build,task,path,row=fixture(tmp_path)
    native=out/'native.json'
    c.write(native,dict(schema='cubutterfly-hadacore-native-profile-v1',gpu_identity={'uuid':'GPU-other'}))
    with pytest.raises(ValueError,match='hardware/schema'):
        c.export_profile(out,build,out/'profile.json',native)
    c.write(native,dict(schema='cubutterfly-hadacore-native-profile-v1',gpu_identity=row['gpu_identity'],
                        library=str(build/'libhadacore_tile.so'),library_sha256='stale'))
    with pytest.raises(ValueError,match='library changed'):
        c.export_profile(out,build,out/'profile.json',native)
    assert not (out/'profile.json').exists()
