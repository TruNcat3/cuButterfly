"""CPU preparation and mocked execution safety; no CUDA runtime is used."""
import copy
import importlib
import json
from pathlib import Path
import sys
import types
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
m=importlib.import_module('run_comparator_extension')


def config():
    s=dict(schema='cubutterfly-semantic-v1',operator=0,storage=6,compute=1,
        shape=[4],batch=2,inverse=0,normalize_inverse=False,placement=0,fft_type=0,
        fft_output_scale=1.,length_mode=0,input_strides=[1],output_strides=[1],
        input_batch_stride=4,output_batch_stride=4,input_orders=[0],output_orders=[0],
        ntt_input_layout=0,ntt_output_layout=0,modulus=None,word_bits=None,matrices=[[]])
    return dict(library='/missing/lib.so',gpu_uuid='GPU-12345678-1234-1234-1234-123456789abc',
                environment={},cases=[dict(id='fft',descriptor=dict(semantics=s),baselines=['gpu-fft'])])


def test_prepare_roundtrip_dtype_binding_and_diagnostic(tmp_path):
    manifest=m.prepare(config(),tmp_path)
    assert manifest['cases'][0]['reference']['kind']=='scipy-pocketfft'
    _,loaded=m.load_prepared(tmp_path)
    p=loaded['fft']
    assert p['source'].dtype==p['expected'].dtype==np.complex64
    assert p['diagnostic_expected'].dtype==np.complex128
    path=tmp_path/manifest['cases'][0]['path']
    path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError,match='checksum'):
        m.load_prepared(tmp_path)


def test_semantic_tamper_is_rejected(tmp_path):
    m.prepare(config(),tmp_path)
    path=tmp_path/'preparation.json'
    data=json.loads(path.read_text())
    data['cases'][0]['semantics_sha256']='bad'
    m.write(path,data)
    with pytest.raises(ValueError,match='binding'):
        m.load_prepared(tmp_path)


def test_fp16_fwht_reference_rounds_each_stage():
    c=config()['cases'][0]
    s=c['descriptor']['semantics']
    s.update(operator=2,storage=0,compute=0,shape=[4])
    a=np.array([[2048,1,1,1]],dtype=np.float16)
    result,info=m.primary_reference(dict(s=s,source=a,expected=a.astype(np.float64)))
    assert result.dtype==np.float16
    # First add rounds 2048+1 to 2048, before the second stage.
    assert result[0,0]==2050
    assert info['kind']=='FP16-per-stage-rounded-Sylvester'


def test_reference_bound_keeps_gap(tmp_path):
    c=config();c['prepare_reference_limit']=1
    manifest=m.prepare(c,tmp_path)
    assert manifest['cases'][0]['gap_status']=='validation-unavailable'
    assert m.load_prepared(tmp_path)[1]=={}


def fake_runtime(monkeypatch):
    monkeypatch.setattr(m.runner,'exclusive_gpu',lambda uuid:None)
    props=types.SimpleNamespace(uuid=config()['gpu_uuid'],name='mock',total_memory=1,major=8,minor=0)
    from contextlib import contextmanager
    stream=types.SimpleNamespace(current=False)
    def synchronize():
        assert stream.current, 'synchronize must use campaign stream context'
    stream.synchronize=synchronize
    @contextmanager
    def current_stream(selected):
        assert selected is stream
        stream.current=True
        try:
            yield
        finally:
            stream.current=False
    torch=types.SimpleNamespace(__version__='mock',version=types.SimpleNamespace(cuda='mock'),
        cuda=types.SimpleNamespace(get_device_properties=lambda _:props,Stream=lambda:stream,stream=current_stream))
    cb=types.SimpleNamespace(__file__=__file__,Library=lambda p:object())
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'cubutterfly',cb)
    return stream


@pytest.mark.parametrize('busy_guard',[1,2,3])
def test_busy_never_publishes_complete_row(tmp_path,monkeypatch,busy_guard):
    m.prepare(config(),tmp_path)
    fake_runtime(monkeypatch)
    guards=[]
    def guard(uuid):
        guards.append(uuid)
        if len(guards)==busy_guard:
            raise RuntimeError('busy')
    monkeypatch.setattr(m.runner,'exclusive_gpu',guard)
    monkeypatch.setattr(m.runner,'run_case',lambda *a,**k:dict(id='fft',status='pass'))
    assert m.run(tmp_path)==76
    assert not (tmp_path/'rows/fft.json').exists()
    assert json.loads((tmp_path/'pending.json').read_text())['pending']==['fft']


def test_fresh_pair_and_identical_resume(tmp_path,monkeypatch):
    m.prepare(config(),tmp_path)
    campaign_stream=fake_runtime(monkeypatch)
    monkeypatch.setattr(m.runner,'exclusive_gpu',lambda uuid:None)
    calls=[]
    def execute(*args,**kwargs):
        assert campaign_stream.current, 'runner must execute under the explicit campaign stream'
        calls.append(kwargs)
        assert kwargs['prepared']['expected'].dtype==np.complex64
        assert kwargs['selection_policy']==0 and kwargs['timing'] is True
        return dict(id='fft',status='accuracy-review-required',mapping=dict(actual='fresh'),baselines=[dict(status='unsupported')])
    monkeypatch.setattr(m.runner,'run_case',execute)
    assert m.run(tmp_path)==0
    row=json.loads((tmp_path/'rows/fft.json').read_text())
    assert row['guards']==dict(before=True,after=True)
    assert row['row']['mapping']==dict(actual='fresh')
    assert m.run(tmp_path)==0
    assert len(calls)==1
    row['identity_sha256']='wrong'
    m.write(tmp_path/'rows/fft.json',row)
    with pytest.raises(ValueError,match='identity'):
        m.run(tmp_path)


def test_prepare_does_not_import_torch(tmp_path):
    import subprocess
    code = ("import sys; import run_comparator_extension as m; "
            f"m.prepare({config()!r}, {str(tmp_path)!r}); "
            "assert 'torch' not in sys.modules")
    subprocess.run([sys.executable,'-c',code],cwd=Path(m.__file__).parent,check=True)


def test_wrong_uuid_rejects_execution(tmp_path,monkeypatch):
    m.prepare(config(),tmp_path)
    fake_runtime(monkeypatch)
    monkeypatch.setattr(m.runner,'same_gpu_uuid',lambda *a:False)
    with pytest.raises(ValueError,match='UUID'):
        m.run(tmp_path)
    assert not (tmp_path/'rows').exists()


@pytest.mark.parametrize('names', [['gpufft'], ['gpu-fft','gpu-fft'], []])
def test_prepare_rejects_invalid_baselines(tmp_path,names):
    c=config();c['cases'][0]['baselines']=names
    with pytest.raises(ValueError,match='baseline names'):
        m.prepare(c,tmp_path)
    assert not (tmp_path/'preparation.json').exists()


def test_environment_identity_excludes_unrelated_and_sensitive_inherited_values(monkeypatch):
    monkeypatch.setenv('SECRET_ACCESS_TOKEN','do-not-record')
    monkeypatch.setenv('UNRELATED_SETTING','ignore')
    monkeypatch.setenv('OMP_NUM_THREADS','4')
    identity=m.environment_identity({'EXPLICIT_SETTING':'chosen'})
    assert 'SECRET_ACCESS_TOKEN' not in identity['effective']
    assert 'UNRELATED_SETTING' not in identity['effective']
    assert identity['effective']['OMP_NUM_THREADS']=='4'
    assert identity['configured']=={'EXPLICIT_SETTING':'chosen'}


def test_fingerprints_bind_only_selected_dependencies():
    files=m.source_fingerprints(config())
    assert 'gpufft_baseline.py' in files
    assert 'cpu_external_baselines.py' in files
    assert 'hadacore_baseline.py' not in files
    assert 'benchmark_external_fht.py' not in files


def test_initial_busy_check_precedes_torch_import(tmp_path,monkeypatch):
    import builtins
    m.prepare(config(),tmp_path)
    original=builtins.__import__
    def guarded_import(name,*args,**kwargs):
        if name=='torch':
            raise AssertionError('Torch imported before initial exclusivity check')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded_import)
    def busy(uuid):
        raise RuntimeError('occupied')
    monkeypatch.setattr(m.runner,'exclusive_gpu',busy)
    assert m.run(tmp_path)==76
    pending=json.loads((tmp_path/'pending.json').read_text())
    assert pending['phase']=='before-torch-import'
    assert pending['pending']==['fft']


def bound_config(tmp_path):
    c=config()
    library=tmp_path/'native.so';library.write_bytes(b'native')
    c['library']=str(library)
    profiles=tmp_path/'profiles';profiles.mkdir()
    provider=tmp_path/'provider.so';provider.write_bytes(b'provider')
    profile=profiles/'model.json'
    profile.write_text(json.dumps({'provider_library':str(provider)}))
    c['environment']['CUBUTTERFLY_PROFILE_ROOT']=str(profiles)
    c['runtime_bindings']={}
    return c,profile,provider


@pytest.mark.parametrize('change',['tamper','missing','new-json'])
def test_runtime_bindings_reject_before_gpu(tmp_path,monkeypatch,change):
    c,profile,provider=bound_config(tmp_path)
    m.prepare(c,tmp_path)
    if change=='tamper':
        provider.write_bytes(b'changed')
    elif change=='missing':
        provider.unlink()
    else:
        (profile.parent/'extra.json').write_text('{}')
    def forbidden(*args,**kwargs):
        pytest.fail('GPU guard reached before runtime binding verification')
    monkeypatch.setattr(m.runner,'exclusive_gpu',forbidden)
    with pytest.raises(ValueError,match='runtime binding'):
        m.run(tmp_path)
    assert not (tmp_path/'rows').exists()


def test_runtime_bindings_midcase_change_never_publishes(tmp_path,monkeypatch):
    c,profile,provider=bound_config(tmp_path)
    m.prepare(c,tmp_path)
    fake_runtime(monkeypatch)
    def execute(*args,**kwargs):
        profile.write_text('{}')
        return dict(id='fft',status='pass')
    monkeypatch.setattr(m.runner,'run_case',execute)
    with pytest.raises(ValueError,match='runtime binding'):
        m.run(tmp_path)
    assert not (tmp_path/'rows/fft.json').exists()
    assert json.loads((tmp_path/'pending.json').read_text())['pending']==['fft']


def test_runtime_bindings_identity_and_supplied_hash_validation(tmp_path,monkeypatch):
    c,profile,provider=bound_config(tmp_path)
    c['runtime_bindings']={str(provider):'wrong'}
    with pytest.raises(ValueError,match='runtime binding checksum'):
        m.prepare(c,tmp_path)
    c['runtime_bindings']={}
    manifest=m.prepare(c,tmp_path)
    bindings=manifest['config']['runtime_bindings']
    assert bindings[str(provider)]==m.sha(provider)
    fake_runtime(monkeypatch)
    monkeypatch.setattr(m.runner,'run_case',lambda *a,**k:dict(id='fft',status='pass'))
    assert m.run(tmp_path)==0
    assert json.loads((tmp_path/'rows/fft.json').read_text())['identity']['runtime_bindings']==bindings
    assert m.run(tmp_path)==0


def test_runtime_bindings_ignore_historical_json_provenance(tmp_path):
    c,profile,provider=bound_config(tmp_path)
    profile.write_text(json.dumps({'provenance':'/sdb/old/nonexistent.json',
                                  'native_rows':{'library':str(provider)}}))
    manifest=m.prepare(c,tmp_path)
    assert str(provider) in manifest['config']['runtime_bindings']
    assert '/sdb/old/nonexistent.json' not in manifest['config']['runtime_bindings']


@pytest.mark.parametrize('root',['/sdb','/mnt/sdb'])
def test_runtime_bindings_reject_forbidden_provider_before_access(tmp_path,root):
    c,profile,provider=bound_config(tmp_path)
    profile.write_text(json.dumps({'provider_library':root+'/provider.so'}))
    with pytest.raises(ValueError,match='outside allowed storage'):
        m.prepare(c,tmp_path)
