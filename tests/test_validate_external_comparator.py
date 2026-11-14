"""CPU/mock checks for the external-only validation entry point."""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import sys
import types
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
m=importlib.import_module('validate_external_comparator')


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    s=dict(schema='cubutterfly-semantic-v1',operator=0,storage=7,compute=2,
        shape=[4],batch=2,inverse=1,normalize_inverse=True,placement=0,fft_type=0,
        fft_output_scale=1.,length_mode=0,input_strides=[2],output_strides=[3],
        input_batch_stride=11,output_batch_stride=13,input_orders=[0],output_orders=[0],
        ntt_input_layout=0,ntt_output_layout=0,modulus=None,word_bits=None,matrices=[[]])
    config=dict(library='/never-load-own.so',gpu_uuid='GPU-12345678-1234-1234-1234-123456789abc',
        environment={},cases=[dict(id='fft',descriptor=dict(semantics=s),baselines=['gpu-fft'])])
    directory=tmp_path/'prepared'
    m.campaign.prepare(config,directory)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',config['gpu_uuid'])
    return directory,tmp_path/'external-validation',config


@pytest.fixture
def runtime(tmp_path,monkeypatch):
    props=types.SimpleNamespace(uuid='GPU-12345678-1234-1234-1234-123456789abc',name='mock',total_memory=1)
    state=types.SimpleNamespace(current=False,calls=[])
    def sync():
        assert state.current
    stream=types.SimpleNamespace(synchronize=sync)
    @contextmanager
    def context(selected):
        assert selected is stream
        state.current=True
        try:
            yield
        finally:
            state.current=False
    class Host:
        def __init__(self,array):
            self.array=array
        def to(self,device):
            assert state.current and device=='cuda'
            return self.array
    torch=types.SimpleNamespace(__version__='mock',version=types.SimpleNamespace(cuda='mock'),
        from_numpy=Host,cuda=types.SimpleNamespace(get_device_properties=lambda _:props,
            Stream=lambda:stream,stream=context))
    monkeypatch.setitem(sys.modules,'torch',torch)
    library=tmp_path/'external.so';library.write_bytes(b'cpu-placeholder')
    monkeypatch.setitem(sys.modules,'gpufft_baseline',types.SimpleNamespace(DEFAULT_LIBRARY=library,__file__=__file__))
    monkeypatch.setattr(m.runner,'exclusive_gpu',lambda uuid:None)
    def execute(case,torch,stream,source,expected,input_saved,output_initial,**kwargs):
        assert state.current
        assert kwargs['timing'] is True and kwargs['baseline_names']==['gpu-fft']
        assert source.dtype==expected.dtype==np.complex128
        assert input_saved.nbytes==18*16
        assert output_initial.nbytes==23*16
        state.calls.append(case['id'])
        return [dict(name='gpu-fft',status='accuracy-review-required',correctness=dict(l2_relative=1e-15),
                     timing=dict(median_ms=.01,protocol='20 complete calls per CUDA graph; 7 replays; 10 warmup'))]
    monkeypatch.setattr(m.runner,'run_baselines',execute)
    return state


def test_external_only_stream_graph_protocol_and_resume(prepared,runtime):
    directory,output,_=prepared
    assert m.validate(directory,'gpu-fft',output)==0
    envelope=json.loads((output/'rows/fft.json').read_text())
    row=envelope['row']
    assert row['purpose']=='validation-only'
    assert row['graph_status']=='direct-and-graph-replay-verified'
    result=row['baselines'][0]
    assert 'CUDA graph' in result['validation_timing_diagnostic']['protocol']
    assert 'timing' not in result
    assert envelope['guards']==dict(before=True,after=True)
    assert 'main' not in envelope['identity']['library']
    assert m.validate(directory,'gpu-fft',output)==0
    assert runtime.calls==['fft']


@pytest.mark.parametrize('busy_at',[1,2,3])
def test_busy_guards_never_publish_row(prepared,runtime,monkeypatch,busy_at):
    directory,output,_=prepared
    calls=[]
    def guard(uuid):
        calls.append(uuid)
        if len(calls)==busy_at:
            raise RuntimeError('busy')
    monkeypatch.setattr(m.runner,'exclusive_gpu',guard)
    assert m.validate(directory,'gpu-fft',output)==76
    assert not (output/'rows/fft.json').exists()
    assert json.loads((output/'pending.json').read_text())['pending']==['fft']


def test_independent_output_and_declared_backend_required(prepared):
    directory,output,_=prepared
    with pytest.raises(ValueError,match='independent'):
        m.validate(directory,'gpu-fft',directory/'new-output')
    with pytest.raises(ValueError,match='not declared'):
        m.validate(directory,'hadacore',output)


def test_library_change_blocks_resume(prepared,runtime):
    directory,output,_=prepared
    assert m.validate(directory,'gpu-fft',output)==0
    Path(sys.modules['gpufft_baseline'].DEFAULT_LIBRARY).write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity mismatch'):
        m.validate(directory,'gpu-fft',output)


def test_physical_buffers_preserve_stride_and_inplace_alias(prepared):
    directory,_,_=prepared
    _,loaded=m.campaign.load_prepared(directory)
    payload=loaded['fft']
    source,output=m.physical_buffers(payload)
    assert source.nbytes==18*16 and output.nbytes==23*16
    np.testing.assert_array_equal(source.view(np.complex128)[payload['input_indices']],payload['source'].reshape(-1))
    payload['s']['placement']=1
    source,output=m.physical_buffers(payload)
    assert source.nbytes==output.nbytes==23*16
    np.testing.assert_array_equal(source,output)


def test_unsupported_metadata_is_retained(prepared,runtime,monkeypatch):
    directory,output,_=prepared
    monkeypatch.setattr(m.runner,'run_baselines',lambda *a,**kw:[dict(name='gpu-fft',status='missing-baseline',
        support_status='unsupported-semantic',support_metadata=dict(rank=2),reason='unsupported rank')])
    assert m.validate(directory,'gpu-fft',output)==0
    row=json.loads((output/'rows/fft.json').read_text())['row']
    assert row['baselines'][0]['support_metadata']==dict(rank=2)
    assert row['graph_status']=='not-completed'


def test_event_only_protocol_is_not_claimed_as_graph_coverage(prepared,runtime,monkeypatch):
    directory,output,_=prepared
    monkeypatch.setattr(m.runner,'run_baselines',lambda *a,**kw:[dict(name='gpu-fft',status='accuracy-review-required',
        correctness=dict(l2_relative=1e-15),timing=dict(median_ms=.01,protocol='7 single event calls; reset enqueued before start event; 10 warmup'))])
    assert m.validate(directory,'gpu-fft',output)==0
    row=json.loads((output/'rows/fft.json').read_text())['row']
    assert row['status']=='graph-validation-gap'
    assert row['graph_status']=='not-covered-by-recorded-protocol'
    assert 'single event' in row['baselines'][0]['validation_timing_diagnostic']['protocol']


def test_initial_busy_before_torch_import(prepared,monkeypatch):
    import builtins
    directory,output,_=prepared
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name=='torch':
            raise AssertionError('Torch imported while GPU busy')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    def busy(uuid):
        raise RuntimeError('occupied')
    monkeypatch.setattr(m.runner,'exclusive_gpu',busy)
    assert m.validate(directory,'gpu-fft',output)==76
