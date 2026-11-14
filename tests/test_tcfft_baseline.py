import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tcfft_baseline", ROOT / "scripts/tcfft_baseline.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


@pytest.mark.parametrize("n,batch", [(256, 32), (512, 16), (1024, 8), (131072, 1),
                                    (262144, 1), (524288, 1), (16777216, 1),
                                    (33554432, 1), (67108864, 1), (134217728, 1)])
def test_explicit_supported_plans(n, batch):
    assert adapter.support(n, batch)["supported"]


@pytest.mark.parametrize("n,batch", [(256, 1), (512, 1), (1024, 1), (4096, 2),
                                    (65536, 1), (131072, 0), (16777216, 256),
                                    (134217728, 16)])
def test_unsafe_or_uninitialized_plans_rejected(n, batch):
    assert not adapter.support(n, batch)["supported"]


@pytest.mark.parametrize("options", [{"precision": "bf16"}, {"accumulation": "fp32"},
                                     {"direction": "inverse"}, {"normalization": "inverse"}, {"rank": 2}])
def test_semantic_substitution_rejected(options):
    assert not adapter.support(131072, 1, **options)["supported"]


def test_codegen_preserves_device_arithmetic_and_uses_stream():
    upstream = ROOT / "external/tcFFT/tcfft_half.cu"
    if not upstream.exists():
        pytest.skip("pinned optional upstream checkout absent")
    original = upstream.read_text()
    generated = adapter.transformed_source(original)
    assert generated.split("void tcfftExecStream(")[0] == original.split("void tcfftExec(")[0]
    assert generated.split("void tcfftCreate(")[1] == original.split("void tcfftCreate(")[1]
    assert generated.count(", stream>>>") == 2
    assert "cudaFuncSetAttribute(" not in generated
    with pytest.raises(ValueError, match="pinned upstream"):
        adapter.transformed_source(original + "\n")


def test_support_check_does_not_load_cuda(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("CUDA library load during CPU preparation")
    monkeypatch.setattr(adapter.ctypes, "CDLL", fail)
    assert adapter.support(131072, 1)["supported"]
    with pytest.raises(ValueError):
        adapter.Plan("unused", 256, 1)


def comparison_module():
    import sys
    sys.path.insert(0, str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('run_tcfft_comparison', ROOT / 'scripts/run_tcfft_comparison.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_matrix_keeps_unsupported_points():
    module = comparison_module()
    assert len(module.CASES) == 9
    assert [(n, b) for n, b in module.CASES if adapter.support(n, b)['supported']] == [(262144, 4), (262144, 32)]


def test_preparation_keeps_frozen_inputs_and_no_mapping(monkeypatch, tmp_path):
    module = comparison_module()
    config = dict(public_manifest='public.json',tcfft_library='tc.so',cufft_library='cu.so',gpu_uuid='GPU-test')
    cases = [dict(id=f'n{n}-b{b}', n=n,batch=b,storage=4,compute=0,inverse=False,
                  normalize_inverse=False,input_file='input.npy',input_sha256='fixed',
                  reference_file='reference.npy',reference_sha256='fixed') for n,b in module.CASES]
    public = dict(cases=cases,library='lib.so')
    monkeypatch.setattr(module, 'read', lambda path: config if str(path)=='config.json' else public)
    monkeypatch.setattr(module, 'sha', lambda path: 'fixed')
    monkeypatch.setattr(module.tcfft.ctypes, 'CDLL', lambda *a: pytest.fail('CPU preparation loaded CUDA'))
    prepared=[]
    def reference(e, output):
        prepared.append(e['id'])
        e.update(rounded_reference_file='rounded.npy',rounded_reference_sha256='fixed')
    monkeypatch.setattr(module,'prepare_rounded_reference',reference)
    result=module.prepare('config.json',tmp_path)
    assert prepared == ['n262144-b4','n262144-b32']
    assert not result['gpu_execution']
    assert len(result['cases']) == 9
    assert all(e['input_sha256']=='fixed' and 'mapping' not in e for e in result['cases'])
    monkeypatch.setattr(module,'read',lambda path: result if str(path)=='cached.json' else
                        config if str(path)=='config.json' else public)
    reused=module.prepare('config.json',tmp_path,reference_manifest='cached.json')
    assert len(prepared)==2  # Compatible references did not invoke CPU recomputation.
    assert reused['cases'][-1]['rounded_reference_file']=='rounded.npy'
    result['cases'][-1]['input_sha256']='different'
    with pytest.raises(ValueError,match='cached reference semantics changed'):
        module.prepare('config.json',tmp_path,reference_manifest='cached.json')
    cases[0]['compute']=1
    with pytest.raises(ValueError,match='semantics'):
        module.prepare('config.json',tmp_path)


def test_diagnostic_reports_exact_byte_differences():
    import numpy as np
    spec=importlib.util.spec_from_file_location('diagnose_tcfft_execution',ROOT/'scripts/diagnose_tcfft_execution.py')
    diagnostic=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnostic)
    report=diagnostic.byte_difference(np.array([1,2,3],np.uint8),np.array([1,4,5],np.uint8),limit=1)
    assert report == dict(equal=False,different_bytes=2,first_positions=[1],actual_bytes=[2],expected_bytes=[4])
    assert diagnostic.byte_difference(np.zeros(2,np.float16),np.zeros(2,np.float16))['equal']
    with pytest.raises(ValueError,match='shape mismatch'):
        diagnostic.byte_difference(np.zeros(2,np.uint8),np.zeros(3,np.uint8))


def test_sm80_bridge_selects_upstream_ampere_fragment_layouts():
    bridge=(ROOT/'benchmarks/tcfft/bridge.cuh').read_text()
    assert 'properties.major != 8' in bridge
    for name in ('256_0','512_0','1024_0','256_1','512_1'):
        assert f'= layer_{name}_A100<' in bridge
    assert adapter.support(262144,4)['device_major_required']==8


def test_comparison_cpu_reference_cached_and_bound(tmp_path):
    import numpy as np
    module=comparison_module()
    source=tmp_path/'input.npy'
    np.save(source,np.array([[[1.,0.],[0.,0.],[0.,0.],[0.,0.]]],np.float16))
    entry=dict(id='impulse',input_file=str(source),batch=1,n=4)
    module.prepare_rounded_reference(entry,tmp_path)
    cached=Path(entry['rounded_reference_file'])
    assert entry['rounded_reference_sha256']==module.sha(cached)
    assert np.array_equal(np.load(cached),np.ones((1,4),np.complex64))
    module.validate(dict(bindings={str(cached):entry['rounded_reference_sha256']}))
    cached.write_bytes(b'changed')
    with pytest.raises(ValueError,match='binding changed'):
        module.validate(dict(bindings={str(cached):entry['rounded_reference_sha256']}))


def test_comparison_unsupported_case_uses_no_gpu(tmp_path, monkeypatch):
    module=comparison_module()
    monkeypatch.setattr(module,'validate',lambda m:None)
    entry=dict(id='unsupported',tcfft_support=dict(supported=False,reasons=['length']))
    module.run(dict(config={},runtime={},cases=[entry]),tmp_path,'unsupported')
    assert module.read(tmp_path/'unsupported/comparison.json')['gpu_execution'] is False


def test_comparison_mismatch_preserves_stage_and_full_output(tmp_path):
    import numpy as np
    module=comparison_module()
    actual=np.zeros(100,np.uint8);expected=actual.copy();actual[2:90]=3
    report=module.record_byte_mismatch(actual,expected,tmp_path,'tcfft','recreated-eager')
    assert report['stage']=='recreated-eager'
    assert report['different_bytes']==88
    assert report['first_positions']==list(range(2,66))
    assert np.array_equal(np.load(report['actual_file']),actual)
    assert np.array_equal(np.load(report['expected_file']),expected)
    assert module.read(tmp_path/'tcfft-recreated-eager-mismatch.json')==report


@pytest.mark.parametrize('backends', [(), ('tcfft','tcfft'), ('unknown',)])
def test_comparison_rejects_invalid_backend_selection_before_loading_cuda(tmp_path, backends):
    module=comparison_module()
    with pytest.raises(ValueError,match='nonempty unique subset'):
        module.run({},tmp_path,'case',backends=backends)


def test_comparison_cli_forwards_isolated_diagnostic_without_loading_cuda(monkeypatch,tmp_path):
    import sys
    module=comparison_module()
    manifest={'prepared': True}
    monkeypatch.setattr(module,'read',lambda path:manifest)
    calls=[]
    monkeypatch.setattr(module,'run',lambda *args,**kwargs:calls.append((args,kwargs)))
    monkeypatch.setattr(sys,'argv',['run_tcfft_comparison.py','--run','--output',str(tmp_path),
        '--case','n262144-b4','--backends','tcfft','cufft','--diagnostic-only'])
    module.main()
    assert calls==[((manifest,tmp_path,'n262144-b4'),
                    dict(backends=['tcfft','cufft'],diagnostic_only=True,synchronize_setup=False))]


@pytest.mark.parametrize('verification_fails', [False, True])
def test_capture_releases_graph_before_plan_close_even_on_error(verification_fails):
    from contextlib import contextmanager
    from types import SimpleNamespace
    module=comparison_module()
    events=[]
    class Graph:
        def replay(self): events.append('replay')
        def __del__(self): events.append('destroy-graph')
    @contextmanager
    def capture(graph, stream):
        events.append('capture')
        yield
        events.append('end-capture')
    torch=SimpleNamespace(cuda=SimpleNamespace(CUDAGraph=Graph,graph=capture))
    stream=SimpleNamespace(synchronize=lambda:events.append('synchronize'))
    def verify():
        events.append('verify')
        if verification_fails: raise ValueError('injected mismatch')
    try:
        module.verify_capture(torch,stream,lambda:events.append('execute'),
                              lambda:events.append('poison'),verify)
    except ValueError as error:
        assert verification_fails and str(error)=='injected mismatch'
    finally:
        events.append('close-plan')
    assert events==['capture','execute','end-capture','poison','replay',
                    'synchronize','verify','destroy-graph','close-plan']


@pytest.mark.parametrize('enabled', [False, True])
def test_setup_sync_is_explicit_and_optional(enabled):
    from types import SimpleNamespace
    module=comparison_module()
    calls=[]
    torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:calls.append('device-sync')))
    module.synchronize_plan_setup(torch,enabled)
    assert calls==(['device-sync'] if enabled else [])


def test_setup_sync_cli_is_forwarded(monkeypatch,tmp_path):
    import sys
    module=comparison_module()
    monkeypatch.setattr(module,'read',lambda path:{})
    calls=[]
    monkeypatch.setattr(module,'run',lambda *args,**kwargs:calls.append(kwargs))
    monkeypatch.setattr(sys,'argv',['run_tcfft_comparison.py','--run','--output',str(tmp_path),
        '--case','n262144-b4','--backends','tcfft','--diagnostic-only','--synchronize-setup'])
    module.main()
    assert calls[0]['synchronize_setup'] is True


def test_bridge_waits_for_initialization_before_publishing_plan():
    bridge=(ROOT/'benchmarks/tcfft/bridge.cuh').read_text()
    create=bridge.split('extern "C" int tcfft_baseline_create',1)[1].split('extern "C" int tcfft_baseline_execute',1)[0]
    assert create.index('cudaMemcpy(p->reverse') < create.index('cudaStreamSynchronize(nullptr)') < create.index('*result = p;')
    assert 'code = cudaStreamSynchronize(nullptr);\n    if (code != cudaSuccess) { baseline_free(p); return code; }' in create
    execute=bridge.split('extern "C" int tcfft_baseline_execute',1)[1]
    assert 'cudaStreamSynchronize' not in execute
