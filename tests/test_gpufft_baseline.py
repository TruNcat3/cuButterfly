import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gpufft_baseline', ROOT/'scripts/gpufft_baseline.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


@pytest.fixture(autouse=True)
def cpu_library_presence(tmp_path, monkeypatch):
    # Capability queries check presence only, and must never load this file.
    library = tmp_path/'prepared.so'
    library.touch()
    monkeypatch.setattr(g, 'DEFAULT_LIBRARY', library)


def semantic(**changes):
    s = dict(operator=0, fft_type=0, length_mode=0, shape=[1024], batch=3,
             storage=6, compute=1, inverse=0, normalize_inverse=False)
    s.update(changes)
    return s


@pytest.mark.parametrize('storage,compute', [(6,0),(6,1),(7,0),(7,2)])
@pytest.mark.parametrize('inverse,norm', [(0,False),(0,True),(1,False),(1,True)])
def test_native_precision_direction_normalization(storage, compute, inverse, norm):
    assert g.support(semantic(storage=storage, compute=compute, inverse=inverse,
                              normalize_inverse=norm))['supported']


@pytest.mark.parametrize('changes', [dict(shape=[1]),dict(shape=[3]),dict(shape=[2**29]),
    dict(shape=[32,32]),dict(shape=[]),dict(batch=0),dict(batch=True),dict(batch=65536),
    dict(shape=[512],batch=2**22),dict(storage=4),dict(compute=2),dict(operator=1),
    dict(fft_type=1),dict(length_mode=1),dict(inverse=True),dict(normalize_inverse=1)])
def test_missing_semantics_are_rejected(changes):
    assert not g.support(semantic(**changes))['supported']


def test_cpu_support_never_loads_cuda(monkeypatch):
    def fail(*a, **kw):
        raise AssertionError('CUDA loaded during support query')
    monkeypatch.setattr(g.ctypes, 'CDLL', fail)
    assert g.support(semantic(shape=[512],batch=65536))['supported']
    with pytest.raises(ValueError):
        g.Plan('unused',semantic(shape=[3]))


@pytest.mark.parametrize('inverse', [False,True])
@pytest.mark.parametrize('normalize', [False,True])
def test_documented_upstream_sign_and_order_adaptation(inverse, normalize):
    # Independent direct-sum mathematical contract documented in upstream
    # test/test_fft.cu: forward_value / inverse_value. This is a CPU derivation,
    # not a validation of the compiled CUDA kernels.
    n=16
    a=np.random.default_rng(7).normal(size=n)+1j*np.random.default_rng(9).normal(size=n)
    rev=np.array([int(f'{k:04b}'[::-1],2) for k in range(n)])
    if inverse:
        packed=a[rev]
        output=np.array([sum(packed[k]*np.exp(2j*np.pi*j*rev[k]/n) for k in range(n)) for j in range(n)])
        if normalize:
            output/=n
        expected=np.fft.ifft(a)*(1 if normalize else n)
    else:
        packed=np.array([sum(a[j]*np.exp(-2j*np.pi*j*rev[k]/n) for j in range(n)) for k in range(n)])
        output=packed[rev]
        expected=np.fft.fft(a)
    np.testing.assert_allclose(output,expected,atol=5e-14)


def test_missing_library_is_explicit(tmp_path):
    assert g.support(semantic(), tmp_path/'missing.so')['status'] == 'missing-external-dependency'
