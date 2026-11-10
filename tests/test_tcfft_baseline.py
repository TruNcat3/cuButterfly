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
