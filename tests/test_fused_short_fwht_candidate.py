import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "experiments/tools"))
from cubutterfly import fused_short_fwht as provider
import benchmark_fused_short_fwht_20261012 as bench


def test_real_short_requests_not_removed():
    assert bench.CASES == [(32, 128, 4096), (32, 65536, 4096), (64, 172, 11008), (64, 88064, 11008)]


def test_selective_prepare_does_not_import_tensor_cuda(tmpdir, monkeypatch):
    calls = []
    def compile_only(command, **kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(b"test module")
    monkeypatch.setattr(provider.subprocess, "run", compile_only)
    result = provider.prepare(tmpdir / "module.so")
    assert len(calls) == 1 and "-arch=sm_80" in calls[0]
    assert result["public_search_integrated"] is False
    assert result["kernel_launches"] == 1 and result["accumulation"] == "fp32"
    assert result["candidate_descriptor"]["operator"] == "fwht"
    assert result["candidate_descriptor"]["requires_linked_provider"] is True


def test_descriptor_has_explicit_semantics_and_fallback():
    descriptor = provider.descriptor()
    assert descriptor["storage_precision"] == "fp16"
    assert descriptor["accumulation"] == "fp32"
    assert descriptor["layout"] == "contiguous-last-axis"
    assert descriptor["fallback"] == "portable-low-precision-fwht"


def test_cpu_input_rejected_before_launch():
    import torch
    instance = object.__new__(provider.FusedShortFWHT)
    with pytest.raises(ValueError, match="CUDA FP16"):
        instance(torch.zeros(1, 32, dtype=torch.float16), 0.5)


def test_source_uses_scale_before_single_half_store():
    source = provider.SOURCE.read_text()
    assert "__float2half_rn(low * scale)" in source
    assert "__float2half_rn(high * scale)" in source
    assert "__shfl_xor_sync" in source
    assert "__syncthreads" not in source
    assert "4096" not in source and "11008" not in source
