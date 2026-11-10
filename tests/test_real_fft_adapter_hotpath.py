"""Exercise adapter geometry and ownership without starting CUDA."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


SPEC = importlib.util.spec_from_file_location(
    "real_fft_hotpath", Path(__file__).resolve().parents[1]
    / "results/e13_assets/cubutterfly_real_fft.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def adapter():
    api = SimpleNamespace(
        check=lambda status, operation: None,
        lib=SimpleNamespace(cubutterflyExecute=lambda *args: 0,
                            cubutterflySetStream=lambda *args: 0))
    def make_plan(self, inverse):
        return SimpleNamespace(inverse=inverse, stream=7, context=None, plan=None)
    with patch.object(torch.cuda, "is_available", return_value=True), \
            patch.object(MODULE, "_Api", return_value=api), \
            patch.object(MODULE.RealFFTAdapter, "_make_plan", make_plan), \
            patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=7)):
        instance = MODULE.RealFFTAdapter("unused.so", (8, 8), 4, torch.device("cuda:0"))
        instance.device = torch.device("cpu")
        yield instance


def test_flat_input_and_output_skip_views(adapter):
    x = torch.randn(4, 8, 8)
    flat, leading = adapter._flatten_batch(x, (8, 8), torch.float32, "rfftn")
    assert flat is x and leading == (4,)
    output = torch.empty(4, 8, 5, dtype=torch.complex64)
    with patch.object(adapter, "_execute", return_value=output) as execute:
        assert adapter.rfftn(x) is output
        assert execute.call_args[0][1] is x
    real = torch.empty(4, 8, 8)
    with patch.object(adapter, "_execute", return_value=real):
        assert adapter.irfftn(output) is real


@pytest.mark.parametrize("leading", [(2, 2), (1, 4)])
def test_leading_dimensions_are_restored(adapter, leading):
    x = torch.randn(*leading, 8, 8)
    half = adapter.rfftn(x)
    assert half.shape == (*leading, 8, 5)
    assert adapter.irfftn(half).shape == (*leading, 8, 8)


def test_noncontiguous_input_still_copies(adapter):
    x = torch.randn(4, 8, 8).transpose(-1, -2)
    flat, leading = adapter._flatten_batch(x, (8, 8), torch.float32, "rfftn")
    assert leading == (4,) and flat.is_contiguous()
    assert flat.data_ptr() != x.data_ptr()
    torch.testing.assert_close(flat, x)


def test_default_outputs_have_independent_storage(adapter):
    x = torch.randn(4, 8, 8)
    first, second = adapter.rfftn(x), adapter.rfftn(x)
    assert first.data_ptr() != second.data_ptr()
    first, second = adapter.irfftn(first), adapter.irfftn(second)
    assert first.data_ptr() != second.data_ptr()


@pytest.mark.parametrize("tensor", [torch.empty(4, 8, 8, dtype=torch.float64),
                                   torch.empty(4, 8, 7), torch.empty(3, 8, 8),
                                   torch.empty(4, 8, 8, device="meta")])
def test_public_validation_remains(adapter, tensor):
    with pytest.raises(ValueError):
        adapter.rfftn(tensor)


def test_execute_validation_and_stream_switch_remain(adapter):
    with pytest.raises(ValueError):
        adapter._execute(adapter._forward, torch.empty(4, 8, 8).transpose(-1, -2))
    with pytest.raises(ValueError):
        adapter._execute(adapter._forward, torch.empty(4, 8, 8), torch.empty(4, 8, 5))
    with patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=9)) as current, \
            patch.object(adapter.api.lib, "cubutterflySetStream", return_value=0) as setter:
        adapter.rfftn(torch.empty(4, 8, 8))
        current.assert_called_once_with(0)
        setter.assert_called_once_with(adapter._forward.context, 9)
        assert adapter._forward.stream == 9
