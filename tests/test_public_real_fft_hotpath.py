"""CPU tests for public adapter pointer, layout and ownership contracts."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from cubutterfly.real_fft import RealFFT


@pytest.fixture
def adapter():
    instance = RealFFT.__new__(RealFFT)
    instance.torch = torch
    instance.shape, instance.half_shape, instance.batch = (8, 8), (8, 5), 6
    instance.device, instance.stream = torch.device("cpu"), 7
    instance.plans = [SimpleNamespace(execute=Mock()), SimpleNamespace(execute=Mock())]
    instance.reuse_buffers, instance.buffers = False, {}
    with patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=7)):
        yield instance


@pytest.mark.parametrize("inverse", [False, True])
def test_contiguous_input_keeps_original_tensor_and_leading_dims(adapter, inverse):
    value = torch.empty(2, 3, 8, 5 if inverse else 8,
                        dtype=torch.complex64 if inverse else torch.float32)
    output = adapter.irfftn(value) if inverse else adapter.rfftn(value)
    assert adapter.plans[int(inverse)].execute.call_args[0][0] is value
    assert output.shape == (2, 3, 8, 8 if inverse else 5)
    again = adapter.irfftn(value) if inverse else adapter.rfftn(value)
    assert output.data_ptr() != again.data_ptr()


def test_noncontiguous_input_still_converted_once(adapter):
    value = torch.randn(2, 3, 8, 8).transpose(-1, -2)
    adapter.rfftn(value)
    sent = adapter.plans[0].execute.call_args[0][0]
    assert sent.is_contiguous() and sent.shape == value.shape
    assert sent.data_ptr() != value.data_ptr()
    torch.testing.assert_close(sent, value)


@pytest.mark.parametrize("value", [torch.empty(2, 3, 8, 7), torch.empty(3, 8, 8),
    torch.empty(2, 3, 8, 8, dtype=torch.float64), torch.empty(2, 3, 8, 8, device="meta")])
def test_input_checks_unchanged(adapter, value):
    with pytest.raises(ValueError, match="input"):
        adapter.rfftn(value)


def test_output_layout_alias_and_stream_checks_unchanged(adapter):
    value = torch.empty(2, 3, 8, 8)
    with pytest.raises(ValueError, match="output"):
        adapter.rfftn(value, out=torch.empty(6, 8, 5).transpose(1, 2))
    storage = torch.empty(6 * 8 * 10)
    aliased_input = storage[:6 * 8 * 8].reshape(2, 3, 8, 8)
    aliased_output = storage.view(torch.complex64).reshape(6, 8, 5)
    with pytest.raises(ValueError, match="aliases"):
        adapter.rfftn(aliased_input, out=aliased_output)
    with patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=9)):
        with pytest.raises(ValueError, match="stream"):
            adapter.rfftn(value)
