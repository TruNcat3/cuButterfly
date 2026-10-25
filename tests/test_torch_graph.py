"""CPU contract tests; actual CUDA capture is validated by the FNO runner."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("graph_contract", ROOT / "python/cubutterfly/torch_graph.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_import_does_not_load_torch():
    source = "import sys; sys.path.insert(0, %r); import cubutterfly.torch_graph; assert 'torch' not in sys.modules" % str(ROOT / "python")
    subprocess.run([sys.executable, "-c", source], check=True)


def test_constructor_rejects_cpu_before_cuda_work():
    with pytest.raises(ValueError, match="CUDA Tensor"):
        MODULE.InferenceGraph(lambda x: x, torch.ones(3))


@pytest.fixture
def graph():
    instance = MODULE.InferenceGraph.__new__(MODULE.InferenceGraph)
    instance.torch = torch
    instance.function = torch.nn.Identity().eval()
    instance.shape, instance.dtype, instance.device = (3,), torch.float32, torch.device("cpu")
    instance.stream = SimpleNamespace(cuda_stream=7)
    instance._input, instance._output = torch.empty(3), torch.empty(3)
    instance._graph = SimpleNamespace(replay=lambda: instance._output.copy_(instance._input * 2))
    with patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=7)):
        yield instance


def test_changing_inputs_copy_replay_and_own_outputs(graph):
    first_input = torch.tensor([1., 2., 3.])
    first = graph(first_input)
    second = graph(torch.tensor([4., 5., 6.]))
    torch.testing.assert_close(first, first_input * 2)
    torch.testing.assert_close(second, torch.tensor([8., 10., 12.]))
    assert first.data_ptr() != second.data_ptr() != graph._output.data_ptr()
    assert graph.contract()["input_copy_included"]


@pytest.mark.parametrize("value", [torch.ones(4), torch.ones(3, dtype=torch.float64),
                                   torch.ones(3, requires_grad=True), [1, 2, 3]])
def test_rejects_contract_changes(graph, value):
    with pytest.raises(ValueError):
        graph(value)


def test_rejects_stream_switch(graph):
    with patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=9)):
        with pytest.raises(ValueError, match="declared capture stream"):
            graph(torch.ones(3))


def test_rejects_training_after_capture(graph):
    graph.function.train()
    with pytest.raises(ValueError, match="eval mode"):
        graph(torch.ones(3))


def test_trial_rotation_covers_all_positions():
    spec = importlib.util.spec_from_file_location("fno_graph_runner", ROOT / "paper/tools/run_fno_graph_integration_20261013.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    orders = [runner.variant_order(index) for index in range(4)]
    assert len({tuple(order) for order in orders}) == 4
    assert all(len({order[position] for order in orders}) == 4 for position in range(4))


def test_weight_digest_includes_extra_state():
    spec = importlib.util.spec_from_file_location("fno_graph_digest", ROOT / "paper/tools/run_fno_graph_integration_20261013.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    model = SimpleNamespace(state_dict=lambda: {"weights": torch.ones(3),
        "_extra_state": {"mode": "dense", "module": torch.nn.Identity, "activation": torch.relu}})
    original = runner.weight_digest(model)
    model.state_dict = lambda: {"weights": torch.ones(3), "_extra_state": {"mode": "factorized"}}
    assert original != runner.weight_digest(model)


@pytest.mark.parametrize("inverse", [False, True])
def test_public_patch_delegates_original_tensor(inverse):
    spec = importlib.util.spec_from_file_location("fno_patch_hotpath", ROOT / "paper/tools/run_fno_graph_integration_20261013.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    instance = runner.public_patch(object, torch)()
    instance.calls = {"rfftn": 0, "irfftn": 0}
    tensor = SimpleNamespace(is_cuda=True, ndim=4,
        dtype=torch.complex64 if inverse else torch.float32,
        shape=(2, 3, 8, 5 if inverse else 8))
    def execute(value):
        assert value is tensor
        return value
    adapter = SimpleNamespace(rfftn=execute, irfftn=execute)
    def select(shape, batch):
        assert shape == (8, 8) and batch == 6
        return adapter
    instance._adapter = select
    if inverse:
        assert instance.irfftn(tensor, s=(8, 8), dim=(-2, -1), norm="forward") is tensor
    else:
        assert instance.rfftn(tensor, dim=(-2, -1), norm="forward") is tensor
    assert instance.calls["irfftn" if inverse else "rfftn"] == 1
    with pytest.raises(ValueError, match="contract"):
        instance.rfftn(tensor, dim=(-2, -1), norm="backward")
