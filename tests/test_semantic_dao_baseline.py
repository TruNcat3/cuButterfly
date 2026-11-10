"""Dao semantic support and binding checks with no CUDA execution."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("semantic_dao_baseline", ROOT / "scripts/semantic_dao_baseline.py")
dao = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dao)


def semantics(**changes):
    result = {"schema": "cubutterfly-semantic-v1", "operator": 2, "storage": 2, "compute": 1,
              "shape": [4, 8], "batch": 3, "inverse": 0, "normalize_inverse": False,
              "length_mode": 0, "placement": 0, "input_orders": [0, 0], "output_orders": [0, 0]}
    result.update(changes)
    return result


def inventory(**changes):
    result = {"available": True, "contract_verified": True, "version": "1.1.0",
              "python_files": [{"path": str(Path(__file__).resolve()), "sha256": "test-python"}],
              "native_files": [{"path": "test-native.so", "sha256": "test-native"}],
              "sources": [{"path": "test-source", "sha256": "test-source"}]}
    result.update(changes)
    return result


@pytest.mark.parametrize("changes,reason", [
    ({"operator": 0}, "FWHT operator only"),
    ({"storage": 3}, "no precision conversion"),
    ({"storage": 0, "compute": 0}, "native low-precision"),
    ({"storage": 1, "compute": 0}, "native low-precision"),
    ({"compute": 2}, "other compute requests"),
    ({"shape": [4, 7]}, "power-of-two"),
    ({"shape": [256, 256]}, "65536 exceeds"),
    ({"shape": [True]}, "power-of-two"),
    ({"shape": []}, "power-of-two"),
    ({"batch": (1 << 31)}, "signed 32-bit"),
    ({"batch": True}, "signed 32-bit"),
    ({"length_mode": 1}, "embedding"),
    ({"normalize_inverse": 1}, "boolean"),
])
def test_support_rejects_exact_semantic_mismatches_before_backend_inventory(monkeypatch, changes, reason):
    def forbidden():
        raise AssertionError("semantic rejection must precede backend/device inspection")
    monkeypatch.setattr(dao, "backend_inventory", forbidden)
    report = dao.support(semantics(**changes))
    assert not report["supported"]
    assert report["status"] == "unsupported-semantic"
    assert reason in report["reason"]


@pytest.mark.parametrize("storage", [0, 1, 2])
def test_verified_fp32_arithmetic_contract_and_full_batch(storage):
    original = semantics(storage=storage, shape=[128, 256], batch=37)
    report = dao.support(original, inventory=inventory())
    assert report["supported"]
    assert report["metadata"]["flattened_points"] == 32768
    assert report["metadata"]["batch"] == 37
    assert report["metadata"]["storage"] == storage
    assert original == semantics(storage=storage, shape=[128, 256], batch=37)


def test_missing_dependency_and_unverified_contract_are_distinct_from_kernel_gaps():
    missing = dao.support(semantics(), inventory=inventory(available=False, reason="package absent"))
    unknown = dao.support(semantics(storage=0), inventory=inventory(contract_verified=False, reason="source changed"))
    assert missing["status"] == "missing-external-dependency"
    assert unknown["status"] == "unverified-external-contract"
    assert missing["reason"] == "package absent" and unknown["reason"] == "source changed"


@pytest.mark.parametrize("inverse,normalized,scale", [(0, False, 1.0), (1, False, 1.0), (1, True, 1 / 32)])
def test_inverse_scale_is_exact(inverse, normalized, scale):
    report = dao.support(semantics(inverse=inverse, normalize_inverse=normalized), inventory=inventory())
    assert report["metadata"]["scale"] == scale


class Tensor:
    def __init__(self, values, dtype="float32", device="cuda:0", contiguous=True, shape=None):
        self.values = values
        self.dtype, self.device, self.contiguous = dtype, device, contiguous
        self.is_cuda = device.startswith("cuda")
        self.shape = shape or (len(values),)

    def numel(self):
        return len(self.values)

    def is_contiguous(self):
        return self.contiguous

    def view(self, *shape):
        return Tensor(self.values, self.dtype, self.device, self.contiguous, shape)

    def copy_(self, other):
        self.values[:] = other.values


@pytest.fixture
def bound_backend(monkeypatch):
    calls = []

    def function(value, scale):
        calls.append({"shape": value.shape, "dtype": value.dtype, "scale": scale})
        return Tensor([x * scale for x in value.values], value.dtype, value.device, shape=value.shape)

    backend = SimpleNamespace(__file__=__file__, __version__="1.1.0", hadamard_transform=function)
    monkeypatch.setattr(dao, "backend_inventory", lambda: inventory())
    monkeypatch.setattr(dao, "import_module", lambda name: backend)
    torch = SimpleNamespace(float16="float16", bfloat16="bfloat16", float32="float32")
    return torch, calls, backend


def test_bind_never_evaluates_input_and_full_call_copies_output(bound_backend):
    torch, calls, _ = bound_backend
    source, output = Tensor(list(range(96))), Tensor([0] * 96)
    core = dao.build_core(torch, semantics(inverse=1, normalize_inverse=True), source, output)
    assert not calls
    core()
    assert calls == [{"shape": (3, 32), "dtype": "float32", "scale": 1 / 32}]
    assert output.values == [x / 32 for x in range(96)]
    assert source.values == list(range(96))
    details = core.metadata()
    assert "copy_backend_result_to_supplied_output" in details["included_operations"]
    assert details["backend_identity"]["native_files"][0]["sha256"] == "test-native"
    details["batch"] = 1
    assert core.metadata()["batch"] == 3
    core.close()
    with pytest.raises(RuntimeError, match="closed"):
        core()


def test_small_length_pad_cost_and_inplace_copy_are_explicit(bound_backend):
    torch, calls, _ = bound_backend
    tensor = Tensor([1, 2, 3, 4])
    core = dao.build_core(torch, semantics(shape=[2, 2], batch=1, placement=1), tensor, tensor)
    assert "backend_zero_pad_to_8_and_output_crop" in core.metadata()["included_operations"]
    assert core.metadata()["backend_internal_points"] == 8
    core()
    assert calls[0]["shape"] == (1, 4)


@pytest.mark.parametrize("changes,reason", [
    ({"dtype": "float16"}, "dtype differs"),
    ({"device": "cpu"}, "packed CUDA"),
    ({"contiguous": False}, "packed CUDA"),
])
def test_tensor_contract_rejects_conversion_and_cpu(bound_backend, changes, reason):
    torch, calls, _ = bound_backend
    with pytest.raises(ValueError, match=reason):
        dao.build_core(torch, semantics(), Tensor(list(range(96)), **changes), Tensor([0] * 96))
    assert not calls


def test_batch_shrink_and_import_identity_rejected(bound_backend):
    torch, calls, backend = bound_backend
    with pytest.raises(ValueError, match="complete logical batch"):
        dao.build_core(torch, semantics(), Tensor([0] * 32), Tensor([0] * 32))
    backend.__version__ = "other"
    with pytest.raises(RuntimeError, match="version differs"):
        dao.build_core(torch, semantics(), Tensor([0] * 96), Tensor([0] * 96))
    assert not calls


def test_unverified_backend_cannot_be_built(bound_backend, monkeypatch):
    torch, calls, _ = bound_backend
    monkeypatch.setattr(dao, "backend_inventory", lambda: inventory(contract_verified=False, reason="not audited"))
    with pytest.raises(dao.UnsupportedBaseline) as failure:
        dao.build_core(torch, semantics(), Tensor([0] * 96), Tensor([0] * 96))
    assert failure.value.report["status"] == "unverified-external-contract"
    assert not calls
