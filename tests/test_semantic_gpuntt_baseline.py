"""CPU-only bridge guards and native binding protocol; no CUDA launches."""
import ctypes as ct
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("semantic_gpuntt_baseline", ROOT / "scripts/semantic_gpuntt_baseline.py")
gpu = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gpu)


def semantics(**changes):
    result = {"schema": "cubutterfly-semantic-v1", "operator": 1, "shape": [8], "batch": 3,
              "storage": 8, "word_bits": 32, "compute": 3, "modulus": 17,
              "inverse": 0, "normalize_inverse": False, "length_mode": 0,
              "ntt_input_layout": 0, "ntt_output_layout": 0}
    result.update(changes)
    return result


@pytest.mark.parametrize("changes,reason", [
    ({"operator": 2}, "NTT only"),
    ({"shape": [2, 4]}, "rank one"),
    ({"shape": [7]}, "power-of-two"),
    ({"shape": [1]}, "logN"),
    ({"shape": [1 << 29]}, "logN"),
    ({"shape": [True]}, "power-of-two"),
    ({"length_mode": 1}, "embedding"),
    ({"batch": 65536}, "no shrink"),
    ({"batch": True}, "no shrink"),
    ({"storage": 2}, "UINT32 or UINT64"),
    ({"compute": 0}, "must match exactly"),
    ({"word_bits": 64}, "must match exactly"),
    ({"modulus": 1 << 30}, "arithmetic range"),
    ({"modulus": 17.0}, "arithmetic range"),
    ({"inverse": 1}, "canonical normalization"),
    ({"normalize_inverse": 1}, "boolean"),
    ({"ntt_input_layout": 1}, "APPT"),
])
def test_semantic_rejections_precede_native_or_device_access(monkeypatch, changes, reason):
    def forbidden(*args):
        raise AssertionError("reject before any native/device access")
    monkeypatch.setattr(gpu, "_identity", forbidden)
    report = gpu.support(semantics(**changes))
    assert not report["supported"]
    assert report["status"] == "unsupported-semantic"
    assert reason in report["reason"]


class FakeBridge:
    def __init__(self):
        self.calls = []

    def validate(self, s):
        self.calls.append(("Validate", s["modulus"]))
        if s["modulus"] == 25:
            raise gpu.BridgeError(1, "modulus must be prime")
        return 9

    def call(self, name, *args):
        self.calls.append((name, args))
        if name == "Create":
            args[0]._obj.value = 123
            args[-2]._obj.value = 9
            args[-1]._obj.value = 128


@pytest.fixture
def binding(monkeypatch):
    bridge = FakeBridge()
    identity = {"binary": {"sha256": "new-binary"}, "vendor_revision": gpu.SOURCE_REVISION,
                "vendor_archive": {"sha256": "pinned-vendor"}, "root_helper_library": {"sha256": "frozen-main"},
                "sources": [{"sha256": "source"}]}
    monkeypatch.setattr(gpu, "_identity", lambda path: (bridge, identity))
    return bridge


def test_prime_failure_and_missing_binding_are_distinct(binding, monkeypatch):
    report = gpu.support(semantics(modulus=25))
    assert report["status"] == "unsupported-semantic" and "prime" in report["reason"]
    def absent(path):
        raise OSError("shared bridge missing")
    monkeypatch.setattr(gpu, "_identity", absent)
    report = gpu.support(semantics())
    assert report["status"] == "missing-external-binding"
    assert "shared bridge missing" in report["reason"]
    assert not gpu.backend_inventory()["available"]


@pytest.mark.parametrize("storage,bits,compute", [(8, 32, 3), (9, 64, 4)])
def test_exact_unsigned_domain_and_public_root(binding, storage, bits, compute):
    report = gpu.support(semantics(storage=storage, word_bits=bits, compute=compute, batch=65535))
    assert report["supported"] and report["metadata"]["root"] == 9
    assert report["metadata"]["batch"] == 65535
    assert binding.calls == [("Validate", 17)]
    assert gpu.backend_inventory()["root_helper_library"]["sha256"] == "frozen-main"


class Tensor:
    def __init__(self, pointer, dtype="uint32", elements=24, device="cuda:0", contiguous=True):
        self.pointer, self.dtype, self.elements = pointer, dtype, elements
        self.device, self.contiguous = device, contiguous
        self.is_cuda = device.startswith("cuda")

    def numel(self):
        return self.elements

    def is_contiguous(self):
        return self.contiguous

    def data_ptr(self):
        return self.pointer


def test_full_call_uses_supplied_pointers_and_stream_without_setup_execution(binding):
    torch = SimpleNamespace(uint32="uint32", uint64="uint64")
    core = gpu.build_core(torch, semantics(), Tensor(100), Tensor(200), SimpleNamespace(cuda_stream=77))
    assert [call[0] for call in binding.calls] == ["Validate", "Create"]
    create = binding.calls[-1][1]
    assert create[1:6] == (3, 3, 32, 17, 0)
    assert create[6].value == 77
    core()
    name, execute = binding.calls[-1]
    assert name == "Execute"
    assert tuple(value.value for value in execute) == (123, 100, 200, 77)
    details = core.metadata()
    assert details["included_operations"] == ["vendor_GPU_NTT", "forward_output_bit_reversal"]
    assert details["execute_allocations"] is False
    assert details["workspace_bytes"] == 128
    details["root"] = 0
    assert core.metadata()["root"] == 9
    core.close()
    assert binding.calls[-1][0] == "Destroy"
    core.close()
    with pytest.raises(RuntimeError, match="closed"):
        core()


def test_inverse_normalization_and_aliasing_preserved(binding):
    torch = SimpleNamespace(uint32="uint32", uint64="uint64")
    tensor = Tensor(100)
    core = gpu.build_core(torch, semantics(inverse=1, normalize_inverse=True), tensor, tensor, 0)
    assert core.metadata()["included_operations"] == ["inverse_input_bit_reversal", "vendor_GPU_INTT_with_inverse_N"]
    core()
    assert tuple(value.value for value in binding.calls[-1][1]) == (123, 100, 100, None)
    core.close()


@pytest.mark.parametrize("changes", [
    {"dtype": "float32"}, {"dtype": "int32"}, {"elements": 8},
    {"device": "cpu"}, {"contiguous": False},
])
def test_tensor_contract_never_promotes_or_shrinks(binding, changes):
    torch = SimpleNamespace(uint32="uint32", uint64="uint64")
    with pytest.raises(ValueError, match="full packed CUDA batch"):
        gpu.build_core(torch, semantics(), Tensor(100, **changes), Tensor(200), 77)
    assert all(call[0] != "Create" for call in binding.calls)


def test_ctypes_signatures_and_revision(monkeypatch):
    class Function:
        def __init__(self, result):
            self.result = result
        def __call__(self, *args):
            return self.result
    dll = SimpleNamespace(**{
        "gpunttSemantic" + name: Function(result) for name, result in {
            "Validate": 0, "Create": 0, "Execute": 0, "Destroy": 0,
            "AbiVersion": 1, "SourceRevision": gpu.SOURCE_REVISION.encode(), "LastError": b"",
        }.items()})
    monkeypatch.setattr(gpu.ct, "CDLL", lambda path: dll)
    bridge = gpu.Bridge(__file__)
    assert bridge.dll.gpunttSemanticExecute.argtypes == [ct.c_void_p] * 4
    assert bridge.dll.gpunttSemanticValidate.argtypes[-1] == ct.POINTER(ct.c_uint64)
    dll.gpunttSemanticSourceRevision.result = b"other"
    with pytest.raises(ValueError, match="revision differs"):
        gpu.Bridge(__file__)


def test_opt_in_actual_cpu_validation():
    library = os.environ.get("GPUNTT_TEST_LIBRARY")
    if not library:
        pytest.skip("set GPUNTT_TEST_LIBRARY for native CPU-only root validation")
    report = gpu.support(semantics(), library)
    assert report["supported"], report
    assert report["metadata"]["root"] == 9
    inverse = gpu.support(semantics(inverse=1, normalize_inverse=True), library)
    assert inverse["supported"] and inverse["metadata"]["root"] == 9
    for changes in ({"modulus": 25}, {"modulus": 19}):
        report = gpu.support(semantics(**changes), library)
        assert not report["supported"] and report["status"] == "unsupported-semantic"
