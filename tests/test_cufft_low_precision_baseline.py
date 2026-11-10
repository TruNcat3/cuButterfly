"""CPU-only support, binding, build-contract and native descriptor checks."""
import ctypes as ct
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/cufft_low_precision_baseline.py"
SPEC = importlib.util.spec_from_file_location("cufft_low_baseline", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
LIBRARY = Path(os.environ.get("CUFFT_LOW_BASELINE_LIBRARY", ROOT / "build/cufft_low_precision_sm80/libcufft_low_precision_baseline.so"))


@pytest.mark.parametrize("precision,token,minimum", [("fp16", "CUDA_C_16F", 53), ("bf16", "CUDA_C_16BF", 80)])
@pytest.mark.parametrize("shape", [[65536], [64, 1024]])
@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_native_type_contract(precision, token, minimum, shape, direction):
    with mock.patch.object(ct, "CDLL", side_effect=AssertionError("support must not load CUDA")):
        capability = M.support(shape, 16, precision=precision, direction=direction)
    assert capability["supported"]
    assert capability["input_type"] == capability["output_type"] == capability["execution_type"] == token
    assert capability["compute_precision"] == "vendor-native"
    assert capability["compute_family"] == "fft-l" and capability["minimum_sm"] == minimum
    assert not capability["gpu_execution_verified"] and capability["normalization"] == "none"
    d = M.descriptor(shape, 16, precision=precision, direction=direction)
    assert d.direction == (-1 if direction == "forward" else 1)
    assert d.precision == M.PRECISIONS[precision] and d.rank == len(shape)
    assert capability["buffer_bytes"] == 16 * 65536 * 4


@pytest.mark.parametrize("kwargs", [dict(compute="fp32"), dict(accumulation="fp32"), dict(accumulation="fp16"),
    dict(normalization="inverse"), dict(normalization="ortho"), dict(placement="in-place"),
    dict(element_stride=2), dict(batch_stride=65537), dict(input_order="bit-reversed"),
    dict(output_order="bit-reversed"), dict(direction="unknown"), dict(precision="fp32"), dict(sm=52)])
def test_requests_are_rejected_without_substitution(kwargs):
    capability = M.support([65536], 16, **kwargs)
    assert not capability["supported"] and capability["reasons"]
    with pytest.raises(ValueError):
        M.descriptor([65536], 16, **kwargs)


@pytest.mark.parametrize("shape,batch", [([], 1), ([16, 16, 16], 1), ([3], 1), ([0], 1), ([True], 1),
    ([1 << 32], 1), ([1 << 16, 1 << 16], 1), ([1 << 31], 2), ([16], 0), ([16], True), ([16], 1.5)])
def test_checked_rank_and_size_bounds(shape, batch):
    assert not M.support(shape, batch)["supported"]


def test_boundaries_and_bfloat_arithmetic_are_explicit():
    assert M.support([1 << 31], 1)["supported"]
    assert M.support([1], (1 << 32) - 1)["supported"]
    assert M.support([16], 1, precision="bf16", sm=80)["supported"]
    assert not M.support([16], 1, precision="bf16", sm=75)["supported"]
    assert "mixture" in M.support([16], 1, precision="bf16")["arithmetic_contract"]
    assert "unspecified" in M.support([16], 1)["arithmetic_contract"]


def test_invalid_plan_request_never_loads_library():
    with mock.patch.object(M, "load") as load:
        with pytest.raises(ValueError):
            M.Plan("unused", [16], 1, stream=0, compute="fp32")
        with pytest.raises(ValueError):
            M.Plan("unused", [16], 1, stream=-1)
        load.assert_not_called()


def test_binding_execution_is_stream_bound_and_has_no_implicit_work():
    plan = M.Plan.__new__(M.Plan)
    plan.lib = mock.Mock()
    plan.lib.cufft_low_execute.return_value = 0
    plan.lib.cufft_low_destroy.return_value = 0
    plan.handle, plan.stream = ct.c_void_p(123), 456
    plan.execute(1024, 4096, stream=456)
    plan.lib.cufft_low_execute.assert_called_once_with(plan.handle, 1024, 4096)
    with pytest.raises(ValueError):
        plan.execute(1024, 4096, stream=0)
    with pytest.raises(ValueError):
        plan.execute(1 << 80, 4096, stream=456)
    plan.close()
    plan.close()
    plan.lib.cufft_low_destroy.assert_called_once()
    with pytest.raises(RuntimeError):
        plan.execute(1024, 4096, stream=456)


@pytest.mark.skipif(not LIBRARY.exists(), reason="optional native SM80 bridge has not been CPU-compiled")
def test_native_cpu_validation_matches_python_without_cuda_initialization(tmp_path):
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler needed for the no-CUDA interposer")
    guard = tmp_path / "no_cuda_guard.so"
    subprocess.run([compiler, "-shared", "-fPIC", str(M.SOURCE / "no_cuda_guard.c"), "-o", str(guard)], check=True)
    code = '''
import ctypes as ct, sys
sys.path.insert(0, sys.argv[1])
import cufft_low_precision_baseline as m
lib = m.load(sys.argv[2])
assert lib.cufft_low_header_version() >= 11200
assert lib.cufft_low_cuda_header_version() >= 12040
assert ct.sizeof(m.Descriptor) == 48 and ct.sizeof(m.Info) == 32
for precision in ("fp16", "bf16"):
    for shape in ([65536], [64,1024]):
        d = m.descriptor(shape, 16, precision=precision)
        assert lib.cufft_low_validate(ct.byref(d)) == 0
        for field, value in (("normalization",1), ("rank",3), ("precision",99),
                             ("batch",0), ("batch",1<<32), ("abi_version",99), ("reserved",1)):
            invalid = m.Descriptor.from_buffer_copy(d)
            setattr(invalid, field, value)
            assert lib.cufft_low_validate(ct.byref(invalid)) == -1
            handle = ct.c_void_p(999)
            assert lib.cufft_low_create(ct.byref(invalid), 0, ct.byref(handle)) == -1
            assert not handle.value
        d.dimensions[0] = 3
        assert lib.cufft_low_validate(ct.byref(d)) == -1
        d.dimensions[0] = 1 << 32
        assert lib.cufft_low_validate(ct.byref(d)) == -1
        d.dimensions[0] = 1 << 31
        d.batch = 2
        assert lib.cufft_low_validate(ct.byref(d)) == -1
assert lib.cufft_low_validate(None) == -1
assert lib.cufft_low_execute(None, 0, 0) == -1
assert lib.cufft_low_destroy(None) == 0
'''
    environment = dict(os.environ, LD_PRELOAD=str(guard))
    subprocess.run([sys.executable, "-c", code, str(ROOT / "scripts"), str(LIBRARY)],
                   env=environment, check=True, capture_output=True, text=True)


def test_cli_support_and_rejection_do_not_query_gpu():
    assert M.main(["--shape", "64", "1024", "--precision", "bf16"]) == 0
    assert M.main(["--compute", "fp32"]) == 2
    assert M.main(["--direction", "inverse", "--normalization", "inverse"]) == 2


@pytest.mark.skipif(not LIBRARY.exists(), reason="optional native bridge build manifest is absent")
def test_build_manifest_is_cpu_only_and_pins_sources():
    manifest = json.loads((LIBRARY.parent / "build.json").read_text())
    assert manifest["gpu_initialization"] is False and manifest["gpu_execution"] is False
    assert manifest["architecture"] == "sm_80"
    assert manifest["library_sha256"] == M.sha256(LIBRARY)
    assert manifest["source_sha256"] == {name: M.sha256(M.SOURCE / name) for name in ("bridge.cu", "bridge.h")}
    assert manifest["binding_sha256"] == M.sha256(SCRIPT)
    assert set(manifest["vendor_header_sha256"]) == {"cufft.h", "cufftXt.h", "library_types.h", "cuda_runtime_api.h"}
    assert "--cudart=shared" in manifest["command"] and "-lcufft" in manifest["command"]


@pytest.mark.parametrize("target", [80, 120, "120"])
def test_build_explicit_architecture_and_compiler(tmp_path, monkeypatch, target):
    nvcc = tmp_path / "cuda/bin/nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    calls = []

    def compile_bridge(command, check):
        assert check is True
        calls.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"mock compiled bridge")

    monkeypatch.setattr(M.subprocess, "check_output", lambda command, text: "mock nvcc")
    monkeypatch.setattr(M.subprocess, "run", compile_bridge)
    monkeypatch.setattr(M, "sha256", lambda path: "mock-sha256")
    library = M.build(tmp_path / "output", nvcc=nvcc, cuda_arch=target)
    receipt = json.loads((library.parent / "build.json").read_text())
    assert calls[0][0] == str(nvcc.resolve())
    assert f"-arch=sm_{target}" in calls[0]
    assert receipt["architecture"] == f"sm_{target}"


@pytest.mark.parametrize("target", [True, 0, -80, 8.0, "sm_80", "80a", "", "８０"])
def test_build_rejects_invalid_architecture_before_writing(tmp_path, target):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="numeric SM target"):
        M.build(output, nvcc=tmp_path / "missing-nvcc", cuda_arch=target)
    assert not output.exists()


def test_cli_forwards_architecture_and_explicit_nvcc(tmp_path):
    with mock.patch.object(M, "build", return_value=tmp_path / "lib.so") as build:
        assert M.main(["--build", str(tmp_path), "--nvcc", "/explicit/cuda/bin/nvcc",
                       "--cuda-arch", "120"]) == 0
    build.assert_called_once_with(tmp_path, "/explicit/cuda/bin/nvcc", cuda_arch=120)
    with mock.patch.object(M, "build", return_value=tmp_path / "lib.so") as build:
        assert M.main(["--build", str(tmp_path)]) == 0
    build.assert_called_once_with(tmp_path, "/usr/local/cuda-12.4/bin/nvcc", cuda_arch=80)


def test_unsupported_architecture_compile_failure_is_not_retried(tmp_path, monkeypatch):
    nvcc = tmp_path / "nvcc"
    nvcc.touch()
    monkeypatch.setattr(M.subprocess, "check_output", lambda command, text: "mock nvcc")
    failure = subprocess.CalledProcessError(1, [str(nvcc), "-arch=sm_120"])
    with mock.patch.object(M.subprocess, "run", side_effect=failure) as compile_bridge:
        with pytest.raises(subprocess.CalledProcessError) as caught:
            M.build(tmp_path / "output", nvcc=nvcc, cuda_arch=120)
    assert caught.value is failure
    assert compile_bridge.call_count == 1
    assert not (tmp_path / "output/build.json").exists()
