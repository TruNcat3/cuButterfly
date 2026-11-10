"""CPU-only coverage of target parsing and compiler/MathDx failure gates."""
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
HELPER = ROOT / "cmake/CuFFTDxTarget.cmake"


def resolve(tmp_path, architectures):
    script = tmp_path / "resolve.cmake"
    script.write_text(f'include("{HELPER}")\ncubutterfly_cufftdx_target(arch sm {architectures})\nmessage(STATUS "target=${{arch}},sm=${{sm}}")\n')
    return subprocess.run(["cmake", "-P", str(script)], text=True, capture_output=True)


@pytest.mark.parametrize("architectures,expected", [
    ("70", "70,sm=700"), ("80-real;80-virtual", "80,sm=800"),
    ("120", "120,sm=1200"), ("120-virtual", "120,sm=1200"),
])
def test_numeric_target_mapping(tmp_path, architectures, expected):
    result = resolve(tmp_path, architectures)
    assert result.returncode == 0, result.stderr
    assert f"target={expected}" in result.stdout


@pytest.mark.parametrize("architectures", ["", "native", "all", "sm120", "90a", "0", "80;120", "120-real;90-virtual"])
def test_unsupported_target_is_rejected(tmp_path, architectures):
    result = resolve(tmp_path, architectures)
    assert result.returncode != 0
    assert "cuFFTDx" in result.stderr


@pytest.mark.parametrize("codes,compile_status,error", [
    ("sm_80", 0, "does not advertise"),
    ("sm_120", 1, "cuFFTDx target probe failed"),
    ("sm_120", 0, None),
])
def test_probe_checks_compiler_and_mathdx_without_gpu(tmp_path, codes, compile_status, error):
    nvcc = tmp_path / "nvcc"
    log = tmp_path / "calls"
    nvcc.write_text(f'''#!/bin/sh
[ -z "$CUDA_VISIBLE_DEVICES" ] || exit 9
printf '%s\\n' "$*" >> '{log}'
if [ "$1" = --list-gpu-code ]; then echo '{codes}'; exit 0; fi
exit {compile_status}
''')
    nvcc.chmod(0o755)
    result = subprocess.run([
        "cmake", "-DCUBUTTERFLY_PROBE_ARCHITECTURES=120-real;120-virtual",
        f"-DCUBUTTERFLY_PROBE_NVCC={nvcc}",
        f"-DCUBUTTERFLY_PROBE_MATHDX={tmp_path / 'mathdx'}",
        f"-DCUBUTTERFLY_PROBE_DIRECTORY={tmp_path / 'probe'}",
        "-P", str(HELPER),
    ], text=True, capture_output=True)
    if error:
        assert result.returncode != 0
        assert error in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert "-arch=sm_120" in log.read_text()
        assert "SM<1200>" in (tmp_path / "probe/cufftdx_target_probe.cu").read_text()
