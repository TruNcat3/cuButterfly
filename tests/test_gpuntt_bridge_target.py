"""Architecture propagation without CUDA compilation or GPU access."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks/gpuntt_crypto/build_semantic_bridge.sh"


def executable(path, text):
    path.write_text("#!/bin/sh\nset -eu\n" + text)
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("architecture", [None, "120"])
def test_target_reaches_cmake_sass_ptx_and_receipt(tmp_path, architecture):
    vendor = tmp_path / "vendor"
    for relative in ("src/lib/ntt_merge/ntt.cu", "src/include/gpuntt/ntt_merge/ntt.cuh"):
        path = vendor / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// frozen mock vendor\n")
    helper = tmp_path / "libcuntt.so"
    helper.write_text("mock helper")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "commands"
    executable(binaries / "git", '''case "$*" in
*rev-parse*) echo d03c5eaeadaa780d153496afcb3b6a9b79a13a63;;
*status*) exit 0;;
*) exit 1;;
esac
''')
    executable(binaries / "cmake", '''printf '%s\\n' "$*" >> "$MOCK_LOG"
if [ "$1" = --build ]; then
    mkdir -p "$2"
    echo mock-vendor > "$2/libgpuntt_crypto_vendor.a"
fi
''')
    nvcc = executable(binaries / "nvcc", '''printf '%s\\n' "$*" >> "$MOCK_LOG"
while [ "$#" -gt 0 ]; do
    if [ "$1" = -o ]; then echo mock-binary > "$2"; exit 0; fi
    shift
done
exit 1
''')
    build = tmp_path / "bridge build"
    output = build / "bridge.so"
    env = os.environ.copy()
    env.pop("CUDA_ARCH", None)
    env.update(PATH=f"{binaries}:{env['PATH']}", GPUNTT_SOURCE_DIR=str(vendor),
               GPUNTT_BUILD_DIR=str(build), GPUNTT_OUTPUT=str(output),
               CUNTT_LIBRARY=str(helper), NVCC=str(nvcc), PYTHON=sys.executable,
               MOCK_LOG=str(log))
    if architecture is not None:
        env["CUDA_ARCH"] = architecture
    result = subprocess.run(["bash", str(SCRIPT)], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    expected = architecture or "80"
    commands = log.read_text()
    assert f"-DCMAKE_CUDA_ARCHITECTURES={expected}" in commands
    assert f"--generate-code=arch=compute_{expected},code=sm_{expected}" in commands
    assert f"--generate-code=arch=compute_{expected},code=compute_{expected}" in commands
    receipt = json.loads(Path(str(output) + ".manifest.json").read_text())
    assert receipt["architecture"] == f"sm{expected} + compute{expected} PTX"
    assert output.read_text().strip() == "mock-binary"


@pytest.mark.parametrize("architecture", ["", "sm120", "120-real", "80;120", "0", "-1", "0120"])
def test_invalid_target_fails_before_vendor_or_build(tmp_path, architecture):
    build = tmp_path / "must not exist"
    env = os.environ.copy()
    env.update(CUDA_ARCH=architecture, GPUNTT_BUILD_DIR=str(build),
               GPUNTT_SOURCE_DIR=str(tmp_path / "absent vendor"))
    result = subprocess.run(["bash", str(SCRIPT)], env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "Invalid CUDA_ARCH" in result.stderr
    assert not build.exists()
