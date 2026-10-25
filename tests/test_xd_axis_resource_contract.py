from pathlib import Path
import shutil
import subprocess

import pytest


def test_cpu_only_axis_resource_contract(tmpdir):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the CPU-only xD contract test")
    root = Path(__file__).resolve().parents[1]
    executable = Path(str(tmpdir)) / "xd_axis_resource_contract"
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
         "-I", str(root / "src"),
         str(root / "tests" / "test_xd_axis_resource_contract.cpp"),
         "-o", str(executable)],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
