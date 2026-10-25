"""CPU contract tests; CUBUTTERFLY_TEST_MODULE_COMPILER selects a compiler snapshot."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
POINTS = [
    pytest.param(dict(backend="shared-iterative", operator="fft", precision="fp32", logN=8,
                      stage_partition=[4, 4], threads=128), id="shared-iterative"),
    pytest.param(dict(fft_core="register-tile", precision="fp32", logN=21, local_stages=10,
                      prefix_threads=256, prefix_ept=32, suffix_threads=256, suffix_ept=16), id="register-tile"),
    pytest.param(dict(backend="factor-streamed", precision="fp32", logN=18, stage_partition=[18],
                      factor_partition=[6, 6, 6], factor_columns=4, factor_ept=8,
                      data_tiles_per_cta=1, prefetch_depth=0), id="factor-streamed"),
]


@pytest.fixture
def runtime_compiler():
    path = Path(os.environ.get("CUBUTTERFLY_TEST_MODULE_COMPILER", ROOT / "scripts/compile_module.py")).resolve()
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("runtime_linkage_compiler", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize("point", POINTS)
def test_shared_runtime_command_identity_manifest_and_cache_hit(runtime_compiler, point, tmpdir, monkeypatch):
    module = runtime_compiler
    temporary = Path(str(tmpdir))
    calls = []
    identities = []
    original_sha256 = module.hashlib.sha256

    def recording_sha256(data=b""):
        if data.startswith(b'{"abi":'):
            identity = json.loads(data)
            if "point" in identity and "compiler" in identity:
                identities.append(identity)
        return original_sha256(data)

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if command[1:] == ["--version"]:
            return SimpleNamespace(returncode=0, stdout="fake nvcc\n", stderr="")
        assert "--shared" in command
        assert [value for value in command if value.startswith("--cudart")] == ["--cudart=shared"]
        Path(command[-1]).write_bytes(b"fake shared-runtime module")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.hashlib, "sha256", recording_sha256)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.delenv("CUBUTTERFLY_COMPILE_BUDGET_FILE", raising=False)
    kwargs = dict(root=ROOT, mathdx=temporary / "mathdx", nvcc=Path("fake-nvcc"),
                  sm=80, cache=temporary / "cache")
    output = module.compile_mapping(point, **kwargs)
    manifest = json.loads(output.with_name("manifest.json").read_text())
    assert manifest["cuda_runtime"] == "shared"
    assert manifest["mapping"] == point
    assert manifest["identity"] == output.parent.name
    assert manifest["module_sha256"] == original_sha256(output.read_bytes()).hexdigest()
    compile_calls = [command for command in calls if command[1:] != ["--version"]]
    assert len(compile_calls) == 1
    assert manifest["command"] == compile_calls[0]
    assert module.compile_mapping(dict(point), **kwargs) == output
    assert len([command for command in calls if command[1:] != ["--version"]]) == 1
    assert len(identities) == 2
    assert identities[0] == identities[1]
    assert all(identity["cuda_runtime"] == "shared" for identity in identities)
    assert all(identity["abi"] == (2 if point.get("backend") == "shared-iterative" else 1)
               for identity in identities)
