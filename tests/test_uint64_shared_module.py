import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compile_module
import resident_mapping


@pytest.mark.parametrize("operator,left", [("subset-zeta", "false"), ("superset-zeta", "true"), ("xor-zeta", "false")])
@pytest.mark.parametrize("resident", [False, True])
def test_uint64_shared_uses_full_width_trait_and_sized_buffers(operator, left, resident):
    point = dict(backend="shared-iterative", operator=operator, precision="uint64",
                 logN=8, stage_partition=[4, 4], threads=128)
    if resident:
        point["local_stage_partitions"] = [[2, 2], [2, 2]]
    parts, _ = resident_mapping.validate_resident_axes(
        point, [4, 4], backend="shared-iterative", operator=operator)
    assert bool(parts) == resident
    source = compile_module.render_shared(point, 80, "uint64-test")
    assert f"using Operator=BooleanZetaOperator<{left},std::uint64_t>;" in source
    assert "using Value=Operator::Value;" in source
    assert 'static_assert(sizeof(Value)==8,"UINT64 zeta requires eight-byte values");' in source
    assert "2ULL*(1ULL<<4)*sizeof(Value)" in source
    assert "static_cast<Value*>(a.io.workspace)+0*extent" in source
    assert "const Operator op{bool(a.io.inverse)};" in source
    assert ("shared_resident_kernel<Operator" in source) == resident
    old = compile_module.render_shared(dict(point, precision="uint32"), 80, "uint32-test")
    assert f"using Operator=BooleanZetaOperator<{left}>;" in old


def test_uint64_shared_cache_keeps_distinct_integer_width(tmp_path, monkeypatch):
    units = []

    def fake_run(command, **kwargs):
        if command[1:] == ["--version"]:
            return SimpleNamespace(returncode=0, stdout="test nvcc", stderr="")
        unit = pathlib.Path(command[command.index("-o") - 1])
        units.append(unit.read_text())
        pathlib.Path(command[-1]).write_bytes(b"test module")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(compile_module.subprocess, "run", fake_run)
    point = dict(backend="shared-iterative", operator="subset-zeta", logN=8,
                 stage_partition=[4, 4], threads=128)
    modules = [compile_module.compile_mapping(
        dict(point, precision=precision), root=ROOT, mathdx=tmp_path / "mathdx",
        nvcc=pathlib.Path("fake-nvcc"), sm=80, cache=tmp_path / "cache")
        for precision in ("uint32", "uint64")]
    assert modules[0] != modules[1]
    assert len(units) == 2
    assert "BooleanZetaOperator<false,std::uint64_t>" in units[1]
    manifest = json.loads((modules[1].parent / "manifest.json").read_text())
    assert manifest["mapping"]["precision"] == "uint64"
