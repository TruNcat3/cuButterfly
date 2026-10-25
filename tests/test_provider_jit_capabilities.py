"""Provider legality is a CPU check, independent of NVCC and GPU access."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compile_module
import semantic_compile_broker
from cufftdx_capabilities import load_database


@pytest.fixture
def tmp_path(tmpdir):
    return Path(str(tmpdir))


@pytest.fixture
def mathdx(tmp_path):
    root = tmp_path / "mathdx"
    database = root / "include/cufftdx/include/database"
    (database / "detail").mkdir(parents=True)
    (database / "detail/block_fft.hpp").write_text("// no architecture aliases\n")
    records = database / "records/800"
    records.mkdir(parents=True)
    for precision, real, shapes in (("fp64", "double", [(32, 32), (1024, 16), (4096, 16)]),
                                    ("fp32", "float", [(1024, 32), (4096, 32)])):
        for suffix, direction in (("fwd", "forward"), ("inv", "inverse")):
            text = "\n".join(f"""template<> struct block_fft_record<{size}, {real}, fft_type::c2c,
                fft_direction::{direction}, 800> {{
                  static constexpr bool defined = true;
                  using blobs = type_list<block_fft_implementation<{ept}, {ept}, {size // ept}, 1, 8192, 1>>;
                }};""" for size, ept in shapes)
            (records / f"database_{precision}_{suffix}.hpp.inc").write_text(text)
    return root


def register_point(suffix=10, ept=32, precision="fp64"):
    local = 16 - suffix
    prefix_ept = 1 << (local // 2)
    return dict(backend="online-reorder", fft_core="register-tile", precision=precision,
                logN=16, batch=64, local_stages=local, prefix_ept=prefix_ept,
                prefix_threads=32, suffix_ept=ept, suffix_threads=(1 << suffix) // ept * 2)


def compile_kwargs(mathdx, tmp_path):
    return dict(root=ROOT, mathdx=mathdx, nvcc=Path("must-not-run-nvcc"),
                sm=80, cache=tmp_path / "cache")


@pytest.mark.parametrize("suffix", [10, 12])
def test_known_fp64_ept32_rejected_before_nvcc_or_cache(mathdx, tmp_path, monkeypatch, suffix):
    def unexpected(*args, **kwargs):
        raise AssertionError("NVCC must not run for a known missing provider implementation")
    monkeypatch.setattr(compile_module.subprocess, "run", unexpected)
    point = register_point(suffix)
    before = dict(point)
    with pytest.raises(ValueError, match="provider-implementation-missing.*fp64.*EPT=32"):
        compile_module.compile_mapping(point, **compile_kwargs(mathdx, tmp_path))
    assert point == before
    assert not (tmp_path / "cache").exists()
    with pytest.raises(ValueError, match="provider-implementation-missing"):
        compile_module.render_register_fft(point, 80, "test", load_database(mathdx))


@pytest.mark.parametrize("suffix", [10, 12])
def test_fp64_ept16_and_fp32_ept32_are_provider_supported(mathdx, suffix):
    db = load_database(mathdx)
    for point in (register_point(suffix, 16), register_point(suffix, 32, "fp32")):
        source = compile_module.render_register_fft(point, 80, "test", db)
        assert "cubutterfly_module_v1" in source
        assert compile_module.provider_capabilities(point, 80, db)[0]["status"] == "supported"


def test_small_fp64_ept32_is_not_blanket_rejected(mathdx):
    point = dict(register_point(), logN=9, local_stages=4, prefix_ept=4,
                 suffix_ept=32, suffix_threads=16)
    assert "ElementsPerThread<32>" in compile_module.render_register_fft(
        point, 80, "test", load_database(mathdx))


def test_default_direct_render_checks_configured_database(mathdx, monkeypatch):
    monkeypatch.setenv("CUBUTTERFLY_MATHDX_ROOT", str(mathdx))
    with pytest.raises(ValueError, match="provider-implementation-missing"):
        compile_module.render_register_fft(register_point(), 80, "test")


def test_both_directions_are_required(mathdx):
    inverse = mathdx / "include/cufftdx/include/database/records/800/database_fp64_inv.hpp.inc"
    inverse.write_text(inverse.read_text().replace("<16, 16,", "<8, 8,"))
    db = load_database(mathdx)
    assert db.check(800, "fp64", 1024, 16, "forward") == "supported"
    with pytest.raises(ValueError, match="forward and inverse"):
        compile_module.render_register_fft(register_point(ept=16), 80, "test", db)


def test_factor_cores_check_every_factor_and_native_cores_bypass_provider(mathdx):
    point = dict(backend="factor-streamed", fft_core="cufftdx-block", precision="fp64",
                 logN=22, factor_partition=[10, 12], factor_ept=32, factor_columns=1)
    db = load_database(mathdx)
    with pytest.raises(ValueError, match="provider-implementation-missing"):
        compile_module.render_factor_fft(point, 80, "test", db)
    checks = compile_module.provider_capabilities(dict(point, factor_ept=16), 80, db)
    assert [entry["size"] for entry in checks] == [1024, 4096]
    class UnexpectedDatabase:
        def check_both(self, *args):
            raise AssertionError("native factors must not query cuFFTDx")
    native = dict(point, fft_core="register-tile", logN=20, factor_partition=[10, 10])
    assert "NativeFactor<10,32" in compile_module.render_factor_fft(native, 80, "test", UnexpectedDatabase())
    # The native prefix EPT32 is independent of its cuFFTDx suffix EPT16.
    prefix = dict(register_point(ept=16), logN=20, local_stages=10,
                  prefix_ept=32, prefix_threads=128)
    assert compile_module.provider_capabilities(prefix, 80, db)[0]["role"] == "suffix"


def fake_nvcc(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[1:] == ["--version"]:
            return SimpleNamespace(stdout="fake nvcc", stderr="", returncode=0)
        Path(command[-1]).write_bytes(b"fake module")
        return SimpleNamespace(stdout="", stderr="", returncode=0)
    monkeypatch.setattr(compile_module.subprocess, "run", run)
    return calls


def test_unknown_is_compiled_and_audited(tmp_path, monkeypatch):
    calls = fake_nvcc(monkeypatch)
    point = register_point()
    module = compile_module.compile_mapping(point, **compile_kwargs(tmp_path / "missing", tmp_path))
    manifest = json.loads(module.with_name("manifest.json").read_text())
    assert len(calls) == 2
    assert manifest["mapping"] == point
    audit = manifest["provider_capabilities"]
    assert audit["checks"][0]["status"] == "unknown"
    assert audit["database"]["diagnostics"]


def test_mathdx_inc_changes_jit_identity(mathdx, tmp_path, monkeypatch):
    fake_nvcc(monkeypatch)
    kwargs = compile_kwargs(mathdx, tmp_path)
    point = register_point(ept=16)
    first = compile_module.compile_mapping(point, **kwargs)
    path = mathdx / "include/cufftdx/include/database/records/800/database_fp64_fwd.hpp.inc"
    path.write_text(path.read_text() + "\n// provider database revision\n")
    assert compile_module.compile_mapping(point, **kwargs) != first


def test_broker_identity_covers_parser_and_mathdx_inc(mathdx, tmp_path):
    root = tmp_path / "source"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for name in ("compile_module.py", "resident_mapping.py", "cufftdx_capabilities.py"):
        (scripts / name).write_text("# initial\n")
    config = dict(root=str(root), mathdx=str(mathdx))
    first = semantic_compile_broker._sources(config)
    (scripts / "cufftdx_capabilities.py").write_text("# parser revision\n")
    second = semantic_compile_broker._sources(config)
    assert second != first
    table = mathdx / "include/cufftdx/include/database/records/800/database_fp64_inv.hpp.inc"
    table.write_text(table.read_text() + "\n// table revision\n")
    assert semantic_compile_broker._sources(config) != second
