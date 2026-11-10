"""An explicit native core fills a lowering gap without disguising cuFFTDx."""
import json
from pathlib import Path
import shutil

import pytest

from test_provider_jit_capabilities import (
    compile_module, compile_kwargs, fake_nvcc, mathdx, register_point,
)


@pytest.mark.parametrize("suffix", [10, 12])
@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_native_suffix_keeps_requested_ept_and_geometry(suffix, precision, mathdx):
    point = dict(register_point(suffix, 32, precision), suffix_core="native",
                 exchange_chunks=[0, 2])
    source = compile_module.render_register_fft(point, 80, "native-test")
    assert f"NativeBlock<{suffix},32,2,Inverse,Complex>" in source
    assert "launch_grouped_suffix_core<" in source
    assert "cufftdx::Block()" not in source
    assert compile_module.provider_capabilities(point, 80) == []
    assert point["suffix_ept"] == 32
    assert point["suffix_threads"] == (1 << suffix) // 32 * 2


def test_native_and_provider_have_distinct_cache_identity(mathdx, tmp_path, monkeypatch):
    fake_nvcc(monkeypatch)
    point = register_point(ept=16)
    kwargs = compile_kwargs(mathdx, tmp_path)
    implicit = compile_module.compile_mapping(point, **kwargs)
    explicit = compile_module.compile_mapping(dict(point, suffix_core="cufftdx-block"), **kwargs)
    native = compile_module.compile_mapping(dict(point, suffix_core="native"), **kwargs)
    assert implicit == explicit and native != implicit
    manifest = json.loads(native.with_name("manifest.json").read_text())
    assert manifest["mapping"]["suffix_core"] == manifest["suffix_core"] == "native"
    assert manifest["provider_capabilities"]["checks"] == []


def test_missing_provider_is_not_silently_replaced(mathdx, tmp_path, monkeypatch):
    calls = fake_nvcc(monkeypatch)
    kwargs = compile_kwargs(mathdx, tmp_path)
    point = register_point(ept=32)
    with pytest.raises(ValueError, match="provider-implementation-missing"):
        compile_module.compile_mapping(point, **kwargs)
    assert not calls
    native = compile_module.compile_mapping(dict(point, suffix_core="native"), **kwargs)
    assert json.loads(native.with_name("manifest.json").read_text())["mapping"]["suffix_ept"] == 32


def test_native_header_is_in_module_identity(mathdx, tmp_path, monkeypatch):
    fake_nvcc(monkeypatch)
    root = tmp_path / "source"
    (root / "src").mkdir(parents=True)
    source = Path(compile_module.__file__).resolve().parents[1]
    for name in ("fft_register_tile.cuh", "fft_grouped_suffix.cuh", "fft_native_block.cuh"):
        shutil.copy2(source / "src" / name, root / "src" / name)
    kwargs = dict(compile_kwargs(mathdx, tmp_path), root=root)
    point = dict(register_point(), suffix_core="native")
    before = compile_module.compile_mapping(point, **kwargs)
    header = root / "src/fft_native_block.cuh"
    header.write_text(header.read_text() + "\n// changed native implementation\n")
    after = compile_module.compile_mapping(point, **kwargs)
    assert before != after
