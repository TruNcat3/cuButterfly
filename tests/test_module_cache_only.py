"""Exercise real cache identity and hit/miss control flow without invoking nvcc."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import compile_module


def test_cache_only_miss_hit_and_source_identity(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    (root / 'include').mkdir(parents=True)
    (root / 'src').mkdir()
    for name in ('operator_traits.cuh', 'ntt_traits.cuh', 'shared_iterative.cuh'):
        (root / 'src' / name).write_text('// frozen template\n')
    calls = []
    def compiler_version_only(command, **kwargs):
        calls.append(command)
        assert command == ['nvcc', '--version'], 'cache-only attempted actual compilation'
        return SimpleNamespace(stdout='mock nvcc version', returncode=0)
    monkeypatch.setattr(compile_module.subprocess, 'run', compiler_version_only)
    monkeypatch.setattr(compile_module, 'load_database', lambda path: object())
    monkeypatch.setattr(compile_module, 'provider_capabilities', lambda *args: [])
    def forbidden_render(*args):
        pytest.fail('cache-only miss generated source')
    monkeypatch.setattr(compile_module, 'render_shared', forbidden_render)
    monkeypatch.setenv('CUBUTTERFLY_JIT_CACHE_ONLY', '1')
    point = dict(backend='shared-iterative', operator='fft', precision='fp16', logN=12,
                 stage_partition=[12], threads=128)
    args = dict(root=root, mathdx=tmp_path / 'mathdx', nvcc=Path('nvcc'), sm=80,
                cache=tmp_path / 'cache')
    with pytest.raises(RuntimeError, match='cache-miss.*forbids compilation'):
        compile_module.compile_mapping(point, **args)
    directories = list(args['cache'].iterdir())
    assert len(directories) == 1
    assert sorted(p.name for p in directories[0].iterdir()) == ['compile.lock']
    module = directories[0] / 'module.so'
    module.write_bytes(b'CPU fixture, never loaded')
    assert compile_module.compile_mapping(point, **args) == module
    # Cache-only must still hash actual source inputs before accepting a hit.
    (root / 'src' / 'shared_iterative.cuh').write_text('// changed template\n')
    with pytest.raises(RuntimeError, match='cache-miss'):
        compile_module.compile_mapping(point, **args)
    assert len(calls) == 3
