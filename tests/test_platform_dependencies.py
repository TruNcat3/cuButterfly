"""Provisioning contracts; all subprocess/network operations are mocked."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import platform_dependencies as deps


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(deps, 'ROOT', tmp_path / 'repo')
    args = SimpleNamespace(prefix=tmp_path / 'env', python=tmp_path / 'env/bin/python',
                           nvcc=tmp_path / 'cuda/bin/nvcc', mathdx=tmp_path / 'mathdx/nvidia/mathdx',
                           build_dir=tmp_path / 'build', output=tmp_path / 'out', jobs=3)
    for file in (args.python, args.nvcc, args.build_dir / 'libcuntt.so', args.build_dir / 'libvkfft_xd_bridge.so'):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('fixture')
    (args.prefix / 'conda-meta').mkdir()
    for path in deps._gpuntt_sources():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('source fixture')
    target = dict(cuda_arch=120, mathdx_version='25.6.0')
    calls = []
    def run(command, env=None, label=''):
        assert env['CUDA_VISIBLE_DEVICES'] == ''
        assert env['PATH'].split(':')[:2] == [str(args.prefix / 'bin'), str(args.nvcc.parent)]
        assert 'nvidia-smi' not in command
        calls.append((command, env, label))
    return args, target, calls, run


def packages(**extra):
    return dict.fromkeys(['cmake', 'ninja', 'packaging', 'wheel', 'setuptools', 'numpy', 'torch'], '1') | extra


def existing_mathdx(args):
    (args.mathdx / 'include').mkdir(parents=True)
    (args.mathdx / 'include/cufftdx.hpp').write_text('header')
    metadata = args.mathdx.parent.parent / 'nvidia_mathdx-25.6.0.dist-info/METADATA'
    metadata.parent.mkdir()
    metadata.write_text('Version: 25.6.0\n')


def test_environment_reuses_present_packages_and_pinned_sources(setup, monkeypatch):
    args, target, calls, run = setup
    existing_mathdx(args)
    monkeypatch.setattr(deps, '_packages', lambda a: packages())
    monkeypatch.setattr(deps, '_torch_target', lambda *a: None)
    source = deps.ROOT / 'external/VkFFT'
    (source / '.git').mkdir(parents=True)
    monkeypatch.setattr(deps, '_read', lambda cmd, args: deps.VKFFT if cmd[-1] == 'HEAD' else '')
    deps.prepare_environment(args, target, run)
    assert calls == []


def test_torch_mismatch_never_replaces_existing_package(setup, monkeypatch):
    args, target, calls, run = setup
    monkeypatch.setattr(deps, '_packages', lambda a: packages())
    monkeypatch.setattr(deps, '_read', lambda *a: 'sm_80 sm_90')
    with pytest.raises(ValueError, match='Existing Torch was preserved'):
        deps.prepare_environment(args, target, run)
    assert calls == []


def test_source_mismatch_is_not_checked_out_or_reset(setup, monkeypatch):
    args, target, calls, run = setup
    source = deps.ROOT / ('external/GPU-NTT-' + deps.GPUNTT)
    (source / '.git').mkdir(parents=True)
    monkeypatch.setattr(deps, '_read', lambda cmd, args: 'wrong' if cmd[-1] == 'HEAD' else '')
    with pytest.raises(ValueError, match='differs or is dirty'):
        deps.prepare_baseline_sources(args, target, run)
    assert calls == []


def test_partial_gpuntt_output_preserved_with_new_build_path(setup, monkeypatch):
    args, target, calls, run = setup
    binary = args.build_dir / 'libgpuntt_semantic_bridge.so'
    binary.write_text('interrupted')
    monkeypatch.setattr(deps, '_packages', lambda a: packages())
    artifact = args.prefix / 'dao.so'
    artifact.write_text('extension')
    monkeypatch.setattr(deps, '_read', lambda *a: json.dumps([str(artifact)]))
    def build(command, env=None, label=''):
        run(command, env, label)
        if label == 'gpuntt-build':
            make_gpuntt(args, target, Path(env['GPUNTT_OUTPUT']), receipt=False)
    result = deps.build_baselines(args, target, build)
    assert result['gpuntt_library'] != binary
    assert args.build_dir / 'baselines' in result['gpuntt_library'].parents
    assert binary.read_text() == 'interrupted'
    assert calls[0][2] == 'gpuntt-build'
    assert calls[0][1]['GPUNTT_BUILD_DIR'].startswith(str(result['gpuntt_library'].parent))


def make_gpuntt(args, target, binary=None, receipt=True):
    binary = binary or args.build_dir / 'libgpuntt_semantic_bridge.so'
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text('binary')
    helper = args.build_dir / 'libcuntt.so'
    def entry(path):
        return dict(path=str(path), sha256=deps._hash(path), bytes=path.stat().st_size)
    manifest = dict(schema='cubutterfly-gpuntt-build-v1', vendor_revision=deps.GPUNTT,
                    architecture='sm120 + compute120 PTX', binary=entry(binary),
                    root_helper_library=entry(helper), vendor_archive=entry(binary),
                    sources=[entry(path) for path in deps._gpuntt_sources()],
                    build_command=f'{args.nvcc} --generate-code=arch=compute_120,code=sm_120 --generate-code=arch=compute_120,code=compute_120')
    deps._write(Path(str(binary) + '.manifest.json'), manifest)
    if receipt:
        deps._write(Path(str(binary) + '.platform.json'), dict(identity=deps._gpuntt_identity(args, target)))
    return binary


def test_complete_baselines_reuse_without_execution_and_detect_mutation(setup, monkeypatch):
    args, target, calls, run = setup
    binary = make_gpuntt(args, target)
    artifact = args.prefix / 'dao.so'
    artifact.write_text('extension')
    deps._write(args.prefix / 'share/cuButterfly/dao-platform-build.json',
                dict(identity=deps._identity(args, target, deps.DAO),
                     artifacts=[dict(path=str(artifact), sha256=deps._hash(artifact))]))
    monkeypatch.setattr(deps, '_packages', lambda a: packages(**{'fast-hadamard-transform': '1'}))
    result = deps.build_baselines(args, target, run)
    assert result['gpuntt_library'] == binary
    assert calls == []
    binary.write_text('modified')
    def rebuild(command, env=None, label=''):
        run(command, env, label)
        assert label == 'gpuntt-build'
        make_gpuntt(args, target, Path(env['GPUNTT_OUTPUT']), receipt=False)
    replacement = deps.build_baselines(args, target, rebuild)
    assert replacement['gpuntt_library'] != binary
    assert binary.read_text() == 'modified'
    assert len(calls) == 1


@pytest.mark.parametrize('stale_receipt', [False, True])
def test_unreceipted_or_stale_dao_is_rebuilt_without_dependencies(setup, monkeypatch, stale_receipt):
    args, target, calls, run = setup
    make_gpuntt(args, target)
    monkeypatch.setattr(deps, '_packages', lambda a: packages(**{'fast-hadamard-transform': '1'}))
    receipt = args.prefix / 'share/cuButterfly/dao-platform-build.json'
    if stale_receipt:
        deps._write(receipt, dict(identity={'old': True}, artifacts=[]))
    artifact = args.prefix / 'dao.so'
    artifact.write_text('rebuilt pinned extension')
    monkeypatch.setattr(deps, '_read', lambda *a: json.dumps([str(artifact)]))
    deps.build_baselines(args, target, run)
    assert len(calls) == 1
    assert calls[0][2] == 'dao-build'
    assert '--no-deps' in calls[0][0]
    assert '--force-reinstall' in calls[0][0]
    assert 'torch' not in calls[0][0]
    assert json.loads(receipt.read_text())['identity'] == deps._identity(args, target, deps.DAO)


def test_failed_dao_install_can_retry_without_false_receipt(setup, monkeypatch):
    args, target, calls, run = setup
    make_gpuntt(args, target)
    monkeypatch.setattr(deps, '_packages', lambda a: packages())
    def fail(command, env=None, label=''):
        run(command, env, label)
        raise RuntimeError('interrupted pip')
    receipt = args.prefix / 'share/cuButterfly/dao-platform-build.json'
    for _ in range(2):
        with pytest.raises(RuntimeError, match='interrupted pip'):
            deps.build_baselines(args, target, fail)
        assert not receipt.exists()
    assert len(calls) == 2
    assert all(call[2] == 'dao-build' for call in calls)
    assert calls[0][1]['MAX_JOBS'] == '3'


def test_reads_hide_gpu_without_querying_device(setup, monkeypatch):
    args, _, _, _ = setup
    def check(command, env, text):
        assert env['CUDA_VISIBLE_DEVICES'] == ''
        assert command == [str(args.python), '-c', 'metadata-only']
        return 'ok\n'
    monkeypatch.setattr(deps.subprocess, 'check_output', check)
    assert deps._read([args.python, '-c', 'metadata-only'], args) == 'ok'


def test_absent_packages_only_are_installed_cpu_only(setup, monkeypatch):
    args, target, calls, run = setup
    existing_mathdx(args)
    monkeypatch.setattr(deps, '_packages', lambda a: packages(ninja=None, torch=None))
    monkeypatch.setattr(deps, '_torch_target', lambda *a: None)
    monkeypatch.setattr(deps, '_source', lambda *a: None)
    deps.prepare_environment(args, target, run)
    assert [call[2] for call in calls] == ['python-prerequisites', 'torch']
    assert calls[0][0][4:] == ['ninja']
    assert '--upgrade' not in calls[0][0]
    assert calls[1][0][-1] == 'https://download.pytorch.org/whl/cu128'


def test_mathdx_mismatch_preserves_directory(setup, monkeypatch):
    args, target, calls, run = setup
    existing_mathdx(args)
    target['mathdx_version'] = 'different'
    monkeypatch.setattr(deps, '_packages', lambda a: packages())
    monkeypatch.setattr(deps, '_torch_target', lambda *a: None)
    with pytest.raises(ValueError, match='Existing MathDx'):
        deps.prepare_environment(args, target, run)
    assert calls == []
    assert (args.mathdx / 'include/cufftdx.hpp').read_text() == 'header'


def test_forbidden_path_spelling_is_rejected_without_filesystem_probe(monkeypatch):
    monkeypatch.setattr(Path, 'resolve', lambda *a: pytest.fail('resolved forbidden mount'))
    with pytest.raises(ValueError, match='must not use'):
        deps._hash('/sdb/private/file')


def test_runbook_native_manifest_is_adopted_without_rebuild(setup, monkeypatch):
    args, target, calls, run = setup
    binary = make_gpuntt(args, target, receipt=False)
    artifact = args.prefix / 'dao.so'
    artifact.write_text('extension')
    deps._write(args.prefix / 'share/cuButterfly/dao-platform-build.json',
                dict(identity=deps._identity(args, target, deps.DAO),
                     artifacts=[dict(path=str(artifact), sha256=deps._hash(artifact))]))
    monkeypatch.setattr(deps, '_packages', lambda a: packages(**{'fast-hadamard-transform': '1'}))
    result = deps.build_baselines(args, target, run)
    assert result['gpuntt_library'] == binary
    assert calls == []
    receipt = json.loads(Path(str(binary) + '.platform.json').read_text())
    assert receipt['identity'] == deps._gpuntt_identity(args, target)
    assert 'adopted native CPU build provenance' in receipt['provenance']
    assert 'no calibration' in receipt['provenance']


def test_wrong_native_compiler_is_not_adopted(setup):
    args, target, _, _ = setup
    binary = make_gpuntt(args, target, receipt=False)
    manifest = Path(str(binary) + '.manifest.json')
    record = json.loads(manifest.read_text())
    record['build_command'] = record['build_command'].replace(str(args.nvcc), '/other/nvcc')
    deps._write(manifest, record)
    with pytest.raises(ValueError, match='compiler/target/source provenance'):
        deps._validate_gpuntt(binary, args.build_dir / 'libcuntt.so', 120, args.nvcc)


def test_source_change_changes_content_identity(setup):
    args, target, _, _ = setup
    original = deps._gpuntt_identity(args, target)
    deps._gpuntt_sources()[0].write_text('changed implementation')
    assert deps._gpuntt_identity(args, target) != original


def test_compile_policy_and_foreign_profiles_are_cleared(setup, monkeypatch):
    args, _, _, _ = setup
    names = ['CUBUTTERFLY_JIT_CACHE_ONLY', 'CUBUTTERFLY_COMPILE_BUDGET_FILE',
             'CUBUTTERFLY_COMPILE_BUDGET_SECONDS', 'CUBUTTERFLY_HARDWARE_MODEL']
    for name in names:
        monkeypatch.setenv(name, 'foreign')
    env = deps._env(args)
    assert not any(name in env for name in names)
    assert env['CUDA_VISIBLE_DEVICES'] == ''
