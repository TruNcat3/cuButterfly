"""CPU-only contracts for the finite PRO6000 migration entry point."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import research_platform as platform
import run_current_recalibration as calibration
import semantic_campaign_manifest as manifests

TARGET = ROOT / 'config/platforms/pro6000_server96.json'


def test_plan_is_subprocess_and_gpu_free_and_freezes_scope(tmp_path, monkeypatch, capsys):
    for name in ('run', 'check_output', 'Popen'):
        monkeypatch.setattr(platform.subprocess, name,
                            lambda *a, **kw: pytest.fail('plan launched a subprocess'))
    monkeypatch.setattr(platform, 'detect', lambda *a: pytest.fail('plan queried GPU'))
    for mode, count in [('quick', 12), ('full', 66)]:
        assert platform.main(['plan', '--mode', mode, '--output', str(tmp_path / 'out')]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result['gpu_access'] is False
        assert result['target']['modes'][mode]['requests'] == count
        assert '--cuda-arch' in result['commands']['build']
        assert '120' in result['commands']['build']
    assert not (tmp_path / 'out').exists()


def test_quick_is_unchanged_subset_of_frozen_full_population():
    target = platform.config(TARGET)
    full = manifests.load_config(ROOT / target['modes']['full']['config'])
    quick = manifests.load_config(ROOT / target['modes']['quick']['config'])
    assert len(full['populations']) == 66
    assert len(quick['populations']) == 12
    assert all(item in full['populations'] for item in quick['populations'])


@pytest.mark.parametrize('changed', [dict(name='NVIDIA A100-PCIE-80GB'),
    dict(compute_cap='8.0'), dict(memory_mib=81920), dict(memory_mib=110000)])
def test_device_identity_rejects_wrong_model_sm_or_memory(changed):
    observed = dict(name='NVIDIA RTX PRO 6000 Blackwell Server Edition',
                    compute_cap='12.0', memory_mib=96000)
    target = platform.config(TARGET)
    platform.device_matches(target, observed)
    with pytest.raises(ValueError):
        platform.device_matches(target, dict(observed, **changed))


def test_cpu_environment_hides_gpu_and_discards_foreign_profiles(tmp_path, monkeypatch):
    for name in ('CUBUTTERFLY_HARDWARE_MODEL', 'CUBUTTERFLY_PROFILE_ROOT',
                 'CUBUTTERFLY_HARDWARE_PROFILE', 'CUBUTTERFLY_COST_MODEL', 'CUBUTTERFLY_REGISTRY'):
        monkeypatch.setenv(name, '/foreign/a100')
    args = SimpleNamespace(gpu='GPU-test', mathdx=tmp_path / 'mathdx', nvcc=tmp_path / 'cuda/bin/nvcc',
                           output=tmp_path / 'out', build_dir=tmp_path / 'build')
    env = platform.environment(args)
    assert env['CUDA_VISIBLE_DEVICES'] == ''
    assert not any(name in env for name in ('CUBUTTERFLY_HARDWARE_MODEL',
        'CUBUTTERFLY_PROFILE_ROOT', 'CUBUTTERFLY_HARDWARE_PROFILE',
        'CUBUTTERFLY_COST_MODEL', 'CUBUTTERFLY_REGISTRY'))
    assert platform.environment(args, gpu=True)['CUDA_VISIBLE_DEVICES'] == 'GPU-test'


@pytest.mark.parametrize('path', ['/sdb', '/sdb/output', '/mnt/sdb', '/mnt/sdb/output'])
def test_unsafe_paths_are_rejected(path):
    with pytest.raises(ValueError, match='local SSD or NAS'):
        platform.safe_path(path)


@pytest.fixture
def entry(tmp_path, monkeypatch):
    build = tmp_path / 'build'
    build.mkdir()
    (build / 'libcuntt.so').write_bytes(b'mock library')
    output = tmp_path / 'output'
    monkeypatch.setattr(calibration, 'validate_build', lambda path: 120)
    calls = []
    monkeypatch.setattr(platform.subprocess, 'run', lambda command, **kwargs: calls.append((command[:], kwargs)))
    args = ['--build-dir', str(build), '--output', str(output), '--gpu', 'GPU-test']
    return SimpleNamespace(build=build, output=output, args=args, calls=calls)


def test_run_without_prepare_never_queries_gpu_or_executes(entry, monkeypatch):
    monkeypatch.setattr(platform, 'detect', lambda *a: pytest.fail('GPU query before preparation'))
    with pytest.raises(ValueError, match='run prepare first'):
        platform.main(['run', *entry.args])
    assert entry.calls == []


def ready(entry, monkeypatch, gaps=None):
    folder = entry.output / 'quick'
    folder.mkdir(parents=True)
    platform.write(folder / 'precompiled.json', {'status': 'cpu-compiled', 'cuda_arch': 120, 'modules': []})
    monkeypatch.setattr(platform, 'detect', lambda *a: {'uuid': 'GPU-test'})
    monkeypatch.setattr(platform, 'doctor', lambda *a: dict(status='cpu-prerequisites-present', baseline_gaps=gaps or []))
    return folder


def test_missing_baselines_block_default_run(entry, monkeypatch):
    ready(entry, monkeypatch, ['Dao FWHT absent'])
    with pytest.raises(ValueError, match='baseline prerequisites missing'):
        platform.main(['run', *entry.args])
    assert entry.calls == []


@pytest.mark.parametrize('resume', [False, True])
def test_run_prepares_then_resumes_then_reports(entry, monkeypatch, resume):
    folder = ready(entry, monkeypatch)
    if resume:
        platform.write(folder / 'run/status.json', {'status': 'checkpoint'})
    assert platform.main(['run', *entry.args]) == 0
    commands = [command for command, kwargs in entry.calls]
    assert len(commands) == 3
    assert '--prepare-only' in commands[0]
    assert ('--resume' in commands[0]) == resume
    assert '--resume' in commands[1] and '--prepare-only' not in commands[1]
    assert commands[2][1].endswith('summarize_semantic_campaign.py')
    assert entry.calls[0][1]['env']['CUDA_VISIBLE_DEVICES'] == 'GPU-test'
    assert entry.calls[2][1]['env']['CUDA_VISIBLE_DEVICES'] == ''


def test_changed_prepared_binding_blocks_before_gpu_or_execution(entry, monkeypatch):
    folder = ready(entry, monkeypatch)
    platform.write(folder / 'binding.json', {'old': 'different-library'})
    monkeypatch.setattr(platform, 'detect', lambda *a: pytest.fail('queried GPU for changed binding'))
    with pytest.raises(ValueError, match='prepared build/config changed'):
        platform.main(['run', *entry.args])
    assert entry.calls == []


def test_prepare_orders_cpu_steps_and_scrubs_inprocess_environment(entry, monkeypatch):
    import os
    import run_semantic_campaign as semantic
    events = []
    monkeypatch.setenv('CUBUTTERFLY_HARDWARE_MODEL', '/foreign/a100.json')
    monkeypatch.setenv('CUBUTTERFLY_PROFILE_ROOT', '/foreign/a100-profile')
    monkeypatch.setattr(platform, 'detect', lambda *a: pytest.fail('prepare queried GPU'))
    monkeypatch.setattr(semantic, 'validate_fingerprints', lambda *a: events.append('validate'))
    def execute(command, **kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        assert 'CUBUTTERFLY_HARDWARE_MODEL' not in kwargs['env']
        if command[1].endswith('semantic_campaign_manifest.py'):
            events.append('manifest')
            platform.write(entry.output / 'quick/manifest.json', {'metrics': {'unique_cases': 12}})
        elif command[1].endswith('semantic_preparation_queue.py'):
            events.append('references')
        else:
            pytest.fail('unexpected preparation command')
    def compile_cpu(*args):
        assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
        assert 'CUBUTTERFLY_HARDWARE_MODEL' not in os.environ
        assert 'CUBUTTERFLY_PROFILE_ROOT' not in os.environ
        events.append('precompile')
    monkeypatch.setattr(platform.subprocess, 'run', execute)
    monkeypatch.setattr(platform, 'precompile', compile_cpu)
    with mock.patch.dict(os.environ):
        assert platform.main(['prepare', *entry.args]) == 0
    assert events == ['manifest', 'validate', 'references', 'precompile']


def test_unsafe_target_is_rejected_before_config_read(monkeypatch):
    monkeypatch.setattr(platform, 'config', lambda *a: pytest.fail('unsafe target read before rejection'))
    with pytest.raises(ValueError, match='local SSD or NAS'):
        platform.main(['plan', '--target', '/sdb/forbidden-platform.json'])


def test_changed_precompiled_module_blocks_gpu(entry, monkeypatch):
    folder = ready(entry, monkeypatch)
    module = entry.output / 'module.so'
    module.write_bytes(b'changed')
    platform.write(folder / 'precompiled.json', dict(status='cpu-compiled', cuda_arch=120,
        modules=[dict(path=str(module), sha256='wrong')]))
    monkeypatch.setattr(platform, 'detect', lambda *a: pytest.fail('queried GPU for stale module'))
    with pytest.raises(ValueError, match='module missing or changed'):
        platform.main(['run', *entry.args])


def test_cpu_process_environment_is_restored(monkeypatch):
    import os
    monkeypatch.setenv('CUBUTTERFLY_HARDWARE_MODEL', '/foreign/profile')
    before = dict(os.environ)
    with pytest.raises(RuntimeError), platform.process_environment({'CUDA_VISIBLE_DEVICES': ''}):
        assert 'CUBUTTERFLY_HARDWARE_MODEL' not in os.environ
        raise RuntimeError('compilation failed')
    assert dict(os.environ) == before


@pytest.mark.parametrize('nvcc_codes,torch_flags,expected', [
    ('sm_80 sm_120', 'sm_70 sm_80 sm_90', 'blocked-prerequisites'),
    ('sm_80 sm_90', 'sm_80 sm_120', 'blocked-prerequisites'),
    ('sm_80 sm_120', 'sm_80 sm_120', 'cpu-prerequisites-present'),
])
def test_doctor_checks_compiler_and_torch_targets_without_gpu(
        tmp_path, monkeypatch, nvcc_codes, torch_flags, expected):
    prefix = tmp_path / 'prefix'
    (prefix / 'conda-meta').mkdir(parents=True)
    python = prefix / 'bin/python'
    python.parent.mkdir()
    python.touch()
    mathdx = tmp_path / 'mathdx'
    (mathdx / 'include').mkdir(parents=True)
    (mathdx / 'include/cufftdx.hpp').touch()
    vkfft, gpuntt = tmp_path / 'vkfft.so', tmp_path / 'gpuntt.so'
    vkfft.touch()
    gpuntt.touch()
    args = SimpleNamespace(prefix=prefix, python=python, mathdx=mathdx,
        nvcc=tmp_path / 'cuda/bin/nvcc', build_dir=tmp_path / 'build',
        output=tmp_path / 'output', gpu='GPU-must-not-query',
        vkfft_library=vkfft, gpuntt_library=gpuntt)
    calls = []
    def query(command, **kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        calls.append(command)
        if command == [str(args.nvcc), '--list-gpu-code']:
            return nvcc_codes
        assert command[:2] == [str(python), '-c']
        code = command[2]
        assert 'torch.cuda' not in code
        assert 'nvidia-smi' not in code
        assert 'cuda.init' not in code
        if '_cuda_getArchFlags' in code:
            return json.dumps({'cuda': '12.8', 'arch_flags': torch_flags})
        assert 'importlib.metadata' in code
        return json.dumps({'numpy': '2.0', 'torch': '2.5.1',
                           'fast-hadamard-transform': '1.0'})
    monkeypatch.setattr(platform.subprocess, 'check_output', query)
    monkeypatch.setattr(platform.subprocess, 'run', lambda *a, **kw: pytest.fail('doctor ran subprocess'))
    monkeypatch.setattr(platform, 'detect', lambda *a: pytest.fail('doctor queried GPU'))
    result = platform.doctor(args, platform.config(TARGET))
    assert result['status'] == expected
    assert result['gpu_access'] is False
    assert result['baseline_gaps'] == []
    assert len(calls) == 3
    checks = {row['name']: row['ok'] for row in result['checks']}
    assert checks['torch'] is True
    assert checks['torch_target'] == ('sm_120' in torch_flags.split())
    assert checks['nvcc_target'] == ('sm_120' in nvcc_codes.split())
