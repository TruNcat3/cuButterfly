"""CPU-only integration contracts for installation -> preparation -> measurement."""
import copy
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import research_platform as platform
import platform_workflow as workflow


def arguments(tmp_path, **updates):
    args = SimpleNamespace(target=ROOT / 'config/platforms/pro6000_server96.json', mode='quick',
        prefix=tmp_path / 'env', python=tmp_path / 'env/bin/python',
        build_dir=tmp_path / 'build', output=tmp_path / 'out', nvcc=tmp_path / 'cuda/bin/nvcc',
        mathdx=tmp_path / 'mathdx/nvidia/mathdx', json_root=None, jobs=4,
        vkfft_library=None, gpuntt_library=None, gpu=None, gpus=None,
        with_vkfft=False, allow_baseline_gaps=False, dry_run=False, prepare_only=False, wait_hours=1)
    vars(args).update(updates)
    return args


def test_all_dry_plan_does_not_query_install_or_write(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(platform.subprocess, 'check_output', lambda *a, **k: pytest.fail('queried hardware'))
    monkeypatch.setattr(workflow, 'Commands', lambda *a: pytest.fail('created command executor'))
    assert platform.main(['all', '--dry-run', '--output', str(tmp_path / 'missing'), '--gpus', '0,1']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['mode'] == 'full' and result['requests'] == 66
    assert result['gpu_access'] is False
    assert not (tmp_path / 'missing').exists()


def test_nested_launcher_environment_matches_direct_watcher(tmp_path, monkeypatch):
    args = arguments(tmp_path, gpu='GPU-fake')
    monkeypatch.setenv('LD_LIBRARY_PATH', '/some/runtime')
    direct = platform.environment(args, gpu=True)
    with platform.process_environment(platform.environment(args)):
        nested = platform.environment(args, gpu=True)
    assert direct == nested


def test_cpu_pipeline_overlaps_independent_preparation_and_reuses_build(tmp_path, monkeypatch):
    args = arguments(tmp_path, prepare_only=True)
    target = platform.config(args.target)
    calls = []
    overlap = threading.Barrier(2)
    monkeypatch.setattr(workflow, 'build_identity', lambda a, target: {'source': 'same', 'python': str(a.python)})
    monkeypatch.setattr(workflow.subprocess, 'check_output', lambda *a, **k: 'sm_120')
    def create_environment(*unused):
        calls.append('environment')
        if not args.python.exists():
            args.python.parent.mkdir(parents=True)
            (args.python.parent / 'python3.11').write_text('interpreter')
            args.python.symlink_to('python3.11')
    monkeypatch.setattr(workflow.dependencies, 'prepare_environment', create_environment)
    monkeypatch.setattr(workflow.dependencies, 'prepare_baseline_sources', lambda *a: calls.append('sources'))
    monkeypatch.setattr(workflow, 'select_devices', lambda *a: pytest.fail('CPU-only queried GPU'))
    class Runner:
        def __init__(self, *a): pass
        def stop(self): pass
        def __call__(self, command, **kwargs):
            assert '--jobs' in command
            action = command[2]
            calls.append(action)
            if action == 'build':
                for name in ('libcuntt.so', 'libvkfft_xd_bridge.so', 'CMakeCache.txt'):
                    (args.build_dir / name).write_text(name)
            elif action == 'prepare':
                assert int(command[command.index('--jobs') + 1]) == 2
                overlap.wait(timeout=5)
            else:
                pytest.fail(action)
    def baselines(baseline_args, *unused):
        assert baseline_args.jobs == 2
        overlap.wait(timeout=5)
        return dict(vkfft_library=args.build_dir / 'libvkfft_xd_bridge.so',
                    gpuntt_library=args.build_dir / 'libgpuntt_semantic_bridge.so')
    monkeypatch.setattr(workflow, 'Commands', Runner)
    monkeypatch.setattr(workflow.dependencies, 'build_baselines', baselines)
    assert workflow.workflow(copy.copy(args), target) == 0
    assert workflow.workflow(copy.copy(args), target) == 0
    assert calls.count('build') == 1
    assert calls.count('prepare') == 2
    assert platform.read(args.output / 'workflow/status.json')['status'] == 'cpu-ready'


def test_changed_build_fails_before_any_rebuild(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    platform.write(args.output / 'workflow/build.json', {'identity': {'version': 1}})
    monkeypatch.setattr(workflow, 'build_identity', lambda *a: {'version': 2})
    with pytest.raises(ValueError, match='Build inputs changed'):
        workflow.ensure_build(args, {}, lambda *a, **k: pytest.fail('rebuilt bound artifact'))


def test_same_workflow_lock_rejects_concurrent_launcher(tmp_path):
    path = tmp_path / 'workflow.lock'
    with workflow.lock(path), pytest.raises(ValueError, match='Another workflow owns'):
        with workflow.lock(path):
            pytest.fail('duplicate owner')


def test_cancel_reaps_only_registered_cpu_process_before_return(tmp_path):
    args = arguments(tmp_path)
    runner = workflow.Commands(args)
    errors = []
    def execute():
        try:
            runner([sys.executable, '-c', 'import time; time.sleep(30)'], label='cancel-test')
        except Exception as error:
            errors.append(error)
    thread = threading.Thread(target=execute)
    thread.start()
    child = None
    deadline = time.monotonic() + 5
    while child is None and time.monotonic() < deadline:
        with runner.mutex:
            child = next(iter(runner.children), None)
        if child is None:
            time.sleep(.01)
    try:
        assert child is not None
        # Same operation as the signal handler; actual cleanup belongs to wait.
        runner.cancelled.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert child.poll() is not None
        assert not runner.children
        assert len(errors) == 1
    finally:
        runner.stop()
        thread.join(timeout=5)


def test_physical_gpu_aliases_cannot_duplicate_measurement(tmp_path, monkeypatch):
    args = arguments(tmp_path, gpus='0,GPU-same')
    monkeypatch.setattr(workflow.subprocess, 'check_output', lambda *a, **k: 'GPU-same')
    monkeypatch.setattr(platform, 'detect', lambda *a: {'uuid': 'GPU-same'})
    with pytest.raises(ValueError, match='same physical GPU'):
        workflow.select_devices(args, {})


def populate_cpu(args):
    folder = args.output / args.mode
    for name in ('binding.json', 'manifest.json', 'precompiled.json'):
        platform.write(folder / name, {'name': name})
    (folder / 'prepared').mkdir()
    (args.output / 'modules').mkdir()


def test_devices_share_only_cpu_inputs_not_calibration_or_results(tmp_path):
    args = arguments(tmp_path)
    populate_cpu(args)
    first = workflow.prepare_device(args, {'uuid': 'GPU-first'})
    second = workflow.prepare_device(args, {'uuid': 'GPU-second'})
    assert first.output != second.output
    for device in (first, second):
        assert (device.output / args.mode / 'prepared').resolve() == args.output / args.mode / 'prepared'
        assert (device.output / 'modules').resolve() == args.output / 'modules'
        assert platform.read(device.output / args.mode / 'binding.json') == {'name': 'binding.json'}
    platform.write(first.output / args.mode / 'run/status.json', {'status': 'pending'})
    assert not (second.output / args.mode / 'run/status.json').exists()


def test_watcher_calls_guarded_driver_directly_and_exports_after_report(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    populate_cpu(args)
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
    monkeypatch.setattr(workflow, 'export_runtime', lambda a: calls.append(('export', {})) or '/runtime.sh')
    result = workflow.measure_device(args, platform.config(args.target), {'uuid': 'GPU-fake-test'}, run)
    assert '--prepare-run-only' in calls[0][0]
    watcher, kwargs = calls[1]
    assert kwargs['gpu_driver'] is True
    assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == 'GPU-fake-test'
    delegated = watcher[watcher.index('--') + 1:]
    assert delegated[1].endswith('run_full_semantic_refresh.py')
    assert '--resume' in delegated
    assert '--wait-hours' in watcher
    assert calls[2][0][2] == 'report'
    assert calls[3][0] == 'export'
    assert result['status'] == 'complete'


def test_runtime_cannot_publish_incomplete_measurement(tmp_path):
    args = arguments(tmp_path)
    platform.write(args.output / args.mode / 'run/status.json', {'status': 'waiting-for-gpu', 'remaining': 2})
    with pytest.raises(ValueError, match='before the finite comparison completes'):
        workflow.export_runtime(args)


def test_precompile_uses_jobs_deduplicates_and_orders_receipt(tmp_path, monkeypatch):
    import compile_module
    import research_compile_requests
    args = arguments(tmp_path, jobs=2)
    original_read = platform.read
    monkeypatch.setattr(platform, 'read', lambda p: {'points': [dict(mapping='a'), dict(mapping='b'), dict(mapping='a')]})
    monkeypatch.setattr(research_compile_requests, 'compile_request', lambda point, policy: {'key': point['mapping']})
    overlap = threading.Barrier(2)
    def compile_one(request, **kwargs):
        overlap.wait(timeout=5)
        path = tmp_path / (request['key'] + '.so'); path.write_text(request['key'])
        return path
    monkeypatch.setattr(compile_module, 'compile_mapping', compile_one)
    platform.precompile(args, {'cuda_arch': 120})
    receipt = original_read(args.output / args.mode / 'precompiled.json')
    assert [item['request']['key'] for item in receipt['modules']] == ['a', 'b']


@pytest.mark.parametrize('flag', ['--prepare-only', '--dry-run', '--gpus'])
def test_all_only_options_cannot_silently_affect_other_commands(flag):
    argv = ['plan', flag] + (['0'] if flag == '--gpus' else [])
    with pytest.raises(ValueError, match='belong to the all action'):
        platform.main(argv)
