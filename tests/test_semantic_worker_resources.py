"""Resource guards preserve complete candidate selection and uncommitted work."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import semantic_worker_resources as resource
import semantic_resource_compile_proxy as proxy
import semantic_gpu_pipeline as gpu


@pytest.fixture(autouse=True)
def isolate_helper_events(monkeypatch):
    monkeypatch.delenv('CUBUTTERFLY_PREPARE_EVENTS', raising=False)
    monkeypatch.delenv('CUBUTTERFLY_HELPER_ATTEMPT', raising=False)


def test_linux_resource_snapshot(tmp_path):
    process = tmp_path / '42'
    process.mkdir()
    (process / 'maps').write_text('0001 /cache/a/module.so\n0002 [heap]\n0003 /cache/a/module.so\n')
    (process / 'status').write_text('VmRSS:\t11 kB\nVmSize:\t22 kB\n')
    (process / 'fd').mkdir()
    (process / 'fd' / '0').touch()
    (tmp_path / 'sys/vm').mkdir(parents=True)
    (tmp_path / 'sys/vm/max_map_count').write_text('65530\n')
    measured = resource.snapshot(42, tmp_path)
    assert measured == dict(pid=42, vma_count=3, rss_bytes=11264, virtual_bytes=22528,
                            fd_count=1, module_mappings=2, max_map_count=65530)


@pytest.mark.parametrize('field,value,reason', [('vma_count', 45000, 'vma-count'),
                                              ('rss_bytes', 8 * 1024**3, 'rss-bytes')])
def test_resource_thresholds(field, value, reason):
    sample = dict(vma_count=0, rss_bytes=0, max_map_count=65530)
    sample[field] = value
    assert resource.exceeded(sample, resource.limits()).startswith(reason)


def test_kernel_vma_limit_reserves_quarter_even_with_large_requested_budget():
    sample = dict(vma_count=750, rss_bytes=0, max_map_count=1000)
    assert resource.exceeded(sample, resource.limits(max_vmas=900))
    sample['vma_count'] = 749
    assert resource.exceeded(sample, resource.limits(max_vmas=900)) is None


def test_completed_and_attempt_budgets():
    sample = dict(vma_count=1, rss_bytes=0, max_map_count=65530)
    assert resource.exceeded(sample, resource.limits(), completed=128).startswith('completed-cases')
    assert resource.exceeded(sample, resource.limits(), attempts=64).startswith('plan-attempts')


def test_proxy_forwards_exact_arguments_without_gpu_import(monkeypatch):
    monkeypatch.setattr(proxy, 'snapshot', lambda pid: dict(vma_count=1, rss_bytes=0, max_map_count=65530))
    arguments = ['compile_module.py', '--mapping', '{"test":1}']
    received = []
    monkeypatch.setattr(proxy.broker, 'main', lambda args: received.append(args) or 9)
    assert proxy.main(arguments) == 9
    assert received == [arguments]


def test_proxy_cpu_script_preserves_interpreter_arguments_and_exit(monkeypatch):
    monkeypatch.setattr(proxy, 'snapshot', lambda pid: dict(vma_count=1, rss_bytes=0, max_map_count=65530))
    arguments = ['/a path/rank_static_candidates.py', '--request', '/request path/input.json',
                 '--output', '/output path/result.json']
    received = []
    monkeypatch.setattr(proxy.broker, 'main', lambda args: pytest.fail('CPU helper sent to compile broker'))
    monkeypatch.setattr(proxy.subprocess, 'run', lambda args, **kwargs:
                        received.append((args, kwargs)) or SimpleNamespace(returncode=7))
    assert proxy.main(arguments) == 7
    assert received == [([sys.executable, *arguments], {'check': False})]


def test_proxy_main_argv_uses_transparent_cpu_interpreter(monkeypatch, tmp_path):
    import subprocess

    helper = tmp_path / 'ordinary CPU helper.py'
    helper.write_text('import json, sys\nprint(json.dumps(sys.argv[1:]))\nraise SystemExit(9)\n')
    arguments = ['--request', 'path with spaces', '--output', 'result.json']
    completed = subprocess.run([sys.executable, str(Path(proxy.__file__)), str(helper), *arguments],
                               text=True, capture_output=True, check=False)
    assert completed.returncode == 9
    assert json.loads(completed.stdout) == arguments
    assert completed.stderr == ''


@pytest.mark.parametrize('exit_code', [0, 9])
def test_first_cpu_helper_creates_log_before_any_broker_request(monkeypatch, tmp_path, exit_code):
    import subprocess
    events = tmp_path / 'new-campaign/measurement/compile_events/first-1.jsonl'
    helper = tmp_path / 'helper.py'
    helper.write_text('import os\nfrom pathlib import Path\n'
        'assert Path(os.environ["CUBUTTERFLY_PREPARE_EVENTS"]).is_file()\n'
        f'raise SystemExit({exit_code})\n')
    monkeypatch.setenv('CUBUTTERFLY_PREPARE_EVENTS', str(events))
    monkeypatch.setenv('CUBUTTERFLY_HELPER_ATTEMPT', 'fresh-attempt')
    assert not events.parent.exists()
    result = subprocess.run([sys.executable, proxy.__file__, str(helper)],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == exit_code, result.stderr
    records = [json.loads(line) for line in events.read_text().splitlines()]
    assert [r['kind'] for r in records] == ['cpu_helper_started', 'cpu_helper_finished']
    assert all(r['attempt_token'] == 'fresh-attempt' for r in records)
    assert records[-1]['returncode'] == exit_code
    assert records[-1]['termination_signal'] is None
    assert records[-1]['received_signals'] == []
    gpu.check_preparation_events(events)


def test_bad_log_destination_prevents_helper_spawn(monkeypatch, tmp_path):
    parent = tmp_path / 'not-directory'
    parent.write_text('existing unrelated data')
    monkeypatch.setenv('CUBUTTERFLY_PREPARE_EVENTS', str(parent / 'events.jsonl'))
    monkeypatch.setattr(proxy.subprocess, 'Popen', lambda *a, **k: pytest.fail('spawned without a log'))
    with pytest.raises(FileExistsError):
        proxy.run_cpu_helper(['helper.py'])
    assert parent.read_text() == 'existing unrelated data'


def test_receipt_write_failure_reaps_started_cpu_helper(monkeypatch, tmp_path):
    from contextlib import contextmanager
    import subprocess
    monkeypatch.setenv('CUBUTTERFLY_PREPARE_EVENTS', str(tmp_path / 'events.jsonl'))
    helper = tmp_path / 'helper.py'
    helper.write_text('import time\ntime.sleep(60)\n')
    class BrokenLog:
        def write(self, value):
            raise OSError('simulated receipt write failure')
    @contextmanager
    def broken_log(path):
        yield BrokenLog()
    monkeypatch.setattr(proxy, 'open_event_log', broken_log)
    children = []
    spawn = subprocess.Popen
    def observed(*args, **kwargs):
        child = spawn(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(proxy.subprocess, 'Popen', observed)
    try:
        with pytest.raises(OSError, match='receipt write failure'):
            proxy.run_cpu_helper([str(helper)])
        assert len(children) == 1 and children[0].poll() is not None
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)


def test_proxy_runs_actual_cpu_static_ranker_not_broker(tmp_path):
    import subprocess

    request = tmp_path / 'request.json'
    request.write_text('{"schema":"invalid-test-request"}')
    output = tmp_path / 'ranking.json'
    completed = subprocess.run([sys.executable, str(Path(proxy.__file__)),
                               str(Path(proxy.__file__).with_name('rank_static_candidates.py')),
                               '--request', str(request), '--output', str(output)],
                              text=True, capture_output=True, check=False)
    assert completed.returncode == 1
    assert 'static ranking error: unsupported static-ranking request schema' in completed.stderr
    assert 'unrecognized arguments' not in completed.stderr
    assert not output.exists()


def test_proxy_compiler_path_still_goes_to_broker_validation(monkeypatch):
    monkeypatch.setattr(proxy, 'snapshot', lambda pid: dict(vma_count=1, rss_bytes=0, max_map_count=65530))
    arguments = ['/wrong root/compile_module.py', '--mapping', '{}']
    monkeypatch.setattr(proxy.subprocess, 'run', lambda *args, **kwargs:
                        pytest.fail('compiler bypassed broker validation'))
    received = []
    monkeypatch.setattr(proxy.broker, 'main', lambda args: received.append(args) or 4)
    assert proxy.main(arguments) == 4
    assert received == [arguments]


def test_proxy_resource_event_discards_selection_without_querying_broker(monkeypatch, tmp_path):
    events = tmp_path / 'fresh/compile_events/events.jsonl'
    monkeypatch.setenv('CUBUTTERFLY_PREPARE_EVENTS', str(events))
    monkeypatch.setattr(proxy, 'snapshot', lambda pid: dict(vma_count=45000, rss_bytes=0, max_map_count=65530))
    monkeypatch.setattr(proxy.broker, 'main', lambda args: pytest.fail('broker queried after resource limit'))
    assert proxy.main([]) == 75
    with pytest.raises(resource.WorkerRecycle):
        gpu.check_preparation_events(events)
    assert json.loads(events.read_text())['kind'] == 'worker_resource_limit'


def test_real_compile_failure_precedes_resource_event(tmp_path):
    events = tmp_path / 'events.jsonl'
    events.write_text(json.dumps({'kind': 'worker_resource_limit', 'reason': 'vma'}) + '\n' +
                      json.dumps({'kind': 'compile_failed'}) + '\n')
    with pytest.raises(gpu.PreparationFailed):
        gpu.check_preparation_events(events)


def test_unrecorded_spawn_failure_cannot_be_hidden_by_pending_candidate(tmp_path):
    events = tmp_path / 'events.jsonl'
    events.write_text(json.dumps({'kind': 'compile_pending', 'request_id': 'one'}) + '\n')
    with pytest.raises(RuntimeError, match='unrecorded specialization'):
        gpu.check_preparation_events(events, 'could not start specialization compiler')


def test_cpp_stderr_capture_closes_successful_plan_after_silent_candidate_error(tmp_path, capfd):
    closed = []

    class Imported:
        def __enter__(self): return self
        def __exit__(self, *args): pass

    def plan(*args, **kwargs):
        os.write(2, b'cuButterfly: rejected measured candidate: could not start specialization compiler\n')
        return SimpleNamespace(close=lambda: closed.append(True))

    cb = SimpleNamespace(Descriptor=SimpleNamespace(from_json=lambda *args: Imported()), Plan=plan)
    with pytest.raises(RuntimeError, match='unrecorded specialization'):
        gpu.build_plan(cb, None, {}, SimpleNamespace(cuda_stream=1), tmp_path / 'absent')
    assert closed == [True]
    assert 'could not start specialization compiler' in capfd.readouterr().err
    os.write(2, b'stderr restored\n')
    assert capfd.readouterr().err == 'stderr restored\n'
