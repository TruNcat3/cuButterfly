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


def test_proxy_resource_event_discards_selection_without_querying_broker(monkeypatch, tmp_path):
    events = tmp_path / 'events.jsonl'
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
