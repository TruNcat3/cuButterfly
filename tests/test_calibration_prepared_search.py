"""CPU preparation must preserve exact scoring, selection and input identities."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import calibrate_local_hardware as calibration
from calibration_cpu_cache import DiskScoreCache


def test_online_search_consumes_offline_scores_without_recomputing(tmp_path, monkeypatch):
    class Predictor:
        def __init__(self, model): pass
        def bind(self, model): return lambda prepared: prepared['value'] * model['factor']
        def prepare(self, sample): return sample.copy()
    monkeypatch.setattr(calibration.stage_cost_model, 'PreparedPredictor', Predictor)
    points = [dict(operator='fwht', logN=12, batch=b) for b in (3, 1, 2, 4)]
    context, model = {'identity': 'current'}, {'factor': 2}
    with DiskScoreCache(tmp_path, context) as offline:
        offline.bind(model)
        for p in points:
            offline.put(calibration.candidate_id(p), None, p['batch'] * 2)
    with DiskScoreCache(tmp_path, context) as online:
        cache = calibration._SearchPredictionCache({}, lambda p: pytest.fail('unexpected projection'), online)
        cache.bind(model)
        assert [p['batch'] for p in calibration.heapq.nsmallest(4, points, key=cache.score)] == [1, 2, 3, 4]
        assert cache.stats['disk_score_hits'] == 4
        assert cache.stats['feature_preparations'] == 0


def test_inplace_descriptor_change_cannot_publish_stale_prepared_features(tmp_path, monkeypatch):
    class Predictor:
        def __init__(self, model): pass
        def bind(self, model): return lambda sample: sample['value'] * model['factor']
        def prepare(self, sample): return sample.copy()
    monkeypatch.setattr(calibration.stage_cost_model, 'PreparedPredictor', Predictor)
    point = {'operator': 'fwht', 'logN': 12}
    key = calibration.candidate_id(point)
    descriptors = {key: {'value': 3}}
    with DiskScoreCache(tmp_path, {}) as disk:
        cache = calibration._SearchPredictionCache(descriptors, lambda p: descriptors[key], disk)
        cache.bind({'factor': 2})
        assert cache.score(point) == 6
        descriptors[key]['value'] = 4
        assert cache.score(point) == 8
        cache.bind({'factor': 3})
        assert cache.score(point) == 12


def test_sparse_online_persistence_keeps_full_ranking_and_resume_exact(tmp_path, monkeypatch):
    class Predictor:
        def __init__(self, model): pass
        def bind(self, model): return lambda sample: sample['batch'] * model['factor']
        def prepare(self, sample): return sample.copy()
    monkeypatch.setattr(calibration.stage_cost_model, 'PreparedPredictor', Predictor)
    points = [dict(operator='fwht', logN=12, batch=b) for b in (3, 1, 2, 4)]
    context, model = {'identity': 'current'}, {'factor': 2}
    with DiskScoreCache(tmp_path, context) as disk:
        cache = calibration._SearchPredictionCache({}, dict, disk, persist_frontier=False)
        cache.bind(model)
        chosen = calibration.heapq.nsmallest(2, points, key=cache.score)
        assert [p['batch'] for p in chosen] == [1, 2]
        assert cache.stats['feature_preparations'] == len(points)
        assert disk.stats['puts'] == 0
        cache.persist(chosen)
        assert disk.stats['puts'] == 2
    with DiskScoreCache(tmp_path, context) as disk:
        resumed = calibration._SearchPredictionCache({}, dict, disk, persist_frontier=False)
        resumed.bind(model)
        assert calibration.heapq.nsmallest(2, points, key=resumed.score) == chosen
        assert resumed.stats['disk_score_hits'] == 2
        assert resumed.stats['feature_preparations'] == 2
        resumed.bind({'factor': -1})
        assert [p['batch'] for p in calibration.heapq.nsmallest(2, points, key=resumed.score)] == [4, 3]
        assert disk.stats['puts'] == 0


def test_compact_coverage_is_complete_and_atomic(tmp_path, monkeypatch):
    path = tmp_path / 'search_coverage.json'
    document = dict(schema='cubutterfly-search-coverage-v1', workloads=[dict(
        candidates=[dict(id='a', status='budget-omitted'), dict(id='b', status='screened')],
        model_updates=[{'factor': 0.10000000000000002}], complete=False)])
    calibration.write_checkpoint(path, document)
    assert json.loads(path.read_text()) == document
    assert path.stat().st_size < len(json.dumps(document, indent=2))
    original = path.read_bytes()
    def interrupted(*args):
        raise OSError('injected interrupted rename')
    monkeypatch.setattr(Path, 'replace', interrupted)
    with pytest.raises(OSError, match='interrupted rename'):
        calibration.write_checkpoint(path, dict(document, workloads=[]))
    assert path.read_bytes() == original


def test_prepared_inventory_reuses_exact_order_and_rejects_mutation(tmp_path):
    workload = dict(operator='fwht', logN=12)
    points = [dict(workload, batch=b) for b in (3, 1, 2)]
    path = tmp_path / 'inventory.jsonl.gz'
    with gzip.open(path, 'wt') as stream:
        for p in points: stream.write(json.dumps(p)+'\n')
    manifest = {'cells': [dict(workload=workload, complete=False,
        inventory=dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))]}
    assert calibration.prepared_inventory((tmp_path, manifest), workload) == points
    with gzip.open(path, 'wt') as stream: stream.write('{}\n')
    with pytest.raises(ValueError, match='digest mismatch'):
        calibration.prepared_inventory((tmp_path, manifest), workload)


def test_preparation_context_checks_build_and_scoring_sources(tmp_path, monkeypatch):
    args = SimpleNamespace(cpu_preparation_input=tmp_path/'manifest.json')
    monkeypatch.setattr(calibration, 'cpu_preparation_context', lambda args, measurement: {'measurement': measurement})
    args.cpu_preparation_input.write_text(json.dumps(dict(schema='cubutterfly-calibration-cpu-input-v1',
        context={'measurement': {'build':'old'}})))
    with pytest.raises(ValueError, match='identity changed'):
        calibration.load_cpu_preparation(args, {'build':'new'})


def test_inventory_only_needs_matching_context_but_no_score_receipt(tmp_path, monkeypatch):
    args = SimpleNamespace(cpu_preparation_input=tmp_path/'manifest.json',
                           cpu_preparation_inventory_only=True)
    monkeypatch.setattr(calibration, 'cpu_preparation_context', lambda args, measurement: {'measurement': measurement})
    document = dict(schema='cubutterfly-calibration-cpu-input-v1', context={'measurement': {'build':'same'}})
    args.cpu_preparation_input.write_text(json.dumps(document))
    assert calibration.load_cpu_preparation(args, {'build':'same'}) == (tmp_path, document)
    with pytest.raises(ValueError, match='identity changed'):
        calibration.load_cpu_preparation(args, {'build':'different'})


def test_preparation_must_be_published_before_online_search(tmp_path, monkeypatch):
    args = SimpleNamespace(cpu_preparation_input=tmp_path/'manifest.json', cpu_preparation_dir=tmp_path/'prepared')
    monkeypatch.setattr(calibration, 'cpu_preparation_context', lambda args, measurement: {'measurement': measurement})
    args.cpu_preparation_input.write_text(json.dumps(dict(schema='cubutterfly-calibration-cpu-input-v1',
        context={'measurement': {'build':'same'}})))
    with pytest.raises(ValueError, match='complete matching scores'):
        calibration.load_cpu_preparation(args, {'build':'same'})
