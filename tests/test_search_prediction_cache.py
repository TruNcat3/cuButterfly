import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import calibrate_local_hardware as calibration


def test_cache_preserves_new_model_and_resolved_descriptor_authority(monkeypatch):
    calls = []
    class Predictor:
        def __init__(self, model):
            pass
        def prepare(self, sample):
            calls.append(sample)
            if sample.get('invalid'):
                raise ValueError('unresolved')
            return sample.copy()
        def bind(self, model):
            return lambda sample: sample['value'] * model['factor']
    monkeypatch.setattr(calibration.stage_cost_model, 'PreparedPredictor', Predictor)
    point = {'operator': 'fwht', 'logN': 12}
    key = calibration.candidate_id(point)
    descriptors = {}
    def projected(p):
        return descriptors.get(key, {'value': 5})
    cache = calibration._SearchPredictionCache(descriptors, projected)
    cache.bind({'factor': 2})
    assert cache.score(point) == cache.score(point) == 10
    assert len(calls) == 1
    cache.bind({'factor': 3})
    assert cache.score(point) == 15
    assert len(calls) == 1
    descriptors[key] = {'status': 'resolved', 'value': 1}
    assert cache.score(point) == 3
    assert len(calls) == 2
    descriptors[key] = {'status': 'unavailable'}
    assert cache.score(point) == float('inf')
    descriptors[key] = {'status': 'resolved', 'invalid': True}
    assert cache.score(point) == float('inf')
    descriptors[key] = {'status': 'resolved', 'value': 2}
    assert cache.score(point) == 6


def test_shortlist_keeps_stable_ties_and_scores_every_candidate():
    values = [5, 1, 1, 2, 0, 1, 0, 0]
    points = [SimpleNamespace(index=i, cost=v) for i, v in enumerate(values)]
    visited = []
    def score(point):
        visited.append(point.index)
        return point.cost
    result = calibration.heapq.nsmallest(4, points, key=score)
    assert [p.index for p in result] == [p.index for p in sorted(points, key=lambda p:p.cost)[:4]]
    assert visited == list(range(len(points)))


def test_empirical_ranking_requires_complete_scope_or_certified_curves():
    service = {'curves': {'good': {}, 'unvalidated': {}},
               'validated': False, 'calibration_status': 'incomplete-unsupported-adapter'}
    assert not calibration._stage_service_usable_for_ranking({'stage_service': service})
    scope = {'schema': 'cubutterfly-stage-ranking-scope-v1',
             'validated_curve_keys': ['good'], 'validated_curve_count': 1}
    service['ranking_scope'] = scope
    assert calibration._stage_service_usable_for_ranking({'stage_service': service})
    scope['validated_curve_keys'] = ['missing']
    assert not calibration._stage_service_usable_for_ranking({'stage_service': service})
    scope['validated_curve_keys'] = ['good', 'good']
    scope['validated_curve_count'] = 2
    assert not calibration._stage_service_usable_for_ranking({'stage_service': service})
