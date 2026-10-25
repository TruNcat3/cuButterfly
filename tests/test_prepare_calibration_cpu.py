import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_calibration_cpu as preparation


def package(tmp_path, monkeypatch):
    root = tmp_path / 'input'
    root.mkdir()
    def frozen(name, value, compressed=False):
        path = root / name
        if compressed:
            with gzip.open(path, 'wt') as out:
                out.write('\n'.join(json.dumps(row) for row in value))
        else:
            path.write_text(json.dumps(value))
        return dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    points = [dict(operator='fft', precision='fp32', logN=10, batch=1,
                   backend='shared-iterative', tile_threads=32, score=value) for value in (2, 1, 1, 4, 5)]
    document = dict(schema=preparation.SCHEMA, context={'frozen': 'identity'},
        profile=frozen('profile.json', {'hardware': {}}),
        training=frozen('training.json', [dict(correct=False, samples=[{}], median_kernel_ms=9),
                                       dict(correct=True, samples=[{'old': 'sample'}], median_kernel_ms=2)]),
        descriptors=frozen('descriptors.json', {}),
        cells=[dict(workload={'operator': 'fft'}, inventory=frozen('inventory.gz', points, True), complete=False)])
    path = root / 'manifest.json'
    path.write_text(json.dumps(document))
    models = []
    monkeypatch.setattr(preparation.stage_cost_model, 'fit', lambda rows, profile: models.append(rows) or {'model': 1})
    class Scorer:
        def __init__(self, *args): pass
        def bind(self, model): pass
        def score(self, point):
            assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
            return point['score']
    monkeypatch.setattr(preparation, '_SearchPredictionCache', Scorer)
    monkeypatch.setattr(preparation, 'stratified_candidates', lambda points, budget: points)
    broker = tmp_path / 'broker'
    broker.mkdir()
    (broker / 'config.json').write_text('{"config": {"sm":80}}')
    monkeypatch.setattr(preparation.broker_api, 'status', lambda path: {'pending': 2})
    return path, document, points, models, broker


def test_full_ranking_stable_and_durable_before_broker(tmp_path, monkeypatch):
    path, document, points, models, broker = package(tmp_path, monkeypatch)
    output = tmp_path / 'output'
    requested = []
    ready = set()
    monkeypatch.setattr(preparation.research_compile_requests, 'compile_request', lambda point, policy: {'id': point['score']})
    def request(broker, config, mapping, events, timeout):
        requested.append(mapping['id'])
        if mapping['id'] not in ready:
            raise RuntimeError('pending')
        return tmp_path / 'module.so'
    monkeypatch.setattr(preparation.broker_api, 'runtime_request', request)
    def work(broker, jobs, retry_failed):
        assert jobs == 2 and retry_failed is False
        state = json.loads((output / 'status.json').read_text())
        assert state['scoring_complete'] and not state['compile_ready'] and not state['gpu_verified']
        ready.update(requested)
        return {'pending': 0}
    monkeypatch.setattr(preparation.broker_api, 'work', work)
    result = preparation.prepare(path, output, broker, jobs=2, compile_top=2)
    assert result['compile_ready'] and not result['gpu_verified']
    assert result['pid'] == os.getpid()
    assert result['scored_candidates'] == result['total_candidates'] == 5
    assert result['current_cell_index'] == 0
    ready_cell = json.loads((output / 'cell-0-ready.json').read_text())
    assert ready_cell['source_sha256'] == document['cells'][0]['inventory']['sha256']
    assert ready_cell['model_epoch'] == result['model_epoch']
    assert ready_cell['scores']['sha256'] == hashlib.sha256((output / ready_cell['scores']['path']).read_bytes()).hexdigest()
    assert models == [[{'sample': {'old': 'sample'}, 'median_kernel_ms': 2}]]
    ranking = json.loads((output / 'cell-0-ranking.json').read_text())
    assert len(ranking) == 5
    assert [row['inventory_index'] for row in ranking[:2]] == [1, 2]
    assert result['existing_broker_queue'] == {'pending': 2}
    assert set(requested) == {1, 4}  # top scores plus fourth stratified point.
    # Repeat uses durable scores, while still preserving the complete ranking.
    monkeypatch.setattr(preparation._SearchPredictionCache, 'score', lambda *args: pytest.fail('cached score recomputed'))
    repeated = preparation.prepare(path, output, broker, skip_compile=True)
    assert repeated['score_cache']['hits'] == 5


@pytest.mark.parametrize('mode', ['cannot-project', 'no-jit', 'compile-failed'])
def test_negative_results_are_explicit(tmp_path, monkeypatch, mode):
    path, _, _, _, broker = package(tmp_path, monkeypatch)
    def project(point, policy):
        if mode == 'cannot-project':
            raise ValueError('missing axes')
        return None if mode == 'no-jit' else {'module': 1}
    monkeypatch.setattr(preparation.research_compile_requests, 'compile_request', project)
    def pending(*args, **kwargs):
        raise RuntimeError('compile failed; see existing record')
    monkeypatch.setattr(preparation.broker_api, 'runtime_request', pending)
    monkeypatch.setattr(preparation.broker_api, 'work', lambda *args, **kwargs: {'failed': 1})
    result = preparation.prepare(path, tmp_path / 'output', broker, compile_top=1)
    rows = json.loads((tmp_path / 'output/compilation.json').read_text())
    assert {row['status'] for row in rows} == {mode if mode != 'compile-failed' else 'failed'}
    assert result['scoring_complete']
    assert result['compile_ready'] == (mode == 'no-jit')


def test_all_input_hashes_checked_even_complete_cell(tmp_path, monkeypatch):
    path, document, _, _, broker = package(tmp_path, monkeypatch)
    document['cells'][0]['complete'] = True
    path.write_text(json.dumps(document))
    (path.parent / 'inventory.gz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        preparation.prepare(path, tmp_path / 'output', broker, skip_compile=True)
    assert json.loads((tmp_path / 'output/status.json').read_text())['status'] == 'failed'


def test_skip_compile_and_completed_cells_touch_no_broker(tmp_path, monkeypatch):
    path, document, _, _, broker = package(tmp_path, monkeypatch)
    document['cells'][0]['complete'] = True
    path.write_text(json.dumps(document))
    monkeypatch.setattr(preparation.broker_api, 'work', lambda *args, **kwargs: pytest.fail('broker called'))
    state = preparation.prepare(path, tmp_path / 'output', broker, skip_compile=True)
    assert state['status'] == 'scoring-complete' and state['cells'][0]['status'] == 'skipped-complete'
    assert not state['compile_ready'] and not state['gpu_verified']


def test_unvalidated_service_is_not_a_ranking_prior():
    profile = {'hardware': {}, 'stage_service': {'curves': {'key': {}}}}
    assert 'stage_service' not in preparation.ranking_profile(profile)
    assert 'stage_service' in profile


def test_real_scorer_and_resolved_projection_match_current_model(tmp_path):
    profile = dict(sm_count=10, capabilities=dict(global_feedback_bytes_per_second=1e12,
        equivalent_butterflies_per_second=1e11, interstage_shared_bytes_per_second=1e13,
        cta_barriers_per_second=1e9))
    point = dict(operator='fft', precision='fp32', logN=12, batch=1,
                 backend='shared-iterative', tile_threads=64)
    groups = [dict(stage_count=4, grid_ctas=256, threads=64, core='scalar', live_shared_bytes=256)]
    descriptor = dict(status='resolved', mapping_json='{}', execution_groups_json=groups)
    descriptors = {preparation.candidate_id(point): descriptor}
    root = tmp_path / 'input'
    root.mkdir()
    references = {}
    for name, value in (('profile', profile), ('training', []), ('descriptors', descriptors)):
        path = root / (name + '.json')
        path.write_text(json.dumps(value))
        references[name] = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    inventory = root / 'inventory.gz'
    with gzip.open(inventory, 'wt') as out:
        out.write(json.dumps(point) + '\n')
    manifest = dict(schema=preparation.SCHEMA, context={'real-scorer': 1}, **references,
                    cells=[dict(workload=point, complete=False,
                        inventory=dict(path=inventory.name, sha256=hashlib.sha256(inventory.read_bytes()).hexdigest()))])
    (root / 'manifest.json').write_text(json.dumps(manifest))
    state = preparation.prepare(root / 'manifest.json', tmp_path / 'output', tmp_path / 'unused-broker', skip_compile=True)
    score = json.loads((tmp_path / 'output/cell-0-scores.json').read_text())[0]['score']
    model = preparation.stage_cost_model.fit([], profile)
    assert score == preparation.stage_cost_model.predict(preparation.project_point(point, descriptors), model)
    assert score > 0 and state['scoring_complete'] and not state['gpu_verified']
