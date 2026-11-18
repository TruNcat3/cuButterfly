import copy
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/tools'))
import pro6000_parallel_static_ranker as parallel
spec=importlib.util.spec_from_file_location('static_rank_fixtures',ROOT/'tests/test_static_candidate_ranking.py')
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)

def request(tmp_path):
    samples=[fixtures.sample(),fixtures.sample((4,8)),fixtures.sample(),fixtures.sample(),fixtures.sample(),fixtures.sample(),fixtures.sample(),fixtures.sample()]
    samples[3]['physical_descriptor_status']='unavailable'
    groups=json.loads(samples[4]['execution_groups_json']);groups[0]['threads']=2048;samples[4]['execution_groups_json']=groups
    samples[5]['execution_groups_json']='{}'
    samples[6]['mapping_json']=json.dumps(dict(backend='shared-iterative',stage_overlap=True,batch_tile_count=4))
    req=fixtures.request(tmp_path,samples)
    for item,index in zip(req['candidates'],[101,4,78,2,999,7,14,3]):item['index']=index
    return req

@pytest.mark.parametrize('workers',[1,2,4,8])
def test_full_partition_scores_reasons_and_ties_equal_serial(tmp_path,workers):
    req=request(tmp_path);original=copy.deepcopy(req)
    assert parallel.rank_parallel(req,workers)==parallel.serial.rank(req)
    assert req==original
    result=parallel.rank_parallel(req,workers)
    ties=[r['index'] for r in result['ranked'] if r['index'] in [101,78,3]]
    assert ties==[101,78,3]

def test_validated_model_and_fallback_confidences_equal_serial(tmp_path):
    req=request(tmp_path);req['timing_protocol']='direct-launch'
    model=fixtures.model_fixture(fixtures.sample())
    (tmp_path/'cost_model.json').write_text(json.dumps(model))
    result=parallel.rank_parallel(req,4)
    assert result==parallel.serial.rank(req)
    assert result['mode']=='mixed-static-model-and-hardware-rate-prior'
    assert {row['confidence'] for row in result['ranked']}=={'validated-static-model','hardware-rate-prior'}

def test_thirty_two_actual_shards_preserve_global_stable_order(tmp_path):
    req=fixtures.request(tmp_path,[fixtures.sample() for _ in range(64)])
    for i,item in enumerate(req['candidates']):item['index']=64-i
    result=parallel.rank_parallel(req,32)
    assert result==parallel.serial.rank(req)
    assert [row['index'] for row in result['ranked']]==list(range(64,0,-1))

def test_duplicate_across_shards_is_rejected_before_scoring(tmp_path):
    req=request(tmp_path);req['candidates'][-1]['index']=req['candidates'][0]['index']
    with pytest.raises(ValueError,match='distinct'):parallel.rank_parallel(req,4)

def test_profile_change_during_scoring_is_rejected(tmp_path,monkeypatch):
    req=request(tmp_path);original=parallel.serial.rank
    def changed(value):
        result=original(value);(tmp_path/'hardware_profile.json').write_text('{}');return result
    monkeypatch.setattr(parallel.serial,'rank',changed)
    with pytest.raises(ValueError,match='profile changed'):parallel.rank_parallel(req,1)

def prepare_cache(tmp_path,monkeypatch):
    profile=tmp_path/'profile';profile.mkdir();req=request(profile)
    source=tmp_path/'request.json';source.write_text(json.dumps(req));output=tmp_path/'ranking.json'
    monkeypatch.setattr(parallel,'dependencies',lambda:{'fixture':'original'})
    return req,source,output,tmp_path/'cache'

def test_exact_cache_hit_skips_parsing_and_scoring(tmp_path,monkeypatch):
    req,source,output,cache=prepare_cache(tmp_path,monkeypatch)
    first=parallel.run(source,output,2,cache);expected=output.read_bytes();assert not first['cache_hit']
    monkeypatch.setattr(parallel,'rank_parallel',lambda *a:pytest.fail('cache hit must not score'))
    monkeypatch.setattr(parallel.serial,'_read',lambda *a:pytest.fail('cache hit must not parse large request'))
    second=parallel.run(source,output,2,cache)
    assert second['cache_hit'] and output.read_bytes()==expected

@pytest.mark.parametrize('change',['profile','code','request'])
def test_cache_identity_changes_force_fresh_scoring(tmp_path,monkeypatch,change):
    req,source,output,cache=prepare_cache(tmp_path,monkeypatch);parallel.run(source,output,2,cache)
    if change=='profile':
        p=Path(req['profile_dir'])/'hardware_profile.json';d=json.loads(p.read_text());d['capabilities']['global_feedback_bytes_per_second']*=2;p.write_text(json.dumps(d))
    elif change=='code':monkeypatch.setattr(parallel,'dependencies',lambda:{'fixture':'changed'})
    else:req['candidates'][0]['sample']['batch']=8;source.write_text(json.dumps(req))
    result=parallel.run(source,output,2,cache)
    assert not result['cache_hit'] and json.loads(output.read_text())==parallel.serial.rank(req)

def test_corrupted_cache_is_archived_and_recomputed(tmp_path,monkeypatch):
    req,source,output,cache=prepare_cache(tmp_path,monkeypatch);parallel.run(source,output,2,cache)
    path=next(cache.glob('*.json.gz'))
    with gzip.open(path,'rt') as stream:d=json.load(stream)
    d['result']['ranked'][0]['predicted_kernel_ms']*=2
    with gzip.open(path,'wt') as stream:json.dump(d,stream)
    audit=parallel.run(source,output,2,cache)
    assert audit['cache_error'] and not audit['cache_hit'] and list(cache.glob('*.invalid-*'))
    assert json.loads(output.read_text())==parallel.serial.rank(req)
