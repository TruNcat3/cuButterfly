"""Portable main-cohort evidence must reject narrowed or relabeled results."""
import copy
import csv
import gzip
import importlib.util
import json
import io
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('full66',ROOT/'experiments/tools/report_pro6000_full66.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def store(root,value):
    raw=json.dumps(value,sort_keys=True).encode();compressed=gzip.compress(raw,mtime=0)
    name=m.sha(raw)+'.json.gz';(root/name).write_bytes(compressed)
    return dict(path=name,sha256=m.sha(compressed),uncompressed_sha256=m.sha(raw))


def fixture(tmp_path,count=66,negative=False):
    cases=[];rows=[]
    timing=dict(median_ms=2.0,samples_ms=[2.0]*7,protocol=next(p for p in m.PROTOCOLS if 'CUDA graph' in p))
    for i in range(66):
        s=dict(operator=i%6,storage=9 if i%6==1 else 6,shape=[16],batch=i+1,inverse=0,
               modulus=17 if i%6==1 else None,input_orders=[0],output_orders=[0])
        if negative and i==1:s.update(shape=[4096],batch=1,modulus=576460756061519873)
        identifier=m.digest(s);descriptor={'semantics':s}
        cases.append(dict(id=identifier,descriptor=descriptor))
        integer=s['storage']==9
        correctness=dict(exact=True,mismatches=0) if integer else dict(verdict='accuracy-review-required',max_abs=1e-9)
        status='pass' if integer else 'accuracy-review-required'
        applicable={0:{'cufft','vkfft'},1:{'gpu-ntt'},2:{'daofht'}}.get(s['operator'],set())
        baselines=[dict(name=name,status=status,correctness=copy.deepcopy(correctness),timing=copy.deepcopy(timing))
                   if name in applicable else dict(name=name,status='not-applicable') for name in m.BASELINES]
        rows.append(dict(id=identifier,descriptor=descriptor,plan_descriptor=dict(semantics=s,selection={'algorithm_policy':1}),
                         mapping={'backend':'fixture'},status=status,correctness=correctness,
                         mapping_replay_correctness=copy.deepcopy(correctness),timing=copy.deepcopy(timing),
                         baselines=baselines))
        if negative and i==1:
            rows[-1]['status']='correctness-failure'
            bad=next(b for b in baselines if b['name']=='gpu-ntt')
            bad.update(status='correctness-failure',error='integer mismatch at 1: 2 != 3',correctness={'exact':False,'mismatches':4094})
            bad.pop('timing')
    population=dict(schema='cubutterfly-semantic-manifest-v1',cases=cases)
    population['manifest_sha256']=m.digest(population)
    population_record=store(tmp_path,population)
    calibration=dict(cost_model_status='calibrated-local-warning',service_gaps=[],selector_replay={'status':'passed'})
    identity=dict(manifest_sha256=m.digest(population),gpu_uuid=m.GPU,library_sha256='library',
                  gpu_name='NVIDIA RTX PRO 6000 Blackwell Server Edition',gpu_total_memory=101973950464,compute_capability='12.0',
                  compile_mode='research',timing=True,calibration_sha256=m.digest(calibration),
                  runtime_environment={'CUBUTTERFLY_SEARCH_BUDGET':'16'})
    entries=[dict(id=r['id'],path='rows/'+r['id']+'.json',sha256=m.digest(r),status=r['status']) for r in rows[:count]]
    report=dict(schema='cubutterfly-semantic-results-v1',identity=identity,rows=entries,rows_sha256=m.digest(entries))
    record=store(tmp_path,report)
    index=dict(parent_manifest_sha256=population_record['uncompressed_sha256'],observed_cases=count,declared_cases=66,
               remaining=[r['id'] for r in rows[count:]],negative_results=sum(m.known_negative(r) for r in rows[:count]),
               rows=[dict(id=r['id'],results='/original/results.json',results_sha256=record['uncompressed_sha256'],
                          row_sha256=m.digest(r),negative_result=m.known_negative(r),selection_policy=1) for r in rows[:count]])
    source=dict(original_path='/original/results.json',original_sha256=record['uncompressed_sha256'],
                checkpoint=record,population=population_record,
                rows={r['id']:store(tmp_path,dict(row=r,sha256=m.digest(r))) for r in rows[:count]})
    manifest=dict(schema='cubutterfly-pro6000-full66-publication-v1',population=population_record,
                  requested_baselines=list(m.BASELINES),
                  observations=store(tmp_path,index),sources=[source],calibration_manifest=store(tmp_path,calibration),
                  gpu_uuid=m.GPU,library_sha256='library')
    save(tmp_path,manifest)
    return manifest,population,report,index,rows


def save(root,manifest):
    (root/'manifest.json').write_text(json.dumps(manifest))


def rewrite_row(root,manifest,report,index,row):
    source=manifest['sources'][0];source['rows'][row['id']]=store(root,dict(row=row,sha256=m.digest(row)))
    entry=next(e for e in report['rows'] if e['id']==row['id']);entry.update(sha256=m.digest(row),status=row['status'])
    report['rows_sha256']=m.digest(report['rows']);source['checkpoint']=store(root,report)
    source['original_sha256']=source['checkpoint']['uncompressed_sha256']
    for item in index['rows']:
        item['results_sha256']=source['original_sha256']
        if item['id']==row['id']:item['row_sha256']=m.digest(row)
    manifest['observations']=store(root,index);save(root,manifest)


def test_all_six_operators_and_original_arrays_are_kept(tmp_path):
    fixture(tmp_path);_,_,rows,index=m.verify(tmp_path)
    assert len(rows)==66 and not index['remaining']
    cases,pairs,text=m.render(tmp_path)
    assert all(name in text for name in m.OPS)
    assert 'diagnostic floating error review' in pairs and 'integer exact with exact replay' in pairs
    assert 'No aggregate across source cohorts' in text


def test_partial_population_requires_explicit_preview_and_cannot_claim_complete(tmp_path):
    fixture(tmp_path,10)
    with pytest.raises(AssertionError,match='incomplete'):m.verify(tmp_path)
    assert len(m.verify(tmp_path,allow_partial=True)[2])==10
    assert '10/66' in m.render(tmp_path,allow_partial=True)[2]


def test_complete_population_keeps_known_baseline_failure_without_a_ratio(tmp_path):
    fixture(tmp_path,negative=True)
    assert m.verify(tmp_path)[3]['negative_results']==1
    _,pairs,text=m.render(tmp_path)
    failures=[row for row in csv.DictReader(io.StringIO(pairs)) if row['status']=='correctness-failure']
    assert len(failures)==1 and failures[0]['baseline']=='gpu-ntt'
    assert failures[0]['baseline_over_own']=='' and failures[0]['ratio_role']=='excluded'
    assert 'correctness-failure' in text and 'retained negative requests 1' in text


@pytest.mark.parametrize('change',['semantics','replay','protocol','median','integer-error','new-baseline-failure'])
def test_refreshing_every_checksum_cannot_hide_invalid_evidence(tmp_path,change):
    manifest,_,report,index,rows=fixture(tmp_path)
    row=rows[1] if change=='integer-error' else rows[0]
    if change=='semantics':row['plan_descriptor']['semantics']=dict(row['descriptor']['semantics'],inverse=1)
    elif change=='replay':row['mapping_replay_correctness']={}
    elif change=='protocol':row['timing']['protocol']='one convenient sample'
    elif change=='median':row['timing']['median_ms']=1.0
    elif change=='integer-error':row['correctness']['mismatches']=1
    else:row['baselines'][0]['status']='runtime-error'
    rewrite_row(tmp_path,manifest,report,index,row)
    with pytest.raises(AssertionError):m.verify(tmp_path)


def test_duplicate_observed_requests_cannot_replace_pending_population(tmp_path):
    manifest,_,_,index,_=fixture(tmp_path);index['rows'][-1]=index['rows'][0]
    manifest['observations']=store(tmp_path,index);save(tmp_path,manifest)
    with pytest.raises(AssertionError):m.verify(tmp_path)


def test_own_gap_retains_request_and_all_four_unattempted_comparators(tmp_path):
    manifest,_,report,index,rows=fixture(tmp_path)
    row=rows[0]
    for key in ('plan_descriptor','timing','correctness','mapping_replay_correctness'):
        row.pop(key)
    row.update(status='missing-lowering',error='explicit fixture lowering gap',baselines=[])
    index['rows'][0]['selection_policy']=None
    rewrite_row(tmp_path,manifest,report,index,row)
    cases,pairs,_=m.render(tmp_path)
    assert 'missing-lowering' in cases and pairs.count('not-reached-after-own-gap')==4


@pytest.mark.parametrize('field,value',[('gpu_name','NVIDIA A100'),('compute_capability','8.0'),('gpu_total_memory',84974239744)])
def test_other_target_hardware_cannot_be_published_as_pro6000(tmp_path,field,value):
    manifest,_,report,index,_=fixture(tmp_path)
    report['identity'][field]=value
    source=manifest['sources'][0];source['checkpoint']=store(tmp_path,report)
    source['original_sha256']=source['checkpoint']['uncompressed_sha256']
    for item in index['rows']:item['results_sha256']=source['original_sha256']
    manifest['observations']=store(tmp_path,index);save(tmp_path,manifest)
    with pytest.raises(AssertionError):m.verify(tmp_path)


def test_only_explicit_cpu_ranking_environment_may_differ():
    first=dict(library_sha256='lib',runtime_environment={'CUBUTTERFLY_SEARCH_BUDGET':'16'},manifest_sha256='full')
    current=copy.deepcopy(first);current['manifest_sha256']='one-request'
    current['runtime_environment']['CUBUTTERFLY_STATIC_RANK_WORKERS']='8'
    m.compatible(first,current)
    current['runtime_environment']['CUBUTTERFLY_SEARCH_BUDGET']='1'
    with pytest.raises(AssertionError):m.compatible(first,current)


def test_refreshed_checkpoint_hash_cannot_replace_frozen_cpu_ranker(tmp_path):
    manifest,_,report,index,_=fixture(tmp_path)
    source=manifest['sources'][0]
    report['identity']['runtime_environment'].update(CUBUTTERFLY_STATIC_RANKER='/original/helper.py',CUBUTTERFLY_STATIC_RANKER_SHA256='different-helper')
    source['checkpoint']=store(tmp_path,report);source['original_sha256']=source['checkpoint']['uncompressed_sha256']
    for item in index['rows']:item['results_sha256']=source['original_sha256']
    manifest['observations']=store(tmp_path,index);save(tmp_path,manifest)
    with pytest.raises(AssertionError,match='CPU helper'):m.verify(tmp_path)


def test_complete_array_audit_cannot_hide_unaudited_requests(tmp_path):
    manifest,_,_,_,_=fixture(tmp_path)
    manifest['numerical_array_audit']=store(tmp_path,dict(status='complete-recorded-array-metrics-replayed',cases=[]))
    save(tmp_path,manifest)
    with pytest.raises(AssertionError,match='Array audit population'):m.verify(tmp_path)


def test_explicit_partial_export_does_not_satisfy_full_publication_gate(tmp_path):
    manifest,_,_,_,_=fixture(tmp_path)
    manifest['publication_gate']='partial-preview';save(tmp_path,manifest)
    with pytest.raises(AssertionError,match='partial preview'):m.verify(tmp_path)


def test_calibration_warning_cannot_be_reclassified_with_refreshed_file_hash(tmp_path):
    manifest,_,_,_,_=fixture(tmp_path)
    calibration=m.unpack(tmp_path,manifest['calibration_manifest']);calibration['cost_model_status']='validated'
    manifest['calibration_manifest']=store(tmp_path,calibration);save(tmp_path,manifest)
    with pytest.raises(AssertionError):m.verify(tmp_path)


def test_known_forward_ntt_failure_does_not_admit_new_inverse_failures():
    s=dict(operator=1,storage=9,modulus=576460756061519873,inverse=0,input_orders=[0],output_orders=[0],shape=[4096],batch=1)
    row=dict(status='correctness-failure',descriptor={'semantics':s},correctness={'exact':True,'mismatches':0},
             mapping_replay_correctness={'exact':True,'mismatches':0},baselines=[dict(name='gpu-ntt',status='correctness-failure',error='integer mismatch at 1: 2 != 3')])
    assert m.known_negative(row)
    row['descriptor']['semantics']['inverse']=1
    assert not m.known_negative(row)
