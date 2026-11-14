"""CPU-only audit of comparator snapshots, including gaps and error warnings."""
import copy
import importlib
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
m=importlib.import_module('summarize_comparator_extension')


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def campaign(tmp_path):
    s=dict(operator=0,storage=6,shape=[16],batch=1,inverse=0,placement=0,normalize_inverse=False)
    cases=[dict(id=i,descriptor=dict(semantics=copy.deepcopy(s)),baselines=['gpu-fft','vkfft']) for i in ('first','second','pending')]
    config=dict(cases=cases)
    manifest=dict(config=config,config_sha256=m.digest(config),cases=[dict(id=c['id'],status='prepared',
        semantics_sha256=m.digest(c['descriptor']['semantics']),descriptor_sha256=m.digest(c['descriptor']),sha256=c['id']) for c in cases])
    save(tmp_path/'preparation.json',manifest)
    identity=dict(config_sha256=manifest['config_sha256'],preparation_sha256=m.sha(tmp_path/'preparation.json'),gpu_uuid='mock',protocol=dict(float_verdict='error-report'))
    for case in cases[:2]:
        row=dict(id=case['id'],descriptor=case['descriptor'],status='accuracy-review-required',
            timing=dict(median_ms=2),correctness=dict(l2_relative=1e-7),mapping=dict(backend='temporal-tile',stage_partition=[4]),
            baselines=[dict(name='gpu-fft',status='accuracy-review-required',timing=dict(median_ms=8),correctness=dict(l2_relative=2e-7)),
                       dict(name='vkfft',status='missing-baseline',reason='unsupported')])
        save(tmp_path/'rows'/(case['id']+'.json'),dict(identity=identity,identity_sha256=m.digest(identity),
            row=row,row_sha256=m.digest(row),guards=dict(before=True,after=True),prepared_sha256=case['id']))
    return tmp_path


def rewrite_row(path,change):
    envelope=json.loads(path.read_text())
    change(envelope)
    envelope['row_sha256']=m.digest(envelope['row'])
    envelope['identity_sha256']=m.digest(envelope['identity'])
    save(path,envelope)


def test_all_scheduled_backends_and_missing_remain(campaign):
    rows,groups,_=m.summarize(campaign)
    assert len(rows)==6
    assert sum(r['baseline_status']=='missing' for r in rows)==2
    g=next(g for g in groups if g['backend']=='gpu-fft')
    assert g['scheduled_count']==3 and g['paired_count']==2
    assert g['gap_or_pending_count']==1
    assert g['geometric_mean_baseline_over_own']==pytest.approx(4)
    assert 'NOT accuracy acceptance' in g['interpretation']
    assert next(g for g in groups if g['backend']=='vkfft')['paired_count']==0


def test_large_error_withholds_whole_group_without_dropping_pair(campaign):
    rewrite_row(campaign/'rows/second.json',lambda e:e['row']['baselines'][0]['correctness'].update(l2_relative=.8))
    rows,groups,_=m.summarize(campaign)
    g=next(g for g in groups if g['backend']=='gpu-fft')
    assert g['paired_count']==2 and g['warning_count']==1
    assert g['geometric_mean_baseline_over_own'] is None
    assert next(r for r in rows if r['case_id']=='second' and r['backend']=='gpu-fft')['baseline_over_own']==4


def test_failure_is_not_success_and_blocks_group_mean(campaign):
    rewrite_row(campaign/'rows/second.json',lambda e:e['row']['baselines'][0].update(status='correctness-failure'))
    rows,groups,_=m.summarize(campaign)
    g=next(g for g in groups if g['backend']=='gpu-fft')
    assert g['paired_count']==1 and g['geometric_mean_baseline_over_own'] is None
    assert next(r for r in rows if r['case_id']=='second' and r['backend']=='gpu-fft')['baseline_over_own'] is None


@pytest.mark.parametrize('change,match',[
    (lambda e:e['guards'].update(after=False),'guards'),
    (lambda e:e['identity'].update(gpu_uuid='foreign'),'mixed'),
    (lambda e:e.update(prepared_sha256='foreign'),'array binding'),
    (lambda e:e['row']['descriptor']['semantics'].update(batch=9),'descriptor binding')])
def test_invalid_measurement_envelope_rejected(campaign,change,match):
    rewrite_row(campaign/'rows/second.json',change)
    with pytest.raises(ValueError,match=match):
        m.summarize(campaign)


def test_row_checksum_is_verified(campaign):
    path=campaign/'rows/first.json'
    envelope=json.loads(path.read_text());envelope['row']['timing']['median_ms']=.001
    save(path,envelope)
    with pytest.raises(ValueError,match='checksum'):
        m.summarize(campaign)


def test_outputs_are_cpu_only_and_source_rows_unchanged(campaign,tmp_path):
    source={p.name:p.read_bytes() for p in (campaign/'rows').glob('*.json')}
    output=tmp_path/'summary'
    m.write_outputs(campaign,output)
    assert {p.name:p.read_bytes() for p in (campaign/'rows').glob('*.json')}==source
    assert (output/'comparisons.csv').read_text().count('\n')==7
    assert 'without an accuracy acceptance threshold' in (output/'summary.md').read_text()
    assert 'large' not in (output/'groups.csv').read_text()


@pytest.mark.parametrize('status',['runtime-error','baseline-error','resource-limit','runner-error'])
def test_execution_errors_block_group_mean(campaign,status):
    rewrite_row(campaign/'rows/second.json',lambda e:e['row']['baselines'][0].update(status=status))
    _,groups,_=m.summarize(campaign)
    g=next(g for g in groups if g['backend']=='gpu-fft')
    assert g['paired_count']==1
    assert g['warning_count']==1
    assert g['geometric_mean_baseline_over_own'] is None


@pytest.mark.parametrize('scope,reason',[
    ('hardware-rate-prior','fallback'),
    ('model-selection','runtime fingerprint missing or incompatible')])
def test_uncalibrated_selection_is_diagnostic(campaign,scope,reason):
    rewrite_row(campaign/'rows/first.json',lambda e:e['row'].update(selection_scope=scope,selection_reason=reason))
    rows,groups,_=m.summarize(campaign)
    r=next(r for r in rows if r['case_id']=='first')
    assert r['selection_scope']==scope and r['selection_reason']==reason
    g=next(g for g in groups if g['backend']=='gpu-fft')
    assert g['paired_count']==2 and g['geometric_mean_baseline_over_own'] is None
    assert 'NOT formal calibrated comparison' in g['interpretation']
    assert reason in g['selection_reasons']


def test_stopped_campaign_scope_withholds_all_means_preserving_points(campaign,tmp_path):
    scope=dict(status='exploratory-stopped',reason='runtime mismatch; exclude from calibrated comparison')
    save(campaign/'SCOPE.json',scope)
    rows,groups=m.write_outputs(campaign,tmp_path/'tables')
    assert len(rows)==6
    assert sum(r['baseline_over_own'] is not None for r in rows)==2
    assert all(g['geometric_mean_baseline_over_own'] is None for g in groups)
    assert all('NOT formal calibrated comparison' in g['interpretation'] for g in groups)
    text=(tmp_path/'tables/summary.md').read_text()
    assert 'exploratory-stopped' in text
    assert 'never an acceptance rule' in text


def test_mixed_arithmetic_policies_have_separate_groups(campaign,tmp_path):
    path=campaign/'preparation.json'
    manifest=json.loads(path.read_text())
    case=manifest['config']['cases'][1]
    case['descriptor']['semantics']['arithmetic_policy']=1
    record=manifest['cases'][1]
    record['semantics_sha256']=m.digest(case['descriptor']['semantics'])
    record['descriptor_sha256']=m.digest(case['descriptor'])
    manifest['config_sha256']=m.digest(manifest['config'])
    save(path,manifest)
    for row_path in (campaign/'rows').glob('*.json'):
        def update(envelope):
            envelope['identity'].update(config_sha256=manifest['config_sha256'],
                                        preparation_sha256=m.sha(path))
            if envelope['row']['id']=='second':
                envelope['row']['descriptor']=case['descriptor']
        rewrite_row(row_path,update)
    rows,groups=m.write_outputs(campaign,tmp_path/'policy-tables')
    fft_groups=[g for g in groups if g['backend']=='gpu-fft']
    assert {g['arithmetic_policy']:g['scheduled_count'] for g in fft_groups}=={
        'STRICT':2,'ALLOW_FP16_MMA':1}
    assert next(r for r in rows if r['case_id']=='second')['arithmetic_policy']=='ALLOW_FP16_MMA'
    assert 'arithmetic_policy' in (tmp_path/'policy-tables/comparisons.csv').read_text().splitlines()[0]
    assert 'permits native or MMA execution' in (tmp_path/'policy-tables/summary.md').read_text()


def test_mapping_summary_retains_actual_core_and_organization():
    mapping=dict(backend='temporal-tile',local_core={'kind':'native','provider':'public'},
                 arithmetic_policy=1,prefetch_depth=2,groups_per_cta=4,groups=[2,2],
                 data_tiles_per_cta=3,stage_partition=[4,4],factor_partition=[2,2])
    assert json.loads(m.mapping_summary(mapping))==mapping
