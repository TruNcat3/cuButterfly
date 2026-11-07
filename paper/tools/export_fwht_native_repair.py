#!/usr/bin/env python3
"""Export frozen, portable FWHT explicit-mapping cross-build confirmation."""
import hashlib
import json
import math
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parents[2]
COHORT=Path('results/fwht_native_repair_20261107')
SIDES=['old_original','new_same_mapping','new_screen_choice','dao']
INTERFACES=['native-return','supplied-output']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def gm(values):return math.exp(statistics.mean(math.log(x) for x in values))
def float32_reference(source,semantics):
    """Axis-order Sylvester FWHT; every add/sub and per-axis inverse scale FP32."""
    import numpy as np
    assert source.dtype==np.float32 and semantics['operator']==2
    work=np.array(source,dtype=np.float32,copy=True).reshape(semantics['batch'],*semantics['shape'])
    for axis,n in enumerate(semantics['shape'],1):
        assert n>0 and n&(n-1)==0
        moved=np.moveaxis(work,axis,-1).copy();rows=moved.reshape(-1,n)
        half=1
        while half<n:
            blocks=rows.reshape(-1,2*half)
            left=blocks[:,:half].copy();right=blocks[:,half:].copy()
            np.add(left,right,out=blocks[:,:half]);np.subtract(left,right,out=blocks[:,half:])
            half*=2
        if semantics['inverse'] and semantics['normalize_inverse']:
            np.multiply(rows,np.float32(1/n),out=rows)
        work=np.moveaxis(rows.reshape(moved.shape),-1,axis)
    assert work.dtype==np.float32
    return np.ascontiguousarray(work)

def aggregate(rows):
    output={}
    for interface in INTERFACES:
        trials=[]
        for trial in (1,2,3):
            bycase=[]
            for row in rows:
                r={side:next(v for v in row['measurements'] if v['trial']==trial and v['interface']==interface and v['side']==side) for side in SIDES}
                bycase.append(r)
            groups={}
            for side in ['old_original','new_same_mapping']:
                ratios=[r['dao']['median_ms']/r[side]['median_ms'] for r in bycase]
                groups[side]=dict(dao_over_cu_gm=gm(ratios),wins=sum(x>1 for x in ratios),losses=sum(x<1 for x in ratios),ties=sum(x==1 for x in ratios),ratios=ratios,max_abs=max(r[side]['correctness_fp32_reference']['max_abs'] for r in bycase),max_l2_relative=max(r[side]['correctness_fp32_reference']['l2_relative'] for r in bycase))
            oldnew=[r['old_original']['median_ms']/r['new_same_mapping']['median_ms'] for r in bycase]
            numeric={}
            for metric in ['max_abs','l2_relative']:
                pairs=[[r['old_original']['correctness_fp32_reference'][metric],r['new_same_mapping']['correctness_fp32_reference'][metric]] for r in bycase]
                numeric[metric]=dict(improved=sum(b<a for a,b in pairs),equal=sum(a==b for a,b in pairs),worsened=sum(b>a for a,b in pairs),pairs=pairs)
            trials.append(dict(trial=trial,by_side=groups,old_over_new_same_gm=gm(oldnew),old_over_new_same_ratios=oldnew,numeric_comparison=numeric))
        output[interface]=dict(processes=trials,median_process_gm={s:statistics.median(t['by_side'][s]['dao_over_cu_gm'] for t in trials) for s in ['old_original','new_same_mapping']},median_process_old_over_new_same_gm=statistics.median(t['old_over_new_same_gm'] for t in trials))
    return output

def export(root=ROOT):
    root=Path(root);sources=[]
    def read(relative):
        path=root/relative;sources.append(dict(path=str(relative),sha256=sha(path)));return json.loads(path.read_text())
    previous_portable=read(Path('paper/generated/fwht_return_contract_20261107.json'))
    previous_precision_correction=dict(path='paper/generated/fwht_return_contract_20261107.json',sha256=sources[-1]['sha256'],actual_reference_precision='float64 CPU promoted transform',label_correction='Prior FP32 CPU-error wording was incorrect: actual GPU outputs are float32, original CPU reference and original numeric errors use float64. Prior raw values remain unchanged as higher precision diagnostics.')
    screen=read(COHORT/'screen/guard/release/report.json')
    screen_summary=read(COHORT/'screen/summary.json')
    receipt=read(COHORT/'build/receipt.json')
    selection=read(COHORT/'confirmation/selection.json')
    prepared=read(COHORT/'confirmation/prepared.json')
    reports=[];guards=[]
    for trial in (1,2,3):
        report=read(COHORT/f'confirmation/process{trial}/release/report.json')
        guard=read(COHORT/f'confirmation/process{trial}/status.json')
        assert report['status']=='complete-confirmation' and len(report['records'])==80 and guard['state']=='complete'
        assert report['new_library_sha256']==receipt['library_sha256']
        assert report['prepared_sha256']==sha(root/COHORT/'confirmation/prepared.json')
        reports.append(report);guards.append(guard)
    audit=read(COHORT/'confirmation/fp32_reference_audit/report.json')
    audit_index={(r['trial'],r['case_id'],r['side'],r['interface']):r for r in audit['records']}
    for entry in audit['reference_records']+[r['error_artifact'] for r in audit['records']]:
        assert sha(root/entry['path'])==entry['sha256']
    rows=[]
    for frozen in selection['rows']:
        c=frozen['case'];records=[]
        for trial,report in enumerate(reports,1):
            for r in report['records']:
                if r['case_id']!=c['id']:continue
                assert len(r['samples_ms'])==7 and statistics.median(r['samples_ms'])==r['median_ms']
                artifact=r['error_artifact'];relative=COHORT/f'confirmation/process{trial}/release/errors'/Path(artifact['path']).name
                assert sha(root/relative)==artifact['sha256']
                a=dict(r);a['trial']=trial;a['correctness_fp64_reference_diagnostic']=a.pop('correctness');ar=audit_index[(trial,c['id'],r['side'],r['interface'])];a['correctness_fp32_reference']=ar['correctness_fp32_reference'];a['fp32_reference_error_artifact']=ar['error_artifact'];a['reference_precision']='float32 CPU axis-order Sylvester; FP64 reference retained separately as higher precision diagnostic';a['error_artifact']=dict(artifact,path=str(relative));records.append(a)
        assert len(records)==24
        tables={}
        for interface in INTERFACES:
            sides={s:[next(r for r in records if r['trial']==t and r['side']==s and r['interface']==interface)['median_ms'] for t in (1,2,3)] for s in SIDES}
            paired=[a/b for a,b in zip(sides['dao'],sides['new_same_mapping'])]
            tables[interface]=dict(new_same_us=statistics.median(sides['new_same_mapping'])*1000,dao_us=statistics.median(sides['dao'])*1000,process_paired_dao_over_new_same=paired,median_paired_dao_over_new_same=statistics.median(paired))
        rows.append(dict(case_id=c['id'],semantics=c['descriptor']['semantics'],input=dict(path=str(Path(c['prepared_file']).relative_to(root)),sha256=c['prepared_sha256']),original_mapping=frozen['original_mapping'],diagnostic_screen_selected_variant=frozen['screen_selected_variant'],diagnostic_screen_selected_mapping=frozen['screen_selected_mapping'],measurements=records,table=tables))
    return dict(schema='fwht-native-repair-confirmation-v1',status='complete-independent-confirmation',interpretation='Explicit replay of historical AUTO-selected saved mappings across builds; not current new AUTO or DEFAULT results. Old-build supplied-output screen choice is diagnostic only; no registry promotion.',statistic='Each process: median of7 Graph replay event intervals divided20; per-case displayed times: median of3 process medians; ratio: median of3 process-paired ratios. GM computed across ten request ratios separately per process, never after averaging ratios.',boundary=reports[0]['timer_boundary'],protocol=reports[0]['protocol'],numeric_policy='Primary same-precision FP32 CPU reference audit; full distributions and arrays, finite required, no threshold. Original float64 CPU reference errors retained separately as higher precision diagnostic. Earlier FP32 CPU-error labeling was incorrect.',reference_precision=audit['reference_precision'],prior_portable_precision_correction=previous_precision_correction,fp32_reference_audit=audit,measurement_gpu_uuid=reports[0]['gpu_uuid'],old_source_provenance_gpu_uuid=selection['old_identity']['gpu_uuid'],old_library_sha256=reports[0]['old_library_sha256'],new_library_sha256=receipt['library_sha256'],build_receipt=receipt,cross_build_replay=prepared['cross_build_replay'],counts=dict(requests=10,processes=3,records=240,errors=240,failures=0),process_pids=[r['pid'] for r in reports],guards=guards,sources=sources,screen=dict(status=screen['status'],report_sha256=next(s['sha256'] for s in sources if s['path']==str(COHORT/'screen/guard/release/report.json')),counts=screen_summary['counts'],purpose='selection diagnostic only'),rows=rows,aggregate=aggregate(rows))

if __name__=='__main__':
    output=ROOT/'paper/generated/fwht_native_repair_20261107.json'
    output.write_text(json.dumps(export(),indent=2,allow_nan=False)+'\n')
    print(output)
