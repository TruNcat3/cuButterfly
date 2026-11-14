#!/usr/bin/env python3
"""Validate and measure public AUTO/ALLOW selection against a paired profile.

Imports and default inspection are CPU-only; --execute explicitly uses the GPU.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
import calibrate_hadacore_native as native

ROOT=Path(__file__).resolve().parents[1]
sha,read,write=native.sha,native.read,native.write


def winner(profile,n,b,library):
    p=read(profile);log=n.bit_length()-1
    natives=[r for r in p.get('native_rows',[]) if r['log_n']==log and r['batch']==b and r['library_sha256']==sha(library)]
    mmas=[r for r in p.get('rows',[]) if r['log_n']==log and r['batch']==b and r.get('status')=='validated-exact-upstream']
    if not natives or not mmas:return dict(kind='fallback',reason='no matching calibrated native/MMA pair')
    if len(natives)!=1:raise ValueError('ambiguous native profile rows')
    base=natives[0];candidates=[('native',base)]+[('mma',r) for r in mmas if r.get('timing_protocol')==base['timing_protocol']]
    if len(candidates)==1:return dict(kind='fallback',reason='no same-protocol MMA candidate')
    for kind,row in candidates:
        evidence=Path(row['evidence_path'])
        if sha(evidence)!=row['evidence_sha256']:raise ValueError('profile evidence changed')
        e=read(evidence)
        if e['gpu_uuid']!=p['gpu_uuid'] or e.get('guards')!={'before':True,'after':True}:raise ValueError('profile evidence identity/guards mismatch')
        if not isinstance(row['median_ms'],(float,int)) or not math.isfinite(row['median_ms']) or row['median_ms']<=0:raise ValueError('invalid profile timing')
        if e['timing']['median_ms']!=row['median_ms'] or e['timing']['protocol']!=row['timing_protocol']:raise ValueError('profile timing/evidence mismatch')
        if sha(e['error_artifact'])!=e['error_artifact_sha256']:raise ValueError('profile error artifact changed')
    kind,row=min(candidates,key=lambda pair:pair[1]['median_ms'])
    return dict(kind=kind,row=row,evidence=read(row['evidence_path']))


def check_mapping(mapping,selected,reason):
    local=mapping.get('local_core',{})
    if selected['kind']=='fallback':
        if local.get('name')=='hadacore-local-tile':raise AssertionError('MMA selected without calibrated pair')
        if 'retained native selection' not in reason:raise AssertionError('native fallback reason missing')
        return
    row=selected['row']
    if selected['kind']=='mma':
        if local.get('name')!='hadacore-local-tile' or local.get('chunks')!=row['chunks'] or local.get('groups_per_block')!=row['groups_per_block']:
            raise AssertionError('public MMA core/geometry differs from measured best candidate')
    else:
        expected=dict(row['mapping']);actual=dict(mapping)
        expected.pop('arithmetic_policy',None);actual.pop('arithmetic_policy',None)
        if actual!=expected:raise AssertionError('public native mapping differs from measured best candidate')


def run_one(prepared,output,n,b,library,profile):
    p,row,x,cpu=native.load_case(prepared,n,b)
    selected=winner(profile,n,b,library)
    if read(profile)['gpu_uuid']!=p['gpu_uuid']:raise ValueError('profile/prepared GPU mismatch')
    report=dict(schema='hadacore-public-validation-v1',status='running',case=row['case'],
        gpu_uuid=p['gpu_uuid'],profile=str(Path(profile).resolve()),profile_sha256=sha(profile),
        library=str(Path(library).resolve()),library_sha256=sha(library),runner_sha256=sha(__file__),
        frontend_sha256=sha(native.FRONTEND),payload_sha256=row['artifact_sha256'],
        selected_kind=selected['kind'],guards={'before':False,'after':False})
    sys.path.insert(0,str(ROOT/'python'));sys.path.insert(0,str(ROOT/'scripts'))
    import run_semantic_campaign as campaign
    plan=graph=None;uuid=p['gpu_uuid']
    try:
        campaign.exclusive_gpu(uuid);report['guards']['before']=True
        os.environ['CUDA_VISIBLE_DEVICES']=uuid;os.environ['CUBUTTERFLY_HADACORE_PROFILE']=str(Path(profile).resolve())
        import torch
        import cubutterfly as cb
        if Path(cb.__file__).resolve()!=native.FRONTEND.resolve():raise RuntimeError('local frontend not loaded')
        props=torch.cuda.get_device_properties(0)
        if not campaign.same_gpu_uuid(uuid,str(props.uuid)):raise ValueError('actual GPU UUID mismatch')
        report['gpu_identity']=dict(uuid=uuid,name=props.name,memory_bytes=props.total_memory,sm=props.major*10+props.minor)
        stream=torch.cuda.Stream();lib=cb.Library(library)
        transform=cb.Transform(op=cb.Operator.FWHT,storage=cb.DataType.FP16,compute=cb.ComputeType.NATIVE,
            extents=[n],batch=b,inverse=False,normalize_inverse=False,in_place=False,
            arithmetic_policy=cb.ArithmeticPolicy.ALLOW_FP16_MMA)
        with cb.Descriptor(lib,transform) as d:plan=cb.Plan(lib,d,stream=stream.cuda_stream)
        report['mapping']=json.loads(plan.mapping_json());report['selection_reason']=plan.selection_reason()
        report['descriptor']=json.loads(plan.descriptor_json())
        check_mapping(report['mapping'],selected,report['selection_reason']);report['selection_correct']=True
        if plan.input_size()!=n*b*2 or plan.output_size()!=n*b*2:raise ValueError('packed buffer size mismatch')
        with torch.cuda.stream(stream):
            def guarded():
                raw=torch.full((n*b+128,),0x3555,dtype=torch.int16,device='cuda')
                return raw,raw[64:-64].view(torch.float16).reshape(b,n)
            sr,src=guarded();dr,dst=guarded();src.copy_(torch.from_numpy(x));saved=sr.clone()
            workspace=torch.empty(max(1,plan.workspace_size()),dtype=torch.uint8,device='cuda')
            if plan.workspace_size():plan.set_workspace(workspace,plan.workspace_size())
            call=lambda:plan.execute(src,dst)
            call();stream.synchronize();direct=dst.cpu().numpy().copy()
            def verify():
                stream.synchronize()
                for raw in (sr,dr):
                    if not bool(torch.all(raw[:64]==0x3555)) or not bool(torch.all(raw[-64:]==0x3555)):raise AssertionError('guard corrupted')
                if not torch.equal(sr,saved):raise AssertionError('input modified')
                if not np.array_equal(dst.cpu().numpy().view(np.uint16),direct.view(np.uint16)):raise AssertionError('repeat/graph mismatch')
            for _ in range(3):call();verify()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):call()
            dst.fill_(float('nan'));graph.replay();verify();graph.reset();graph=None
            report.update(exact_repeat=True,input_immutable=True,buffer_guards=True,graph_reproduced=True)
            report['timing']=campaign.time_execute(torch,stream,call,lambda:None,False,verify,lambda:dst.fill_(float('nan')))
            report['finite']=bool(np.isfinite(direct).all());report['cpu_error']=campaign.check_full(direct,cpu,0,'error-report',True)
            if selected['kind']!='fallback':
                evidence=selected['evidence'];report['selected_evidence_path']=selected['row']['evidence_path']
                report['selected_evidence_sha256']=selected['row']['evidence_sha256']
                if evidence['payload_sha256']!=row['artifact_sha256']:raise ValueError('calibration/public input mismatch')
                with np.load(evidence['error_artifact'],allow_pickle=False) as z:
                    expected=z['actual'].copy()
                    upstream=z['upstream'].copy() if 'upstream' in z else None
                if not np.array_equal(direct.view(np.uint16),expected.view(np.uint16)):raise AssertionError('public output differs from selected candidate')
                report['exact_selected_match']=True
                if upstream is not None:
                    report['exact_upstream_match']=bool(np.array_equal(direct.view(np.uint16),upstream.view(np.uint16)))
                    if not report['exact_upstream_match']:raise AssertionError('public MMA differs from upstream')
            error=Path(output)/f'N{n}xB{b}-errors.npz';error.parent.mkdir(parents=True,exist_ok=True)
            np.savez(error,actual=direct,cpu=cpu,error=direct.astype(np.float64)-cpu.astype(np.float64))
            report['error_artifact']=str(error.resolve());report['error_artifact_sha256']=sha(error)
            report['status']=('validated-public-selection' if selected['kind']!='fallback' else 'validated-native-fallback') if report['finite'] else 'nonfinite'
    except Exception as exc:report.update(status='failed',error=str(exc));raise
    finally:
        try:campaign.exclusive_gpu(uuid);report['guards']['after']=True
        except Exception as exc:report.update(status='interrupted',after_guard_error=str(exc))
        if graph is not None:graph.reset()
        if plan is not None:plan.close()
        write(native.result_path(output,n,b),report)
    return report


def execute(prepared,output,library,profile,one=None):
    tasks=[one] if one else native.cases(prepared)
    report=dict(status='running',completed=0,total=len(tasks),rows=[])
    for n,b in tasks:
        path=native.result_path(output,n,b)
        if path.exists():
            old=read(path);_,case,_,_=native.load_case(prepared,n,b)
            if old.get('status') in ('validated-public-selection','validated-native-fallback') and old.get('profile_sha256')==sha(profile) and old.get('library_sha256')==sha(library) and old.get('runner_sha256')==sha(__file__) and old.get('frontend_sha256')==sha(native.FRONTEND) and old.get('payload_sha256')==case['artifact_sha256'] and old.get('guards')=={'before':True,'after':True} and sha(old['error_artifact'])==old['error_artifact_sha256']:
                report['rows'].append(dict(case=old['case'],status='reused'));report['completed']+=1;continue
            report.update(status='blocked-existing-result',blocked_case=f'N{n}xB{b}');break
        try:r=run_one(prepared,output,n,b,library,profile)
        except Exception as exc:report.update(status='stopped',blocked_case=f'N{n}xB{b}',reason=str(exc));break
        if r['status'] not in ('validated-public-selection','validated-native-fallback'):
            report.update(status='stopped',blocked_case=r['case'],reason=r['status']);break
        report['rows'].append(dict(case=r['case'],status=r['status']));report['completed']+=1
        write(Path(output)/'checkpoint.json',report)
    else:report['status']='complete'
    write(Path(output)/'checkpoint.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','output','library','profile'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--case');a=p.parse_args()
    one=None
    if a.case:
        import re
        m=re.fullmatch(r'N(\d+)xB(\d+)',a.case)
        if not m:p.error('case must be N{N}xB{B}')
        one=(int(m[1]),int(m[2]))
    if a.execute:
        result=execute(a.prepared,a.output,str(a.library),str(a.profile),one)
    else:result=dict(status='prepared',gpu_execution=False,cases=[one] if one else native.cases(a.prepared))
    print(json.dumps(result,indent=2))
    if result['status'] in ('stopped','blocked-existing-result'):sys.exit(1)
if __name__=='__main__':main()
