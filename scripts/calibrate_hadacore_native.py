#!/usr/bin/env python3
"""Measure STRICT native FWHT endpoints against prepared FP16 CPU references.

CPU-only by default; exported native_rows are separate from MMA profiles.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FRONTEND=ROOT/'python/cubutterfly/__init__.py'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def load_case(prepared,n,b):
    p=read(Path(prepared)/'prepare.json')
    row=next((r for r in p['cases'] if (r['N'],r['B'])==(n,b)),None)
    if row is None:raise ValueError('case not prepared')
    payload=Path(prepared)/f'N{n}xB{b}.npz'
    if sha(payload)!=row['artifact_sha256']:raise ValueError('prepared payload hash mismatch')
    with np.load(payload,allow_pickle=False) as z:x,y=z['input'].copy(),z['reference'].copy()
    if x.shape!=(b,n) or y.shape!=(b,n) or x.dtype!=np.float16 or y.dtype!=np.float16:raise ValueError('prepared shape/dtype mismatch')
    if hashlib.sha256(x.tobytes()).hexdigest()!=row['input_hash'] or hashlib.sha256(y.tobytes()).hexdigest()!=row['reference_hash']:raise ValueError('array hash mismatch')
    return p,row,x,y

def result_path(output,n,b):return Path(output)/'rows'/f'N{n}xB{b}.json'
def identity(prepared,n,b,library):
    p,row,x,y=load_case(prepared,n,b)
    environment={k:os.environ.get(k) for k in ('CUBUTTERFLY_PROFILE_ROOT','CUBUTTERFLY_REGISTRY',
        'CUBUTTERFLY_COMPILE_MODE','CUBUTTERFLY_TEMPLATE_ROOT')}
    registry=environment['CUBUTTERFLY_REGISTRY']
    if registry:
        if registry.startswith(('/sdb','/mnt/sdb')):raise ValueError('forbidden registry storage')
        environment['registry_sha256']=sha(registry)
    return dict(calibration_environment=environment,timing_source_sha256=sha(ROOT/'scripts/run_semantic_campaign.py'),case=row['case'],gpu_uuid=p['gpu_uuid'],payload_sha256=row['artifact_sha256'],
                library=str(Path(library).resolve()),library_sha256=sha(library),
                runner_sha256=sha(__file__),frontend_sha256=sha(FRONTEND),arithmetic_policy='STRICT')

def checked(prepared,output,n,b,library):
    row=read(result_path(output,n,b));expected=identity(prepared,n,b,library)
    for k,v in expected.items():
        if row.get(k)!=v:raise ValueError('native identity mismatch: '+k)
    if row.get('status')!='measured-error-report':raise ValueError('native result is not measured')
    for key in ('finite','exact_repeat','input_immutable','buffer_guards','graph_reproduced'):
        if row.get(key) is not True:raise ValueError('native validation missing: '+key)
    if row.get('guards')!={'before':True,'after':True}:raise ValueError('native exclusive guards incomplete')
    if sha(row['error_artifact'])!=row['error_artifact_sha256']:raise ValueError('native error artifact changed')
    timing=row['timing']
    if not math.isfinite(timing['median_ms']) or timing['median_ms']<=0 or len(timing['samples_ms'])!=7 or any(not math.isfinite(v) or v<=0 for v in timing['samples_ms']):raise ValueError('invalid native timing')
    if not row.get('mapping') or not row.get('gpu_identity'):raise ValueError('missing mapping/hardware')
    return row

def run_one(prepared,output,n,b,library):
    binding=identity(prepared,n,b,library);p,row,x,expected=load_case(prepared,n,b)
    report=dict(binding,status='running',guards={'before':False,'after':False},
                reference='CPU radix2-stage-FP16; full error report without acceptance threshold')
    # Read/import the checked-in frontend, preserving installation profile env.
    sys.path.insert(0,str(ROOT/'python'));sys.path.insert(0,str(ROOT/'scripts'))
    import run_semantic_campaign as campaign
    uuid=binding['gpu_uuid'];plan=None;graph=None
    try:
        campaign.exclusive_gpu(uuid);report['guards']['before']=True
        os.environ['CUDA_VISIBLE_DEVICES']=uuid
        import torch
        import cubutterfly as cb
        if Path(cb.__file__).resolve()!=FRONTEND.resolve():raise RuntimeError('local frontend not loaded')
        props=torch.cuda.get_device_properties(0)
        if not campaign.same_gpu_uuid(uuid,str(props.uuid)):raise ValueError('actual GPU differs from prepared identity')
        report['gpu_identity']=dict(uuid=uuid,name=props.name,memory_bytes=props.total_memory,sm=props.major*10+props.minor)
        report['environment']={k:v for k,v in os.environ.items() if k.startswith('CUBUTTERFLY_')}
        stream=torch.cuda.Stream();lib=cb.Library(library)
        transform=cb.Transform(op=cb.Operator.FWHT,storage=cb.DataType.FP16,compute=cb.ComputeType.NATIVE,
            extents=[n],batch=b,inverse=False,normalize_inverse=False,in_place=False,
            arithmetic_policy=cb.ArithmeticPolicy.STRICT)
        with cb.Descriptor(lib,transform) as descriptor:plan=cb.Plan(lib,descriptor,stream=stream.cuda_stream)
        report['mapping']=json.loads(plan.mapping_json());report['descriptor']=json.loads(plan.descriptor_json())
        if plan.input_size()!=n*b*2 or plan.output_size()!=n*b*2:raise ValueError('packed caller-owned storage differs from request')
        with torch.cuda.stream(stream):
            def guarded():
                raw=torch.full((n*b+128,),0x3555,dtype=torch.int16,device='cuda')
                return raw,raw[64:-64].view(torch.float16).reshape(b,n)
            srcraw,src=guarded();outraw,out=guarded();src.copy_(torch.from_numpy(x));saved=srcraw.clone()
            workspace=torch.empty(max(1,plan.workspace_size()),dtype=torch.uint8,device='cuda')
            if plan.workspace_size():plan.set_workspace(workspace,plan.workspace_size())
            call=lambda:plan.execute(src,out)
            call();stream.synchronize();direct=out.cpu().numpy().copy()
            def verify():
                stream.synchronize()
                for raw in (srcraw,outraw):
                    if not bool(torch.all(raw[:64]==0x3555)) or not bool(torch.all(raw[-64:]==0x3555)):raise AssertionError('buffer guards failed')
                if not torch.equal(srcraw,saved):raise AssertionError('input mutated')
                if not np.array_equal(out.cpu().numpy().view(np.uint16),direct.view(np.uint16)):raise AssertionError('repeat/graph mismatch')
            for _ in range(3):call();verify()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):call()
            out.fill_(float('nan'));graph.replay();verify();graph.reset();graph=None
            report.update(exact_repeat=True,buffer_guards=True,input_immutable=True,graph_reproduced=True)
            report['timing']=campaign.time_execute(torch,stream,call,lambda:None,False,verify,lambda:out.fill_(float('nan')))
            report['finite']=bool(np.isfinite(direct).all())
            report['cpu_error']=campaign.check_full(direct,expected,0,'error-report',True)
            error_path=Path(output)/f'N{n}xB{b}-errors.npz';error_path.parent.mkdir(parents=True,exist_ok=True)
            np.savez(error_path,actual=direct,reference=expected,error=direct.astype(np.float64)-expected.astype(np.float64))
            report['error_artifact']=str(error_path.resolve());report['error_artifact_sha256']=sha(error_path)
            report['status']='measured-error-report' if report['finite'] else 'nonfinite'
    except Exception as exc:report.update(status='failed',error=str(exc));raise
    finally:
        try:campaign.exclusive_gpu(uuid);report['guards']['after']=True
        except Exception as exc:report.update(status='interrupted',after_guard_error=str(exc))
        if graph is not None:graph.reset()
        if plan is not None:plan.close()
        report['ended_unix']=time.time();write(result_path(output,n,b),report)
    return report

def cases(prepared):
    return [(r['N'],r['B']) for r in read(Path(prepared)/'prepare.json')['cases'] if r['status']=='prepared']

def export(prepared,output,library,path):
    rows=[];pending=[];gpu_identity=None
    for n,b in cases(prepared):
        try:r=checked(prepared,output,n,b,library)
        except (OSError,KeyError,ValueError) as exc:pending.append(dict(log_n=n.bit_length()-1,batch=b,reason=str(exc)));continue
        if gpu_identity is not None and r['gpu_identity']!=gpu_identity:raise ValueError('mixed native GPUs')
        gpu_identity=r['gpu_identity'];evidence=result_path(output,n,b)
        rows.append(dict(log_n=n.bit_length()-1,batch=b,mapping=r['mapping'],median_ms=r['timing']['median_ms'],
            evidence_path=str(evidence.resolve()),evidence_sha256=sha(evidence),library_sha256=sha(library),timing_protocol=r['timing']['protocol']))
    if not rows:raise ValueError('no current validated native measurements')
    result=dict(schema='cubutterfly-hadacore-native-profile-v1',gpu_uuid=gpu_identity['uuid'],sm=gpu_identity['sm'],
                gpu_identity=gpu_identity,library=str(Path(library).resolve()),library_sha256=sha(library),native_rows=rows,pending=pending)
    write(path,result);return result

def execute(prepared,output,library):
    state=dict(status='running',completed=0,total=len(cases(prepared)),jobs={})
    checkpoint=Path(output)/'native-checkpoint.json'
    for n,b in cases(prepared):
        name=f'N{n}xB{b}';p=result_path(output,n,b)
        if p.exists():
            try:checked(prepared,output,n,b,library)
            except (OSError,KeyError,ValueError) as exc:
                state.update(status='blocked-existing-result',blocked_job=name,reason=str(exc));write(checkpoint,state);return state
            state['jobs'][name]='reused';state['completed']+=1;write(checkpoint,state);continue
        log=Path(output)/'logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(Path(__file__).resolve()),'--execute','--case',name,'--prepared',str(prepared),'--output',str(output),'--library',str(library)]
        with log.open('ab') as f:sub=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        try:
            if sub.returncode:raise ValueError('native subprocess failed; see '+str(log))
            checked(prepared,output,n,b,library)
        except (OSError,KeyError,ValueError) as exc:
            state.update(status='stopped',blocked_job=name,reason=str(exc));write(checkpoint,state);return state
        state['jobs'][name]='measured';state['completed']+=1;write(checkpoint,state)
    state['status']='complete';write(checkpoint,state);return state

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--library',type=Path,required=True)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--execute',action='store_true');mode.add_argument('--export-only',action='store_true')
    p.add_argument('--case');p.add_argument('--profile',type=Path);a=p.parse_args()
    if a.export_only:
        if not a.profile:p.error('--export-only requires --profile')
        r=export(a.prepared,a.output,a.library,a.profile)
    elif a.execute:
        if a.case:
            import re
            match=re.fullmatch(r'N(\d+)xB(\d+)',a.case)
            if match is None:p.error('case must be N{N}xB{B}')
            r=run_one(a.prepared,a.output,int(match[1]),int(match[2]),str(a.library))
        else:r=execute(a.prepared,a.output,a.library)
    else:r=dict(status='prepared',gpu_execution=False,cases=cases(a.prepared))
    print(json.dumps(r,indent=2))
    if r.get('status') in ('failed','nonfinite','interrupted','stopped','blocked-existing-result'):sys.exit(1)
if __name__=='__main__':main()
