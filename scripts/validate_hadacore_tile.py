#!/usr/bin/env python3
"""Prepare and (when explicitly requested) validate the HadaCore tile ABI.

Preparation is CPU-only.  GPU execution is deliberately opt-in and one case at
 a time; this script never selects a device or starts a measurement by default.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROVIDER = ROOT / 'scripts' / 'hadacore_tile_provider.py'
BUILD = ROOT / 'build' / 'hadacore_tile'
OUT = ROOT / 'results' / 'hadacore_tile_validation_20261111'
UUID = None
CONTRACT = 'fp16-mma-orthonormal-final-scale'
QUICK = [(n,b) for n in (256,4096,32768) for b in (1,16,256)]
FULL = [(n,b) for n in (2**k for k in range(1,16)) for b in (1,2,4,16,64,256)]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fwht(x):
    x = np.asarray(x, dtype=np.float16).copy()
    h=1
    while h < x.shape[-1]:
        for i in range(0,x.shape[-1],2*h):
            a=x[...,i:i+h].copy(); b=x[...,i+h:i+2*h].copy()
            x[...,i:i+h]=(a+b).astype(np.float16); x[...,i+h:i+2*h]=(a-b).astype(np.float16)
        h*=2
    return x
def input_data(n,b):
    # Fixed integer pattern avoids platform RNG differences.
    a=(np.arange(n*b,dtype=np.int64)*1103515245+12345)%65521
    return ((a.astype(np.float32)-32760)/32760).astype(np.float16).reshape(b,n)
def parse_case(value):
    if isinstance(value, tuple): return value
    if 'x' not in value: raise ValueError('case must be N{n}xB{b}, e.g. N256xB1')
    n,b=(int(v[1:]) for v in value.split('x'))
    if (n,b) not in FULL: raise ValueError('case outside bounded full matrix')
    return n,b
def descriptor(n,b,groups_per_block=1,chunks=None):
    import importlib.util
    spec=importlib.util.spec_from_file_location('hadacore_tile_provider',PROVIDER); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    log=int(math.log2(n)); return m.descriptor(log,b,arithmetic_contract=CONTRACT,chunks=chunks,groups_per_block=groups_per_block)
def target_uuid(value=None):
    value = value or os.environ.get('CUBUTTERFLY_GPU_UUID')
    if not isinstance(value, str) or not value.strip():
        raise ValueError('target GPU UUID required: use --gpu-uuid or CUBUTTERFLY_GPU_UUID')
    return value.strip()


def prepare(cases, out=OUT, build=BUILD, device_uuid=UUID):
    device_uuid = target_uuid(device_uuid)
    out.mkdir(parents=True,exist_ok=True); rows=[]
    for n,b in cases:
        item={'case':f'N{n}xB{b}','N':n,'B':b,'dtype':'float16','reference':'radix2-stage-FP16','input_hash':None,'reference_hash':None}
        x=input_data(n,b); y=fwht(x); path=out/f'N{n}xB{b}.npz'; np.savez(path,input=x,reference=y)
        item['input_hash']=hashlib.sha256(x.tobytes()).hexdigest(); item['reference_hash']=hashlib.sha256(y.tobytes()).hexdigest(); item['artifact_sha256']=sha(path)
        try: item['descriptor']=descriptor(n,b)
        except ValueError as e: item.update(status='unsupported',reason=str(e))
        else: item['status']='prepared'
        rows.append(item)
    tasks=[]
    for item in rows:
        if item['status'] != 'prepared':
            continue
        for groups_per_block in (1, 2, 4):
            # Keep every requested geometry explicit.  A small case may have
            # fewer groups than the block reuse factor; the GPU runner records
            # that as a legal one-block execution rather than padding input.
            tasks.append(dict(case=item['case'], N=item['N'], B=item['B'],
                              groups_per_block=groups_per_block,
                              arithmetic_contract=CONTRACT, gpu_uuid=device_uuid,
                              status='pending-gpu-validation'))
    report={'schema':'hadacore-tile-validation-v1','gpu_execution':False,'gpu_uuid':device_uuid,
            'contract':CONTRACT,'cases':rows,'gpu_tasks':tasks,
            'provider_sha256':sha(PROVIDER)}
    if (build/'build.json').exists():
        report['build_sha256']=sha(build/'build.json')
    (out/'prepare.json').write_text(json.dumps(report,indent=2)+'\n'); return report
def validate_inputs(n,b,output,build,device_uuid):
    prepared=json.loads((output/'prepare.json').read_text())
    if prepared['gpu_uuid'] != device_uuid: raise ValueError('prepared target UUID mismatch')
    if prepared['provider_sha256'] != sha(PROVIDER): raise ValueError('provider source changed; prepare again')
    if prepared.get('build_sha256') != sha(build/'build.json'):
        raise ValueError('build changed or not bound during preparation; prepare again')
    row=next((r for r in prepared['cases'] if (r['N'],r['B'])==(n,b)),None)
    if not row: raise ValueError('run requires prior --prepare case')
    if row['status']!='prepared': raise ValueError(row.get('reason','unsupported'))
    payload=output/f'N{n}xB{b}.npz'
    if sha(payload)!=row['artifact_sha256']: raise ValueError('payload hash mismatch')
    build_record=json.loads((build/'build.json').read_text())
    for name,digest in build_record['artifacts'].items():
        if sha(build/name)!=digest: raise ValueError('build artifact hash mismatch: '+name)
    with np.load(payload,allow_pickle=False) as z:
        x,y=z['input'].copy(),z['reference'].copy()
    if hashlib.sha256(x.tobytes()).hexdigest()!=row['input_hash'] or hashlib.sha256(y.tobytes()).hexdigest()!=row['reference_hash']:
        raise ValueError('input/reference array hash mismatch')
    return row,x,y,build_record


def run_one(n,b,groups_per_block=1,output=OUT,device_uuid=UUID,build=BUILD,
            execute=False,upstream_library=None):
    device_uuid = target_uuid(device_uuid)
    output,build=Path(output),Path(build)
    if not execute:
        prepared=json.loads((output/'prepare.json').read_text())
        if prepared['gpu_uuid']!=device_uuid: raise ValueError('prepared target UUID mismatch')
        return dict(status='queued',case=f'N{n}xB{b}',groups_per_block=groups_per_block,
                    gpu_uuid=device_uuid,measurement='pending; GPU execution requires explicit --execute')
    # Validate all disk artifacts before importing torch or initializing CUDA.
    row,x,cpu,record=validate_inputs(n,b,output,build,device_uuid)
    if upstream_library is None: raise ValueError('explicit upstream comparator library required')
    upstream_library=Path(upstream_library)
    sys.path.insert(0,str(ROOT/'scripts'))
    import run_semantic_campaign as campaign
    import hadacore_tile_provider as provider
    import hadacore_baseline as upstream
    upstream_record=json.loads((upstream_library.parent/'build.json').read_text())
    for name,digest in upstream_record['artifacts'].items():
        if sha(upstream_library.parent/name)!=digest: raise ValueError('upstream artifact hash mismatch: '+name)
    if upstream_record['commit']!=upstream.COMMIT: raise ValueError('upstream commit mismatch')
    report=dict(schema='hadacore-tile-validation-v2',status='running',case=row['case'],
        gpu_uuid=device_uuid,groups_per_block=groups_per_block,contract=CONTRACT,
        build_sha256=sha(build/'build.json'),provider_sha256=sha(PROVIDER),
        runner_sha256=sha(__file__),payload_sha256=row['artifact_sha256'],
        upstream_library_sha256=sha(upstream_library),guards={'before':False,'after':False},
        arithmetic_reference='hash-pinned upstream FP16 MMA + FP32 output scale',
        cpu_reference_scope='diagnostic radix2-stage-FP16; different rounding contract')
    destination=output/'rows'/f'{row["case"]}-groups{groups_per_block}.json'
    plan=baseline=graph=None
    try:
        campaign.exclusive_gpu(device_uuid);report['guards']['before']=True
        os.environ['CUDA_VISIBLE_DEVICES']=device_uuid
        import torch
        if not campaign.same_gpu_uuid(device_uuid,str(torch.cuda.get_device_properties(0).uuid)):
            raise ValueError('actual GPU UUID differs from prepared target')
        props=torch.cuda.get_device_properties(0)
        report['gpu_identity']=dict(uuid=device_uuid,name=props.name,memory_bytes=props.total_memory,
                                    sm=props.major*10+props.minor)
        report['library']=str((build/'libhadacore_tile.so').resolve())
        report['library_sha256']=sha(build/'libhadacore_tile.so')
        stream=torch.cuda.Stream()
        semantic=dict(schema='cubutterfly-semantic-v1',operator=2,storage=0,compute=0,
            length_mode=0,shape=[n],batch=b,input_orders=[0],output_orders=[0],
            placement=0,inverse=0,normalize_inverse=False)
        plan=provider.TilePlan(build/'libhadacore_tile.so',int(math.log2(n)),b,
            arithmetic_contract=CONTRACT,groups_per_block=groups_per_block,scale=math.sqrt(n))
        baseline=upstream.Plan(upstream_library,semantic)
        if not hasattr(plan,'resources'):
            raise RuntimeError('local provider lacks runtime resources() ABI; rebuild before calibration')
        report['resources']=plan.resources()
        report['mapping']=plan.info
        with torch.cuda.stream(stream):
            # 128-byte guards preserve vector alignment of the interior pointers.
            def guarded():
                raw=torch.full((n*b+128,),0x3555,dtype=torch.int16,device='cuda')
                return raw,raw[64:-64].view(torch.float16).reshape(b,n)
            srcraw,src=guarded(); outraw,out=guarded(); refraw,ref=guarded()
            src.copy_(torch.from_numpy(x)); saved=srcraw.clone()
            call=lambda:plan.execute(src.data_ptr(),out.data_ptr(),stream=stream.cuda_stream)
            refcall=lambda:baseline.execute(src.data_ptr(),ref.data_ptr(),stream=stream.cuda_stream)
            refcall();call();stream.synchronize()
            direct=out.cpu().numpy().copy(); reference=ref.cpu().numpy().copy()
            def verify():
                stream.synchronize()
                for raw in (srcraw,outraw,refraw):
                    if not bool(torch.all(raw[:64]==0x3555)) or not bool(torch.all(raw[-64:]==0x3555)):
                        raise AssertionError('buffer guard corrupted')
                if not torch.equal(srcraw,saved): raise AssertionError('input changed')
                if not np.array_equal(out.cpu().numpy().view(np.uint16),direct.view(np.uint16)):
                    raise AssertionError('repeat/graph differs from direct')
            for _ in range(3):call();verify()
            report.update(exact_repeat=True,input_immutable=True,buffer_guards=True)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):call()
            out.fill_(float('nan'));graph.replay();verify();report['graph_reproduced']=True
            graph.reset();graph=None
            report['timing']=campaign.time_execute(torch,stream,call,lambda:None,False,verify,lambda:out.fill_(float('nan')))
            def verify_upstream():
                verify()
                if not np.array_equal(ref.cpu().numpy().view(np.uint16),reference.view(np.uint16)):
                    raise AssertionError('upstream replay differs from direct')
            report['upstream_timing']=campaign.time_execute(torch,stream,refcall,lambda:None,False,verify_upstream)
            verify_upstream();report['upstream_graph_reproduced']=True
            report['comparison']=campaign.check_full(direct,reference,0,'error-report',True)
            report['exact_upstream_match']=bool(np.array_equal(direct.view(np.uint16),reference.view(np.uint16)))
            report['cpu_diagnostic']=campaign.check_full(direct,cpu,0,'error-report',True)
            report['finite']=bool(np.isfinite(direct).all())
            error_path=output/f'{row["case"]}-groups{groups_per_block}-errors.npz'
            np.savez(error_path,actual=direct,
                     upstream=reference,cpu_stage_fp16=cpu,error_upstream=direct.astype(np.float64)-reference.astype(np.float64),
                     error_cpu=direct.astype(np.float64)-cpu.astype(np.float64))
            report['error_artifact']=str(error_path)
            report['error_artifact_sha256']=sha(error_path)
            verify()
            report['status']='validated-exact-upstream' if report['finite'] and report['exact_upstream_match'] else 'accuracy-review-required'
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        try:
            campaign.exclusive_gpu(device_uuid);report['guards']['after']=True
        except Exception as exc:
            report.update(status='interrupted',after_guard_error=str(exc))
        if graph is not None:graph.reset()
        if plan is not None:plan.close()
        if baseline is not None:baseline.close()
        report['ended_unix']=time.time()
        destination.parent.mkdir(parents=True,exist_ok=True)
        tmp=destination.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(destination)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true')
    p.add_argument('--case');p.add_argument('--full',action='store_true');p.add_argument('--execute',action='store_true')
    p.add_argument('--groups-per-block',type=int,default=1);p.add_argument('--build',type=Path,default=BUILD)
    p.add_argument('--output',type=Path,default=OUT);p.add_argument('--gpu-uuid',default=UUID)
    p.add_argument('--upstream-library',type=Path);a=p.parse_args()
    try:
        a.gpu_uuid = target_uuid(a.gpu_uuid)
    except ValueError as exc:
        p.error(str(exc))
    cases=[parse_case(a.case)] if a.case else FULL if a.full else QUICK
    if a.prepare or not a.run:
        print(json.dumps(prepare(cases,a.output,a.build,a.gpu_uuid),indent=2));return
    if not a.case: p.error('--run requires one explicit --case')
    print(json.dumps(run_one(*cases[0],groups_per_block=a.groups_per_block,output=a.output,
        device_uuid=a.gpu_uuid,build=a.build,execute=a.execute,upstream_library=a.upstream_library),indent=2))
if __name__=='__main__':main()
