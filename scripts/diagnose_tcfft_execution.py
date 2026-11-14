#!/usr/bin/env python3
"""Diagnose tcFFT determinism without changing kernels or comparison timing."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT)]


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def byte_difference(actual, expected, limit=64):
    import numpy as np
    actual=np.asarray(actual).view(np.uint8).reshape(-1)
    expected=np.asarray(expected).view(np.uint8).reshape(-1)
    if actual.shape != expected.shape: raise ValueError('byte comparison shape mismatch')
    positions=np.flatnonzero(actual!=expected)
    first=positions[:limit]
    return dict(equal=not bool(positions.size),different_bytes=int(positions.size),
        first_positions=first.tolist(),actual_bytes=actual[first].tolist(),expected_bytes=expected[first].tolist())


def run(a):
    import numpy as np
    from tcfft_baseline import Plan, support
    from run_semantic_campaign import exclusive_gpu
    from benchmark.review_fft_l_cpu_reference import metrics
    from cufft_low_precision_baseline import sha256
    a.output.mkdir(parents=True,exist_ok=True)
    info=support(a.n,a.batch)
    if not info['supported']: raise ValueError(info['reasons'])
    data=np.load(a.input,allow_pickle=False)
    if data.dtype!=np.float16 or data.size!=a.n*a.batch*2:
        raise ValueError('input must contain exactly N*batch FP16 complex pairs')
    reference=np.load(a.reference,allow_pickle=False).reshape(a.batch,a.n)
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_uuid
    exclusive_gpu(a.gpu_uuid)
    import torch
    rows=[];plan=None;stage='setup';original=None;previous=None
    identity=dict(n=a.n,batch=a.batch,gpu_uuid=a.gpu_uuid,library=str(a.library.resolve()),
        library_sha256=sha256(a.library),input_sha256=sha256(a.input),reference_sha256=sha256(a.reference),
        capability=info,threshold=None,scope='diagnostic; no performance measurement')
    stream=torch.cuda.Stream()
    try:
        with torch.cuda.stream(stream):
            raw=np.ascontiguousarray(data).view(np.uint8).reshape(-1)
            initial=np.full(raw.size+128,0xA5,np.uint8);initial[64:-64]=raw
            full_x=torch.from_numpy(initial).to('cuda');full_y=torch.full_like(full_x,0xA5)
            x=full_x[64:-64];y=full_y[64:-64]
            identity.update(stream=stream.cuda_stream,gpu_name=torch.cuda.get_device_name(0),
                gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
            if not stream.cuda_stream: raise ValueError('diagnostic requires nondefault stream')
            def snapshot(name):
                nonlocal original,previous
                stream.synchronize()
                output=y.cpu().numpy().copy()
                np.save(a.output/(name+'-output.npy'),output,allow_pickle=False)
                pairs=output.view(np.float16).astype(np.float64).reshape(a.batch,a.n,2)
                actual=pairs[...,0]+1j*pairs[...,1]
                row=dict(stage=name,status='error-report-only',all_outputs_finite=bool(np.isfinite(actual).all()),
                    frozen_reference_error=metrics(actual,reference),
                    input_immutable=np.array_equal(full_x.cpu().numpy(),initial),
                    output_guards_intact=bool(torch.all(full_y[:64]==0xA5)) and bool(torch.all(full_y[-64:]==0xA5)))
                if original is not None: row['versus_first_eager']=byte_difference(output,original)
                if previous is not None: row['versus_previous_stage']=byte_difference(output,previous)
                rows.append(row);write(a.output/(name+'.json'),row)
                if original is None: original=output
                previous=output
                if not row['input_immutable'] or not row['output_guards_intact']:
                    raise ValueError('input immutability or output guard failure')
            plan=Plan(a.library,a.n,a.batch)
            for name,pattern in [('same-plan-eager1',0x11),('same-plan-eager2',0x22)]:
                stage=name;y.fill_(pattern)
                plan.execute(x.data_ptr(),y.data_ptr(),stream=stream.cuda_stream);snapshot(stage)
            stage='output-after-plan-close';plan.close();plan=None;snapshot(stage)
            stage='recreated-plan-eager';plan=Plan(a.library,a.n,a.batch);y.fill_(0x33)
            plan.execute(x.data_ptr(),y.data_ptr(),stream=stream.cuda_stream);snapshot(stage)
            stage='capture';graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                plan.execute(x.data_ptr(),y.data_ptr(),stream=stream.cuda_stream)
            capture_row=dict(stage='capture',status='complete',note='capture records commands; outputs checked on replay')
            rows.append(capture_row);write(a.output/'capture.json',capture_row)
            for name,pattern in [('capture-replay1',0x44),('capture-replay2',0x55)]:
                stage=name;y.fill_(pattern);graph.replay();snapshot(stage)
            del graph
    except Exception as ex:
        row=dict(stage=stage,status='failed',error=repr(ex));rows.append(row)
        write(a.output/(stage+'-failure.json'),row)
    finally:
        try:
            stream.synchronize()
            if plan is not None:plan.close()
        except Exception as ex:
            rows.append(dict(stage='cleanup',status='failed',error=repr(ex)))
        try:exclusive_gpu(a.gpu_uuid)
        except Exception as ex:rows.append(dict(stage='exclusive-gpu-final',status='failed',error=repr(ex)))
        write(a.output/'diagnostic.json',dict(identity=identity,stages=rows))
    return 1 if any(r['status']=='failed' for r in rows) else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input','reference','library','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--n',type=int,required=True);p.add_argument('--batch',type=int,required=True)
    p.add_argument('--gpu-uuid',required=True)
    return run(p.parse_args())

if __name__=='__main__':sys.exit(main())
