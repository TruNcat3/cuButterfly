#!/usr/bin/env python3
"""Controlled normalization/buffer/filter-cache experiment; launch is explicit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'paper/tools'))
import run_xd_application_breakdown_20261012 as base
sys.path.insert(0,str(ROOT/'python'))
from cubutterfly import AlgorithmPolicy
from cubutterfly.real_fft import RealFFT

INPUT=base.DEFAULT_OUT/'manifest.json'
OUTPUT=ROOT/'results/paper_targeted_20261012/filter_normalization_optimization_v2'


def configuration():
    import inspect
    import cupy
    from cupyx.scipy import fft as cpfft
    from cupyx.scipy.signal import _signaltools_core
    env,identity=base.identity()
    identity['source_hashes'][str(Path(__file__).resolve())]=base.digest(Path(__file__))
    for path in (ROOT/'python/cubutterfly/__init__.py',ROOT/'python/cubutterfly/real_fft.py'):
        identity['source_hashes'][str(path)]=base.digest(path)
    manifest=base.verify(INPUT)
    identity['cupy_backend']=dict(version=cupy.__version__,
        convolution_core_sha256=base.digest(Path(inspect.getfile(_signaltools_core))),
        actual_real_next_fast_len={str(n):int(cpfft.next_fast_len(n+30,real=True)) for n in (1024,2048)})
    return env,dict(identity=identity,input_manifest=str(INPUT),input_manifest_sha256=base.digest(INPUT),
                    filters=manifest['filters'],contracts=['forward_each_call','backward_each_call','backward_fixed_filter_cache'],
                    reuse_buffers=[False,True],tolerance='unassessed, finite errors only',
                    timing='plan/setup/filter-cache creation excluded; complete eager call, no inner synchronization')


def prepare(output):
    output.mkdir(parents=True,exist_ok=True)
    _,spec=configuration()
    path=output/'spec.json'
    if path.exists() and json.loads(path.read_text())!=spec:
        raise RuntimeError('immutable spec changed; use new output root')
    base.save(path,spec)
    return path


def make_call(image,kernel,shape,forward,filter_forward,inverse,torch,norm,cached=False):
    pad=lambda v:torch.nn.functional.pad(v,(0,shape[1]-v.shape[-1],0,shape[0]-v.shape[-2]))
    hf=filter_forward(pad(kernel)) if cached else None
    def call():
        xf=forward(pad(image))
        spectrum=hf if cached else filter_forward(pad(kernel))
        product=xf*spectrum
        if norm=='forward': product*=np.prod(shape)
        full=inverse(product)
        return base.filter_api.same_crop(full,image.shape,kernel.shape).clone()
    return call


def run(args):
    spec=json.loads(args.spec.read_text())
    if configuration()[1]!=spec: raise RuntimeError('spec or inputs changed')
    import torch
    import cupy as cp
    from cupyx.scipy import fft as cpfft
    from cupyx.scipy.signal import fftconvolve
    device=torch.device('cuda:0')
    benchmark=base.load_module('xd_gpu_identity',ROOT/'scripts/benchmark_xd_real.py')
    gpu=benchmark._gpu_identity()
    if str(gpu.get('uuid','')).removeprefix('GPU-')!=spec['identity']['gpu_uuid'].removeprefix('GPU-'):
        raise RuntimeError('wrong calibrated GPU')
    output=Path(os.environ.get('XD_REPORT_DIR','.'))/'result.json'
    if output.exists(): raise RuntimeError('prior result exists')
    report=dict(spec_sha256=base.digest(args.spec),gpu=gpu,identity=spec['identity'],rows=[],
                torch=torch.__version__,cupy=cp.__version__,accepted=None)
    policy=AlgorithmPolicy.MEASURE if os.environ.get('CUBUTTERFLY_E13_ALGORITHM_POLICY')=='measure' else AlgorithmPolicy.DEFAULT
    for item in spec['filters']:
        data=np.load(item['artifact'])
        image=torch.as_tensor(data['image'],device=device)
        kernel=torch.as_tensor(data['kernel'],device=device)
        shape=base.filter_api.padding_shape(image.shape[-2:],kernel.shape)
        batch=item['contract'].get('batch',1)
        for norm,cached in [('forward',False),('backward',False),('backward',True)]:
            for reuse in (False,True):
                adapter=RealFFT(base.LIB,shape,batch,device,norm=norm,reuse_buffers=reuse,policy=policy)
                # Separate filter adapter prevents reused image/filter spectra
                # aliasing for B1, including the fixed-filter-cache contract.
                ha=RealFFT(base.LIB,shape,1,device,norm=norm,reuse_buffers=reuse,policy=policy)
                plans=base.require_plans([adapter,ha],args.require_cache)
                own=make_call(image,kernel,shape,adapter.rfftn,ha.rfftn,adapter.irfftn,torch,norm,cached)
                tf=lambda v:torch.fft.rfftn(v,dim=(-2,-1),norm=norm)
                hf=lambda v:torch.fft.rfftn(v,dim=(-2,-1),norm=norm)
                ti=lambda v:torch.fft.irfftn(v,s=shape,dim=(-2,-1),norm=norm)
                # Match output-buffer contract on Torch as well.
                if reuse:
                    xb=torch.empty((*image.shape[:-2],shape[0],shape[1]//2+1),dtype=torch.complex64,device=device)
                    hb=torch.empty((shape[0],shape[1]//2+1),dtype=torch.complex64,device=device)
                    ib=torch.empty((batch,*shape),dtype=torch.float32,device=device) if image.ndim==3 else torch.empty(shape,dtype=torch.float32,device=device)
                    tf=lambda v:torch.fft.rfftn(v,dim=(-2,-1),norm=norm,out=xb)
                    hf=lambda v:torch.fft.rfftn(v,dim=(-2,-1),norm=norm,out=hb)
                    ti=lambda v:torch.fft.irfftn(v,s=shape,dim=(-2,-1),norm=norm,out=ib)
                same=make_call(image,kernel,shape,tf,hf,ti,torch,norm,cached)
                stream=cp.cuda.ExternalStream(torch.cuda.current_stream(device).cuda_stream)
                with cp.cuda.Device(0),stream:
                    cx,ch=cp.from_dlpack(image),cp.from_dlpack(kernel)
                    methods=[('cubutterfly',own),('torch_cufft_same_padding',same)]
                    # fftconvolve is the original uncached contract only.
                    if not cached and not reuse:
                        methods.append(('cupy_fftconvolve_native_padding',lambda:fftconvolve(cx,ch if batch==1 else ch[None],mode='same',axes=(-2,-1))))
                    for name,call in methods:
                        actual=call()
                        torch.cuda.synchronize(device)
                        array=cp.asnumpy(actual) if isinstance(actual,cp.ndarray) else actual.cpu().numpy()
                        error=base.distribution(array,data['reference'])
                        row=dict(contract=item['contract'],norm=norm,filter_spectrum_cached=cached,reuse_buffers=reuse,
                            method=name,padding=list(shape),cupy_actual_next_fast_len=[cpfft.next_fast_len(int(n+k-1),real=True) for n,k in zip(image.shape[-2:],kernel.shape)],
                            error=error,accepted=None,plans=plans if name=='cubutterfly' else None,
                            timing=None if args.plan_only else base.timing(call,torch,device,3,5,5))
                        report['rows'].append(row)
                        base.save(output,report)
                        if not error['finite']: raise RuntimeError('nonfinite output')
                adapter.close();ha.close()
    report['status']='complete';base.save(output,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--launch',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--spec',type=Path)
    parser.add_argument('--until');parser.add_argument('--plan-only',action='store_true');parser.add_argument('--require-cache',action='store_true')
    args=parser.parse_args()
    if args.prepare: print(prepare(args.output));return
    if args.run: run(args);return
    if not args.until: parser.error('--launch requires finite --until')
    path=prepare(args.output);env,spec=configuration();seed=None
    for policy in ('measure','default'):
        report_dir=args.output/f'{policy}_guard'
        command=[base.PYTHON,str(Path(__file__).resolve()),'--run','--spec',str(path.resolve())]+(['--plan-only'] if policy=='measure' else ['--require-cache'])
        record=dict(policy=policy,report_dir=str(report_dir.resolve()),command=command)
        if (report_dir/'release').exists(): seed=base.reuse_release(record,base.fingerprint(spec),seed);continue
        wrapper=[base.PYTHON,str(ROOT/'paper/tools/run_application_batch_20261012.py'),'--cache-child','--policy',policy]
        if seed: wrapper+=['--cache-seed',str(seed)]
        guard=[base.PYTHON,str(ROOT/'scripts/run_isolated_gpu.py'),'--gpu-uuid',spec['identity']['gpu_uuid'],'--report-dir',str(report_dir.resolve()),'--until',args.until,'--idle-seconds','3','--']+wrapper+['--']+command
        with (args.output/'queue.log').open('a') as log: result=subprocess.run(guard,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError('guard failed/expired; artifacts preserved')
        seed=base.accept_release(record,base.fingerprint(spec),seed)


if __name__=='__main__': main()
