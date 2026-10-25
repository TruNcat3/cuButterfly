#!/usr/bin/env python3
"""Bounded compact xD comparison and filter diagnostics; no implicit GPU launch.

Prepare CPU artifacts first. --launch uses the existing isolated GPU/cache
workflow: MEASURE prepares a cache, then DEFAULT must replay its frozen copy.
Segment event timings are diagnostic and must not be summed as application time.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
try:
    from scipy import fft
    if not all(hasattr(fft, name) for name in ('rfftn', 'irfftn', 'next_fast_len')):
        raise ImportError('legacy SciPy exposes no scipy.fft namespace')
except (ImportError, AttributeError):
    class _NumpyFFT:
        rfftn = staticmethod(np.fft.rfftn)
        irfftn = staticmethod(np.fft.irfftn)

        @staticmethod
        def next_fast_len(value, real=False):
            value = max(1, int(value))
            return 1 << (value - 1).bit_length()

    fft = _NumpyFFT()


def _legacy_fft_size(values, s, axes):
    if s is not None:
        return int(np.prod(tuple(int(value) for value in s)))
    axes = tuple(range(values.ndim)) if axes is None else tuple(axes)
    return int(np.prod(tuple(values.shape[axis] for axis in axes)))


if not hasattr(fft, '_cubutterfly_norm_compat'):
    _rfftn = fft.rfftn
    _irfftn = fft.irfftn

    def _rfftn_norm(values, s=None, axes=None, norm=None):
        if norm in (None, 'backward'):
            return _rfftn(values, s=s, axes=axes, norm=None)
        if norm == 'forward':
            return _rfftn(values, s=s, axes=axes, norm=None) / _legacy_fft_size(values, s, axes)
        return _rfftn(values, s=s, axes=axes, norm=norm)

    def _irfftn_norm(values, s=None, axes=None, norm=None):
        if norm in (None, 'backward'):
            return _irfftn(values, s=s, axes=axes, norm=None)
        if norm == 'forward':
            return _irfftn(values, s=s, axes=axes, norm=None) * _legacy_fft_size(values, s, axes)
        return _irfftn(values, s=s, axes=axes, norm=norm)

    fft.rfftn, fft.irfftn = _rfftn_norm, _irfftn_norm
    fft._cubutterfly_norm_compat = True

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper/tools'))
from run_application_batch_20261012 import (LIB, PYTHON, TARGETS, accept_release,
    digest, effective_environment, fingerprint, reuse_release, save)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FILTER_RUNNER = ROOT / 'paper/experiments/e13/run_fft_filter_end_to_end.py'
filter_api = load_module('xd_filter_reference', FILTER_RUNNER)
CASES = [(shape, batch) for shape in ((1024, 1024), (2048, 2048),
         (64, 64, 64), (128, 128, 128)) for batch in (1, 4)]
STAGES = ('pad', 'forward', 'filterFFT', 'multiply', 'normalize', 'inverse', 'crop')
DEFAULT_OUT = ROOT / 'results/paper_targeted_20261012/xd_application_breakdown_v1'


def seeded_rng(seed: int):
    """Return a deterministic RNG on both current and legacy NumPy hosts."""
    factory = getattr(np.random, 'default_rng', None)
    return factory(seed) if factory is not None else np.random.RandomState(seed)


def distribution(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape:
        raise ValueError('reference shape mismatch')
    # The shared real-valued distribution must not discard imaginary errors.
    if np.iscomplexobj(actual) or np.iscomplexobj(reference):
        actual = np.stack((actual.real, actual.imag), axis=-1)
        reference = np.stack((reference.real, reference.imag), axis=-1)
    return filter_api.error_distribution(actual, reference)


def identity():
    uuid, calibration_path = TARGETS['fft_filter']
    calibration = json.loads(calibration_path.read_text())
    if calibration['measurement_identity']['library_sha256'] != digest(LIB):
        raise RuntimeError('frozen runtime/calibration mismatch')
    env, env_identity = effective_environment(calibration, uuid, 8 * 1024**3)
    paths = [Path(__file__), FILTER_RUNNER,
             ROOT/'scripts/benchmark_xd_real.py',
             ROOT/'results/e13_assets/cubutterfly_real_fft.py',
             ROOT/'paper/tools/run_application_batch_20261012.py',
             ROOT/'scripts/run_isolated_gpu.py']
    registry = Path(env['CUBUTTERFLY_REGISTRY'])
    return env, dict(library=str(LIB), library_sha256=digest(LIB), gpu_uuid=uuid,
        calibration_path=str(calibration_path), calibration_sha256=digest(calibration_path),
        registry_path=str(registry), registry_sha256=digest(registry),
        source_hashes={str(p): digest(p) for p in paths},
        effective_environment=env_identity, xd_tune_max_bytes=8*1024**3,
        calibration_usage='rank1 registry where compatible; xD whole-plan MEASURE cache replay',
        semantics=dict(storage='FP32 / compact complex64', rank=[2, 3],
                       placement='out-of-place', layout='contiguous', norm='forward'),
        tolerance=dict(atol=None, rtol=None, accepted=None, exploratory=True))


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        verify(manifest_path)
        return manifest_path
    _, provenance = identity()
    records = []
    for index, (shape, batch) in enumerate(CASES):
        path = output/f'fft_{"x".join(map(str,shape))}_b{batch}.npz'
        if path.exists():
            raise RuntimeError(f'refusing to overwrite unconfirmed artifact: {path}')
        source = seeded_rng(20261012+index).standard_normal((batch, *shape)).astype(np.float32)
        dims = tuple(range(1, len(shape)+1))
        reference_half = fft.rfftn(source.astype(np.float64), axes=dims, norm='forward')
        reference_inverse = fft.irfftn(reference_half.astype(np.complex64).astype(np.complex128),
                                      s=shape, axes=dims, norm='forward')
        np.savez(path, input=source, forward_reference=reference_half,
                 inverse_reference=reference_inverse)
        records.append(dict(shape=list(shape), batch=batch, artifact=str(path.resolve()), sha256=digest(path)))
    input_manifest = ROOT/'results/paper_targeted_20261012/application_batch/fft_filter/manifest.json'
    original = json.loads(input_manifest.read_text())
    filters = []
    for contract in original['contracts']:
        if contract['image_shape'][-1] not in (1024, 2048) or contract.get('batch',1) not in (1,4) or contract['filter_shape'][-1] != 31:
            continue
        path = input_manifest.parent/contract['artifact']
        if digest(path) != contract['sha256']:
            raise RuntimeError('filter input artifact mismatch')
        filters.append(dict(contract=contract, artifact=str(path.resolve()), sha256=digest(path)))
    if len(filters) != 4:
        raise RuntimeError('expected four original independent-batch filter payloads')
    manifest = dict(schema='xd-application-breakdown-v1', identity=provenance,
        direct=records, filters=filters, filter_manifest=str(input_manifest),
        filter_manifest_sha256=digest(input_manifest),
        timing_contract='eager GPU resident; plan/first use excluded; unsynchronized inner calls; segment events diagnostic only')
    save(manifest_path, manifest)
    return manifest_path


def verify(path):
    manifest = json.loads(path.read_text())
    _, current = identity()
    if manifest['identity'] != current:
        raise RuntimeError('identity changed; select a new output cohort')
    for row in manifest['direct']+manifest['filters']:
        if digest(row['artifact']) != row['sha256']:
            raise RuntimeError('input artifact hash mismatch')
    if digest(manifest['filter_manifest']) != manifest['filter_manifest_sha256']:
        raise RuntimeError('original filter manifest changed')
    return manifest


def require_plans(adapters, required):
    plans = [plan for adapter in adapters for plan in adapter.plan_metadata()]
    if required and any(plan['selection_source'] != 'user_cache' for plan in plans):
        raise RuntimeError('DEFAULT did not replay every tuned plan')
    return plans


def timing(call, torch, device, warmup, repeats, samples):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize(device)
    gpu, wall = [], []
    for _ in range(samples):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        begin.record()
        for _ in range(repeats):
            call()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter()-start)*1000/repeats)
        gpu.append(begin.elapsed_time(end)/repeats)
    return dict(cuda_event_ms=gpu, wall_ms=wall,
                median_cuda_event_ms=statistics.median(gpu), median_wall_ms=statistics.median(wall))


def filter_call(image, kernel, shape, forward, filter_forward, inverse, torch, events=None):
    def mark(index):
        if events is not None:
            events[index].record()
    mark(0)
    x = torch.nn.functional.pad(image, (0,shape[1]-image.shape[-1],0,shape[0]-image.shape[-2]))
    h = torch.nn.functional.pad(kernel, (0,shape[1]-kernel.shape[-1],0,shape[0]-kernel.shape[-2]))
    mark(1)
    xf = forward(x)
    mark(2)
    hf = filter_forward(h)
    mark(3)
    product = xf*hf
    mark(4)
    product *= np.prod(shape)
    mark(5)
    full = inverse(product)
    mark(6)
    output = filter_api.same_crop(full, image.shape, kernel.shape).clone()
    mark(7)
    return output


def segment_timing(call, torch, samples):
    values = {stage: [] for stage in STAGES}
    for _ in range(samples):
        events = [torch.cuda.Event(enable_timing=True) for _ in range(8)]
        call(events)
        events[-1].synchronize()
        for index, stage in enumerate(STAGES):
            values[stage].append(events[index].elapsed_time(events[index+1]))
    return dict(stage_cuda_event_ms=values,
                stage_median_ms={key: statistics.median(value) for key,value in values.items()},
                warning='instrumented diagnostic only; do not sum or substitute for uninstrumented complete-call timing')


def gpu_run(args):
    manifest = verify(args.manifest)
    import torch
    import cupy as cp
    from cupyx.scipy.signal import fftconvolve
    from cupyx.scipy.fft import get_fft_plan
    sys.path.insert(0, str(ROOT/'results/e13_assets'))
    from cubutterfly_real_fft import RealFFTAdapter
    # Reuse the existing direct-xD benchmark's device identity, not its fixed
    # tolerance or graph-only timing contract (this diagnoses application calls).
    benchmark = load_module('xd_real_benchmark', ROOT/'scripts/benchmark_xd_real.py')
    device = torch.device('cuda:0')
    gpu = benchmark._gpu_identity()
    if str(gpu.get('uuid','')).removeprefix('GPU-') != manifest['identity']['gpu_uuid'].removeprefix('GPU-'):
        raise RuntimeError('wrong physical GPU for calibrated cohort')
    report = dict(manifest_sha256=digest(args.manifest), identity=manifest['identity'], gpu=gpu,
                  torch=torch.__version__, cupy=cp.__version__, plan_only=args.plan_only,
                  timing_contract=manifest['timing_contract'], direct=[], filters=[])
    output = args.result if args.result.is_absolute() else Path(os.environ.get('XD_REPORT_DIR', '.'))/args.result
    output = output.resolve()
    if output.exists():
        raise RuntimeError('refusing to overwrite prior result')
    output.parent.mkdir(parents=True, exist_ok=True)
    def record(collection, row):
        report[collection].append(row)
        save(output, report)
    for case in manifest['direct']:
        shape, batch = tuple(case['shape']), case['batch']
        data = np.load(case['artifact'])
        x = torch.as_tensor(data['input'], device=device)
        half = torch.as_tensor(data['forward_reference'].astype(np.complex64), device=device)
        adapter = RealFFTAdapter(LIB, shape, batch, device, native_real=True, reuse_buffers=False)
        plans = require_plans([adapter], args.require_cache)
        dims = tuple(range(1,len(shape)+1))
        stream = cp.cuda.ExternalStream(torch.cuda.current_stream(device).cuda_stream)
        with cp.cuda.Device(0), stream:
            cx, ch = cp.from_dlpack(x), cp.from_dlpack(half)
            forward_plan = get_fft_plan(cx, axes=dims, value_type='R2C')
            inverse_plan = get_fft_plan(ch, shape=shape, axes=dims, value_type='C2R')
            def cf():
                with forward_plan:
                    return cp.fft.rfftn(cx, axes=dims, norm='forward')
            def ci():
                with inverse_plan:
                    return cp.fft.irfftn(ch, s=shape, axes=dims, norm='forward')
            methods = [('cubutterfly', lambda:adapter.rfftn(x), lambda:adapter.irfftn(half)),
                       ('torch_cufft', lambda:torch.fft.rfftn(x,dim=dims,norm='forward'),
                        lambda:torch.fft.irfftn(half,s=shape,dim=dims,norm='forward')),
                       ('cupy_cufft_explicit_plan', cf, ci)]
            rows = []
            for name, forward, inverse in methods:
                row = dict(method=name, directions={})
                for direction, callback, ref in [('R2C', forward, data['forward_reference']),
                                                  ('C2R', inverse, data['inverse_reference'])]:
                    actual = callback()
                    torch.cuda.synchronize(device)
                    array = cp.asnumpy(actual) if isinstance(actual,cp.ndarray) else actual.cpu().numpy()
                    error = distribution(array, ref)
                    if not error['finite']:
                        raise RuntimeError('nonfinite direct FFT output')
                    row['directions'][direction] = dict(error=error, accepted=None,
                        timing=None if args.plan_only else timing(callback,torch,device,args.warmup,args.repeats,args.samples))
                rows.append(row)
            record('direct',dict(shape=list(shape),batch=batch,plans=plans,methods=rows,
                                 adapter=adapter.plan_metadata(), input_sha256=case['sha256']))
        adapter.close()
    for item in manifest['filters']:
        contract = item['contract']
        data = np.load(item['artifact'])
        image = torch.as_tensor(data['image'],device=device)
        kernel = torch.as_tensor(data['kernel'],device=device)
        shape = filter_api.padding_shape(image.shape[-2:],kernel.shape)
        batch = contract.get('batch',1)
        adapter = RealFFTAdapter(LIB,shape,batch,device,native_real=True,reuse_buffers=False)
        h_adapter = adapter if batch==1 else RealFFTAdapter(LIB,shape,1,device,native_real=True,reuse_buffers=False)
        plans = require_plans([adapter,h_adapter],args.require_cache)
        own = lambda events=None: filter_call(image,kernel,shape,adapter.rfftn,h_adapter.rfftn,adapter.irfftn,torch,events)
        tf = lambda x:torch.fft.rfftn(x,dim=(-2,-1),norm='forward')
        ti = lambda x:torch.fft.irfftn(x,s=shape,dim=(-2,-1),norm='forward')
        same = lambda events=None:filter_call(image,kernel,shape,tf,tf,ti,torch,events)
        def original_same():
            x=torch.nn.functional.pad(image,(0,shape[1]-image.shape[-1],0,shape[0]-image.shape[-2]))
            h=torch.nn.functional.pad(kernel,(0,shape[1]-kernel.shape[-1],0,shape[0]-kernel.shape[-2]))
            full=torch.fft.irfftn(torch.fft.rfftn(x,dim=(-2,-1))*torch.fft.rfftn(h,dim=(-2,-1)),s=shape,dim=(-2,-1))
            return filter_api.same_crop(full,image.shape,kernel.shape).clone()
        stream = cp.cuda.ExternalStream(torch.cuda.current_stream(device).cuda_stream)
        with cp.cuda.Device(0),stream:
            cx,ch = cp.from_dlpack(image),cp.from_dlpack(kernel)
            native = lambda:fftconvolve(cx,ch if batch==1 else ch[None],mode='same',axes=(-2,-1))
            rows=[]
            for name,callback in [('cubutterfly',own),('torch_cufft_same_padding_matched_normalization',same),
                                  ('torch_cufft_same_padding_original_v2_diagnostic',original_same),
                                  ('cupy_fftconvolve_native_padding',native)]:
                actual=callback()
                torch.cuda.synchronize(device)
                array=cp.asnumpy(actual) if isinstance(actual,cp.ndarray) else actual.cpu().numpy()
                error=distribution(array,data['reference'])
                if not error['finite']:
                    raise RuntimeError('nonfinite filter output')
                rows.append(dict(method=name,error=error,accepted=None,
                    timing=None if args.plan_only else timing(callback,torch,device,args.warmup,args.repeats,args.samples),
                    segments=None if args.plan_only or callback not in (own,same) else segment_timing(callback,torch,args.samples)))
            record('filters',dict(contract=contract,padding=list(shape),
                cupy_padding=[fft.next_fast_len(n+k-1,real=True) for n,k in zip(image.shape[-2:],kernel.shape)],
                plans=plans,methods=rows,filter_spectrum_cached=False,input_sha256=item['sha256']))
        adapter.close()
        if h_adapter is not adapter:
            h_adapter.close()
    report['status']='complete'
    save(output,report)


def launch(args):
    if not args.until:
        raise ValueError('--launch requires a finite --until deadline')
    manifest_path=prepare(args.output)
    manifest=verify(manifest_path)
    env,_=identity()
    spec_hash=fingerprint(manifest)
    seed=None
    for phase in ('measure','default'):
        report_dir=args.output/f'{phase}_guard'
        child=[PYTHON,str(Path(__file__).resolve()),'--run','--manifest',str(manifest_path.resolve()),'--result','result.json',
               '--warmup',str(args.warmup),'--repeats',str(args.repeats),'--samples',str(args.samples)]
        child+=['--plan-only'] if phase=='measure' else ['--require-cache']
        record=dict(policy=phase,report_dir=str(report_dir.resolve()),command=child)
        if (report_dir/'release').exists():
            seed=reuse_release(record,spec_hash,seed)
            continue
        wrapper=[PYTHON,str(ROOT/'paper/tools/run_application_batch_20261012.py'),'--cache-child','--policy',phase]
        if seed:
            wrapper+=['--cache-seed',str(seed)]
        command=[PYTHON,str(ROOT/'scripts/run_isolated_gpu.py'),'--gpu-uuid',manifest['identity']['gpu_uuid'],
                 '--report-dir',str(report_dir.resolve()),'--until',args.until,'--idle-seconds','3','--']+wrapper+['--']+child
        with (args.output/'queue.log').open('a') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{phase} guard failed or expired; prior artifacts preserved')
        seed=accept_release(record,spec_hash,seed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--run',action='store_true')
    mode.add_argument('--launch',action='store_true',help='explicit GPU guard launch, never implied by prepare')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--result',type=Path,default=Path('result.json'))
    parser.add_argument('--until')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--require-cache',action='store_true')
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--samples',type=int,default=5)
    args=parser.parse_args()
    if min(args.repeats,args.samples)<1 or args.warmup<0:
        parser.error('invalid timing counts')
    if args.prepare:
        print(prepare(args.output))
    elif args.launch:
        launch(args)
    else:
        if args.manifest is None:
            parser.error('--run requires --manifest')
        gpu_run(args)


if __name__=='__main__':
    main()
