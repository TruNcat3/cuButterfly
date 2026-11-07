#!/usr/bin/env python3
"""Finite native-FP16 factor correctness study; prepare is CPU/compiler only.

CPU reference is the declared storage-rounded scalar reference, not cuFFTDx's
instruction-level oracle. Errors are reported without thresholds. Schedule
invariance, impulse spectra and guard checks are exact acceptance conditions.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
from measure_fp16_local_core import sha256, write_result
from research_compile_requests import compile_request


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_result(path, value)


def matrix():
    tasks = []
    for logn, factors in [(12, [6, 6]), (15, [7, 8]), (18, [6, 6, 6])]:
        schedules = [('d0', 0, False, 1, False), ('d1', 1, False, 1, False),
                     ('d2', 2, False, 1, False), ('static', 1, True, 1, False)]
        if logn == 18:
            schedules += [('slices', 1, False, 2, False), ('overlap', 1, False, 2, True)]
        for name, depth, static, slices, overlap in schedules:
            mapping = dict(kind='butterfly', backend='factor-streamed', fft_core='cufftdx-block',
                           compute_unit='auto', shared_layout='writer-aligned', cross_twiddle='recurrence',
                           stage_partition=[logn], factor_partition=factors, factor_ept=8, factor_columns=8,
                           data_tiles_per_cta=5, prefetch_depth=depth, factor_slices=slices,
                           factor_overlap=overlap, factor_io_policies=['static-unrolled' if static else 'dynamic']*len(factors))
            for inverse in (False, True):
                for data in ('random', 'impulse'):
                    tasks.append(dict(id=f'n{logn}-{name}-i{int(inverse)}-{data}', logN=logn,
                                      n=1 << logn, batch=3, inverse=inverse, in_place=inverse,
                                      element_stride=2, batch_stride=2*(1 << logn)+16,
                                      schedule=name, data=data, mapping=mapping))
    return tasks


def module_point(task):
    return compile_request(dict(operator='fft', precision='fp16', accumulation='native',
                                logN=task['logN'], batch=3, inverse=task['inverse'],
                                normalize_inverse=False, mapping_json=task['mapping']), 'research')


def reference(source, inverse):
    from low_fft_cpu_reference import transform
    return transform(source[..., 0].astype('float32') + 1j*source[..., 1].astype('float32'),
                     dict(operator=0, storage=4, compute=0, shape=[source.shape[1]],
                          batch=3, inverse=inverse, normalize_inverse=False), [0])


def prepare(args):
    import numpy as np
    from compile_module import compile_mapping
    out = args.output
    if (out / 'manifest.json').exists():
        raise FileExistsError('manifest exists; execute it or use a fresh preparation directory')
    save(out / 'status.json', dict(status='preparing', compile_ready=False))
    modules, arrays = {}, {}
    tasks = matrix()
    for t in tasks:
        p = module_point(t)
        key = json.dumps(p, sort_keys=True)
        if key not in modules:
            module = compile_mapping(p, root=ROOT, mathdx=ROOT/'external/mathdx/nvidia/mathdx',
                                     nvcc=Path('/usr/local/cuda/bin/nvcc'), sm=args.sm, cache=out/'modules')
            modules[key] = dict(path=str(module), sha256=sha256(module), point=p)
            save(out/'status.json', dict(status='preparing', compile_ready=False, modules=len(modules)))
        stem = f"n{t['logN']}-i{int(t['inverse'])}-{t['data']}"
        if stem not in arrays:
            n = t['n']
            if t['data'] == 'impulse':
                source = np.zeros((3, n, 2), np.float16)
                source[:, 0, 0] = [1, 2, 3]
            else:
                source = np.random.default_rng(20261105+n).uniform(-.03125, .03125, (3, n, 2)).astype(np.float16)
            ref = reference(source, t['inverse'])
            if not np.isfinite(ref).all():
                raise ValueError('nonfinite CPU reference')
            inp, expected = out/'arrays'/(stem+'-input.npy'), out/'arrays'/(stem+'-reference.npy')
            inp.parent.mkdir(exist_ok=True)
            np.save(inp, source, allow_pickle=False)
            np.save(expected, ref, allow_pickle=False)
            arrays[stem] = dict(input=str(inp), input_sha256=sha256(inp), reference=str(expected),
                                reference_sha256=sha256(expected))
        t.update(arrays[stem], comparison_group=stem)
    # Verify that the public preparation path refuses default normalized inverse.
    bad = dict(operator='fft', precision='fp16', accumulation='native', logN=12,
               inverse=True, normalize_inverse=True, mapping_json=tasks[0]['mapping'])
    try:
        compile_request(bad, 'research')
    except ValueError:
        pass
    else:
        raise RuntimeError('normalized inverse preparation unexpectedly accepted')
    manifest = dict(schema='fp16-factor-correctness-v1', tasks=tasks, modules=list(modules.values()),
                    library=str(args.library), library_sha256=sha256(args.library), gpu_uuid=args.gpu_uuid,
                    script_sha256=sha256(__file__), cache=str(out/'modules'),
                    reference_contract='storage-rounded scalar FFT; errors report-only; no np.fft oracle')
    save(out/'manifest.json', manifest)
    save(out/'status.json', dict(status='ready', compile_ready=True, tasks=len(tasks), modules=len(modules)))


def execute(args):
    import numpy as np
    out = args.output
    manifest_path = out/'manifest.json'
    m = json.loads(manifest_path.read_text())
    identity = sha256(manifest_path)
    if sha256(__file__) != m['script_sha256'] or sha256(m['library']) != m['library_sha256']:
        raise ValueError('script/library changed after preparation')
    for module in m['modules']:
        if sha256(module['path']) != module['sha256']: raise ValueError('module changed')
    os.environ.update(CUDA_VISIBLE_DEVICES=m['gpu_uuid'], CUBUTTERFLY_COMPILE_MODE='research',
                      CUBUTTERFLY_JIT_CACHE_ONLY='1', CUBUTTERFLY_JIT_CACHE=m['cache'],
                      CUBUTTERFLY_TEMPLATE_ROOT=str(ROOT))
    from run_semantic_campaign import exclusive_gpu
    exclusive_gpu(m['gpu_uuid'])
    import torch
    import cubutterfly as cb
    library = cb.Library(m['library'])
    state_path = out/'progress.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(manifest_sha256=identity, rows=[])
    if state['manifest_sha256'] != identity: raise ValueError('resume manifest changed')
    done = {r['id']: r for r in state['rows']}
    for row in state['rows']:
        if sha256(row['output']) != row['output_sha256']: raise ValueError('resume output changed')
    for t in m['tasks']:
        if t['id'] in done: continue
        save(out/'status.json', dict(status='running', task=t['id'], completed=len(done)))
        for name in ('input', 'reference'):
            if sha256(t[name]) != t[name+'_sha256']: raise ValueError(name+' changed')
        exclusive_gpu(m['gpu_uuid'])
        source = np.load(t['input'], allow_pickle=False)
        n, distance = t['n'], t['batch_stride']
        extent = 3*distance
        indices = 32 + np.arange(3)[:,None]*distance + np.arange(n)[None,:]*2
        full = np.full((extent+64,2), 0x3555, np.uint16)
        full[indices] = source.view(np.uint16)
        x = torch.from_numpy(full.copy()).cuda()
        initial_y = np.full_like(full, 0x3555)
        y = x if t['in_place'] else torch.from_numpy(initial_y).cuda()
        stream = torch.cuda.Stream()
        transform = cb.Transform(storage=cb.DataType.COMPLEX_FP16, extents=[n], batch=3,
                                 inverse=t['inverse'], normalize_inverse=False, in_place=t['in_place'],
                                 input_strides=[2], output_strides=[2], input_batch_stride=distance,
                                 output_batch_stride=distance)
        (out/'runs'/t['id']).mkdir(parents=True, exist_ok=True)
        plan = cb.Plan(library, transform, cb.PlanOptions(policy=cb.AlgorithmPolicy.EXPLICIT,
                       mapping=json.dumps(t['mapping'])), stream=stream.cuda_stream,
                       cache_path=str(out/'runs'/t['id']/'plan-cache.json'))
        try:
            workspace = torch.empty(max(1, plan.workspace_size()), dtype=torch.uint8, device='cuda')
            if plan.workspace_size(): plan.set_workspace(workspace, plan.workspace_size())
            # Device copies use the default stream; explicitly finish before the plan stream.
            torch.cuda.synchronize()
            plan.execute_steady(x.data_ptr()+32*4, y.data_ptr()+32*4)
            stream.synchronize()
            actual_full, input_after = y.cpu().numpy(), x.cpu().numpy()
            mask = np.ones(full.shape[0], bool); mask[indices.reshape(-1)] = False
            expected_y = full if t['in_place'] else initial_y
            if not np.array_equal(actual_full[mask], expected_y[mask]): raise RuntimeError('output guards/gaps changed')
            if not t['in_place'] and not np.array_equal(input_after, full): raise RuntimeError('input changed')
            actual = actual_full[indices].copy().view(np.float16)
            if not np.isfinite(actual).all(): raise RuntimeError('nonfinite output')
            if t['data']=='impulse':
                expected = np.zeros_like(actual); expected[:,:,0]=np.arange(1,4)[:,None]
                if not np.array_equal(actual, expected):
                    raise RuntimeError('index-zero batch-distinct impulse spectrum mismatch')
            baseline = out/'baselines'/(t['comparison_group']+'.npy')
            baseline.parent.mkdir(exist_ok=True)
            if baseline.exists():
                if not np.array_equal(actual.view(np.uint16), np.load(baseline).view(np.uint16)):
                    raise RuntimeError('schedule changed output bytes for identical factor arithmetic')
            else: np.save(baseline, actual, allow_pickle=False)
            expected = np.load(t['reference'], allow_pickle=False)
            values = actual[...,0].astype(np.float64)+1j*actual[...,1].astype(np.float64)
            error = np.abs(values-expected)
            result_path = out/'runs'/t['id']/'output.npy'; result_path.parent.mkdir(parents=True,exist_ok=True)
            np.save(result_path, actual, allow_pickle=False)
            row=dict(id=t['id'], output=str(result_path), output_sha256=sha256(result_path),
                     max_abs=float(error.max()), rel_l2=float(np.linalg.norm(error)/max(np.linalg.norm(expected),1e-300)),
                     finite=True, guards=True, schedule_bytes_equal=True,
                     reference_verdict='error-report-only', threshold=None, mapping=json.loads(plan.mapping_json()))
            state['rows'].append(row); done[t['id']]=row
            save(state_path,state)
        finally: plan.close()
        exclusive_gpu(m['gpu_uuid'])
    # Actual public constructor must also reject normalized inverse.
    bad = cb.Transform(storage=cb.DataType.COMPLEX_FP16, extents=[4096], batch=3,
                       inverse=True, normalize_inverse=True)
    try:
        rejected = cb.Plan(library,bad,cb.PlanOptions(policy=cb.AlgorithmPolicy.EXPLICIT,
                           mapping=json.dumps(m['tasks'][0]['mapping'])))
    except Exception as error:
        if 'normalized inverse' not in str(error): raise
    else:
        rejected.close(); raise RuntimeError('public normalized inverse unexpectedly accepted')
    save(out/'status.json',dict(status='complete',completed=len(done),accuracy='error-report-only',exact_contract_checks=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare-only',action='store_true');g.add_argument('--execute',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--library',type=Path,default=ROOT/'results/a100_fp16_factor_20261105/build/libcuntt.so')
    p.add_argument('--gpu-uuid',default='GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02')
    p.add_argument('--sm',type=int,default=80)
    a=p.parse_args();a.output=a.output.resolve();a.library=a.library.resolve();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if a.prepare_only: prepare(a)
            else: execute(a)
        except BaseException as error:
            save(a.output/'error.json',dict(error=repr(error),updated_unix=time.time()))
            save(a.output/'status.json',dict(status='failed',error=repr(error)))
            raise
    return 0

if __name__=='__main__': raise SystemExit(main())
