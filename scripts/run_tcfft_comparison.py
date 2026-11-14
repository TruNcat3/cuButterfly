#!/usr/bin/env python3
"""Finite full-call FP16 comparison. --prepare is CPU-only; --run needs exclusive GPU.

Config: public_manifest, tcfft_library, cufft_library, gpu_uuid, optionally library,
registry, profile_root, module_cache, template_root, mathdx_root. Paths are absolute.
The public manifest supplies the unchanged frozen inputs; manual mappings are unused.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import tcfft_baseline as tcfft

CASES = [(4096,256),(4096,2048),(8192,256),(16384,64),(16384,512),
         (32768,32),(32768,256),(262144,4),(262144,32)]
BACKENDS = ('default', 'cufft', 'tcfft')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def record_byte_mismatch(actual, expected, directory, backend, stage):
    """Save exact failure evidence; no tolerance or numeric-policy changes."""
    import numpy as np
    actual=np.asarray(actual).view(np.uint8).reshape(-1).copy()
    expected=np.asarray(expected).view(np.uint8).reshape(-1).copy()
    if actual.shape!=expected.shape: raise ValueError('diagnostic byte shape mismatch')
    positions=np.flatnonzero(actual!=expected)
    first=positions[:64]
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    stem=backend+'-'+stage+'-mismatch'
    actual_path=directory/(stem+'-actual.npy');expected_path=directory/(stem+'-expected.npy')
    np.save(actual_path,actual,allow_pickle=False);np.save(expected_path,expected,allow_pickle=False)
    report=dict(stage=stage,different_bytes=int(positions.size),total_bytes=int(actual.size),
        first_positions=first.tolist(),actual_bytes=actual[first].tolist(),expected_bytes=expected[first].tolist(),
        actual_file=str(actual_path.resolve()),expected_file=str(expected_path.resolve()))
    write(directory/(stem+'.json'),report)
    return report


def prepare_rounded_reference(e, output):
    import numpy as np
    from low_fft_cpu_reference import transform
    data=np.load(e['input_file'],allow_pickle=False)
    pairs=data.astype(np.float32).reshape(e['batch'],e['n'],2)
    source=pairs[...,0].astype(np.float64)+1j*pairs[...,1].astype(np.float64)
    reference=transform(source,dict(compute=0,storage=4,shape=[e['n']],batch=e['batch'],
        inverse=False,normalize_inverse=False),[0])
    path=output/'prepared'/(e['id']+'-rounded-reference.npy')
    path.parent.mkdir(parents=True,exist_ok=True)
    np.save(path,reference,allow_pickle=False)
    e.update(rounded_reference_file=str(path.resolve()),rounded_reference_sha256=sha(path))


def prepare(config, output, reference_manifest=None):
    c = read(config); public = read(c['public_manifest'])
    reused = read(reference_manifest) if reference_manifest else None
    if reused:
        for source in (ROOT/'scripts/low_fft_cpu_reference.py',ROOT/'benchmark/review_fft_l_cpu_reference.py'):
            relative=source.relative_to(ROOT).as_posix()
            digests=[digest for path,digest in reused['bindings'].items() if path.endswith('/'+relative)]
            if digests != [sha(source)]:
                raise ValueError('cached reference implementation changed: '+str(source))
    cases = []
    for n,b in CASES:
        e = dict(next(e for e in public['cases'] if (e['n'],e['batch']) == (n,b)))
        if (e['storage'],e['compute'],e['inverse'],e['normalize_inverse']) != (4,0,False,False):
            raise ValueError('frozen FP16 request semantics changed')
        e['tcfft_support'] = tcfft.support(n,b)
        cases.append(e)
    runtime = {k: c.get(k,public.get(k)) for k in ('library','registry','profile_root',
        'module_cache','template_root','mathdx_root')}
    bindings = {}
    paths = [config,c['public_manifest'],c['tcfft_library'],c['cufft_library'],runtime['library'],
             __file__,ROOT/'scripts/tcfft_baseline.py',ROOT/'scripts/cufft_low_precision_baseline.py',
             ROOT/'scripts/run_semantic_campaign.py',ROOT/'scripts/low_fft_cpu_reference.py',
             ROOT/'benchmark/review_fft_l_cpu_reference.py']
    for e in cases:
        for kind in ('input','reference'):
            p=e[kind+'_file']
            if sha(p)!=e[kind+'_sha256']: raise ValueError('frozen input/reference changed: '+p)
            paths.append(p)
        if e['tcfft_support']['supported']:
            if reused:
                old=next(item for item in reused['cases'] if item['id']==e['id'])
                for key in ('n','batch','storage','compute','inverse','normalize_inverse','input_sha256'):
                    if old[key]!=e[key]: raise ValueError('cached reference semantics changed: '+key)
                path=old['rounded_reference_file'];digest=old['rounded_reference_sha256']
                if sha(path)!=digest: raise ValueError('cached reference changed: '+path)
                e.update(rounded_reference_file=path,rounded_reference_sha256=digest)
            else:
                prepare_rounded_reference(e, output)
            paths.append(e['rounded_reference_file'])
        else:
            e['comparison_status']='unsupported-tcfft; metadata only, no GPU execution'
    for key in ('registry','profile_root'):
        if runtime[key]:
            p=Path(runtime[key]); paths.extend([p] if p.is_file() else sorted(p.rglob('*.json')))
    for p in paths: bindings[str(Path(p).resolve())]=sha(p)
    return dict(schema='tcfft-full-call-comparison-v2',scope='automatic-library-comparison',
        gpu_execution=False,config=c,runtime=runtime,cases=cases,bindings=bindings,
        precision=dict(storage='fp16',public_compute='native-fp16',tcfft_accumulator='fp16',
                       cufft_compute='vendor-native, unspecified internal arithmetic'),
        timing='shared time_execute: 20 full calls/graph, 7 replays, 10 warmup',
        numeric_policy='error report only; no floating-point error threshold',
        tcfft_scope='device natural-input reorder + upstream FFT + output copy')


def verify_capture(torch, stream, execute, poison, verify):
    """Release the captured raw addresses before returning to plan ownership."""
    graph=torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph,stream=stream): execute()
        poison();graph.replay();stream.synchronize();verify()
    finally:
        # Also covers a verification exception, before the caller closes plan.
        graph=None


def validate(m):
    for p,digest in m['bindings'].items():
        if sha(p)!=digest: raise ValueError('binding changed: '+p)


def synchronize_plan_setup(torch, enabled):
    """Diagnostic setup dependency only; never inside capture or timing."""
    if enabled:
        torch.cuda.synchronize()


def run(m, output, case_id, *, backends=BACKENDS, diagnostic_only=False, synchronize_setup=False):
    if not backends or len(set(backends)) != len(backends) or any(b not in BACKENDS for b in backends):
        raise ValueError('backends must be a nonempty unique subset of '+str(BACKENDS))
    validate(m)
    c=m['config']; runtime=m['runtime']; e=next(e for e in m['cases'] if e['id']==case_id)
    if not e['tcfft_support']['supported']:
        write(output/case_id/'comparison.json',dict(id=case_id,status='unsupported',
            gpu_execution=False,capability=e['tcfft_support']))
        return
    if 'rounded_reference_file' not in e:
        raise ValueError('CPU reference not prepared; use a new --prepare output directory')
    cache=output/case_id/'cache'; cache.mkdir(parents=True,exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES']=c['gpu_uuid']
    os.environ.setdefault('CUBUTTERFLY_COMPILE_MODE','research')
    for key,env in [('registry','CUBUTTERFLY_REGISTRY'),('profile_root','CUBUTTERFLY_PROFILE_ROOT'),
        ('module_cache','CUBUTTERFLY_JIT_CACHE'),('template_root','CUBUTTERFLY_TEMPLATE_ROOT'),
        ('mathdx_root','CUBUTTERFLY_MATHDX_ROOT')]:
        if runtime[key]: os.environ[env]=runtime[key]
    os.environ['CUBUTTERFLY_PLAN_CACHE']=str(cache/'plan.json')
    os.environ['CUBUTTERFLY_JIT_PYTHON']=sys.executable
    sys.path[:0]=[str(ROOT/'python'),str(ROOT)]
    from run_semantic_campaign import exclusive_gpu,time_execute
    from benchmark.review_fft_l_cpu_reference import metrics
    import numpy as np
    data=np.load(e['input_file'],allow_pickle=False)
    reference=np.load(e['rounded_reference_file'],allow_pickle=False)
    frozen_reference=np.load(e['reference_file'],allow_pickle=False)
    exclusive_gpu(c['gpu_uuid'])
    import torch
    import cubutterfly as cb
    import cufft_low_precision_baseline as cufft
    rows=[]; actuals={}
    stream=torch.cuda.Stream()
    for backend in backends:
        row=dict(backend=backend,id=case_id,status='failed',threshold=None,
            input_sha256=e['input_sha256'],scope=m['scope'],timing_scope='complete operator call',
            verification_stage='initial-eager')
        rows.append(row); plan=None
        row['timing_performed']=False
        row['synchronize_setup']=synchronize_setup
        row['setup_synchronizations']=0
        if backend=='tcfft' and not e['tcfft_support']['supported']:
            row.update(status='unsupported',capability=e['tcfft_support'])
            write(output/case_id/(backend+'.json'),row)
            continue
        try:
            with torch.cuda.stream(stream):
                raw=data.view(np.uint8).reshape(-1)
                initial=np.full(raw.size+128,0xA5,np.uint8); initial[64:-64]=raw
                full_x=torch.from_numpy(initial).to('cuda'); full_y=torch.full_like(full_x,0xA5)
                x=full_x[64:-64]; y=full_y[64:-64]
                def create_execution():
                    workspace=None
                    if backend=='default':
                        lib=cb.Library(runtime['library'])
                        plan=cb.Plan(lib,cb.Transform(storage=4,compute=0,extents=[e['n']],batch=e['batch'],
                            inverse=False,normalize_inverse=False),cb.PlanOptions(policy=cb.AlgorithmPolicy.DEFAULT),
                            stream=stream.cuda_stream,cache_path=str(cache/'plan.json'))
                        workspace=torch.empty(max(1,plan.workspace_size()),dtype=torch.uint8,device='cuda')
                        if plan.workspace_size(): plan.set_workspace(workspace,plan.workspace_size())
                        selected=ctypes.c_int();lib._call('PlanGetSelectionSource',plan.native_handle,ctypes.byref(selected))
                        row.update(actual_mapping=json.loads(plan.mapping_json()),selection_source=selected.value,
                                   selection_reason=plan.selection_reason(),descriptor=json.loads(plan.descriptor_json()))
                        execute=lambda:plan.execute_steady(x,y)
                    else:
                        library=c[backend+'_library']
                        plan=(tcfft.Plan(library,e['n'],e['batch']) if backend=='tcfft' else
                              cufft.Plan(library,[e['n']],e['batch'],stream=stream.cuda_stream,precision='fp16'))
                        row['identity']=tcfft.support(e['n'],e['batch']) if backend=='tcfft' else plan.info
                        execute=lambda:plan.execute(x.data_ptr(),y.data_ptr(),stream=stream.cuda_stream)
                    synchronize_plan_setup(torch,synchronize_setup)
                    if synchronize_setup: row['setup_synchronizations']+=1
                    return plan,execute,workspace
                plan,execute,workspace=create_execution()
                execute();stream.synchronize(); direct=y.clone();stream.synchronize()
                def verify():
                    if not torch.equal(y,direct):
                        report=record_byte_mismatch(y.cpu().numpy(),direct.cpu().numpy(),output/case_id,
                            backend,row['verification_stage'])
                        row['byte_mismatch']=report
                        raise ValueError('direct/graph mismatch at '+row['verification_stage']+
                                         ': '+str(report['different_bytes'])+' bytes')
                    if not np.array_equal(full_x.cpu().numpy(),initial): raise ValueError('input/guards modified')
                    if not bool(torch.all(full_y[:64]==0xA5)) or not bool(torch.all(full_y[-64:]==0xA5)):
                        raise ValueError('output guard modified')
                row['verification_stage']='output-lifetime-before-timing'
                plan.close();plan=None
                stream.synchronize();verify()
                row['output_lifetime_before_timing']=True
                row['verification_stage']='recreated-eager'
                plan,execute,workspace=create_execution()
                execute();stream.synchronize();verify()
                # Capture/replay, poison and source immutability gate actual timing.
                row['verification_stage']='first-graph'
                verify_capture(torch,stream,execute,lambda:y.fill_(0xA5),verify)
                if not bool(torch.isfinite(y.view(torch.float16)).all()):
                    raise ValueError('nonfinite output before timing; no precision substitution')
                if not diagnostic_only:
                    row['verification_stage']='timing-replay'
                    row['graph_timing']=time_execute(torch,stream,execute,lambda:None,False,verify,lambda:y.fill_(0xA5))
                    row['timing_performed']=True
                verify(); saved=y.cpu().numpy().copy(); plan.close();plan=None
                # Caller-owned destination must survive destruction of plan scratch.
                stream.synchronize()
                row['verification_stage']=('output-lifetime-after-execution' if diagnostic_only
                                           else 'output-lifetime-after-timing')
                if not np.array_equal(saved,y.cpu().numpy()):
                    row['byte_mismatch']=record_byte_mismatch(y.cpu().numpy(),saved,output/case_id,
                        backend,row['verification_stage'])
                    raise ValueError('output lifetime failed at '+row['verification_stage'])
                values=saved.view(np.float16).astype(np.float64).reshape(e['batch'],e['n'],2)
                actual=values[...,0]+1j*values[...,1]; actuals[backend]=actual
                row.update(status='error-report-only',cpu_rounded_error=metrics(actual,reference),
                    frozen_reference_error=metrics(actual,frozen_reference),all_outputs_finite=bool(np.isfinite(actual).all()),
                    capture_replay=True,input_restored=True,output_lifetime=True)
                np.save(output/case_id/(backend+'-output.npy'),saved,allow_pickle=False)
        except Exception as ex: row.update(status='failed',error=repr(ex))
        finally:
            stream.synchronize()
            if plan is not None: plan.close()
            write(output/case_id/(backend+'.json'),row)
    for row in rows:
        if row['backend'] in actuals:
            row['external_errors']={name:metrics(actuals[row['backend']],value) for name,value in actuals.items()
                                    if name!=row['backend']}
    exclusive_gpu(c['gpu_uuid']);validate(m)
    write(output/case_id/'comparison.json',dict(manifest=m,rows=rows,gpu_uuid=c['gpu_uuid'],
        backends=list(backends),diagnostic_only=diagnostic_only,synchronize_setup=synchronize_setup,
        gpu_name=torch.cuda.get_device_name(0),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--case')
    p.add_argument('--backends',nargs='+',choices=BACKENDS,default=list(BACKENDS),
                   help='ordered backend subset for isolation (default: all three)')
    p.add_argument('--diagnostic-only',action='store_true',help='run correctness gates without timing')
    p.add_argument('--synchronize-setup',action='store_true',
                   help='diagnostic device synchronization after each plan creation, outside timing')
    p.add_argument('--reference-manifest',type=Path,
                   help='reuse hash-checked compatible CPU rounded references during --prepare')
    a=p.parse_args()
    if a.prepare==a.run:p.error('select exactly one of --prepare or --run')
    if a.prepare:
        if not a.config:p.error('--prepare needs --config')
        if (a.output/'manifest.json').exists():p.error('manifest exists; use a new output directory')
        write(a.output/'manifest.json',prepare(a.config,a.output,a.reference_manifest))
    else:
        if not a.case:p.error('--run needs --case (one of frozen nine IDs)')
        run(read(a.output/'manifest.json'),a.output,a.case,
            backends=a.backends,diagnostic_only=a.diagnostic_only,synchronize_setup=a.synchronize_setup)

if __name__=='__main__':main()
