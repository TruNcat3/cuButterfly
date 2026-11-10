#!/usr/bin/env python3
"""Two matched external mechanism pairs, frozen contracts and bounded execution."""
import argparse
import copy
import ctypes as ct
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'python'), str(ROOT/'experiments/tools')]
import run_semantic_campaign as runner
import run_controlled_experiment as controlled
import semantic_preparation_queue as preparation
import semantic_dao_baseline as dao
import semantic_gpuntt_baseline as gpuntt
from run_mechanism_cohort_20261012 import environment, previous
from run_comprehensive_suite import require_exclusive_gpu

OUT = ROOT/'results/paper_targeted_20261012/mechanism/external'


def save(path, value):
    controlled._atomic_write(path, value)


def prepare():
    import numpy as np
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT/'results/paper_targeted_20261012/ntt_extension/manifest.json').read_text())
    ntt = next(c for c in manifest['cases'] if c['descriptor']['semantics']['shape']==[65536] and c['descriptor']['semantics']['batch']==16 and c['descriptor']['semantics']['inverse']==0)
    prepared = preparation.load_case(ROOT/'results/paper_targeted_20261012/ntt_extension/prepared', ntt)
    cases = []
    for name, size, external in [('ntt',65536,'gpu-ntt'), ('fwht',32768,'dao')]:
        descriptor = copy.deepcopy(ntt['descriptor'])
        s = descriptor['semantics']
        s.update(shape=[size], input_batch_stride=size, output_batch_stride=size)
        if name=='fwht':
            s.update(operator=2, storage=2, compute=1, modulus=None, word_bits=None)
        mapping = dict(schema_version=1, kind='ntt' if name=='ntt' else 'butterfly', backend='shared-iterative', compute_unit='radix2', stage_partition=[8,8] if name=='ntt' else [8,7])
        if name=='ntt':
            mapping.update(threads_per_block=128, dataflow_layout='linear', modular_multiply='shoup')
            source, expected = prepared['source'], prepared['expected']
            support = gpuntt.support(s)
        else:
            mapping.update(tile_threads=128, fft_core='scalar', shared_layout='linear', local_exchange='shared')
            source = runner.make_source(s)
            expected = runner.reference(s, source)
            support = dao.support(s)
        path = OUT/(name+'.npz')
        if not path.exists():
            np.savez(path, source=source, expected=expected)
        cases.append(dict(id=name, external=external, descriptor=descriptor, mapping=mapping,
                          support=support, artifact=str(path), artifact_sha256=runner.sha256_file(path)))
    requested = copy.deepcopy(cases[1]['descriptor']['semantics'])
    requested.update(shape=[65536], input_batch_stride=65536, output_batch_stride=65536)
    value = dict(schema='cubutterfly-external-mechanism-v1', cases=cases,
                 dao_n65536_gap=dao.support(requested), gpu_uuid=previous.GPU,
                 library=str(previous.LIB), library_sha256=previous.EXPECTED_LIB,
                 protocol=runner.PROTOCOL, independent_process_trials=3,
                 scope='NTT N65536B16 and separately approved FWHT N32768B16 matched complete calls; no substitute for unsupported Dao N65536.')
    path = OUT/'cohort.json'
    if path.exists() and json.loads(path.read_text())!=value:
        raise RuntimeError('cohort changed')
    save(path,value)
    print(json.dumps({'prepared':2, 'dao_n65536':value['dao_n65536_gap']['reason']}))


def prepare_fwht_report():
    """Freeze a separate float-report directory without changing source data."""
    target = OUT/'fwht_float_report'
    target.mkdir(parents=True, exist_ok=True)
    source = json.loads((OUT/'cohort.json').read_text())
    fwht = next(item for item in source['cases'] if item['id']=='fwht')
    for name in ['fwht.npz']:
        destination = target/name
        if not destination.exists():
            destination.write_bytes((OUT/name).read_bytes())
    value = dict(schema='cubutterfly-external-fwht-float-report-v1', source_cohort=str((OUT/'cohort.json').resolve()),
                 source_case=fwht, operator='FWHT', precision='FP32', shape=[32768], batch=16,
                 mapping=fwht['mapping'], float_verdict='error-report', numerical_acceptance='unassessed',
                 correctness_requirements=['finite output', 'complete input unchanged', 'full error distribution'],
                 timing_requirements={'independent_process_trials':3,'normal_timing_excludes_setup':True},
                 profiles={'logical_calls':1,'warmup':100,'capture':'cudaProfilerApi full synchronized call',
                           'normal_timing_accepted':False},
                 artifact=str((target/'fwht.npz').resolve()), artifact_sha256=runner.sha256_file(target/'fwht.npz'))
    save(target/'cohort.json', value)
    print(json.dumps({'status':'cpu-prepared','output':str(target),'float_verdict':'error-report',
                      'artifact_sha256':value['artifact_sha256']}))


def execute(case_id, implementation, mode, output, float_verdict='legacy-fixed'):
    import numpy as np
    import torch
    import cubutterfly as cb
    cohort = json.loads((OUT/'cohort.json').read_text())
    if runner.sha256_file(previous.LIB)!=cohort['library_sha256']:
        raise RuntimeError('runtime mismatch')
    case = next(c for c in cohort['cases'] if c['id']==case_id)
    if not case['support']['supported']:
        raise RuntimeError(case['support']['reason'])
    if runner.sha256_file(case['artifact'])!=case['artifact_sha256']:
        raise RuntimeError('oracle changed')
    arrays = np.load(case['artifact'])
    s = case['descriptor']['semantics']
    dtype = torch.uint64 if case_id=='ntt' else torch.float32
    source = torch.from_numpy(arrays['source']).to('cuda')
    saved = source.clone()
    destination = torch.empty_like(source)
    stream = torch.cuda.current_stream()
    plan = native = None
    metadata = {}
    try:
        if implementation=='cubutterfly':
            library = cb.Library(previous.LIB)
            descriptor = runner.selection_descriptor(case['descriptor'], mapping=case['mapping'])
            descriptor['selection']['execution_mode'] = int(cb.ExecutionMode.COMPOSITE)
            with cb.Descriptor.from_json(library,descriptor) as desc:
                plan = cb.Plan(library,desc,stream=stream.cuda_stream)
            metadata = dict(mapping=json.loads(plan.mapping_json()), algorithm=plan.algorithm_name(),workspace_bytes=plan.workspace_size())
            expected_mapping=dict(case['mapping'])
            if any(metadata['mapping'].get(k)!=v for k,v in expected_mapping.items() if k not in {'schema_version','kind'}):
                raise RuntimeError('resolved control mapping changed')
            workspace = torch.empty(max(1,plan.workspace_size()),dtype=torch.uint8,device='cuda')
            if plan.workspace_size():
                plan.set_workspace(workspace,plan.workspace_size())
            call=lambda:plan.execute(source,destination)
        else:
            native= gpuntt.build_core(torch,s,source,destination,stream) if case_id=='ntt' else dao.build_core(torch,s,source,destination)
            metadata=native.metadata()
            call=native
        row=dict(case=case_id,implementation=implementation,mode=mode,semantics=s,metadata=metadata,
                 gpu_uuid=previous.GPU,library_sha256=previous.EXPECTED_LIB,artifact_sha256=case['artifact_sha256'],
                 device=torch.cuda.get_device_name(),driver=runner.sha256_file(__file__))
        def verify():
            result=runner.check_full(destination.cpu().numpy(),arrays['expected'],s['storage'],float_verdict,False)
            if not torch.equal(source,saved):
                raise AssertionError('out-of-place input modified')
            return result
        call(); stream.synchronize()
        try:
            row['correctness']=verify()
        except AssertionError as error:
            row.update(status='correctness-failure', error=str(error), performance_accepted=False,
                       correctness=runner.check_full(destination.cpu().numpy(),arrays['expected'],s['storage'],'error-report',True))
            save(output,row)
            raise
        if mode=='timing':
            row['timing']=runner.time_execute(torch,stream,call,lambda:source.copy_(saved),False,verify,lambda:destination.zero_())
        else:
            for _ in range(100):call()
            stream.synchronize()
            cuda=ct.CDLL('libcudart.so.12')
            if cuda.cudaProfilerStart()!=0: raise RuntimeError('profiler start failed')
            call(); stream.synchronize()
            if cuda.cudaProfilerStop()!=0: raise RuntimeError('profiler stop failed')
            row.update(logical_calls=1,warmup=100,capture='synchronized cudaProfilerApi range',normal_timing_accepted=False)
            row['post_capture_correctness']=verify()
        row['status']='accuracy-review-required' if s['storage'] not in (8,9) and float_verdict=='error-report' else 'pass'
        row['float_verdict']=float_verdict
        row['numerical_acceptance_claimed']=row['status']=='pass'
        if s['storage'] not in (8,9):
            row['correctness']=runner.check_full(destination.cpu().numpy(),arrays['expected'],s['storage'],float_verdict,True)
        save(output,row)
    finally:
        if plan:plan.close()
        if native:native.close()


def run(output_dir=OUT, cases=('ntt','fwht'), float_verdict='legacy-fixed'):
    output_dir.mkdir(parents=True,exist_ok=True)
    os.environ.update(environment(output_dir/'events.jsonl'))
    for trial in range(1,4):
        for case in cases:
            order=['cubutterfly','external'] if trial%2 else ['external','cubutterfly']
            for implementation in order:
                path=output_dir/f'{case}.{implementation}.trial{trial}.json'
                if path.exists():continue
                require_exclusive_gpu()
                command=[str(previous.PYTHON),str(Path(__file__).resolve()),'worker','--case',case,'--implementation',implementation,'--float-verdict',float_verdict,'--output',str(path)]
                result=subprocess.run(command,text=True,capture_output=True)
                (output_dir/f'{case}.{implementation}.trial{trial}.log').write_text(result.stdout+'\n'+result.stderr)
                require_exclusive_gpu()
                print(case,implementation,trial,result.returncode,flush=True)
                if result.returncode:
                    if case=='fwht' and path.exists():
                        continue
                    raise SystemExit(result.returncode)
    from profile_mechanism_cohort_20261012 import METRICS
    for case in cases:
        for implementation in ['cubutterfly','external']:
            validation = json.loads((output_dir/f'{case}.{implementation}.trial1.json').read_text())
            if validation['status'] not in {'pass','accuracy-review-required'}:
                continue
            for tool in ['ncu','nsys']:
                stem=output_dir/f'{case}.{implementation}.{tool}'
                record=stem.with_suffix(stem.suffix+'.json')
                if record.exists():continue
                driver=[str(previous.PYTHON),str(Path(__file__).resolve()),'worker','--mode','profile','--case',case,'--implementation',implementation,'--float-verdict',float_verdict,'--output',str(stem)+'.capture.json']
                if tool=='ncu':
                    command=['ncu','--profile-from-start','off','--target-processes','all','--kernel-name-base','demangled','--metrics',','.join(METRICS),'--csv','--log-file',str(stem)+'.csv']+driver
                else:
                    command=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--output',str(stem)]+driver
                require_exclusive_gpu()
                result=subprocess.run(command,text=True,capture_output=True)
                require_exclusive_gpu()
                save(record,dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,logical_calls=1,normal_timing_accepted=False))
                print(case,implementation,tool,result.returncode,flush=True)
                if result.returncode:raise SystemExit(result.returncode)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','prepare-fwht-report','run','worker'])
    parser.add_argument('--case',choices=['ntt','fwht'])
    parser.add_argument('--implementation',choices=['cubutterfly','external'])
    parser.add_argument('--mode',choices=['timing','profile'],default='timing')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--float-verdict',choices=['legacy-fixed','error-report'],default='legacy-fixed')
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='prepare-fwht-report':prepare_fwht_report()
    elif args.action=='run':run(args.output_dir,(args.case,) if args.case else ('ntt','fwht'),args.float_verdict)
    else:
        import torch
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            execute(args.case,args.implementation,args.mode,args.output,args.float_verdict)


if __name__=='__main__':main()
