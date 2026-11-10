#!/usr/bin/env python3
"""CPU-only factor-study preparation; execution uses the common local-core driver."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
from compile_module import compile_mapping
from low_fft_cpu_reference import transform
from measure_fp16_local_core import sha256
from research_compile_requests import compile_request
from run_fp16_local_core_study import save

ROOT = Path(__file__).resolve().parents[1]


def prepare(config, output, library):
    output.mkdir(parents=True, exist_ok=True)
    def state(status, **details):
        save(output/'cpu_preparation.json', dict(status=status, compile_ready=False,
             scoring_complete=False, updated_unix=time.time(), **details))
    if (output/'manifest.json').exists(): raise FileExistsError('manifest already prepared')
    state('preparing')
    try:
        cache=output/'modules'; cache.mkdir(exist_ok=True)
        prepared=output/'prepared'; prepared.mkdir(exist_ok=True)
        plans={}
        for logn, partitions in config['factor_partitions'].items():
            plans[int(logn)]=[]
            for parts in partitions:
                for columns in config['columns']:
                    for schedule in config['schedules']:
                        mapping=dict(kind='butterfly', backend='factor-streamed', fft_core='cufftdx-block',
                                     compute_unit='auto', shared_layout='writer-aligned', cross_twiddle='recurrence',
                                     stage_partition=[int(logn)], factor_partition=parts,
                                     factor_columns=columns, factor_ept=config['ept'],
                                     data_tiles_per_cta=schedule['tiles'], prefetch_depth=schedule['depth'],
                                     factor_io_policies=[schedule['io']]*len(parts))
                        point=compile_request(dict(operator='fft',precision='fp16',accumulation='native',
                            logN=int(logn),inverse=False,normalize_inverse=False,mapping_json=mapping),'research')
                        plans[int(logn)].append(dict(name='-'.join(map(str,parts))+f'-c{columns}-'+schedule['id'],
                                                    mapping=mapping,point=point))
        def compile_plan(plan):
            module=compile_mapping(plan['point'],root=ROOT,mathdx=ROOT/'external/mathdx/nvidia/mathdx',
                                   nvcc=Path('/usr/local/cuda/bin/nvcc'),sm=80,cache=cache)
            return dict(path=str(module),sha256=sha256(module),point=plan['point'])
        flat=[p for ps in plans.values() for p in ps]
        modules=[]
        with ThreadPoolExecutor(max_workers=2) as pool:
            for module in pool.map(compile_plan,flat):
                modules.append(module);state('compiling',modules=len(modules),total_modules=len(flat))
        tasks=[]
        old=ROOT/'results/a100_fp16_local_core_geometry_20261105/prepared'
        for case in config['cases']:
            n,b=case['n'],case['batch'];logn=n.bit_length()-1;stem=f'n{n}-b{b}'
            inp,ref=old/(stem+'-input.npy'),old/(stem+'-reference.npy')
            if not inp.exists() or not ref.exists():
                inp,ref=prepared/(stem+'-input.npy'),prepared/(stem+'-reference.npy')
                x=np.random.default_rng(20261105+n+b).uniform(-.25,.25,(b,n,2)).astype(np.float16)
                np.save(inp,x,allow_pickle=False)
                expected=transform(x[...,0].astype(np.float64)+1j*x[...,1].astype(np.float64),
                    dict(operator=0,storage=4,compute=0,shape=[n],batch=b,inverse=False,normalize_inverse=False),[0])
                if not np.isfinite(expected).all(): raise ValueError('nonfinite CPU reference')
                np.save(ref,expected,allow_pickle=False)
            common=dict(n=n,batch=b,input_file=str(inp),input_sha256=sha256(inp),
                        reference_file=str(ref),reference_sha256=sha256(ref))
            for plan in plans[logn]:
                tasks.append(dict(common,id=stem+'-'+plan['name'],variant='local',mapping=plan['mapping']))
            if logn<=15:
                scalar=dict(kind='butterfly',backend='temporal-tile',fft_core='scalar',tile_threads=256,
                            stage_partition=[logn],compute_unit='radix4',local_exchange='shared',shared_layout='linear')
            else:
                scalar=dict(kind='butterfly',backend='online-reorder',fft_core='scalar',tile_threads=256,
                            local_stages=9,stage_partition=[9,logn-9],compute_unit='radix4',local_exchange='shared',
                            shared_layout='linear',cross_twiddle='table',direct_boundary='direct-strided',
                            reorder_columns=8,data_tiles_per_cta=16,prefix_ept=8,suffix_ept=8)
            tasks.append(dict(common,id=stem+'-scalar',variant='scalar',mapping=scalar))
            tasks.append(dict(common,id=stem+'-cufft',variant='cufft',mapping=None))
            state('preparing-reference',case=stem,modules=len(modules))
        vendor=ROOT/'build/cufft_low_precision_sm80/libcufft_low_precision_baseline.so'
        manifest=dict(schema=config['schema'],config=config,scope=config['scope'],gpu_uuid=config['gpu_uuid'],
            library=str(library),library_sha256=sha256(library),vendor_library=str(vendor),vendor_sha256=sha256(vendor),
            jit_cache=str(cache),preparation=str(prepared),modules=modules,exclusions=[],tasks=tasks,
            worker_sha256=sha256(ROOT/'scripts/measure_fp16_local_core.py'),compiler_sha256=sha256(ROOT/'scripts/compile_module.py'))
        save(output/'manifest.json',manifest)
        save(output/'cpu_preparation.json',dict(status='ready',scoring_complete=True,compile_ready=True,
                                               tasks=len(tasks),modules=len(modules),updated_unix=time.time()))
    except BaseException as error:
        state('failed',error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'config/fft_l_factor_study.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    a=p.parse_args();prepare(json.loads(a.config.read_text()),a.output.resolve(),a.library.resolve())
