#!/usr/bin/env python3
"""Bounded FP32/FP64 factor-template regression; prepare compiles on CPU only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/a100_fp16_factor_20261105'
UUID = 'GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02'
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def point(precision, policy):
    p = dict(backend='factor-streamed', precision=precision, logN=12, stage_partition=[12],
             factor_partition=[6, 6], factor_ept=8, factor_columns=8, data_tiles_per_cta=4, prefetch_depth=1)
    if policy != 'dynamic':
        p['factor_io_policies'] = [policy, policy]
    return p


def mapping(policy):
    m = dict(kind='butterfly', backend='factor-streamed', fft_core='cufftdx-block',
             compute_unit='auto', shared_layout='writer-aligned', cross_twiddle='recurrence',
             stage_partition=[12], factor_partition=[6, 6], factor_ept=8, factor_columns=8,
             data_tiles_per_cta=4, prefetch_depth=1)
    if policy != 'dynamic':
        m['factor_io_policies'] = [policy, policy]
    return m


def prepare(out):
    import numpy as np
    import torch
    from compile_module import compile_mapping
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'modules'
    cache.mkdir(exist_ok=True)
    # Preparation is the only phase permitted to compile.
    os.environ.pop('CUBUTTERFLY_JIT_CACHE_ONLY', None)
    records = []
    for precision, dtype in [('fp32', np.complex64), ('fp64', np.complex128)]:
        rng = np.random.default_rng(20261105)
        random = (rng.uniform(-.125, .125, (3, 4096)) + 1j*rng.uniform(-.125, .125, (3, 4096))).astype(dtype)
        impulse = np.zeros((3, 4096), dtype=dtype)
        impulse[:, 0] = 1
        for profile, data in [('random', random), ('impulse', impulse)]:
            np.save(out / f'{precision}-{profile}-input.npy', data)
            for inverse in [False, True]:
                cpu = torch.from_numpy(data)
                reference = (torch.fft.ifft(cpu) if inverse else torch.fft.fft(cpu)).numpy()
                assert reference.dtype == dtype
                np.save(out / f'{precision}-{profile}-reference-{int(inverse)}.npy', reference)
        for policy in ['dynamic', 'static-unrolled']:
            p = point(precision, policy)
            module = compile_mapping(p, root=ROOT, mathdx=ROOT/'external/mathdx/nvidia/mathdx',
                                     nvcc=Path('/usr/local/cuda/bin/nvcc'), sm=80, cache=cache)
            records.append(dict(precision=precision, policy=policy, point=p,
                                module=str(module), module_sha256=sha(module), mapping=mapping(policy)))
    library = BASE / 'build/libcuntt.so'
    files = {p.name: sha(p) for p in sorted(out.glob('*.npy'))}
    identity = dict(scope='bounded generic factor-template FP32/FP64 regression, not a performance matrix',
                    gpu_uuid=UUID, library=str(library), library_sha256=sha(library),
                    jit_cache=str(cache.resolve()), modules=records, arrays=files,
                    reference='CPU torch.fft; complex64/complex128 retained; normalized inverse',
                    n=4096, batch=3, script_sha256=sha(__file__))
    save(out/'identity.json', identity)
    print('CPU preparation complete:', out/'identity.json', flush=True)


def execute(out):
    import numpy as np
    identity = json.loads((out/'identity.json').read_text())
    if (out/'summary.json').exists():
        raise FileExistsError('summary already exists')
    if sha(identity['library']) != identity['library_sha256'] or sha(__file__) != identity['script_sha256']:
        raise RuntimeError('library/script identity changed after preparation')
    for name, digest in identity['arrays'].items():
        if sha(out/name) != digest:
            raise RuntimeError('prepared array hash mismatch: '+name)
    for record in identity['modules']:
        if sha(record['module']) != record['module_sha256']:
            raise RuntimeError('module hash mismatch')
    os.environ.update(CUDA_VISIBLE_DEVICES=identity['gpu_uuid'], CUBUTTERFLY_COMPILE_MODE='research',
                      CUBUTTERFLY_JIT_CACHE_ONLY='1', CUBUTTERFLY_JIT_CACHE=identity['jit_cache'],
                      CUBUTTERFLY_JIT_PYTHON=sys.executable, CUBUTTERFLY_TEMPLATE_ROOT=str(ROOT),
                      CUBUTTERFLY_MATHDX_ROOT=str(ROOT/'external/mathdx/nvidia/mathdx'))
    from run_semantic_campaign import exclusive_gpu
    exclusive_gpu(identity['gpu_uuid'])
    import torch
    import cubutterfly as cb
    library = cb.Library(identity['library'])
    rows, outputs = [], {}
    stream = torch.cuda.Stream()
    for record in identity['modules']:
        precision, policy = record['precision'], record['policy']
        for inverse in [False, True]:
            plan = None
            try:
                with torch.cuda.stream(stream):
                    transform = cb.Transform(storage=cb.DataType.COMPLEX_FP32 if precision=='fp32' else cb.DataType.COMPLEX_FP64,
                                             extents=[4096], batch=3, inverse=inverse, normalize_inverse=inverse)
                    plan = cb.Plan(library, transform,
                                   cb.PlanOptions(policy=cb.AlgorithmPolicy.EXPLICIT, mapping=json.dumps(record['mapping'])),
                                   stream=stream.cuda_stream, cache_path=str(out/f'{precision}-{policy}-{int(inverse)}-plan.json'))
                    workspace = torch.empty(max(1,plan.workspace_size()), device='cuda',dtype=torch.uint8)
                    if plan.workspace_size(): plan.set_workspace(workspace,plan.workspace_size())
                    actual_mapping=json.loads(plan.mapping_json())
                    for profile in ['random','impulse']:
                        data=np.load(out/f'{precision}-{profile}-input.npy',allow_pickle=False)
                        reference=np.load(out/f'{precision}-{profile}-reference-{int(inverse)}.npy',allow_pickle=False)
                        encoded=data.view(np.uint8).reshape(-1)
                        initial=np.full(encoded.size+128,0xA5,np.uint8); initial[64:-64]=encoded
                        full_x=torch.from_numpy(initial).to('cuda'); expected_x=full_x.clone()
                        full_y=torch.full_like(full_x,0xA5); x=full_x[64:-64]; y=full_y[64:-64]
                        plan.execute_steady(x,y); stream.synchronize(); baseline=y.clone()
                        y.fill_(0xA5);plan.execute_steady(x,y);stream.synchronize()
                        if not torch.equal(y,baseline) or not torch.equal(full_x,expected_x):
                            raise RuntimeError('replay/input bytes changed')
                        if not bool(torch.all(full_y[:64]==0xA5)) or not bool(torch.all(full_y[-64:]==0xA5)):
                            raise RuntimeError('output guards changed')
                        actual=y.cpu().numpy().copy().view(data.dtype).reshape(data.shape)
                        if not np.isfinite(actual).all(): raise RuntimeError('nonfinite output')
                        if profile=='impulse' and not np.array_equal(actual,reference):
                            raise RuntimeError('impulse is not exact')
                        key=(precision,inverse,profile)
                        if policy=='dynamic': outputs[key]=actual.copy()
                        elif actual.tobytes()!=outputs[key].tobytes():
                            raise RuntimeError('dynamic/static output bytes differ')
                        filename=f'{precision}-{policy}-{profile}-{int(inverse)}-output.npy'
                        np.save(out/filename,actual)
                        delta=actual.astype(np.complex128)-reference.astype(np.complex128)
                        denominator=float(np.linalg.norm(reference.astype(np.complex128)))
                        rows.append(dict(precision=precision,policy=policy,inverse=inverse,normalize_inverse=inverse,
                                         profile=profile,mapping=actual_mapping,output=filename,output_sha256=sha(out/filename),
                                         max_abs=float(np.max(np.abs(delta))),relative_l2=float(np.linalg.norm(delta)/denominator),
                                         all_finite=True,guards_passed=True,replay_byte_equal=True,
                                         dynamic_static_byte_equal=True if policy!='dynamic' else None,
                                         impulse_exact=profile=='impulse',accuracy_threshold=None,verdict='error-report-only'))
                exclusive_gpu(identity['gpu_uuid'])
            finally:
                if plan is not None: plan.close()
    for record in identity['modules']:
        if sha(record['module']) != record['module_sha256']: raise RuntimeError('module changed during execution')
    save(out/'summary.json',dict(status='complete',identity_sha256=sha(out/'identity.json'),rows=rows,
                                 dynamic_static_byte_equal=True,impulse_exact=True,accuracy_threshold=None))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path,default=BASE/'fp32_fp64_regression')
    args=parser.parse_args();out=args.output.resolve()
    prepare(out) if args.prepare_only else execute(out)
