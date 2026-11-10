#!/usr/bin/env python3
"""Execute one frozen FP16 study task; importing this module never initializes CUDA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_task(manifest_path, task_id):
    """Resolve manifest-relative paths and validate artifacts without touching CUDA."""
    import numpy as np
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    def resolve(value):
        p = Path(value)
        return str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    for key in ('library', 'vendor_library', 'jit_cache', 'preparation'):
        manifest[key] = resolve(manifest[key])
    if not manifest['gpu_uuid'].startswith('GPU-'):
        raise ValueError('gpu_uuid must identify the target GPU')
    for key, digest in [('library', 'library_sha256'), ('vendor_library', 'vendor_sha256')]:
        if sha256(manifest[key]) != manifest[digest]:
            raise ValueError(key + ' hash mismatch')
    if not Path(manifest['jit_cache']).is_dir():
        raise ValueError('precompiled jit_cache is missing')
    for module in manifest['modules']:
        module['path'] = resolve(module['path'])
        if sha256(module['path']) != module['sha256']:
            raise ValueError('module hash mismatch')
    tasks = manifest['tasks']
    if len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('duplicate task IDs')
    matches = [t for t in tasks if t['id'] == task_id]
    if len(matches) != 1:
        raise ValueError('unknown task ID: ' + task_id)
    task = matches[0]
    if task['variant'] not in ('local', 'scalar', 'cufft'):
        raise ValueError('unsupported variant')
    if any(type(task[k]) is not int or task[k] <= 0 for k in ('n', 'batch')):
        raise ValueError('n and batch must be positive integers')
    if task['variant'] != 'cufft' and not isinstance(task['mapping'], dict):
        raise ValueError('public EXPLICIT mapping required')
    for key in ('input_file', 'reference_file'):
        task[key] = resolve(task[key])
        if not Path(task[key]).is_file():
            raise ValueError(key + ' is missing')
        digest_key = 'input_sha256' if key == 'input_file' else 'reference_sha256'
        if sha256(task[key]) != task[digest_key]:
            raise ValueError(key + ' hash mismatch')
    data = np.load(task['input_file'], allow_pickle=False)
    if data.dtype != np.float16 or data.shape != (task['batch'], task['n'], 2):
        raise ValueError('input must be complete FP16 [batch,n,2]')
    if not np.isfinite(data).all():
        raise ValueError('input contains nonfinite values')
    return manifest, task, np.ascontiguousarray(data)


def configure_environment(manifest):
    os.environ.update(CUDA_VISIBLE_DEVICES=manifest['gpu_uuid'],
                      CUBUTTERFLY_COMPILE_MODE='research',
                      CUBUTTERFLY_JIT_CACHE_ONLY='1',
                      CUBUTTERFLY_JIT_CACHE=manifest['jit_cache'],
                      CUBUTTERFLY_JIT_PYTHON=sys.executable,
                      CUBUTTERFLY_MATHDX_ROOT=str(ROOT / 'external/mathdx/nvidia/mathdx'),
                      CUBUTTERFLY_TEMPLATE_ROOT=str(ROOT))


def resource_limit(error):
    text = str(error).lower()
    return any(marker in text for marker in (
        'cudaerrorlaunchoutofresources', 'cudaerrorinvalidconfiguration',
        'compiled specialization exceeds active-device resources',
        'cuda error: too many resources requested for launch',
        'cuda error: invalid configuration argument',
        'cuda runtime error: too many resources requested for launch',
        'cuda runtime error: invalid configuration argument'))


def write_result(path, record):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run(args):
    import numpy as np
    manifest, task, data = load_task(args.manifest, args.task)
    configure_environment(manifest)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists() or (out / 'output.npy').exists():
        raise FileExistsError('output already contains results; use an independent attempt directory')
    # Atomic claim also rejects retries after an interrupted or failed attempt.
    with (out / 'worker-started.json').open('x') as stream:
        json.dump(dict(task_id=args.task, mode=args.mode, pid=os.getpid()), stream)
    record = dict(task_id=args.task, mode=args.mode, gpu_uuid=manifest['gpu_uuid'],
                  manifest_sha256=sha256(args.manifest), library_sha256=manifest['library_sha256'],
                  vendor_sha256=manifest['vendor_sha256'], compile_mode='research', jit_cache_only=True,
                  design_point=task, modules=manifest['modules'],
                  input_sha256=sha256(task['input_file']), reference_file=task['reference_file'])
    plan = None
    try:
        sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
        from run_semantic_campaign import exclusive_gpu, time_execute
        exclusive_gpu(manifest['gpu_uuid'])
        import torch
        import cubutterfly as cb
        import cufft_low_precision_baseline as vendor
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            encoded = data.view(np.uint8).reshape(-1)
            initial = np.full(encoded.size + 128, 0xA5, np.uint8)
            initial[64:-64] = encoded
            full_x = torch.from_numpy(initial).to('cuda')
            expected_x = full_x.clone()
            full_y = torch.full_like(full_x, 0xA5)
            x = full_x[64:-64].view(torch.float16).reshape(data.shape)
            y = full_y[64:-64].view(torch.float16).reshape(data.shape)
            start = time.monotonic()
            if task['variant'] == 'cufft':
                plan = vendor.Plan(manifest['vendor_library'], [task['n']], task['batch'],
                                   stream=stream.cuda_stream, precision='fp16', direction='forward')
                execute = lambda: plan.execute(x.data_ptr(), y.data_ptr(), stream=stream.cuda_stream)
                mapping = dict(provider='cuFFT', arithmetic='vendor internal arithmetic unspecified')
            else:
                library = cb.Library(manifest['library'])
                transform = cb.Transform(storage=cb.DataType.COMPLEX_FP16, extents=[task['n']],
                                         batch=task['batch'], normalize_inverse=False)
                options = cb.PlanOptions(policy=cb.AlgorithmPolicy.EXPLICIT,
                                         mapping=json.dumps(task['mapping']))
                plan = cb.Plan(library, transform, options, stream=stream.cuda_stream,
                               cache_path=str(out / 'plan-cache.json'))
                workspace = torch.empty(max(1, plan.workspace_size()), device='cuda', dtype=torch.uint8)
                if plan.workspace_size():
                    plan.set_workspace(workspace, plan.workspace_size())
                execute = lambda: plan.execute_steady(x, y)
                mapping = json.loads(plan.mapping_json())
            record.update(setup_seconds=time.monotonic() - start, mapping=mapping)
            execute()
            stream.synchronize()
            direct = y.view(torch.uint8).clone()
            def verify():
                if not torch.equal(y.view(torch.uint8), direct):
                    raise RuntimeError('replay output bytes changed')
                if not torch.equal(full_x, expected_x):
                    raise RuntimeError('input or input guards changed')
                if not bool(torch.all(full_y[:64] == 0xA5)) or not bool(torch.all(full_y[-64:] == 0xA5)):
                    raise RuntimeError('output guards changed')
            if args.mode == 'measure':
                record['graph_timing'] = time_execute(torch, stream, execute, lambda: None, False,
                                                      verify, lambda: y.view(torch.uint8).fill_(0xA5))
            else:
                y.view(torch.uint8).fill_(0xA5)
                stream.synchronize()
                with torch.cuda.nvtx.range('fft_l_local_core'):
                    execute()
                    stream.synchronize()
                record['timing_role'] = 'capture only; NCU duration is not performance timing'
            verify()
            stream.synchronize()
            actual = y.cpu().numpy().copy()
            if not np.isfinite(actual).all():
                raise RuntimeError('nonfinite output')
            plan.close()
            plan = None
        exclusive_gpu(manifest['gpu_uuid'])
        # Confirm immutable inputs to precompiled execution did not change.
        for key, digest in [('library', 'library_sha256'), ('vendor_library', 'vendor_sha256')]:
            if sha256(manifest[key]) != manifest[digest]:
                raise RuntimeError(key + ' changed during execution')
        for module in manifest['modules']:
            if sha256(module['path']) != module['sha256']:
                raise RuntimeError('module changed during execution')
        np.save(out / 'output.npy', actual, allow_pickle=False)
        record.update(status='complete', output_sha256=sha256(out / 'output.npy'),
                      replay_byte_equal=True, input_and_output_guards_passed=True,
                      all_outputs_finite=True, accuracy_verdict='pending offline reference review')
        write_result(out / 'result.json', record)
    except Exception as error:
        limited = resource_limit(error)
        record.update(status='resource-limit' if limited else 'failed',
                      error_type=type(error).__name__, error=str(error))
        write_result(out / 'result.json', record)
        if not limited:
            raise
    finally:
        if plan is not None:
            plan.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', choices=['measure', 'capture'], required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
