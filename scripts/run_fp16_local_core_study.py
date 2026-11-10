#!/usr/bin/env python3
"""Prepare a finite local-core study on CPU; resume isolated GPU requests.

Use watch_idle_campaign.py for idle detection. This driver checkpoints each
request and quarantines attempts interrupted by foreign GPU work. It never
retries numerical, compiler, or trace failures automatically.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from measure_fp16_local_core import sha256, write_result
from run_isolated_gpu import descendant, gpu_pids, stop_owned


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_result(path, value)


def build_library(output):
    # Reuse the established development relink, preserving its original receipt.
    builder = load_module('local_core_builder', ROOT / 'results/a100_fft_l_diagnostic_20261105/build_local_core.py')
    builder.OUT = output / 'build'
    builder.OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        compiled = list(pool.map(builder.compile_one, builder.sources))
    command = shlex.split((builder.BUILD / 'CMakeFiles/cuntt.dir/link.txt').read_text())
    replacements = {'CMakeFiles/cuntt.dir/' + r['source'] + '.o': r['object'] for r in compiled}
    command = [replacements.get(word, word) for word in command]
    library = builder.OUT / 'libcuntt.so'
    command[command.index('-o') + 1] = str(library)
    with (builder.OUT / 'link.log').open('w') as log:
        subprocess.run(command, cwd=builder.BUILD, stdout=log, stderr=subprocess.STDOUT, check=True)
    save(builder.OUT / 'receipt.json', dict(library=str(library), library_sha256=sha256(library),
         compiled=compiled, link=command, scope='diagnostic relink; unchanged existing objects reused'))
    # Verify dataflow projection without creating a CUDA context.
    test = builder.OUT / 'test_fp16_contract'
    subprocess.run(['/usr/bin/c++', '-std=c++17', '-DFP16_LOCAL_CORE_TEST_DATAFLOW',
                    '-I' + str(ROOT / 'include'), '-I' + str(ROOT / 'src'),
                    '-I/usr/local/cuda/include', str(ROOT / 'tests/test_fp16_local_core_contract.cpp'),
                    '-L' + str(builder.OUT), '-lcuntt', '-Wl,-rpath,' + str(builder.OUT),
                    '-o', str(test)], check=True)
    subprocess.run([str(test)], check=True)
    return library



def local_core_compiled_resources(raw):
    """Require both directions and both I/O specializations; retain every entry."""
    entries = []
    seen = set()
    for match in re.finditer(r"^\s*Function\s+(\S+):\s*\n([^\n]+)", raw, re.MULTILINE):
        name, fields = match.groups()
        variant = re.search(r"ELb([01])ELb([01])EEE", name)
        if "low_block_pair" not in name or variant is None:
            raise ValueError("unexpected compiled resource function: " + name)
        key = tuple(int(value) for value in variant.groups())
        if key in seen:
            raise ValueError("duplicate compiled resource variant: " + name)
        values = {}
        for field in ("REG", "STACK", "SHARED", "LOCAL"):
            tokens = re.findall(r"\b" + field + r":([^\s]+)", fields)
            if len(tokens) != 1 or not tokens[0].isdigit():
                raise ValueError("invalid compiled " + field + " resource: " + name)
            values[field.lower()] = int(tokens[0])
        if not 1 <= values["reg"] <= 255:
            raise ValueError("invalid compiled register count: " + name)
        seen.add(key)
        entries.append(dict(function=name, inverse=bool(key[0]), aligned_io=bool(key[1]), **values))
    expected = {(inverse, aligned) for inverse in (0, 1) for aligned in (0, 1)}
    if seen != expected or len(re.findall(r"\bREG:", raw)) != len(entries):
        raise ValueError("expected all four forward/inverse aligned/unaligned compiled resource entries")
    return entries


def prepare(config_path, output):
    import numpy as np
    from compile_module import compile_mapping
    from cufftdx_capabilities import load_database
    from cufftdx_low_module import validate_point, PACKING
    from low_fft_cpu_reference import transform
    config = read(config_path)
    state_path = output / 'cpu_preparation.json'
    def state(status, **details):
        save(state_path, dict(status=status, scoring_complete=False, compile_ready=False,
                             pid=os.getpid(), updated_unix=time.time(), **details))
    if (output / 'manifest.json').exists():
        raise FileExistsError('prepared manifest already exists; do not silently replace the campaign')
    state('preparing')
    try:
        mathdx = ROOT / 'external/mathdx/nvidia/mathdx'
        database = load_database(mathdx)
        hardware = read(ROOT / config['hardware_model'])['hardware']
        cache = output / 'modules'
        cache.mkdir(exist_ok=True)
        points, exclusions = [], []
        for n in sorted({c['n'] for c in config['cases']}):
            for ept in config['ept']:
                for threads in config['threads']:
                    point = dict(backend='temporal-tile', fft_core='cufftdx-block', precision='fp16',
                                 accumulation='native', packing=PACKING, logN=n.bit_length()-1,
                                 tile_threads=threads, prefix_ept=ept)
                    if threads > hardware['max_threads_per_block']:
                        exclusions.append(dict(point=point, reason='hardware thread limit'))
                        continue
                    try:
                        validate_point(point, config['sm'], database)
                    except ValueError as error:
                        exclusions.append(dict(point=point, reason=str(error)))
                    else:
                        points.append(point)
        # Compilation and CPU references run before GPU readiness is released.
        def compile_point(point):
            path = compile_mapping(point, root=ROOT, mathdx=mathdx,
                                   nvcc=Path('/usr/local/cuda/bin/nvcc'), sm=config['sm'], cache=cache)
            raw = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--dump-resource-usage', str(path)], text=True)
            (path.parent / 'resources.txt').write_text(raw)
            resources = local_core_compiled_resources(raw)
            registers = [entry['reg'] for entry in resources]
            return dict(path=str(path), sha256=sha256(path), point=point,
                        compiled_resources=resources,
                        worst_compiled_resources={field: max(entry[field] for entry in resources)
                                                  for field in ('reg', 'stack', 'shared', 'local')},
                        registers=registers, feasible_register_bound=(max(registers)*point['tile_threads'] <= hardware['registers_per_block']))
        with ThreadPoolExecutor(max_workers=config['cpu_jobs']) as pool:
            build_future = pool.submit(build_library, output)
            # Serial modules plus one library build keep NVCC resource use bounded.
            modules = []
            for point in points:
                modules.append(compile_point(point))
                state('compiling', compiled_modules=len(modules), total_modules=len(points))
            preparation = output / 'prepared'
            preparation.mkdir(exist_ok=True)
            tasks = []
            for case in config['cases']:
                n, b = case['n'], case['batch']
                stem = f'n{n}-b{b}'
                rng = np.random.default_rng(config['seed'] + n + b)
                data = rng.uniform(*config['input_range'], (b, n, 2)).astype(np.float16)
                inp, ref = preparation / (stem + '-input.npy'), preparation / (stem + '-reference.npy')
                np.save(inp, data, allow_pickle=False)
                source = data[..., 0].astype(np.float64) + 1j*data[..., 1].astype(np.float64)
                reference = transform(source, dict(operator=0, storage=4, compute=0, shape=[n], batch=b,
                                                   inverse=False, normalize_inverse=False), [0])
                if not np.isfinite(reference).all():
                    raise ValueError('CPU FP16 reference is nonfinite; preserve failed preparation for review')
                np.save(ref, reference, allow_pickle=False)
                common = dict(n=n, batch=b, input_file=str(inp), input_sha256=sha256(inp),
                              reference_file=str(ref), reference_sha256=sha256(ref))
                for module in modules:
                    point = module['point']
                    if 1 << point['logN'] != n: continue
                    if not module['feasible_register_bound']:
                        continue
                    t, e = point['tile_threads'], point['prefix_ept']
                    mapping = dict(kind='butterfly', backend='temporal-tile', fft_core='cufftdx-block',
                                   tile_threads=t, prefix_ept=e, stage_partition=[point['logN']],
                                   compute_unit='radix2', local_exchange='shared', shared_layout='linear')
                    tasks.append(dict(common, id=f'{stem}-t{t}-e{e}', variant='local', mapping=mapping))
                tasks.append(dict(common, id=f'{stem}-scalar', variant='scalar', mapping=dict(
                    kind='butterfly', backend='temporal-tile', fft_core='scalar', tile_threads=256,
                    prefix_ept=16, stage_partition=[n.bit_length()-1], compute_unit='radix4',
                    local_exchange='shared', shared_layout='linear')))
                tasks.append(dict(common, id=f'{stem}-cufft', variant='cufft', mapping=None))
                state('preparing-reference', prepared_case=stem, compiled_modules=len(modules))
            library = build_future.result()
        vendor = ROOT / 'build/cufft_low_precision_sm80/libcufft_low_precision_baseline.so'
        manifest = dict(schema=config['schema'], config=config, scope=config['scope'],
                        gpu_uuid=config['gpu_uuid'], library=str(library), library_sha256=sha256(library),
                        vendor_library=str(vendor), vendor_sha256=sha256(vendor), jit_cache=str(cache),
                        preparation=str(preparation), modules=modules, exclusions=exclusions, tasks=tasks,
                        worker_sha256=sha256(ROOT / 'scripts/measure_fp16_local_core.py'),
                        compiler_sha256=sha256(ROOT / 'scripts/compile_module.py'))
        save(output / 'manifest.json', manifest)
        save(state_path, dict(status='ready', scoring_complete=True, compile_ready=True,
                              tasks=len(tasks), modules=len(modules), updated_unix=time.time()))
    except BaseException as error:
        state('failed', error=repr(error))
        raise


class Interrupted(Exception):
    pass


def foreign(uuid, child=None):
    return [pid for pid in gpu_pids(uuid) if Path(f'/proc/{pid}').exists()
            and (child is None or not descendant(pid, child.pid))]


def run_attempt(command, directory, uuid, timeout=900):
    directory.mkdir(parents=True, exist_ok=False)
    if foreign(uuid): raise Interrupted('GPU acquired by foreign work before launch')
    child = None
    started = time.monotonic()
    try:
        with (directory / 'worker.log').open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid))
            while child.poll() is None:
                if foreign(uuid, child): raise Interrupted('foreign GPU work during request')
                if time.monotonic() - started > timeout: raise TimeoutError('request timeout')
                time.sleep(1)
            if foreign(uuid, child): raise Interrupted('foreign GPU work at request completion')
            if child.returncode: raise RuntimeError(f'worker exit {child.returncode}; see {directory}/worker.log')
    finally:
        if child is not None: stop_owned(child)


def trace_selection(manifest, completed):
    tasks = {t['id']: t for t in manifest['tasks']}
    selected = []
    for case in manifest['config']['trace_cases']:
        matching = [t for t in tasks.values() if (t['n'], t['batch']) == (case['n'], case['batch'])
                    and completed[t['id']].get('status') == 'complete']
        local = [t for t in matching if t['variant'] == 'local']
        if local:
            best = min(local, key=lambda t: completed[t['id']]['graph_timing']['median_ms'])
            selected.append(best['id'])
        selected.extend(t['id'] for t in matching if t['variant'] == 'cufft' or
                        (t['variant'] == 'local' and t['mapping'].get('tile_threads') == 256 and t['mapping'].get('prefix_ept') == 16))
    return list(dict.fromkeys(selected))


def checked_result(attempt, task_id, mode, identity, receipt=None):
    result_path = attempt / 'result.json'
    result = read(result_path)
    if (result.get('task_id') != task_id or result.get('mode') != mode or
            result.get('manifest_sha256') != identity or
            result.get('status') not in ('complete', 'resource-limit')):
        raise ValueError('worker result identity/status mismatch')
    if receipt is not None and receipt.get('result_sha256') != sha256(result_path):
        raise ValueError('saved result hash mismatch')
    if result['status'] == 'complete' and sha256(attempt / 'output.npy') != result['output_sha256']:
        raise ValueError('saved output hash mismatch')
    return result


def run(output):
    manifest_path = output / 'manifest.json'
    manifest = read(manifest_path)
    identity = sha256(manifest_path)
    uuid = manifest['gpu_uuid']
    preparation = read(output / 'cpu_preparation.json')
    if not (preparation.get('scoring_complete') and preparation.get('compile_ready')):
        raise ValueError('CPU preparation has not completed')
    if sha256(ROOT / 'scripts/measure_fp16_local_core.py') != manifest['worker_sha256']:
        raise ValueError('worker changed after CPU preparation')
    if sha256(ROOT / 'scripts/compile_module.py') != manifest['compiler_sha256']:
        raise ValueError('module cache identity changed after CPU preparation')
    tasks = manifest['tasks']
    completed = {}
    def checkpoint(status, phase, **extra):
        save(output / 'status.json', dict(status=status, phase=phase, measured_rows=len(completed),
             remaining=len(tasks)-len(completed) + (1 if phase != 'all' else 0),
             updated_unix=time.time(), pid=os.getpid(), gpu_uuid=uuid, **extra))
    def interrupted(sig, frame): raise Interrupted('received signal ' + str(sig))
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        for task in tasks:
            path = output / 'measure' / task['id'] / 'complete.json'
            if path.exists():
                receipt = read(path)
                if receipt['manifest_sha256'] != identity: raise ValueError('resume identity mismatch')
                completed[task['id']] = checked_result(Path(receipt['attempt']), task['id'], 'measure', identity, receipt)
                continue
            checkpoint('running', 'measure', task_id=task['id'])
            attempt = path.parent / ('attempt-' + str(time.time_ns()))
            command = [sys.executable, str(ROOT / 'scripts/measure_fp16_local_core.py'), '--manifest',
                       str(manifest_path), '--task', task['id'], '--output', str(attempt), '--mode', 'measure']
            run_attempt(command, attempt, uuid)
            result = checked_result(attempt, task['id'], 'measure', identity)
            save(path, dict(attempt=str(attempt), manifest_sha256=identity,
                            result_sha256=sha256(attempt / 'result.json')))
            completed[task['id']] = result
        selected = trace_selection(manifest, completed)
        save(output / 'trace-selection.json', dict(task_ids=selected, manifest_sha256=identity))
        sections = ['LaunchStats', 'Occupancy', 'SpeedOfLight', 'SchedulerStats',
                    'WarpStateStats', 'MemoryWorkloadAnalysis', 'InstructionStats', 'SourceCounters']
        for task_id in selected:
            path = output / 'capture' / task_id / 'complete.json'
            if path.exists():
                receipt = read(path)
                if receipt['manifest_sha256'] != identity: raise ValueError('trace identity mismatch')
                checked_result(Path(receipt['attempt']), task_id, 'capture', identity, receipt)
                if sha256(Path(receipt['attempt']) / 'capture.ncu-rep') != receipt['report_sha256']:
                    raise ValueError('NCU report hash mismatch')
                continue
            checkpoint('running', 'capture', task_id=task_id)
            attempt = path.parent / ('attempt-' + str(time.time_ns()))
            command = ['/usr/sbin/ncu', '--target-processes', 'all', '--nvtx', '--nvtx-include', 'fft_l_local_core/',
                       '--replay-mode', 'kernel', '--cache-control', 'none', '--clock-control', 'base',
                       '--export', str(attempt / 'capture'), '--page', 'raw', '--csv', '--log-file', str(attempt / 'ncu.csv')]
            for section in sections: command += ['--section', section]
            command += [sys.executable, str(ROOT / 'scripts/measure_fp16_local_core.py'), '--manifest', str(manifest_path),
                        '--task', task_id, '--output', str(attempt), '--mode', 'capture']
            run_attempt(command, attempt, uuid)
            result = checked_result(attempt, task_id, 'capture', identity)
            if result['status'] != 'complete': raise RuntimeError('measured candidate failed NCU replay')
            if not (attempt / 'capture.ncu-rep').is_file(): raise RuntimeError('NCU report missing')
            save(path, dict(attempt=str(attempt), manifest_sha256=identity,
                            result_sha256=sha256(attempt / 'result.json'),
                            report_sha256=sha256(attempt / 'capture.ncu-rep')))
        checkpoint('complete-gpu-pending-cpu-review', 'all')
        # Release the CUDA device before offline array/error and trace processing.
        with (output / 'review.log').open('a') as log:
            subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--review', '--output', str(output)],
                             cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return 0
    except Interrupted as error:
        checkpoint('interrupted-by-foreign-gpu-work', 'resume', retryable=True, error=str(error))
        return 75
    except BaseException as error:
        checkpoint('failed', 'review-required', retryable=False, error=repr(error))
        raise
    finally:
        for sig, handler in previous.items(): signal.signal(sig, handler)


def review(output):
    import numpy as np
    manifest = read(output / 'manifest.json')
    identity = sha256(output / 'manifest.json')
    rows = []
    for task in manifest['tasks']:
        receipt = read(output / 'measure' / task['id'] / 'complete.json')
        if receipt['manifest_sha256'] != identity: raise ValueError('review identity mismatch')
        attempt = Path(receipt['attempt'])
        result = checked_result(attempt, task['id'], 'measure', identity, receipt)
        if result['status'] == 'complete':
            if sha256(attempt / 'output.npy') != result['output_sha256']: raise ValueError('output hash mismatch')
            if sha256(task['reference_file']) != task['reference_sha256']: raise ValueError('reference hash mismatch')
            actual = np.load(attempt / 'output.npy', mmap_mode='r')
            reference = np.load(task['reference_file'], mmap_mode='r')
            if (actual.shape != (task['batch'], task['n'], 2) or actual.dtype != np.float16 or
                    reference.shape != (task['batch'], task['n']) or reference.dtype.kind != 'c' or
                    not np.isfinite(actual).all() or not np.isfinite(reference).all()):
                raise ValueError('review output/reference shape, dtype or finiteness mismatch')
            total, ref_total, maximum, absolute = 0., 0., 0., []
            for start in range(0, task['batch'], 16):
                a = actual[start:start+16]
                v = a[...,0].astype(np.float64) + 1j*a[...,1].astype(np.float64)
                r = reference[start:start+16]
                error = np.abs(v-r)
                maximum = max(maximum, float(np.max(error)))
                total += float(np.sum(error**2)); ref_total += float(np.sum(np.abs(r)**2))
                absolute.append(error.reshape(-1))
            result.update(cpu_fp16_max_abs=maximum, cpu_fp16_rel_l2=math.sqrt(total/max(ref_total,1e-300)),
                          absolute_error_quantiles=dict(zip(['p50','p90','p99','max'],
                          np.quantile(np.concatenate(absolute),[.5,.9,.99,1]).tolist())),
                          accuracy_verdict='error-report-only', threshold=None)
        rows.append(result)
    trace_reader = load_module('fft_l_trace_reader', ROOT / 'results/a100_fft_l_trace_20261103/trace.py')
    for task_id in read(output / 'trace-selection.json')['task_ids']:
        attempt = Path(read(output / 'capture' / task_id / 'complete.json')['attempt'])
        summary = trace_reader.parse_report(attempt / 'capture.ncu-rep')
        if summary['kernel_count'] == 0: raise ValueError('empty NCU capture')
        save(attempt / 'summary.json', summary)
    save(output / 'summary.json', dict(scope=manifest['scope'], status='complete-error-report-only', rows=rows,
                                      modules=manifest['modules'], exclusions=manifest['exclusions']))
    state = read(output / 'status.json')
    save(output / 'status.json', dict(state, status='complete-error-report-only', updated_unix=time.time()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--run', action='store_true')
    group.add_argument('--review', action='store_true')
    ap.add_argument('--config', type=Path, default=ROOT / 'config/fft_l_local_core_study.json')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare: prepare(args.config, args.output)
    elif args.review:
        try:
            review(args.output)
            save(args.output / 'cpu_review.json', dict(status='complete', updated_unix=time.time()))
        except BaseException as error:
            save(args.output / 'cpu_review.json', dict(status='failed', error=repr(error), updated_unix=time.time()))
            state = read(args.output / 'status.json')
            save(args.output / 'status.json', dict(state, status='failed-cpu-review', phase='cpu-review',
                                                  retryable=False, error=repr(error), updated_unix=time.time()))
            raise
    else:
        with (args.output / 'driver.lock').open('a') as lock:
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: return 73
            return run(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
