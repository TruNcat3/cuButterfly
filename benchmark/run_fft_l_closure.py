"""Finite source-bound FFT-L study; preparation/import never initializes CUDA.

The 32 raw requests retain explicit storage and compute semantics. Sixteen native
requests may compare with vendor-native cuFFT at storage/transform level only;
vendor internal arithmetic is unspecified. Strict-arithmetic speedups stay null.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import cufft_low_precision_baseline as vendor
import run_semantic_campaign as semantic
import semantic_cpu_pipeline as cpu
from semantic_campaign_manifest import validate_manifest_hash

DEFAULT = ROOT / 'results/a100_closure_20261102/fft_l'
VALID = {'pass', 'accuracy-review-required'}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def vendor_contract(s):
    if s['compute'] != 0:
        return dict(supported=False, reason='cuFFT vendor-native arithmetic cannot honor explicitly requested FP32 arithmetic',
                    strict_arithmetic_match=False, strict_arithmetic_speedup=None)
    capability = vendor.support(s['shape'], s['batch'],
        precision='fp16' if s['storage'] == 4 else 'bf16',
        direction='inverse' if s['inverse'] else 'forward', normalization='none', sm=80)
    scale = 1 / math.prod(s['shape']) if s['inverse'] and s['normalize_inverse'] else 1.0
    capability.update(normalization_adapter_scale=scale * s['fft_output_scale'],
        complete_normalization='inverse' if s['inverse'] and s['normalize_inverse'] else 'none',
        comparison='matching storage, transform, direction and normalization; vendor-native internal arithmetic unspecified',
        strict_arithmetic_match=False, strict_arithmetic_speedup=None)
    return capability


def load_prepared(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'raw_manifest.json').read_text())
    if not validate_manifest_hash(manifest):
        raise ValueError('FFT-L semantic manifest checksum mismatch')
    report = json.loads((directory / 'preparation.json').read_text())
    if report['manifest_sha256'] != manifest['manifest_sha256']:
        raise ValueError('FFT-L preparation manifest mismatch')
    raw = {row['semantic_id']: row for row in report['rows'] if
           row.get('input_profile', {}).get('profile') == 'raw-stress'}
    if len(manifest['cases']) != 32 or len(raw) != 32:
        raise ValueError('expected the existing 32-request raw FFT-L population')
    artifacts = {}
    for case in manifest['cases']:
        s = case['descriptor']['semantics']
        if (s['operator'] != 0 or s['fft_type'] != 0 or s['storage'] not in (4, 5) or
                s['compute'] not in (0, 1) or s['shape'] not in ([4096], [64, 64]) or
                s['batch'] not in (1, 4) or s['placement'] or s['length_mode'] or
                s['input_orders'] != [0] * len(s['shape']) or
                s['output_orders'] != [0] * len(s['shape'])):
            raise ValueError('request outside the finite FFT-L contract')
        if case['id'] != semantic.digest(s) or raw[case['id']]['semantics'] != s:
            raise ValueError('request semantic identity mismatch')
        path = directory / raw[case['id']]['artifact']['path']
        if path.parent.resolve() != directory.resolve() or semantic.sha256_file(path) != raw[case['id']]['artifact']['sha256']:
            raise ValueError('FFT-L artifact identity mismatch')
        with np.load(path, allow_pickle=False) as arrays:
            source, expected, storage = (arrays[key].copy() for key in ('source', 'reference', 'storage'))
        if not np.array_equal(source, semantic.decode_storage(storage, s['storage'])):
            raise ValueError('prepared input words changed')
        artifacts[case['id']] = dict(case_id=case['id'], s=s, source=source, expected=expected,
            encoded_source=storage, input_indices=semantic.layout_indices(s, 'input'),
            output_indices=semantic.layout_indices(s, 'output'),
            artifact_sha256=raw[case['id']]['artifact']['sha256'],
            source_sha256=hashlib.sha256(storage.tobytes()).hexdigest())
    return manifest, artifacts


def drive_steps(generator):
    """Consume the existing GPU snapshots using its unchanged CPU verifier."""
    value = None
    while True:
        try:
            payload = generator.send(value)
        except StopIteration as done:
            return done.value
        if 'batch_snapshots' in payload:
            value = {'batch_results': [cpu.validate_snapshot(item) for item in payload['batch_snapshots']]}
        else:
            value = cpu.validate_snapshot(payload)


def reference_contract():
    return dict(same_precision_cpu_oracle='available-declared-arithmetic',
        implementation='scripts/low_fft_cpu_reference.py',
        arithmetic='native: storage RNE after each primitive; FP32: unfused FP32 products and sums with source-level low-storage boundaries',
        equivalence='declared same-precision CPU algorithm; GPU contraction may differ; vendor internals unknown',
        execution='CPU sidecar finalization after GPU output retention',
        reported_reference='full complex128 mathematical FFT from exact quantized input',
        secondary_reference='mathematical output rounded once to requested storage; diagnostic only',
        acceptance_threshold=None)


def finalize_cpu_reference(prepared_directory, report_path):
    """Review retained outputs on CPU; never rewrite original measurement evidence."""
    from benchmark.review_fft_l_cpu_reference import review
    report_path = Path(report_path)
    reference_source = ROOT / 'scripts/low_fft_cpu_reference.py'
    sidecar = report_path.parent / 'cpu_reference' / str(time.time_ns()) / 'review.json'
    result = review(prepared_directory, report_path, sidecar.parent / 'errors')
    result['reference_contract'] = reference_contract()
    result['reference_source_sha256'] = semantic.sha256_file(reference_source)
    result['review_source_sha256'] = semantic.sha256_file(ROOT / 'benchmark/review_fft_l_cpu_reference.py')
    result['summary']['same_precision_cpu_oracle'] = ('available-declared-arithmetic'
        if result['summary']['references_computed'] == len(result['rows']) else 'partial-mapping-gap')
    write(sidecar, result)
    return sidecar



def quantized_diagnostic(row, prepared, errors, backend):
    artifact = row.get('error_artifact')
    if not artifact:
        return
    with np.load(artifact['path'], allow_pickle=False) as arrays:
        actual = arrays['actual'].copy()
    rounded = semantic.decode_storage(semantic.encode_storage(prepared['expected'], prepared['s']['storage']), prepared['s']['storage'])
    row['quantized_mathematical_reference_artifact'] = semantic.retain_error_artifact(
        errors, prepared['case_id'], backend + '-quantized-mathematical', actual, rounded.reshape(-1))


def run_vendor(case, prepared, torch, stream, library, errors):
    s = prepared['s']
    contract = vendor_contract(s)
    row = dict(name='cufft-xt-low-storage', support_metadata=contract,
        oracle=reference_contract(), strict_arithmetic_speedup=None)
    if not contract['supported']:
        row.update(status='unsupported-semantic', reason=contract['reason'])
        return row
    plan = None
    try:
        initial = prepared['encoded_source'].view(np.uint8).reshape(-1).copy()
        raw_in = np.full(initial.size + 128, 0xA5, dtype=np.uint8)
        raw_in[64:-64] = initial
        gpu_in_full = torch.from_numpy(raw_in).to('cuda')
        gpu_in = gpu_in_full[64:-64]
        gpu_out_full = torch.full_like(gpu_in_full, 0xA5)
        gpu_out = gpu_out_full[64:-64]
        output_initial = np.full(initial.size, 0xA5, dtype=np.uint8)
        plan = vendor.Plan(library, s['shape'], s['batch'], stream=stream.cuda_stream,
            precision=contract['storage_precision'], direction=contract['direction'])
        row['vendor_info'] = plan.info
        from cpu_fftw_baseline import _identity as loaded_symbol_identity
        row['vendor_runtime_library'] = loaded_symbol_identity(plan.lib, 'cufftXtExec', None)
        components = gpu_out.view(torch.float16 if s['storage'] == 4 else torch.bfloat16)
        factor = contract['normalization_adapter_scale']

        def execute():
            plan.execute(gpu_in.data_ptr(), gpu_out.data_ptr(), stream=stream.cuda_stream)
            if factor != 1:
                components.mul_(factor)

        def verify():
            stream.synchronize()
            raw = gpu_out_full.cpu().numpy()
            if not np.all(raw[:64] == 0xA5) or not np.all(raw[-64:] == 0xA5):
                raise AssertionError('vendor output guard bytes modified')
            if not np.array_equal(gpu_in_full.cpu().numpy(), raw_in):
                raise AssertionError('vendor out-of-place input/guard bytes modified')
            result = cpu.validate_snapshot(dict(s=s, expected=prepared['expected'],
                output_raw=raw[64:-64].copy(), output_initial=output_initial,
                detailed='correctness' not in row, float_verdict='error-report',
                error_dir=errors, case_id=case['id'], backend='cufft-xt-low-storage'))
            if 'correctness' not in row:
                row.update({key: value for key, value in result.items() if key not in ('ok', 'status')})
            row.setdefault('validation_statuses', []).append(result['status'])
            if not result['ok']:
                raise AssertionError(result.get('error', 'vendor verification failed'))
            return result['correctness']

        execute()
        row['correctness'] = verify()
        row['timing'] = semantic.time_execute(torch, stream, execute, lambda: None, False,
                                             verify, lambda: gpu_out.fill_(0xA5))
        row.update(status='accuracy-review-required', category='native' if factor == 1 else 'normalization-adapter',
            included_operations=['complete cufftXtExec'] + ([] if factor == 1 else
                ['GPU multiply real/imaginary low-storage components by ' + repr(factor)]))
    except AssertionError as exc:
        row.update(status='correctness-failure', error=str(exc))
    except Exception as exc:
        row.update(status='baseline-error', error=str(exc), exception_type=type(exc).__name__)
    finally:
        if plan is not None:
            stream.synchronize()
            plan.close()
    quantized_diagnostic(row, prepared, errors, 'cufft-xt-low-storage')
    return row


def export_table(report, output):
    rows = []
    for row in report['rows']:
        s, baseline = row['descriptor']['semantics'], row['low_storage_baseline']
        own_ms = row.get('timing', {}).get('median_ms') if row['status'] in VALID else None
        vendor_ms = baseline.get('timing', {}).get('median_ms') if baseline['status'] in VALID else None
        rows.append(dict(case_id=row['id'], storage='fp16' if s['storage'] == 4 else 'bf16',
            compute='native' if s['compute'] == 0 else 'fp32', shape='x'.join(map(str, s['shape'])),
            batch=s['batch'], inverse=s['inverse'], normalization=s['normalize_inverse'],
            cubutterfly_status=row['status'], cubutterfly_ms=own_ms,
            cufft_status=baseline['status'], cufft_ms=vendor_ms,
            storage_semantic_speedup=vendor_ms/own_ms if own_ms and vendor_ms else None,
            strict_arithmetic_speedup=None, same_precision_cpu_oracle='available-declared-arithmetic',
            reference='complex128 mathematical diagnostic; no numerical acceptance threshold',
            mapping=json.dumps(row.get('mapping', {}), sort_keys=True)))
    if rows:
        with (output / 'comparison.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def finalize_report(report, case_ids):
    """Preserve execution failures independently of pending work/support gaps."""
    own_gaps = {'missing-lowering', 'unsupported', 'unsupported-semantic', 'runner-gap', 'validation-unavailable'}
    baseline_gaps = {'unsupported-semantic', 'missing-external-dependency', 'missing-baseline', 'not-applicable'}
    errors = []
    for row in report['rows']:
        for side, observation, gaps in [('cubutterfly', row, own_gaps),
                ('cufft-xt-low-storage', row['low_storage_baseline'], baseline_gaps)]:
            status = observation.get('status')
            if status not in VALID | gaps or status in VALID and not observation.get('timing'):
                errors.append(dict(case_id=row['id'], backend=side, status=status,
                    reason=observation.get('error', observation.get('reason', 'missing measured timing or unknown terminal status'))))
    remaining = sorted(set(case_ids) - {row['id'] for row in report['rows']})
    report['remaining_case_ids'] = remaining
    report['errors'] = errors
    report['summary'] = dict(completed=len(report['rows']), remaining=len(remaining),
        cubutterfly_measured=sum(row['status'] in VALID and bool(row.get('timing')) for row in report['rows']),
        vendor_measured=sum(row['low_storage_baseline']['status'] in VALID and bool(row['low_storage_baseline'].get('timing')) for row in report['rows']),
        storage_semantic_pairs=sum(row['status'] in VALID and row['low_storage_baseline']['status'] in VALID
                                   and bool(row.get('timing')) and bool(row['low_storage_baseline'].get('timing')) for row in report['rows']),
        cubutterfly_support_gaps=sum(row['status'] in own_gaps for row in report['rows']),
        vendor_support_gaps=sum(row['low_storage_baseline']['status'] in baseline_gaps for row in report['rows']),
        execution_error_count=len(errors), strict_arithmetic_pairs=0, same_precision_cpu_oracle='available-declared-arithmetic')
    if errors:
        report['status'] = 'incomplete-with-errors' if remaining else 'complete-with-errors'
        return 2
    if remaining:
        report['status'] = 'cpu-compilation-pending'
        return 75
    report['status'] = 'complete-with-reported-gaps'
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=DEFAULT / 'prepared')
    parser.add_argument('--output', type=Path, default=DEFAULT / 'measurement')
    parser.add_argument('--library', type=Path, default=ROOT / 'build-e13-cufftdx/libcuntt.so')
    parser.add_argument('--vendor-library', type=Path, default=ROOT / 'build/cufft_low_precision_sm80/libcufft_low_precision_baseline.so')
    parser.add_argument('--calibration-manifest', type=Path, default=ROOT / 'results/full_current_20261025/calibration/finalized/calibration_manifest.json')
    parser.add_argument('--compile-broker', type=Path, default=ROOT / 'results/full_current_20261025/compile')
    parser.add_argument('--gpu-uuid', default='GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02')
    parser.add_argument('--finalize-cpu', action='store_true', help='Review OUTPUT/results.json into a new CPU sidecar without CUDA or changing original evidence')
    parser.add_argument('--execute', action='store_true', help='Only this flag initializes CUDA; execute after the main GPU task finishes')
    args = parser.parse_args(argv)
    manifest, prepared = load_prepared(args.prepared)
    if args.finalize_cpu:
        print(finalize_cpu_reference(args.prepared, args.output / 'results.json'))
        return 0
    semantic.validate_fingerprints(manifest, args.library)
    args.output.mkdir(parents=True, exist_ok=True)
    readiness = dict(status='cpu-prepared-gpu-pending', requests=32,
        vendor_storage_semantic_eligible=sum(vendor_contract(c['descriptor']['semantics'])['supported'] for c in manifest['cases']),
        strict_arithmetic_comparisons=0, oracle=reference_contract(),
        input_profiles_selected=['raw-stress'], scaled_controls_executed=False,
        gpu_initialization=False, gpu_execution=False)
    write(args.output / 'readiness.json', readiness)
    if not args.execute:
        calibration = json.loads(args.calibration_manifest.read_text())
        launch = dict(argv=[sys.executable, str(Path(__file__).resolve()), '--execute',
            '--prepared', str(args.prepared.resolve()), '--output', str(args.output.resolve()),
            '--library', str(args.library.resolve()), '--vendor-library', str(args.vendor_library.resolve()),
            '--calibration-manifest', str(args.calibration_manifest.resolve()),
            '--compile-broker', str(args.compile_broker.resolve()), '--gpu-uuid', args.gpu_uuid],
            environment=dict(LD_LIBRARY_PATH=calibration['measurement_identity']['jit_runtime']['ld_library_path'],
                OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                CUBUTTERFLY_SEARCH_BUDGET='512'),
            cwd=str(ROOT), gpu_execution_started=False,
            prerequisite='main composed GPU campaign stopped/finished and target GPU exclusively available',
            compilation_pending_exit_code=75, prepared_requests=32)
        write(args.output / 'launch.json', launch)
        print(json.dumps(readiness, indent=2))
        return 0
    semantic.exclusive_gpu(args.gpu_uuid)
    calibration = json.loads(args.calibration_manifest.read_text())
    environment = semantic.bind_calibration_environment(calibration)
    import semantic_compile_broker as broker
    document = broker._document(args.compile_broker)
    from jit_calibration_identity import validate_timing
    validate_timing(calibration, document)
    config = document['config']
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUBUTTERFLY_PREPARE_BROKER=str(args.compile_broker.resolve()),
        CUBUTTERFLY_JIT_PYTHON=str(ROOT / 'scripts/semantic_resource_compile_proxy.py'),
        CUBUTTERFLY_WORKER_MAX_RSS_BYTES=str(64 * 1024**3),
        CUBUTTERFLY_TEMPLATE_ROOT=config['root'], CUBUTTERFLY_MATHDX_ROOT=config['mathdx'],
        CUBUTTERFLY_NVCC=config['nvcc'], CUBUTTERFLY_JIT_CACHE=config['cache'])
    import torch
    import cubutterfly as cb
    import semantic_gpu_pipeline as gpu
    properties = torch.cuda.get_device_properties(0)
    if not semantic.same_gpu_uuid(args.gpu_uuid, str(getattr(properties, 'uuid', ''))):
        raise ValueError('GPU UUID mismatch')
    identity = dict(manifest_sha256=semantic.digest(manifest), library_sha256=semantic.sha256_file(args.library),
        gpu_uuid=args.gpu_uuid, gpu_name=properties.name, gpu_total_memory=properties.total_memory,
        compute_capability=f'{properties.major}.{properties.minor}', **environment,
        runner_sha256=semantic.sha256_file(__file__), torch=torch.__version__, torch_cuda=torch.version.cuda,
        search_budget=os.environ.get('CUBUTTERFLY_SEARCH_BUDGET', 'unbounded'),
        vendor_library_sha256=semantic.sha256_file(args.vendor_library),
        preparation_sha256=semantic.sha256_file(args.prepared / 'preparation.json'),
        calibration_sha256=semantic.sha256_file(args.calibration_manifest),
        shared_sources={name: semantic.sha256_file(ROOT / 'scripts' / name) for name in
            ('run_semantic_campaign.py', 'semantic_gpu_pipeline.py', 'semantic_cpu_pipeline.py',
             'cufft_low_precision_baseline.py', 'cpu_fftw_baseline.py')})
    limitations = semantic.validate_calibration(calibration, identity)
    path = args.output / 'results.json'
    report = dict(schema='cubutterfly-fft-l-closure-v1', identity=identity, rows=[], attempts=[],
        readiness=readiness, calibration_limitations=limitations, status='running')
    if path.exists():
        report = json.loads(path.read_text())
        if report['identity'] != identity:
            raise ValueError('FFT-L resume identity changed; use fresh output')
    completed = {row['id'] for row in report['rows']}
    library = cb.Library(args.library)
    stream = torch.cuda.Stream()
    errors = args.output / 'errors'
    with torch.cuda.stream(stream):
        for case in manifest['cases']:
            if case['id'] in completed:
                continue
            semantic.exclusive_gpu(args.gpu_uuid)
            events = args.output / 'compile-events' / (case['id'] + '-' + str(time.time_ns()) + '.jsonl')
            events.parent.mkdir(parents=True, exist_ok=True)
            os.environ['CUBUTTERFLY_PREPARE_EVENTS'] = str(events.resolve())
            options = SimpleNamespace(timing=True, float_verdict='error-report', error_dir=errors,
                                      events_path=events, vkfft_library=None, gpuntt_library=None)
            try:
                row = drive_steps(gpu.execution_steps(case, prepared[case['id']], library, torch, stream, options))
            except gpu.PreparationPending as exc:
                report['attempts'].append(dict(case_id=case['id'], status='cpu-compilation-pending', request_ids=exc.request_ids))
                report['status'] = 'cpu-compilation-pending'
                write(path, report)
                continue
            row['oracle'] = reference_contract()
            row['source_sha256'] = prepared[case['id']]['source_sha256']
            row['input_profile'] = 'raw-stress'
            quantized_diagnostic(row, prepared[case['id']], errors, 'cubutterfly')
            row['low_storage_baseline'] = run_vendor(case, prepared[case['id']], torch, stream,
                                                    args.vendor_library, errors)
            report['rows'].append(row)
            report['updated_unix'] = time.time()
            write(path, report)
            export_table(report, args.output)
            print(json.dumps(dict(case_id=case['id'], status=row['status'],
                baseline_status=row['low_storage_baseline']['status'], completed=len(report['rows']), total=32)), flush=True)
    exit_code = finalize_report(report, [case['id'] for case in manifest['cases']])
    write(path, report)
    sidecar = finalize_cpu_reference(args.prepared, path)
    print(json.dumps(dict(cpu_reference_sidecar=str(sidecar), original_results=str(path))), flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
