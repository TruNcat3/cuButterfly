#!/usr/bin/env python3
"""Frozen FP32 batched, GPU-resident, complete 2-D linear convolutions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import signal as _scipy_signal
try:
    from scipy import fft, signal
    if not all(hasattr(fft, name) for name in ('rfftn', 'irfftn', 'next_fast_len')):
        raise ImportError('legacy SciPy exposes no scipy.fft namespace')
except (ImportError, AttributeError):
    # The research host also supports an older SciPy where ``from scipy
    # import fft`` resolves to a scalar helper function.  The CPU reference
    # only needs NumPy's deterministic real FFT and a conservative metadata
    # padding value; GPU timing never uses this fallback.
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
    # Keep the convolution oracle available even when this host exposes an
    # older/scalar ``scipy.fft`` namespace.
    signal = _scipy_signal

ROOT = Path(__file__).resolve().parents[3]
SEED = 20261012
DEFAULT_OUT = ROOT / 'results/paper_targeted_20261012/fft_filter'
LIBRARY = ROOT / 'results/search_budget_20261011/frozen_runtime/libcuntt.so'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def same_crop(full, image_shape, filter_shape):
    start = tuple((int(k) - 1) // 2 for k in filter_shape)
    return full[(...,) + tuple(slice(s, s + n) for s, n in zip(start, image_shape[-len(start):]))]


def padding_shape(image_shape, filter_shape):
    return tuple(1 << (int(n + k - 2).bit_length())
                 for n, k in zip(image_shape, filter_shape))


def cpu_convolve(image, kernel, *, forward_norm=False):
    shape = padding_shape(image.shape[-2:], kernel.shape[-2:])
    norm = 'forward' if forward_norm else 'backward'
    spectrum = fft.rfftn(image, s=shape, axes=(-2, -1), norm=norm) * fft.rfftn(kernel, s=shape, axes=(-2, -1), norm=norm)
    if forward_norm:
        spectrum *= np.prod(shape)
    full = fft.irfftn(spectrum, s=shape, axes=(-2, -1), norm=norm)
    return same_crop(full, image.shape, kernel.shape).copy()


def cpu_fp32_convolve(image, kernel, shape):
    """Same-precision CPU oracle; FFT intermediates stay complex64."""
    import torch
    values = torch.from_numpy(np.asarray(image, dtype=np.float32))
    weights = torch.from_numpy(np.asarray(kernel, dtype=np.float32))
    x = torch.fft.rfftn(values, s=shape, dim=(-2, -1))
    w = torch.fft.rfftn(weights, s=shape, dim=(-2, -1))
    assert x.dtype == w.dtype == torch.complex64
    result = torch.fft.irfftn(x * w, s=shape, dim=(-2, -1))
    assert result.dtype == torch.float32
    return same_crop(result, image.shape, kernel.shape).numpy().copy()


def error_distribution(actual, reference):
    a, r = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.shape != r.shape:
        raise ValueError(f'output/reference shape mismatch: {a.shape} != {r.shape}')
    finite = bool(np.isfinite(a).all() and np.isfinite(r).all())
    if not finite:
        return {'finite': False, 'nonfinite_actual_count': int((~np.isfinite(a)).sum()),
                'nonfinite_reference_count': int((~np.isfinite(r)).sum()),
                'absolute_quantiles': None, 'relative_quantiles': None,
                'rms_error': None, 'relative_l2': None}
    diff = np.abs(a - r)
    relative = diff / np.maximum(np.abs(r), np.finfo(np.float64).tiny)
    quantiles = [0.5, 0.9, 0.99, 0.999, 1.0]
    return {'finite': finite, 'absolute_quantiles': dict(zip(map(str, quantiles), map(float, np.quantile(diff, quantiles)))),
            'relative_quantiles': dict(zip(map(str, quantiles), map(float, np.quantile(relative, quantiles)))),
            'rms_error': float(np.sqrt(np.mean((a-r)**2))),
            'relative_l2': float(np.linalg.norm(a-r) / max(np.linalg.norm(r), np.finfo(float).tiny)),
            'reference_rms': float(np.sqrt(np.mean(r*r))), 'reference_abs_max': float(np.max(np.abs(r)))}


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def rotate_methods(methods, process_index):
    """One-based independent process index balances method positions."""
    if process_index < 1 or not methods:
        raise ValueError('positive process index and nonempty methods required')
    offset = (process_index - 1) % len(methods)
    return methods[offset:] + methods[:offset]


def prepare(out, batches=(1,), sizes=(512, 1024, 2048), filters=(31, 127)):
    out.mkdir(parents=True, exist_ok=True)
    contracts = []
    for n, k, batch in ((n, k, b) for n in sizes for k in filters for b in batches):
            name = f'image{n}_filter{k}' + (f'_batch{batch}' if batch != 1 else '')
            artifact = out / f'{name}.npz'
            # Separate deterministic seeds keep contracts stable across subset runs.
            seed = SEED + n * 1000 + k
            rng = np.random.default_rng(seed)
            image = rng.standard_normal((n, n) if batch == 1 else (batch, n, n)).astype(np.float32)
            # Keep the same shared filter across all batches, including B=1.
            if batch != 1:
                rng = np.random.default_rng(seed)
                rng.standard_normal((n, n))
            kernel = rng.standard_normal((k, k)).astype(np.float32) / np.float32(k)
            reference = (signal.fftconvolve(image.astype(np.float64), kernel.astype(np.float64), mode='same')
                         if batch == 1 else np.stack([signal.fftconvolve(item.astype(np.float64), kernel.astype(np.float64), mode='same') for item in image]))
            if artifact.exists():
                with np.load(artifact) as previous:
                    for key, value in [('image', image), ('kernel', kernel), ('reference', reference)]:
                        if not np.array_equal(previous[key], value):
                            raise RuntimeError(f'existing preparation differs: {artifact}: {key}')
            else:
                np.savez(artifact, image=image, kernel=kernel, reference=reference)
            contracts.append({'id': name, 'seed': seed, 'image_shape': [n, n], 'filter_shape': [k, k],
                              'artifact': artifact.name, 'sha256': sha256(artifact),
                              'cubutterfly_padding': list(padding_shape((n,n),(k,k))),
                              'scipy_reference_fast_padding': [fft.next_fast_len(n+k-1, real=True)] * 2,
                              'cpu_fp32_error': error_distribution(cpu_convolve(image, kernel), reference)})
            if batch != 1:
                contracts[-1].update(batch=batch, filter_contract='one shared spatial filter; spectrum recomputed once per full batch call')
            print(f'prepared {name}', flush=True)
    manifest = {'schema': 1, 'application': '2d_linear_convolution_same', 'seed': SEED,
                'input_dtype': 'float32', 'output_dtype': 'float32', 'oracle_dtype': 'float64',
                'oracle': 'scipy.signal.fftconvolve(mode=same)', 'numpy': np.__version__, 'scipy': scipy.__version__,
                'filter_spectrum_cached': False, 'timing_scope': ['padding', 'image_fft', 'filter_fft', 'multiply', 'inverse_fft', 'crop_copy'],
                'excluded': ['host_device_transfer', 'plan_setup', 'compilation', 'search', 'cpu_validation'],
                'tolerance': None, 'accuracy_status': 'distribution_only_no_accepted_tolerance', 'contracts': contracts}
    if (out / 'manifest.json').exists() and json.loads((out / 'manifest.json').read_text()) != manifest:
        raise RuntimeError('preparation changed; use a new output directory')
    write_json(out / 'manifest.json', manifest)
    return manifest


def measure(call, torch, warmup, repeats, samples, *, protocol='cuda_events_regular_execution'):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    timings = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            output = call()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000 / repeats)
    return {'samples_us': timings, 'median_us': statistics.median(timings),
            'protocol': protocol, 'warmup_calls': warmup, 'calls_per_sample': repeats}, output


def make_torch_graph_call(raw_call, image, kernel, torch, *, fixed_filter=False):
    """Capture a complete torch call and replay it with refreshed inputs.

    ``image`` and ``kernel`` are the tensors read by ``raw_call``.  Separate
    source tensors make each replay's device-to-device input copies explicit
    The raw call returns a crop view. One ownership copy after replay makes
    returned outputs independent across calls, as in historical eager mode.
    """
    source_image, source_kernel = image.clone(), kernel.clone()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.current_stream(image.device)
    torch.cuda.synchronize(image.device)
    for _ in range(3):
        raw_call()
    torch.cuda.synchronize(image.device)
    with torch.cuda.graph(graph, stream=stream):
        captured = raw_call()
    torch.cuda.synchronize(image.device)

    def replay():
        image.copy_(source_image)
        kernel.copy_(source_kernel)
        graph.replay()
        return captured.clone()

    def set_inputs(new_image, new_kernel=None):
        if fixed_filter and new_kernel is not None:
            raise ValueError('cached filter is fixed; kernel replacement requires recapture')
        source_image.copy_(new_image)
        if new_kernel is not None:
            source_kernel.copy_(new_kernel)

    replay.set_inputs = set_inputs

    return replay


def make_cupy_graph_call(raw_call, image, kernel, cp, stream, *, fixed_filter=False):
    """CuPy equivalent of :func:`make_torch_graph_call` on the same stream."""
    source_image, source_kernel = image.copy(), kernel.copy()
    # CuPy does not reserve a graph-private allocator pool automatically.
    # Captured temporary addresses must not be reused by input/output copies.
    capture_pool = cp.cuda.MemoryPool()
    stream.synchronize()
    with stream, cp.cuda.using_allocator(capture_pool.malloc):
        for _ in range(3):
            raw_call()
        stream.synchronize()
        stream.begin_capture()
        captured = raw_call()
        graph = stream.end_capture()
    stream.synchronize()

    def replay():
        cp.copyto(image, source_image)
        cp.copyto(kernel, source_kernel)
        graph.launch(stream)
        return captured.copy()

    def set_inputs(new_image, new_kernel=None):
        if fixed_filter and new_kernel is not None:
            raise ValueError('cached filter is fixed; kernel replacement requires recapture')
        cp.copyto(source_image, new_image)
        if new_kernel is not None:
            cp.copyto(source_kernel, new_kernel)

    replay.set_inputs = set_inputs
    replay.capture_pool = capture_pool

    return replay


def owned_call(raw_call, copier):
    def call():
        return copier(raw_call())
    call.graph_raw = raw_call
    return call


def validate_graph_replay(call, original, changed, reference, changed_reference,
                          as_numpy):
    """Verify new data reaches the graph and previous outputs keep their value."""
    call.set_inputs(original)
    first = call()
    first_snapshot = as_numpy(first).copy()
    call.set_inputs(changed)
    second = call()
    changed_actual = as_numpy(second).copy()
    retained = bool(np.array_equal(as_numpy(first), first_snapshot))
    call.set_inputs(original)
    restored = as_numpy(call()).copy()
    errors = [error_distribution(first_snapshot, reference),
              error_distribution(changed_actual, changed_reference)]
    result = dict(distinct_input_outputs=bool(not np.array_equal(first_snapshot, changed_actual)),
                  previous_output_preserved=retained,
                  restored_input_exact=bool(np.array_equal(first_snapshot, restored)),
                  same_precision_cpu_errors=errors)
    if (not all(error['finite'] for error in errors) or not retained
            or not result['restored_input_exact'] or not result['distinct_input_outputs']):
        raise RuntimeError('graph input/output lifetime validation failed: ' + str(result))
    return result


def require_valid_reports(rows):
    """No floating tolerance is inferred; explicit failures must stop a run."""
    failed = [row['method'] for row in rows
              if not row['error']['finite'] or row['accepted'] is False]
    if failed:
        raise RuntimeError('invalid numerical results retained for: ' + ', '.join(failed))


def runtime_contract(prepared, shape, *, cached_filter, execution_mode):
    """Keep preparation provenance separate from the executed call boundary."""
    contract = {key: value for key, value in prepared.items()
                if key not in ('cpu_fp32_error', 'filter_contract')}
    contract.update(batch=int(prepared.get('batch', 1)),
                    cubutterfly_padding=list(shape), execution_mode=execution_mode,
                    filter_spectrum_cached=bool(cached_filter),
                    filter_contract=('one shared fixed spatial filter; spectrum cached outside timed calls'
                                     if cached_filter else
                                     'one shared spatial filter; spectrum recomputed once per full batch call'))
    return contract


def method_call_contract(shape, *, cached_filter, execution_mode):
    included = ['image_padding', 'image_fft', 'spectrum_multiply', 'inverse_fft',
                'crop', 'independent_output_copy']
    if not cached_filter:
        included[1:1] = ['filter_padding', 'filter_fft']
    if execution_mode == 'graph':
        included[0:0] = ['image_device_input_refresh', 'kernel_device_input_refresh']
    return {'fft_padding': list(shape), 'filter_spectrum_cached': bool(cached_filter),
            'execution_mode': execution_mode, 'timing_scope': included,
            'allocation_policy': ('captured temporaries reused; output ownership copy per call'
                                  if execution_mode == 'graph' else
                                  'application allocations and output ownership copy per call'),
            'excluded': ['host_device_transfer', 'plan_setup', 'compilation', 'search',
                         'graph_capture', 'cpu_validation']
                        + (['filter_spectrum_setup'] if cached_filter else [])}


def run(args):
    import torch
    import cupy as cp
    from cupyx.scipy import fft as cp_fft
    from cupyx.scipy.signal import fftconvolve
    sys.path.insert(0, str(ROOT / 'results/e13_assets'))
    from cubutterfly_real_fft import RealFFTAdapter
    manifest = json.loads((args.output / 'manifest.json').read_text())
    device = torch.device('cuda', args.device)
    torch.cuda.set_device(device)
    experiment_stream = None
    if args.execution_mode == 'graph':
        experiment_stream = torch.cuda.Stream(device=device)
        experiment_stream.wait_stream(torch.cuda.current_stream(device))
        torch.cuda.set_stream(experiment_stream)
    props = torch.cuda.get_device_properties(device)
    if 'A100' not in props.name or props.total_memory < 70 * 1024**3:
        raise RuntimeError('primary experiment requires an assigned A100 80GB')
    identity = {'gpu_name': props.name, 'gpu_uuid': 'GPU-' + str(props.uuid).removeprefix('GPU-'), 'memory_bytes': props.total_memory,
                'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'cupy': cp.__version__,
                'library': str(args.library), 'library_sha256': sha256(args.library),
                'runner_sha256': sha256(__file__), 'adapter_sha256': sha256(ROOT / 'results/e13_assets/cubutterfly_real_fft.py'),
                'compile_mode': os.environ.get('CUBUTTERFLY_COMPILE_MODE', 'research'),
                'algorithm_policy': os.environ.get('CUBUTTERFLY_E13_ALGORITHM_POLICY', 'default'),
                'python': platform.python_version()}
    if args.calibration_identity:
        identity['host_rank1_calibration_reference'] = {
            'path': str(args.calibration_identity.resolve()), 'sha256': sha256(args.calibration_identity),
            'usage': 'host provenance only; public native xD default runtime selector does not consume rank1 calibration'}
    requested = {'warmup': args.warmup, 'repeats': args.repeats, 'samples': args.samples,
                 'torch_control': args.torch_control, 'atol': args.atol, 'rtol': args.rtol,
                 'process_index': args.process_index, 'plan_only': args.plan_only,
                 'padding_policy': args.padding_policy,
                 'cache_filter_spectrum': args.cache_filter_spectrum,
                 'require_cache': args.require_cache,
                 'execution_mode': args.execution_mode}
    for contract in manifest['contracts']:
        if args.contract and contract['id'] not in args.contract:
            continue
        report_dir = Path(os.environ.get('XD_REPORT_DIR', str(args.output)))
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{contract['id']}.{args.run_id}.json"
        shape = tuple(contract['cubutterfly_padding'])
        if args.padding_policy == 'fast':
            shape = tuple(cp_fft.next_fast_len(int(n + k - 1), real=True)
                          for n, k in zip(contract['image_shape'], contract['filter_shape']))
        executed_contract = runtime_contract(contract, shape,
                                             cached_filter=args.cache_filter_spectrum,
                                             execution_mode=args.execution_mode)
        if report_path.exists():
            previous = json.loads(report_path.read_text())
            if (previous['identity'] != identity or previous['requested'] != requested
                    or previous['contract'] != executed_contract
                    or previous.get('preparation_contract') != contract):
                raise RuntimeError(f'incompatible existing report; choose a new --run-id: {report_path}')
            require_valid_reports(previous['rows'])
            print(f"resumed completed {contract['id']}", flush=True)
            continue
        artifact = args.output / contract['artifact']
        if sha256(artifact) != contract['sha256']:
            raise RuntimeError('prepared artifact hash mismatch')
        with np.load(artifact) as data:
            image_cpu, kernel_cpu = data['image'].copy(), data['kernel'].copy()
            image = torch.from_numpy(image_cpu).to(device)
            kernel = torch.from_numpy(kernel_cpu).to(device)
        # Historical input archives retain their FP64 diagnostic arrays. New
        # validation uses a same-precision CPU result, outside GPU timing.
        reference = cpu_fp32_convolve(image_cpu, kernel_cpu, shape)
        changed_cpu = np.roll(image_cpu, shift=(13, -7), axis=(-2, -1)).copy()
        changed_reference = (cpu_fp32_convolve(changed_cpu, kernel_cpu, shape)
                             if args.execution_mode == 'graph' else None)
        setup_start = time.perf_counter()
        batch = int(contract.get('batch', 1))
        reuse_fft_buffers = bool(args.cache_filter_spectrum)
        adapter = RealFFTAdapter(args.library, shape, batch, device,
                                 reuse_buffers=reuse_fft_buffers, native_real=True)
        # A cached B=1 filter must have its own output buffer. Otherwise a
        # subsequent image transform would overwrite the retained spectrum.
        filter_adapter = (adapter if batch == 1 and not reuse_fft_buffers else
                          RealFFTAdapter(args.library, shape, 1, device,
                                         reuse_buffers=reuse_fft_buffers, native_real=True))
        torch.cuda.synchronize()
        setup_seconds = time.perf_counter() - setup_start
        plans = adapter.plan_metadata()
        filter_plans = filter_adapter.plan_metadata()
        if args.require_cache and any(p['selection_source'] != 'user_cache' for p in plans + filter_plans):
            raise RuntimeError('measurement requested tuned-cache replay but a plan did not use user_cache')
        def own():
            padded_image = torch.nn.functional.pad(image, (0,shape[1]-image.shape[-1],0,shape[0]-image.shape[-2]))
            padded_kernel = torch.nn.functional.pad(kernel, (0,shape[1]-kernel.shape[1],0,shape[0]-kernel.shape[0]))
            image_fft = adapter.rfftn(padded_image)
            product = image_fft * filter_adapter.rfftn(padded_kernel)
            product *= shape[0] * shape[1]
            return same_crop(adapter.irfftn(product), image.shape, kernel.shape)
        cached_filter = None
        if args.cache_filter_spectrum:
            padded_kernel = torch.nn.functional.pad(
                kernel, (0, shape[1] - kernel.shape[1], 0, shape[0] - kernel.shape[0]))
            # Filter setup is deliberately outside the timed complete call.
            # The tensor remains alive for every subsequent application
            # invocation, matching a fixed convolution filter in deployment.
            cached_filter = filter_adapter.rfftn(padded_kernel)
            # Both plans use forward normalization. Fold its compensation
            # into the invariant filter, once, rather than traversing every
            # batch spectrum for an extra product scaling kernel. Power-of-two
            # padding moves an exact binary exponent factor. For fast padding
            # this changes rounding order; complete output errors are reported
            # against the same-precision CPU oracle without relaxing a gate.
            cached_filter.mul_(shape[0] * shape[1])

            def own_cached():
                padded_image = torch.nn.functional.pad(
                    image, (0, shape[1] - image.shape[-1], 0, shape[0] - image.shape[-2]))
                product = adapter.rfftn(padded_image) * cached_filter
                return same_crop(adapter.irfftn(product), image.shape, kernel.shape)

            own_call = own_cached
        else:
            own_call = own
        stream = cp.cuda.ExternalStream(torch.cuda.current_stream(device).cuda_stream)
        with cp.cuda.Device(args.device), stream:
            cp_image, cp_kernel = cp.from_dlpack(image), cp.from_dlpack(kernel)
            def baseline():
                return fftconvolve(cp_image, cp_kernel if batch == 1 else cp_kernel[None], mode='same', axes=(-2, -1))
            if args.cache_filter_spectrum:
                cp_shape = tuple(cp_fft.next_fast_len(int(n + k - 1), real=True)
                                 for n, k in zip(image.shape[-2:], kernel.shape))
                cp_kernel_padded = cp.pad(cp_kernel, ((0, cp_shape[0] - kernel.shape[0]),
                                                       (0, cp_shape[1] - kernel.shape[1])))
                cp_cached_filter = cp_fft.rfftn(cp_kernel_padded, s=cp_shape, axes=(-2, -1))

                def baseline_cached():
                    cp_padded = cp.pad(cp_image, ((0, 0),
                                                   (0, cp_shape[0] - image.shape[-2]),
                                                   (0, cp_shape[1] - image.shape[-1]))) if batch != 1 else cp.pad(
                        cp_image, ((0, cp_shape[0] - image.shape[-2]),
                                   (0, cp_shape[1] - image.shape[-1])))
                    cp_product = cp_fft.rfftn(cp_padded, s=cp_shape, axes=(-2, -1)) * cp_cached_filter
                    return same_crop(cp_fft.irfftn(cp_product, s=cp_shape, axes=(-2, -1)),
                                     image.shape, kernel.shape)

                baseline_call = baseline_cached
            else:
                baseline_call = baseline
            methods = [('cubutterfly', owned_call(own_call, lambda x: x.clone())),
                       ('cupy_fftconvolve_native_padding', owned_call(baseline_call, lambda x: x.copy()))]
            if args.torch_control:
                def control():
                    x = torch.nn.functional.pad(image, (0,shape[1]-image.shape[-1],0,shape[0]-image.shape[-2]))
                    h = torch.nn.functional.pad(kernel, (0,shape[1]-kernel.shape[1],0,shape[0]-kernel.shape[0]))
                    full = torch.fft.irfftn(torch.fft.rfftn(x, dim=(-2,-1))*torch.fft.rfftn(h, dim=(-2,-1)), s=shape, dim=(-2,-1))
                    return same_crop(full, image.shape, kernel.shape)
                if args.cache_filter_spectrum:
                    torch_cached_filter = torch.fft.rfftn(padded_kernel, dim=(-2, -1))

                    def control_cached():
                        x = torch.nn.functional.pad(
                            image, (0, shape[1] - image.shape[-1], 0, shape[0] - image.shape[-2]))
                        full = torch.fft.irfftn(
                            torch.fft.rfftn(x, dim=(-2, -1)) * torch_cached_filter,
                            s=shape, dim=(-2, -1))
                        return same_crop(full, image.shape, kernel.shape)

                    control_call = control_cached
                else:
                    control_call = control
                methods.append(('torch_cufft_same_padding_diagnostic',
                                owned_call(control_call, lambda x: x.clone())))
            methods = rotate_methods(methods, args.process_index)
            rows = []
            for name, call in methods:
                graph_capture_seconds = None
                graph_validation = None
                if args.execution_mode == 'graph':
                    graph_start = time.perf_counter()
                    raw_call = getattr(call, 'graph_raw', call)
                    if name.startswith('cupy_'):
                        call = make_cupy_graph_call(raw_call, cp_image, cp_kernel, cp, stream,
                                                    fixed_filter=args.cache_filter_spectrum)
                    else:
                        call = make_torch_graph_call(raw_call, image, kernel, torch,
                                                     fixed_filter=args.cache_filter_spectrum)
                    graph_capture_seconds = time.perf_counter() - graph_start
                    if name.startswith('cupy_'):
                        graph_validation = validate_graph_replay(
                            call, cp.asarray(image_cpu), cp.asarray(changed_cpu),
                            reference, changed_reference, cp.asnumpy)
                    else:
                        graph_validation = validate_graph_replay(
                            call, torch.from_numpy(image_cpu).to(device),
                            torch.from_numpy(changed_cpu).to(device), reference,
                            changed_reference, lambda x: x.detach().cpu().numpy())
                init_start = time.perf_counter()
                first = call()
                torch.cuda.synchronize()
                initialization_seconds = time.perf_counter()-init_start
                timing, output = ((None, first) if args.plan_only else
                                  measure(call, torch, args.warmup, args.repeats, args.samples,
                                          protocol=('cuda_events_cuda_graph_replay'
                                                    if args.execution_mode == 'graph'
                                                    else 'cuda_events_regular_execution')))
                actual = cp.asnumpy(output) if isinstance(output, cp.ndarray) else output.detach().cpu().numpy()
                error = error_distribution(actual, reference)
                accepted = None
                if args.atol is not None and args.rtol is not None:
                    accepted = bool(error['finite'] and np.allclose(actual, reference, atol=args.atol, rtol=args.rtol))
                method_shape = (tuple(cp_fft.next_fast_len(int(n + k - 1), real=True)
                                      for n, k in zip(image.shape[-2:], kernel.shape))
                                if name.startswith('cupy_') else shape)
                rows.append({'method': name, 'timing': timing, 'error': error, 'accepted': accepted,
                             'call_contract': method_call_contract(
                                 method_shape, cached_filter=args.cache_filter_spectrum,
                                 execution_mode=args.execution_mode),
                             'batch': batch, 'amortized_us_per_image': timing['median_us'] / batch if timing else None,
                             'images_per_second': batch * 1e6 / timing['median_us'] if timing else None,
                             'pixels_per_second': batch * int(np.prod(contract['image_shape'])) * 1e6 / timing['median_us'] if timing else None,
                             'first_call_initialization_seconds': initialization_seconds,
                             'graph_capture_setup_seconds': graph_capture_seconds,
                             'graph_changing_input_validation': graph_validation,
                             'numerical_acceptance': ('unassessed_error_report_only' if accepted is None
                                                      else 'passed' if accepted else 'failed'),
                             'timing_status': 'measured_complete_call' if timing else 'not_measured',
                             'formal_performance_eligible': accepted is True and timing is not None})
            report = {'schema': 'fft-filter-complete-call-v2',
                      'contract': executed_contract, 'preparation_contract': contract,
                      'identity': identity, 'complete_filter_call': True,
                      'validation_reference': {'backend': 'torch.fft CPU', 'dtype': 'float32',
                                               'fft_dtype': 'complex64', 'padding': list(shape),
                                               'precision_promotion': False,
                                               'archived_fp64_reference_used': False},
                      'requested': requested,
                      'actual_cubutterfly_padding': list(shape),
                      'method_order': [name for name, _ in methods],
                      'cupy_native_fft_padding': [cp_fft.next_fast_len(int(n+k-1), real=True)
                                                  for n,k in zip(image.shape[-2:],kernel.shape)],
                      'filter_spectrum_cached': bool(args.cache_filter_spectrum),
                      'execution_mode': args.execution_mode,
                      'graph_capture_outside_timing': args.execution_mode == 'graph',
                      'graph_call_contract': ({'input_device_copy_included': True,
                                               'independent_output_copy_included': True,
                                               'capture_stream': 'non-default'}
                                              if args.execution_mode == 'graph' else None),
                      'cached_filter_normalization_compensation': bool(args.cache_filter_spectrum),
                      'cubutterfly_plan_setup_including_selection_seconds': setup_seconds,
                      'selection_search_separate_from_timing': True, 'plan_metadata': plans,
                      'filter_plan_metadata': filter_plans,
                      'tolerance': {'atol': args.atol, 'rtol': args.rtol}, 'rows': rows}
            write_json(report_path, report)
            print(json.dumps({'contract': contract['id'], 'rows': rows}), flush=True)
            require_valid_reports(rows)
        adapter.close()
        if filter_adapter is not adapter:
            filter_adapter.close()


def main():
    os.environ.setdefault('CUBUTTERFLY_COMPILE_MODE', 'research')
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--library', type=Path, default=LIBRARY)
    parser.add_argument('--calibration-identity', type=Path,
                        help='record host calibration provenance, not an input to public xD selector')
    parser.add_argument('--device', type=int, default=0, help='local index within assigned CUDA_VISIBLE_DEVICES')
    parser.add_argument('--contract', action='append')
    parser.add_argument('--run-id', default='process1')
    parser.add_argument('--process-index', type=int, default=1,
                        help='one-based independent process index for method order rotation')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--samples', type=int, default=7)
    parser.add_argument('--torch-control', action='store_true')
    parser.add_argument('--padding-policy', choices=('power2', 'fast'), default='power2',
                        help='Explicit application padding; fast uses the same real next_fast_len as CuPy')
    parser.add_argument('--plan-only', action='store_true', help='Tune/create plans and report complete output errors, without application timing')
    parser.add_argument('--require-cache', action='store_true', help='Fail if public DEFAULT did not replay a prepared plan cache')
    parser.add_argument('--cache-filter-spectrum', action='store_true',
                        help='Precompute the fixed filter spectrum outside timed application calls')
    parser.add_argument('--execution-mode', choices=('eager', 'graph'), default='eager',
                        help='Complete-call execution mode; graph captures each backend call before timing')
    parser.add_argument('--batches', type=int, nargs='+', default=[1], help='Preparation only; true independent images per batch')
    parser.add_argument('--sizes', type=int, nargs='+', default=[512,1024,2048], help='Preparation image edges')
    parser.add_argument('--filters', type=int, nargs='+', default=[31,127], help='Preparation filter edges')
    parser.add_argument('--atol', type=float)
    parser.add_argument('--rtol', type=float)
    args = parser.parse_args()
    if (args.atol is None) != (args.rtol is None):
        parser.error('explicit accuracy acceptance requires both --atol and --rtol')
    if min(args.warmup, args.repeats, args.samples) <= 0:
        parser.error('warmup, repeats, samples must be positive')
    if args.process_index <= 0:
        parser.error('process-index must be positive')
    if min(args.batches + args.sizes + args.filters) <= 0:
        parser.error('batches, sizes and filters must be positive')
    prepare(args.output, args.batches, args.sizes, args.filters) if args.prepare else run(args)


if __name__ == '__main__':
    main()
