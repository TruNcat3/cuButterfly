#!/usr/bin/env python3
"""Eight QuaRot FWHT subcontracts; CUDA Graph diagnostics, never 7B E2E."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('dao', 'fp32_composed', 'public_fp32_final')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cases(path):
    cases = json.loads(Path(path).read_text())['cases']
    expected = {(b, phase, n) for b in (1, 4) for phase in ('prefill', 'decode') for n in (32, 64)}
    actual = {(c['application_batch'], c['phase'], c['N']) for c in cases}
    if len(cases) != 8 or actual != expected:
        raise ValueError('cases must contain exactly the eight recorded QuaRot contracts')
    for c in cases:
        b, n = c['application_batch'], c['N']
        seq = 128 if c['phase'] == 'prefill' else 1
        if c['full_n'] != (32 if n == 32 else 11008) or c['batch'] != b * seq * (128 if n == 32 else 172):
            raise ValueError('case full_n or flattened batch differs from recorded contract')
    return cases


def oracle(torch, source, n, scale):
    values = source.detach().cpu().float().contiguous().reshape(-1, n)
    width = 1
    while width < n:
        grouped = values.reshape(-1, n // (2 * width), 2, width)
        left, right = grouped[:, :, 0, :].clone(), grouped[:, :, 1, :].clone()
        grouped[:, :, 0, :] = left + right
        grouped[:, :, 1, :] = left - right
        width *= 2
    return (values * scale).half()


def errors(torch, actual, reference):
    got, ref = actual.detach().cpu().float().reshape(-1), reference.float().reshape(-1)
    diff = (got - ref).double()
    denominator = torch.linalg.vector_norm(ref.double()).item()
    max_abs = diff.abs().max().item()
    relative_l2 = torch.linalg.vector_norm(diff).item() / max(denominator, 1e-30)
    return {'finite': bool(torch.isfinite(got).all()), 'max_abs': max_abs if math.isfinite(max_abs) else None,
            'relative_l2': relative_l2 if math.isfinite(relative_l2) else None,
            'threshold_assessed': False}


def ms_per_call(elapsed_ms, repeats, graph_calls):
    return elapsed_ms / (repeats * graph_calls)


def checkpoint(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def profile_graph(torch, graph, stream, graph_calls=1):
    try:
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            # Captured outputs are inference tensors; replay must retain that mode.
            with torch.inference_mode(), torch.cuda.stream(stream):
                graph.replay()
            stream.synchronize()
        events = []
        for event in prof.events():
            if event.device_type == torch.autograd.DeviceType.CUDA:
                events.append({'name': event.name, 'duration_us': event.time_range.elapsed_us(),
                               'device_index': event.device_index})
        kernels = [e for e in events if not any(word in e['name'].lower() for word in ('memcpy', 'memset'))]
        return {'status': 'ok', 'boundary': 'one graph replay after timing', 'graph_calls': graph_calls,
                'kernels_per_call': len(kernels) / graph_calls,
                'kernel_count': len(kernels), 'kernel_duration_us': sum(e['duration_us'] for e in kernels),
                'kernel_count_method': 'CUDA activities excluding names containing memcpy or memset',
                'cuda_activity_count': len(events), 'cuda_activity_duration_us': sum(e['duration_us'] for e in events),
                'events': events, 'count_note': 'CUDA activities as emitted by torch.profiler; inspect names for copies/memsets versus compute kernels'}
    except Exception as exc:
        return {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'timings_preserved': True}


def run(args, report):
    import importlib.metadata
    import torch
    import fast_hadamard_transform
    sys.path.insert(0, str(ROOT / 'experiments/workloads/e13'))
    from quarot_adapter import QuaRotFWHTAdapter

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; this script has no CPU timing fallback')
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    props = torch.cuda.get_device_properties(device)
    report.update(gpu={'name': props.name, 'uuid': str(props.uuid), 'total_memory': props.total_memory,
                       'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')},
                  torch=torch.__version__, cuda=torch.version.cuda,
                  dao={'version': importlib.metadata.version('fast-hadamard-transform'),
                       'module': fast_hadamard_transform.__file__,
                       'module_sha256': sha(fast_hadamard_transform.__file__)})
    report['dao']['loaded_extension_sha256'] = {
        name: {'path': module.__file__, 'sha256': sha(module.__file__)}
        for name, module in list(sys.modules.items())
        if 'hadamard' in name and getattr(module, '__file__', '').endswith('.so')}
    stream = torch.cuda.Stream(device=device)
    for index, case in enumerate(load_cases(args.cases)):
        adapters, graphs, outputs = {}, {}, {}
        result = {'case': case, 'rounds': [], 'profiles': {}, 'numerical': {}}
        report['results'].append(result)
        try:
            seq = 128 if case['phase'] == 'prefill' else 1
            generator = torch.Generator(device='cpu').manual_seed(args.seed + index)
            # Attention is [B,S,heads,head_dim] transposed to [B,S,128,32].
            shape = (case['application_batch'], seq, 32, 128) if case['N'] == 32 else (case['application_batch'], seq, 172, 64)
            cpu_base = torch.randn(shape, generator=generator, dtype=torch.float32).half()
            base = cpu_base.to(device)
            source = base.transpose(-1, -2) if case['N'] == 32 else base
            scale = float(1.0 / torch.tensor(case['full_n']).sqrt())
            result.update(scale=scale, source_shape=list(source.shape), source_stride=list(source.stride()),
                          input_sha256=hashlib.sha256(cpu_base.numpy().tobytes()).hexdigest(),
                          input_hash_contract='FP16 contiguous backing allocation bytes; source strides recorded')
            reference = oracle(torch, source, case['N'], scale)
            for variant in VARIANTS[1:]:
                adapters[variant] = QuaRotFWHTAdapter(args.library, device=device,
                                                      compute_contract=variant, reuse_buffers=True)
            functions = {'dao': lambda: fast_hadamard_transform.hadamard_transform(source.contiguous(), scale)}
            for variant, adapter in adapters.items():
                functions[variant] = lambda adapter=adapter: adapter._transform_vectors(source, case['N'], case['full_n'])
            with torch.inference_mode(), torch.cuda.stream(stream):
                for variant in VARIANTS:
                    fn = functions[variant]
                    for _ in range(args.warmup):
                        fn()
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        for _ in range(args.graph_calls):
                            outputs[variant] = fn()
                    graphs[variant] = graph
                    graph.replay()
                    stream.synchronize()
                    result['numerical'][variant] = errors(torch, outputs[variant], reference)
                for round_index in range(args.rounds):
                    offset = round_index % len(VARIANTS)
                    order = VARIANTS[offset:] + VARIANTS[:offset]
                    record = {'round': round_index, 'order': list(order), 'timings_ms_per_call': {}}
                    for variant in order:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record(stream)
                        for _ in range(args.repeats):
                            graphs[variant].replay()
                        end.record(stream)
                        end.synchronize()
                        record['timings_ms_per_call'][variant] = ms_per_call(start.elapsed_time(end), args.repeats, args.graph_calls)
                    result['rounds'].append(record)
            result['medians_ms'] = {v: statistics.median(r['timings_ms_per_call'][v] for r in result['rounds']) for v in VARIANTS}
            result['adapter_plans'] = {v: a.plan_statistics() for v, a in adapters.items()}
            for variant in VARIANTS:
                result['profiles'][variant] = profile_graph(torch, graphs[variant], stream, args.graph_calls)
            result['status'] = 'complete'
            checkpoint(args.output, report)
        finally:
            torch.cuda.synchronize()
            graphs.clear()
            outputs.clear()
            for adapter in adapters.values():
                adapter.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', required=True, type=Path)
    p.add_argument('--output', type=Path, help='new JSON file; default XD_REPORT_DIR/report.json')
    p.add_argument('--cases', type=Path, default=ROOT / 'configs/experiments/quarot_contracts_20261107.json')
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--rounds', type=int, default=7)
    p.add_argument('--graph-calls', type=int, default=20, help='complete transforms per captured graph')
    p.add_argument('--seed', type=int, default=20261107)
    return p


def main():
    args = parser().parse_args()
    if min(args.warmup, args.repeats, args.rounds, args.graph_calls) < 1:
        raise ValueError('warmup, repeats, and rounds must be positive')
    load_cases(args.cases)
    if args.output is None:
        if not os.environ.get('XD_REPORT_DIR'):
            raise ValueError('provide --output or XD_REPORT_DIR')
        args.output = Path(os.environ['XD_REPORT_DIR']) / 'report.json'
    report = {'schema': 'quarot-matched-contracts-v2', 'status': 'partial', 'graph_calls': args.graph_calls, 'results': [],
              'timing_boundary': 'CUDA events around repeated graph replays / (repeats * graph_calls); each graph captures graph_calls complete transforms; includes contiguous, casts and scale required by variant; excludes preparation, warmup, capture, CPU oracle, profiler, input copy, output clone, hadK matmul and 7B model',
              'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__).resolve(), ROOT / 'experiments/workloads/e13/quarot_adapter.py')},
              'library': str(args.library.resolve()), 'library_sha256': sha(args.library),
              'numerical_contract': 'CPU FP32 Sylvester FWHT, FP32 normalization multiplication, final FP16 storage; errors reported without a new threshold',
              'timing_note': 'Events may include idle gaps between host graph replay submissions; graph capture contains graph_calls complete transform subpaths; profile counts cover the whole graph and kernels_per_call divides by graph_calls',
              'selection_environment': {key: os.environ.get(key) for key in ('CUBUTTERFLY_E13_ALGORITHM_POLICY', 'CUBUTTERFLY_SEARCH_BUDGET', 'CUBUTTERFLY_REGISTRY', 'CUBUTTERFLY_PLAN_CACHE')},
              'cases_sha256': sha(args.cases), 'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve destination before any GPU work; never replace an existing report.
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
    try:
        run(args, report)
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        checkpoint(args.output, report)


if __name__ == '__main__':
    main()
