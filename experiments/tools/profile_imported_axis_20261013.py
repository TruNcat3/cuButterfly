#!/usr/bin/env python3
"""Four fixed axis-core cases with complete Graph timing and Driver resources."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[2]
CPP = Path(__file__).with_suffix('.cpp')
GPU = 'GPU-0eae5f07-33de-321b-70a9-ffeae49309e6'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_bridge(args, directory):
    path = directory / 'imported_axis_profile_bridge.so'
    command = ['g++', '-std=c++17', '-O2', '-fPIC', '-shared', '-I' + str(ROOT / 'src'),
               '-I/usr/local/cuda/include', str(CPP), str(args.library.resolve()),
               '-L/usr/local/cuda/lib64/stubs', '-lcuda',
               '-Wl,-rpath,' + str(args.library.resolve().parent), '-o', str(path)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return path, command


class Bridge:
    def __init__(self, path):
        self.dll = ct.CDLL(str(path))
        signatures = {
            'create': ([ct.c_int, ct.c_int], ct.c_void_p), 'error': ([], ct.c_char_p),
            'metadata': ([ct.c_void_p], ct.c_char_p),
            'execute': ([ct.c_void_p] * 4, ct.c_int),
            'capture_resources': ([ct.c_void_p] * 4, ct.c_char_p),
            'destroy': ([ct.c_void_p], None),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.dll, 'imported_axis_' + name)
            function.argtypes, function.restype = arguments, result
            setattr(self, name, function)


def profile_case(torch, np, bridge, case, source, output, stream, host, reference):
    handle = bridge.create(int(case['provider'] == 'vkfft'), int(case['inverse']))
    if not handle:
        raise RuntimeError(bridge.error().decode())
    graph = None
    try:
        def execute():
            if bridge.execute(handle, source.data_ptr(), output.data_ptr(), stream.cuda_stream):
                raise RuntimeError(bridge.error().decode())

        for _ in range(3):
            execute()
        stream.synchronize()
        eager = output.cpu().numpy().copy()
        if not np.array_equal(source.cpu().numpy(), host):
            raise AssertionError('core execution modified input')
        delta = eager - reference
        error = dict(max_abs=float(np.abs(delta).max()),
                     relative_l2=float(np.linalg.norm(delta.ravel()) / np.linalg.norm(reference.ravel())))
        if not np.isfinite(eager).all():
            raise AssertionError('nonfinite core output')
        resources = bridge.capture_resources(handle, source.data_ptr(), output.data_ptr(), stream.cuda_stream)
        if resources:
            resources = json.loads(resources)
        else:
            resources = dict(status='unavailable', reason=bridge.error().decode(), probe_api='cuda-driver')
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            execute()
        with torch.cuda.stream(stream):
            for _ in range(5):
                graph.replay()
        stream.synchronize()
        if not np.array_equal(output.cpu().numpy(), eager):
            raise AssertionError('Graph core output differs from eager')
        samples = []
        repeats = 20
        for _ in range(7):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                begin.record(stream)
                for _ in range(repeats):
                    graph.replay()
                end.record(stream)
            end.synchronize()
            samples.append(begin.elapsed_time(end) * 1000 / repeats)
        if not np.array_equal(source.cpu().numpy(), host):
            raise AssertionError('timed Graph execution modified input')
        return dict(case, status='pass', metadata=json.loads(bridge.metadata(handle)),
                    input_address=source.data_ptr(), output_address=output.data_ptr(),
                    input_preserved=True, graph_matches_eager_exact=True,
                    cpu_scipy_error=error, resources=resources,
                    graph_timing=dict(samples_us=samples, median_us=statistics.median(samples),
                                      min_us=min(samples), max_us=max(samples),
                                      sample_count=7, graph_replays_per_sample=repeats,
                                      scope='complete core Graph only; setup, compile, copies excluded'))
    finally:
        stream.synchronize()
        del graph
        bridge.destroy(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compile-only', action='store_true')
    args = parser.parse_args()
    if not args.compile_only and os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        raise RuntimeError('profiling requires assigned GPU2 UUID under isolation guard')
    if not args.output.is_absolute() and os.environ.get('XD_REPORT_DIR'):
        args.output = Path(os.environ['XD_REPORT_DIR']) / args.output
    report = dict(schema='imported-axis-profile-20261013-v1', status='running', cases=[],
                  shape=[16, 2048, 1025], dtype='complex64', fft_axis=1, scale=1.0,
                  library=str(args.library.resolve()), library_sha256=sha256(args.library),
                  runner_sha256=sha256(__file__), bridge_source_sha256=sha256(CPP),
                  attribution='same addresses/input/layout/scale; core plus its exchange/resource policy differs',
                  limitation='isolated axis evidence, not whole-application timing or unique causal attribution',
                  numerical_acceptance='finite required; same-precision CPU SciPy errors report only')
    try:
        with tempfile.TemporaryDirectory(prefix='cubutterfly-imported-axis-') as directory:
            path, command = compile_bridge(args, Path(directory))
            report.update(bridge_compile_argv=command, bridge_sha256=sha256(path))
            if args.compile_only:
                report['status'] = 'compiled'
            else:
                import numpy as np
                import scipy.fft
                import torch
                torch.cuda.set_device(0)
                torch.set_num_threads(4)
                rng = np.random.default_rng(20261013)
                host = (rng.standard_normal((16, 2048, 1025)).astype(np.float32) +
                        1j * rng.standard_normal((16, 2048, 1025)).astype(np.float32))
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    source = torch.from_numpy(host).cuda()
                    output = torch.empty_like(source)
                stream.synchronize()
                report.update(torch_version=torch.__version__,
                              scipy_version=scipy.__version__,
                              gpu_name=torch.cuda.get_device_name(0),
                              input_sha256=hashlib.sha256(host.tobytes()).hexdigest())
                bridge = Bridge(path)
                for inverse in (False, True):
                    reference = (scipy.fft.ifft(host, axis=1, norm='forward') if inverse
                                 else scipy.fft.fft(host, axis=1))
                    if reference.dtype != np.complex64:
                        raise AssertionError('CPU reference changed requested precision')
                    for provider in ('cufftdx', 'vkfft'):
                        case = dict(provider=provider, inverse=inverse)
                        result = profile_case(torch, np, bridge, case, source, output,
                                              stream, host, reference)
                        report['cases'].append(result)
                        print(json.dumps(result), flush=True)
                report['status'] = 'pass'
                report['median_core_speedup_vkfft'] = {
                    str(inverse): next(row['graph_timing']['median_us'] for row in report['cases']
                                       if row['provider'] == 'cufftdx' and row['inverse'] == inverse) /
                                  next(row['graph_timing']['median_us'] for row in report['cases']
                                       if row['provider'] == 'vkfft' and row['inverse'] == inverse)
                    for inverse in (False, True)}
    except Exception as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], cases=len(report['cases']),
                          output=str(args.output), error=report.get('error'))))
    return 0 if report['status'] in ('pass', 'compiled') else 1


if __name__ == '__main__':
    raise SystemExit(main())
