#!/usr/bin/env python3
"""Finite mixed-provider semantic checks, with no timing or selection claims."""
import argparse
import ctypes as ct
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[2]
CPP = Path(__file__).with_suffix('.cpp')
GPU = 'GPU-0eae5f07-33de-321b-70a9-ffeae49309e6'


def cases():
    rows = []
    for precision, kind, first in itertools.product(
            ('fp32', 'fp64'), ('c2c-forward', 'c2c-inverse', 'r2c', 'c2r'), ('vkfft', 'cufftdx')):
        rows.append(dict(shape=[128, 32] if kind.startswith('c2c') else [64, 16, 32],
                         batch=2, precision=precision, kind=kind, order=[1, 0],
                         cores=[0, 1] if first == 'vkfft' else [1, 0],
                         exchanges=[0, 0], batch_tile=0, c2r_prefix_scale=False,
                         output_scale=0.5, expect_rejected=False, retained=False))
    for precision, first in itertools.product(('fp32', 'fp64'), ('vkfft', 'cufftdx')):
        rows.append(dict(shape=[64, 16, 32], batch=2, precision=precision, kind='c2r',
                         order=[1, 0], cores=[0, 1] if first == 'vkfft' else [1, 0],
                         exchanges=[0, 0], batch_tile=0, c2r_prefix_scale=True,
                         output_scale=0.5, expect_rejected=first == 'cufftdx', retained=False))
    for precision, tile in itertools.product(('fp32', 'fp64'), (0, 1)):
        rows.append(dict(shape=[128, 32], batch=3, precision=precision, kind='c2c-inverse',
                         order=[0, 1], cores=[0, 1], exchanges=[4, 0], batch_tile=tile,
                         c2r_prefix_scale=False, output_scale=0.5,
                         expect_rejected=False, retained=True))
    return rows


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_bridge(args, directory):
    output = directory / 'xd_mixed_validation_bridge.so'
    command = [args.cxx, '-std=c++17', '-O2', '-fPIC', '-shared',
               '-I' + str(ROOT / 'src'), '-I' + str(args.cuda_include), str(CPP),
               str(args.library.resolve()), '-Wl,-rpath,' + str(args.library.resolve().parent),
               '-o', str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output, command


class Bridge:
    def __init__(self, path):
        self.dll = ct.CDLL(str(path))
        signatures = {
            'create': ([ct.POINTER(ct.c_size_t), ct.c_uint, ct.c_size_t,
                        ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_double,
                        ct.POINTER(ct.c_uint), ct.POINTER(ct.c_uint), ct.POINTER(ct.c_size_t),
                        ct.c_size_t, ct.c_int], ct.c_void_p),
            'error': ([], ct.c_char_p), 'mapping': ([ct.c_void_p], ct.c_char_p),
            'workspace_size': ([ct.c_void_p], ct.c_size_t),
            'set_workspace': ([ct.c_void_p, ct.c_void_p, ct.c_size_t], ct.c_int),
            'execute': ([ct.c_void_p] * 4, ct.c_int), 'destroy': ([ct.c_void_p], None),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.dll, 'xd_mixed_' + name)
            function.argtypes, function.restype = arguments, result
            setattr(self, name, function)

    def checked(self, function, *arguments):
        if function(*arguments):
            raise RuntimeError(self.error().decode())


def inputs_and_references(case):
    import numpy as np
    import scipy.fft
    rng = np.random.default_rng(20261013)
    dtype = np.float64 if case['precision'] == 'fp64' else np.float32
    shape = (case['batch'], *case['shape'])
    axes = tuple(range(1, len(shape)))
    sources, references = [], []
    for _ in range(2):
        real = rng.standard_normal(shape).astype(dtype)
        kind = case['kind']
        if kind == 'r2c':
            source = real
            reference = scipy.fft.rfftn(source, axes=axes)
        elif kind == 'c2r':
            source = scipy.fft.rfftn(real, axes=axes)
            reference = scipy.fft.irfftn(source, s=case['shape'], axes=axes, norm='backward')
        else:
            source = real + 1j * rng.standard_normal(shape).astype(dtype)
            reference = (scipy.fft.ifftn(source, axes=axes, norm='backward')
                         if kind == 'c2c-inverse' else scipy.fft.fftn(source, axes=axes))
        reference = reference * dtype(case['output_scale'])
        if source.dtype.itemsize != (16 if source.dtype.kind == 'c' and dtype == np.float64
                                    else 8 if source.dtype.kind == 'c' or dtype == np.float64 else 4):
            raise AssertionError('CPU reference changed requested precision')
        sources.append(source.copy())
        references.append(reference.copy())
    return sources, references


def error_metrics(np, actual, reference):
    delta = np.abs(actual - reference)
    if not np.isfinite(actual).all():
        raise AssertionError('nonfinite output')
    return dict(max_abs=float(delta.max()),
                relative_l2=float(np.linalg.norm((actual - reference).ravel()) /
                                  np.linalg.norm(reference.ravel())))


def validate_case(torch, bridge, case, stream):
    import numpy as np
    rank = len(case['shape'])
    count = rank if case['kind'].startswith('c2c') else rank - 1
    shape = (ct.c_size_t * rank)(*case['shape'])
    cores = (ct.c_uint * count)(*case['cores'])
    exchanges = (ct.c_uint * count)(*case['exchanges'])
    order = (ct.c_size_t * count)(*case['order'])
    inverse = case['kind'] in ('c2r', 'c2c-inverse')
    handle = bridge.create(shape, rank, case['batch'], int(inverse), int(inverse),
                           int(case['precision'] == 'fp64'), int(case['kind'].startswith('c2c')),
                           case['output_scale'], cores, exchanges, order,
                           case['batch_tile'], int(case['c2r_prefix_scale']))
    if not handle:
        reason = bridge.error().decode()
        if case['expect_rejected'] and reason == 'C2R prefix scaling requires an available final fused complex axis':
            return dict(case, status='expected-rejection', reason=reason)
        raise RuntimeError('plan construction failed: ' + reason)
    graph = None
    try:
        if case['expect_rejected']:
            raise AssertionError('unsupported C2R prefix scaling was accepted')
        mapping = json.loads(bridge.mapping(handle))
        mapped = mapping['axes']
        expected_cores = ['vkfft-axis' if case['cores'][axis] else 'cufftdx-axis-tiled'
                          for axis in case['order']]
        if [axis['core'] for axis in mapped] != expected_cores:
            raise AssertionError('mixed provider request fell back: ' + json.dumps(mapping))
        if [axis['axis'] for axis in mapped] != case['order']:
            raise AssertionError('axis order did not reach the lowering')
        if case['retained'] and not (mapped[0]['packed'] and mapped[0]['permutation'] == [1, 0]
                                     and mapped[1]['input_axis_order'] == [1, 0]):
            raise AssertionError('retained permutation was not executed')
        if case['batch_tile'] and mapping['batch_chunks'] != 3:
            raise AssertionError('batch tile did not create three chunks')
        row = dict(case, mapping=mapping, eager=[], graph_original_changed_restored=[])
        sources, references = inputs_and_references(case)
        workspace_bytes = bridge.workspace_size(handle)
        with torch.cuda.stream(stream):
            workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device='cuda')
            source = torch.from_numpy(sources[0]).cuda()
            output = torch.empty(references[0].shape, dtype=torch.from_numpy(references[0]).dtype, device='cuda')
        if workspace_bytes:
            bridge.checked(bridge.set_workspace, handle, workspace.data_ptr(), workspace_bytes)

        def execute():
            bridge.checked(bridge.execute, handle, source.data_ptr(), output.data_ptr(), stream.cuda_stream)

        eager = []
        for index in (0, 1):
            with torch.cuda.stream(stream):
                source.copy_(torch.from_numpy(sources[index]))
                execute()
            stream.synchronize()
            actual = output.cpu().numpy().copy()
            eager.append(actual)
            row['eager'].append(error_metrics(np, actual, references[index]))
            if not np.array_equal(source.cpu().numpy(), sources[index]):
                raise AssertionError('eager execution modified input')
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            execute()
        for index in (0, 1, 0):
            with torch.cuda.stream(stream):
                source.copy_(torch.from_numpy(sources[index]))
                graph.replay()
            stream.synchronize()
            actual = output.cpu().numpy().copy()
            if not np.array_equal(actual, eager[index]):
                raise AssertionError('Graph result differs from same-backend eager')
            if not np.array_equal(source.cpu().numpy(), sources[index]):
                raise AssertionError('Graph execution modified input')
            row['graph_original_changed_restored'].append(dict(
                error_metrics(np, actual, references[index]), exactly_matches_eager=True))
        row.update(status='pass', input_preserved=True, graph_matches_eager_exact=True,
                   workspace_bytes=workspace_bytes)
        return row
    finally:
        stream.synchronize()
        del graph
        bridge.destroy(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--cuda-include', type=Path, default=Path('/usr/local/cuda/include'))
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--list-cases', action='store_true')
    args = parser.parse_args()
    if args.list_cases:
        print(json.dumps(cases(), indent=2))
        return 0
    if not args.library or not args.output:
        parser.error('--library and --output are required')
    if not args.compile_only and os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        raise RuntimeError('validation requires the assigned GPU2 UUID under isolation guard')
    if not args.output.is_absolute() and os.environ.get('XD_REPORT_DIR'):
        args.output = Path(os.environ['XD_REPORT_DIR']) / args.output
    report = dict(schema='xd-vkfft-mixed-validation-20261013-v1', status='running', cases=[],
                  scope='explicit mixed cores in XdRealFftPlan; semantic component evidence, no timing claim',
                  attribution='core-order and retained-layout/batch-tile checks; no performance attribution from these cases',
                  library=str(args.library.resolve()), library_sha256=sha256(args.library),
                  runner_sha256=sha256(__file__), bridge_source_sha256=sha256(CPP),
                  reference='CPU SciPy pocketfft at matching FP32/FP64 precision',
                  numerical_acceptance='finite required; floating errors report only',
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    try:
        with tempfile.TemporaryDirectory(prefix='cubutterfly-vkfft-mixed-') as directory:
            path, command = compile_bridge(args, Path(directory))
            report.update(bridge_compile_argv=command, bridge_sha256=sha256(path))
            if args.compile_only:
                report['status'] = 'compiled'
            else:
                import torch
                torch.cuda.set_device(0)
                torch.set_num_threads(4)
                stream = torch.cuda.Stream()
                bridge = Bridge(path)
                for case in cases():
                    try:
                        result = validate_case(torch, bridge, case, stream)
                    except Exception as error:
                        result = dict(case, status='failed', error_type=type(error).__name__, error=str(error))
                    report['cases'].append(result)
                    print(json.dumps(dict(case=case, status=result['status'], error=result.get('error'))), flush=True)
                report['status'] = 'failed' if any(row['status'] == 'failed' for row in report['cases']) else 'pass'
    except Exception as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], attempted=len(report['cases']),
                          passed=sum(row['status'] == 'pass' for row in report['cases']),
                          expected_rejections=sum(row['status'] == 'expected-rejection' for row in report['cases']),
                          output=str(args.output))))
    return 0 if report['status'] in ('pass', 'compiled') else 1


if __name__ == '__main__':
    raise SystemExit(main())
