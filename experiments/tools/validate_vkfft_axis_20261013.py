#!/usr/bin/env python3
"""Bounded direct-provider validation; run under scripts/run_isolated_gpu.py."""
import argparse
import ctypes
import itertools
import json
import os
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    args = parser.parse_args()
    expected_gpu = 'GPU-00000000-0000-0000-0000-000000000002'
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected_gpu:
        raise RuntimeError('validation requires the explicitly assigned GPU2 UUID')
    report = Path(os.environ['XD_REPORT_DIR'])
    library = ctypes.CDLL(str(args.library.resolve()))
    library.vka_error.restype = ctypes.c_char_p
    library.vka_create.argtypes = [ctypes.c_size_t] * 3 + [ctypes.c_int] * 2
    library.vka_create.restype = ctypes.c_void_p
    library.vka_execute.argtypes = [ctypes.c_void_p] + [ctypes.c_size_t] * 3
    library.vka_execute.restype = ctypes.c_int
    library.vka_metadata.argtypes = [ctypes.c_void_p]
    library.vka_metadata.restype = ctypes.c_char_p
    library.vka_destroy.argtypes = [ctypes.c_void_p]
    torch.set_num_threads(4)
    torch.manual_seed(20261013)
    stream = torch.cuda.Stream()
    records = []

    def execute(plan, source, output):
        status = library.vka_execute(plan, source.data_ptr(), output.data_ptr(), stream.cuda_stream)
        if status:
            raise RuntimeError(library.vka_error().decode())

    def reference(source, inverse):
        return torch.fft.ifft(source, dim=1, norm='forward') if inverse else torch.fft.fft(source, dim=1)

    def error(actual, expected):
        difference = (actual.cpu() - expected).abs()
        return {'max_abs': difference.max().item(),
                'relative_l2': (torch.linalg.vector_norm(difference) /
                                torch.linalg.vector_norm(expected.abs())).item()}

    for length, suffix, prefix, fp64, inverse in itertools.product(
            (2048, 4096), (1, 17, 1025), (1, 2), (False, True), (False, True)):
        dtype = torch.complex128 if fp64 else torch.complex64
        host = torch.randn((prefix, length, suffix), dtype=dtype)
        second_host = torch.randn(host.shape, dtype=dtype)
        source = host.cuda()
        output = torch.empty_like(source)
        torch.cuda.synchronize()
        plan = library.vka_create(length, suffix, prefix * suffix, int(fp64), int(inverse))
        if not plan:
            raise RuntimeError(library.vka_error().decode())
        graph = None
        try:
            record = dict(length=length, suffix=suffix, prefix=prefix,
                          fp64=fp64, inverse=inverse,
                          metadata=json.loads(library.vka_metadata(plan)))
            expected = reference(host, inverse)
            execute(plan, source, output)
            stream.synchronize()
            eager_result = output.cpu()
            record['eager_cpu_error'] = error(output, expected)
            if not torch.equal(source.cpu(), host):
                raise RuntimeError('out-of-place execution modified the input')
            record['input_preserved'] = True

            changed_source = second_host.cuda()
            changed_output = torch.empty_like(output)
            torch.cuda.synchronize()
            expected_second = reference(second_host, inverse)
            execute(plan, changed_source, changed_output)
            stream.synchronize()
            record['changed_address_cpu_error'] = error(changed_output, expected_second)
            if not torch.equal(changed_source.cpu(), second_host):
                raise RuntimeError('changed-address execution modified the input')

            # Capture after warm-up, then replace values in the captured input buffer.
            execute(plan, source, output)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                execute(plan, source, output)
            source.copy_(changed_source)
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            record['changing_input_graph_cpu_error'] = error(output, expected_second)
            if not torch.equal(source.cpu(), second_host):
                raise RuntimeError('graph execution modified the input')
            record['graph_output_changed'] = not torch.equal(output.cpu(), eager_result)
            if not record['graph_output_changed']:
                raise RuntimeError('graph output did not change after input replacement')

            inplace = host.cuda()
            torch.cuda.synchronize()
            execute(plan, inplace, inplace)
            stream.synchronize()
            record['inplace_cpu_error'] = error(inplace, expected)
            records.append(record)
            (report / 'provider_validation.json').write_text(json.dumps(records, indent=2) + '\n')
            print(json.dumps(record), flush=True)
        finally:
            torch.cuda.synchronize()
            del graph
            library.vka_destroy(plan)
    summary = {'cases': len(records), 'input_preserved': True,
               'numerical_errors_are_report_only': True,
               'max_relative_l2': max(value['relative_l2'] for record in records
                                      for key, value in record.items() if key.endswith('_cpu_error'))}
    (report / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
