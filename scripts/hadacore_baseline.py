"""Pinned external complete FWHT comparator; import/support never initializes CUDA.

Tensor-Core normalized subtransforms round differently from radix-2 FP16
butterflies. This is an external numerical comparator, not an exact-rounding oracle.
"""
from __future__ import annotations
import argparse
import copy
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '2391954b19988bd76cf3c2ea84c1ce74b68d568b'
DEFAULT_LIBRARY = os.environ.get('CUBUTTERFLY_HADACORE_LIBRARY', str(ROOT / 'build/hadacore/libhadacore_baseline.so'))
UPSTREAM = 'https://github.com/pytorch-labs/applied-ai'
SUBDIR = 'kernels/cuda/inference/hadamard_transform'
HASHES = {
    'hadamard_transform_cuda.cu': '6eaaedda3f1207112a9c6225f33bc31784eab2271cdd4fdf47f1e384df0a4baa',
    'hadamard_transform.cpp': 'c38c9ae82ec6a03b14d7a7aa6e216908a32a9b80615645222c53f17968b4ec0c',
}
LICENSE_HASH = '6abd46cf7d5771e61b85eaf3c9c91b46ab111d59e73c7877581ccaad3476737e'


def support(s, librarypath=None):
    """Full canonical semantic request; packed natural FP16 native arithmetic only."""
    def reject(reason):
        return dict(supported=False, status='unsupported-semantic', reason=reason)
    if not isinstance(s, dict) or s.get('schema') != 'cubutterfly-semantic-v1':
        return reject('canonical cubutterfly-semantic-v1 dict required')
    for key, value in [('operator', 2), ('storage', 0), ('compute', 0), ('length_mode', 0)]:
        if type(s.get(key)) is not int or s[key] != value:
            return reject(f'{key} must be {value}: FWHT, FP16 storage/native FP16 MMA; no precision conversion or embedding')
    shape = s.get('shape')
    if not isinstance(shape, list) or not shape or any(type(n) is not int or n < 1 or n & (n-1) for n in shape):
        return reject('shape requires positive power-of-two extents')
    n = math.prod(shape)
    if not 2 <= n <= 32768:
        return reject('flattened length must be 2..32768; upstream dispatch does not implement length 1')
    b = s.get('batch')
    if type(b) is not int or b < 1 or b*n > 2147483392 or b*n % 256:
        return reject('full batch*length must be positive, divisible by 256 and <=2147483392; no padding or batch reduction')
    for key in ['input_orders', 'output_orders']:
        if s.get(key) != [0]*len(shape) or any(type(x) is not int for x in s[key]):
            return reject(f'{key} must contain natural order for every axis')
    strides, span = [], 1
    for extent in reversed(shape):
        strides.insert(0, span)
        span *= extent
    for key in ['input_strides', 'output_strides']:
        if key in s and s[key] != strides:
            return reject(f'{key} must be packed row-major')
    for key in ['input_batch_stride', 'output_batch_stride']:
        if key in s and s[key] != n:
            return reject(f'{key} must equal flattened length')
    for key in ['input_extents', 'output_extents']:
        if key in s and s[key] != shape:
            return reject(f'{key} must equal shape')
    if type(s.get('placement')) is not int or s['placement'] not in (0, 1):
        return reject('placement must be out-of-place=0 or in-place=1')
    if type(s.get('inverse')) is not int or s['inverse'] not in (0, 1) or type(s.get('normalize_inverse')) is not bool:
        return reject('canonical inverse and boolean normalize_inverse required')
    if s.get('output_rounding', 0) != 0:
        return reject('final-storage FP32 arithmetic is not supported')
    scale = 1/math.sqrt(n) if s['inverse'] and s['normalize_inverse'] else math.sqrt(n)
    library = Path(librarypath or DEFAULT_LIBRARY)
    if not library.is_file():
        return dict(supported=False, status='missing-external-dependency', reason=f'HadaCore adapter library missing: {library}')
    return dict(supported=True, status='supported', reason='external FP16 numerical comparator; rounding differs from radix-2 butterflies',
                metadata=dict(implementation='HadaCore', backend_kind='external-supplied-output', logical_core_layout='packed-natural', timing_scope='complete upstream transform and required output scale; no allocation or padding', arithmetic_contract='FP16 MMA and normalized subtransform coefficients; rescale in FP32 then FP16 writeback; differs from radix-2 stage rounding', source_hashes=copy.deepcopy(HASHES), library_path=str(library.resolve()), library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(), upstream=UPSTREAM, upstream_commit=COMMIT, flattened_points=n, batch=b,
                              storage_precision='fp16', multiply_precision='fp16', accumulation_precision='fp16 MMA result/accumulator (hardware internal reduction order unspecified)',
                              native_normalization='orthonormal; FP16 rounded coefficients per subtransform',
                              adapter_output_scale=scale, output_contract='caller-supplied packed FP16 output',
                              timed_operations=['unchanged upstream run_fht', 'FP32 scalar rescale of FP16 result then FP16 writeback'],
                              excluded_operations=['build', 'plan initialization', 'allocation'],
                              graph_status='not GPU validated', arithmetic_equivalence='not bitwise/stage-rounding equivalent',
                              semantic=copy.deepcopy(s)))


def transformed_source(source):
    if hashlib.sha256(source.encode()).hexdigest() != HASHES['hadamard_transform_cuda.cu']:
        raise ValueError('upstream CUDA source hash mismatch')
    # Remove only framework includes; all device and host dispatch text is retained.
    for line in ['#include <torch/extension.h>\n', '#include <c10/cuda/CUDAException.h>\n']:
        if source.count(line) != 1:
            raise ValueError('unexpected upstream include')
        source = source.replace(line, '')
    return '#include "shim.cuh"\n' + source + '\n#include "bridge.cuh"\n'


def build(upstream, output, nvcc='/usr/local/cuda-12.4/bin/nvcc'):
    upstream, output = Path(upstream).resolve(), Path(output).resolve()
    commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != COMMIT:
        raise ValueError('upstream commit mismatch')
    for name, expected in HASHES.items():
        if hashlib.sha256((upstream/SUBDIR/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'{name} hash mismatch')
    license_data = (upstream/'LICENSE').read_bytes()
    if hashlib.sha256(license_data).hexdigest() != LICENSE_HASH:
        raise ValueError('license hash mismatch')
    output.mkdir(parents=True, exist_ok=True)
    (output/'LICENSE.HadaCore').write_bytes(license_data)
    for name in ['shim.cuh', 'bridge.cuh']:
        (output/name).write_bytes((ROOT/'benchmarks/hadacore'/name).read_bytes())
    unit = output/'hadacore.cu'
    unit.write_text(transformed_source((upstream/SUBDIR/'hadamard_transform_cuda.cu').read_text()))
    lib = output/'libhadacore_baseline.so'
    command = [nvcc, '-std=c++17', '-O3', '-shared', '-Xcompiler', '-fPIC',
               '-gencode', 'arch=compute_80,code=sm_80', '-gencode', 'arch=compute_80,code=compute_80',
               str(unit), '-o', str(lib)]
    subprocess.run(command, check=True)
    artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [unit, output/'shim.cuh', output/'bridge.cuh', lib]}
    (output/'build.json').write_text(json.dumps(dict(upstream=UPSTREAM, commit=COMMIT, source_hashes=HASHES,
        license_sha256=LICENSE_HASH, command=command, artifacts=artifacts,
        nvcc_version=subprocess.check_output([nvcc, '--version'], text=True),
        gpu_execution=False, changes=['replace framework includes with minimal enum/error shim', 'append independent C ABI and output-rescale kernel'],
        device_core='otherwise unchanged upstream CUDA source', comparison_kind='external-complete-FWHT'), indent=2)+'\n')
    return lib


class Plan:
    """No allocations. Caller owns pointers/device/current context and synchronizes lifetime.

    Explicit stream required; execute is asynchronous. In-place calls need input
    reset outside timing. Only equal or disjoint allocation ranges are supported.
    """
    def __init__(self, library, semantic):
        self.report = support(semantic, librarypath=library)
        if not self.report['supported']:
            raise ValueError(self.report['reason'])
        self.lib = ctypes.CDLL(str(library))
        self.fn = self.lib.hadacore_execute
        self.fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float, ctypes.c_void_p]
        self.fn.restype = ctypes.c_int

    def execute(self, source, destination, *, stream):
        if self.fn is None:
            raise RuntimeError('plan closed')
        if any(type(x) is not int or x < 0 for x in [source, destination, stream]) or not source or not destination:
            raise ValueError('nonzero device pointers and explicit nonnegative stream address required')
        m = self.report['metadata']
        n, b = m['flattened_points'], m['batch']
        if (source == destination) != bool(m['semantic']['placement']):
            raise ValueError('pointer aliasing differs from requested placement')
        if source != destination and abs(source-destination) < 2*n*b:
            raise ValueError('partially overlapping buffers are unsupported')
        if source % 16 or destination % 16:
            raise ValueError('upstream vector loads require 16-byte aligned pointers')
        code = self.fn(source, destination, n*b, n, m['adapter_output_scale'], stream)
        if code:
            raise RuntimeError(f'HadaCore CUDA/status {code}')

    def metadata(self):
        return copy.deepcopy(self.report['metadata'])

    def close(self):
        self.fn = self.lib = None


class _Core:
    def __init__(self, plan, source, output, stream, user_semantic):
        self.plan, self.source, self.output, self.stream = plan, source, output, stream
        self._metadata = plan.metadata()
        self._metadata['core_semantic'] = copy.deepcopy(self._metadata['semantic'])
        self._metadata['semantic'] = copy.deepcopy(user_semantic)
        self._metadata['placement_projection'] = {
            'requested': user_semantic['placement'],
            'core': self._metadata['core_semantic']['placement'],
            'basis': 'actual packed core buffer pointer aliasing; runner owns user placement boundaries',
        }

    def __call__(self):
        if self.plan is None:
            raise RuntimeError('HadaCore core closed')
        self.plan.execute(self.source.data_ptr(), self.output.data_ptr(), stream=self.stream)

    def metadata(self):
        return copy.deepcopy(self._metadata)

    def close(self):
        if self.plan is not None:
            self.plan.close()
        self.plan = self.source = self.output = None


def build_core(torch, s, source, output, stream, librarypath=None):
    report = support(s, librarypath)
    if not report['supported']:
        raise ValueError(report['reason'])
    elements = s['batch'] * report['metadata']['flattened_points']
    for name, tensor in [('source', source), ('output', output)]:
        if tensor.dtype != torch.float16 or not tensor.is_cuda or not tensor.is_contiguous() or tensor.numel() != elements:
            raise ValueError(f'{name} must be complete packed CUDA FP16 tensor; no casts')
    if source.device != output.device:
        raise ValueError('input/output device mismatch')
    stream_address = stream if type(stream) is int else stream.cuda_stream
    if type(stream_address) is not int or stream_address < 0:
        raise ValueError('explicit CUDA stream required')
    core_semantic = copy.deepcopy(s)
    core_semantic['placement'] = int(source.data_ptr() == output.data_ptr())
    return _Core(Plan(librarypath or DEFAULT_LIBRARY, core_semantic),
                 source, output, stream_address, s)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--nvcc', default='/usr/local/cuda-12.4/bin/nvcc')
    args = p.parse_args()
    print(build(args.upstream, args.output, args.nvcc))
