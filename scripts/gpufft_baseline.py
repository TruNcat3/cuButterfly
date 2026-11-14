"""Pinned GPU-FFT complete rank-one C2C adapter; import/support are CPU-only."""
from __future__ import annotations
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '7f05858f7af7c0bbf55a77472426d9ac56ec6b23'
DEFAULT_LIBRARY = Path(os.environ.get('CUBUTTERFLY_GPUFFT_LIBRARY', ROOT / 'build/gpufft/libgpufft_baseline.so'))
DEFAULT_UPSTREAM = ROOT / 'external/GPU-FFT'


def support(s, librarypath=None):
    """Accept canonical packed rank-one semantics; all other layouts are explicit gaps."""
    reasons = []
    shape = s.get('shape')
    n = shape[0] if isinstance(shape, (list, tuple)) and len(shape) == 1 else None
    if type(n) is not int or not 2 <= n <= 2**28 or n & (n - 1):
        reasons.append('requires rank-one power-of-two length in [2,2^28]')
    b = s.get('batch')
    if type(b) is not int or b < 1:
        reasons.append('batch must be a positive integer')
    elif type(n) is int and n > 0 and (n*b >= 2**31 or (n > 512 and b > 65535)):
        reasons.append('upstream batch/grid/index bound exceeded')
    if s.get('operator') != 0 or s.get('fft_type') != 0 or s.get('length_mode') != 0:
        reasons.append('requires C2C FFT without length embedding')
    storage = s.get('storage')
    if storage not in (6, 7) or s.get('compute') not in ((0, 1) if storage == 6 else (0, 2)):
        reasons.append('requires matching native FP32 or FP64 storage/compute')
    if type(s.get('inverse')) is not int or s['inverse'] not in (0, 1) or type(s.get('normalize_inverse')) is not bool:
        reasons.append('requires canonical direction and inverse normalization')
    # This core accepts packed buffers. The central semantic boundary owns any
    # physical layout conversions and requested output scale, timed separately.
    missing = not Path(librarypath or DEFAULT_LIBRARY).is_file()
    status = 'unsupported-semantic' if reasons else ('missing-external-dependency' if missing else 'supported')
    if missing and not reasons:
        reasons.append('GPU-FFT shared library absent; build or set CUBUTTERFLY_GPUFFT_LIBRARY')
    return dict(supported=not reasons, status=status,
                reason='; '.join(reasons), metadata=dict(provider='GPU-FFT', upstream_commit=COMMIT,
                license='Apache-2.0', implementation='packed-rank1-C2C',
                native_order='forward natural to bit-reversed; inverse bit-reversed to natural',
                sign_adaptation='conjugated upstream twiddle table',
                timed_operations=['input copy or bit reversal', 'unchanged upstream GPU_FFT',
                                  'output bit reversal or copy'],
                validation='CPU prepared; GPU correctness and performance pending'))


def build(output, upstream=DEFAULT_UPSTREAM, nvcc='/usr/local/cuda/bin/nvcc', arch='80'):
    upstream, output = Path(upstream).resolve(), Path(output).resolve()
    if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'], text=True).strip() != COMMIT:
        raise ValueError('GPU-FFT checkout differs from pinned commit')
    if subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'], text=True).strip():
        raise ValueError('GPU-FFT checkout must be unmodified')
    output.mkdir(parents=True, exist_ok=True)
    library = output / 'libgpufft_baseline.so'
    bridge = ROOT / 'benchmarks/gpufft/bridge.cu'
    command = [nvcc,'-std=c++17','-O3','-shared','-Xcompiler','-fPIC',
               '-gencode',f'arch=compute_{arch},code=[sm_{arch},compute_{arch}]',
               '-I'+str(upstream/'src/include'), str(upstream/'src/lib/fft.cu'),
               str(bridge), '-o',str(library)]
    subprocess.run(command, check=True)
    (output/'LICENSE.GPU-FFT').write_bytes((upstream/'LICENSE').read_bytes())
    (output/'build.json').write_text(json.dumps(dict(upstream_commit=COMMIT, command=command,
        bridge_sha256=hashlib.sha256(bridge.read_bytes()).hexdigest(), gpu_execution=False),indent=2)+'\n')
    return library


class Plan:
    """Prepare/close outside timing; caller supplies packed pointers and explicit stream.

    Synchronize before close; a plan is not safe for concurrent execution.
    Output scaling/layout adaptation belongs to the timed semantic boundary.
    """
    def __init__(self, library, s):
        info = support(s, library)
        if not info['supported']:
            raise ValueError(info['reason'])
        self.lib = ctypes.CDLL(str(library))
        self.lib.gpufft_create.argtypes = [ctypes.c_int]*5+[ctypes.POINTER(ctypes.c_void_p)]
        self.lib.gpufft_execute.argtypes = [ctypes.c_void_p]*4
        self.lib.gpufft_destroy.argtypes = [ctypes.c_void_p]
        self.handle = ctypes.c_void_p()
        self._check(self.lib.gpufft_create(s['shape'][0].bit_length()-1, s['batch'],
                    64 if s['storage']==7 else 32, s['inverse'], int(s['normalize_inverse']), ctypes.byref(self.handle)))
    @staticmethod
    def _check(code):
        if code:
            raise RuntimeError(f'GPU-FFT adapter status {code}')
    def execute(self, source, destination, *, stream):
        if not self.handle.value:
            raise RuntimeError('plan is closed')
        if type(stream) is not int or stream < 0:
            raise ValueError('explicit nonnegative CUDA stream required')
        self._check(self.lib.gpufft_execute(self.handle, source, destination, stream))
    def close(self):
        if self.handle.value:
            self._check(self.lib.gpufft_destroy(self.handle))
            self.handle = ctypes.c_void_p()


class PackedCore:
    def __init__(self, torch, s, source, output, stream, librarypath=None):
        library = Path(librarypath or DEFAULT_LIBRARY)
        info = support(s, library)
        if not info['supported']:
            raise ValueError(info['reason'])
        dtype = torch.complex128 if s['storage'] == 7 else torch.complex64
        expected = (s['batch'], *s['shape'])
        for tensor in (source, output):
            if tuple(tensor.shape) != expected or tensor.dtype != dtype or not tensor.is_contiguous():
                raise ValueError('GPU-FFT core requires matching packed logical tensors')
            if tensor.device.type != 'cuda':
                raise ValueError('GPU-FFT requires CUDA tensors')
        if source.device != output.device:
            raise ValueError('GPU-FFT tensors must share a device')
        self.source, self.output = source, output
        self.stream = int(stream.cuda_stream)
        self.plan = Plan(library, s)
        self._metadata = dict(info['metadata'], library=str(library.resolve()),
                              library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                              caller_stream=self.stream, shape=list(s['shape']), batch=s['batch'],
                              precision='fp64' if s['storage'] == 7 else 'fp32',
                              inverse=s['inverse'], normalize_inverse=s['normalize_inverse'],
                              core_normalize_inverse=bool(s['inverse'] and s['normalize_inverse']),
                              scratch_bytes=source.numel()*source.element_size(),
                              allocation_policy='scratch and roots allocated before timing')
    def __call__(self):
        self.plan.execute(self.source.data_ptr(), self.output.data_ptr(), stream=self.stream)
    def metadata(self):
        return self._metadata
    def close(self):
        self.plan.close()


def build_core(torch, s, source, output, stream, librarypath=None):
    return PackedCore(torch, s, source, output, stream, librarypath)
