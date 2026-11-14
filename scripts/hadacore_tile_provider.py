"""Experimental packed local FWHT provider; imports and compilation are CPU-only.

The device function is extracted from hash-pinned HadaCore. This optional module
is not public AUTO selection and cannot implement native FP16 stage rounding.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import hadacore_baseline as upstream_contract

CONTRACT = 'fp16-mma-orthonormal-final-scale'
CONFIGS = [(n, 1, 1, 24) for n in range(1, 9)] + [(n, 2, 1, 24) for n in range(1, 9)] + [
    (9, 2, 1, 24), (10, 2, 2, 16), (11, 2, 4, 8), (12, 2, 8, 4),
    (13, 2, 16, 3), (14, 4, 16, 2), (15, 8, 16, 1)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def descriptor(log_n, tile_count, *, arithmetic_contract, chunks=None,
               groups_per_block=1, scale=1.0):
    if arithmetic_contract != CONTRACT:
        raise ValueError('explicit FP16 MMA arithmetic contract required; no native-rounding replacement')
    if type(log_n) is not int or not 1 <= log_n <= 15:
        raise ValueError('local log_n must be in 1..15')
    if type(tile_count) is not int or tile_count < 1:
        raise ValueError('positive complete tile_count required')
    if type(groups_per_block) is not int or not 1 <= groups_per_block <= 2**32-1:
        raise ValueError('groups_per_block must be a positive uint32')
    if not math.isfinite(scale):
        raise ValueError('finite scale required')
    narrowed_scale = ct.c_float(scale).value
    if not math.isfinite(narrowed_scale):
        raise ValueError('scale must be representable as FP32')
    if chunks is None:
        chunks = 1 if log_n <= 8 else next(c for n,c,_,_ in CONFIGS if n == log_n)
    matches = [c for c in CONFIGS if c[:2] == (log_n, chunks)] if type(chunks) is int else []
    if len(matches) != 1:
        raise ValueError('launch geometry is not compiled')
    n,c,w,b = matches[0]
    elements, group_elements = tile_count * 2**n, 256*c*w
    if elements > (2**64-1)//2 or elements % group_elements:
        raise ValueError('complete groups required; no hidden padding or batch reduction')
    groups = elements // group_elements
    blocks = (groups-1)//groups_per_block+1
    if blocks > 2**31-1:
        raise ValueError('grid exceeds CUDA x dimension')
    return dict(provider='hadacore-local-tile-v1', arithmetic_contract=CONTRACT,
                log_n=n, tile_count=tile_count, chunks_per_warp=c, warps_per_block=w,
                blocks_per_sm=b, threads=32*w, group_elements=group_elements,
                shared_bytes=2*group_elements, stages=n, groups=groups, blocks=blocks,
                groups_per_block=groups_per_block, scale=narrowed_scale, elements=elements,
                layout='packed-natural-contiguous', input_stride=1, output_stride=1,
                tile_stride=2**n, workspace_bytes=0, public_search_integrated=False,
                gpu_validated=False, prefetch_depth=0,
                dataflow='local device subtransform; sequential reuse of one shared tile group',
                output_scale='FP32 multiplication of FP16 MMA output fused at global store, then FP16 rounding')


def device_source(source):
    if hashlib.sha256(source.encode()).hexdigest() != upstream_contract.HASHES['hadamard_transform_cuda.cu']:
        raise ValueError('upstream source hash mismatch')
    def replace_once(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('source extraction anchor changed: ' + old[:80])
        source = source.replace(old, new)
    for line in ['#include <torch/extension.h>\n', '#include <c10/cuda/CUDAException.h>\n']:
        replace_once(line, '')
    # Keep all upstream helpers and the entire arithmetic body, exclude host
    # dispatch. A local device function receives block-relative pointers/scratch.
    source = source[:source.index('constexpr int ceil_div')]
    replace_once('__global__ void __launch_bounds__(32 * warps_per_block, blocks_per_sm)',
                 '__device__ __forceinline__ void')
    replace_once('hadamard_transform_kernel(b16* a, b16* out, int total_num_chunks)',
                 'cubutterfly_hadacore_device(b16* a, b16* out, int total_num_chunks, b32* bfrag_arr, float output_scale)')
    replace_once('uint blockid = blockIdx.x * warps_per_block + threadIdx.x / 32;',
                 'uint blockid = threadIdx.x / 32;')
    replace_once('    extern __shared__ b32 bfrag_arr[]; // num_chunks * warps_per_block * 128\n', '')
    # Change only final global stores. Intermediate shared stores remain exact.
    replace_once('store[(real_row + (reg % 2) * 4) + (real_col + (reg / 2) * 8) * 8] = b_frag_all[k][j];',
                 'store[(real_row + (reg % 2) * 4) + (real_col + (reg / 2) * 8) * 8] = '
                 'log_had_size <= 8 ? cubutterfly_hada_scale_pair(b_frag_all[k][j], output_scale) : b_frag_all[k][j];')
    replace_once('                        store[rowidx * 128 + colidx] = data;',
                 '                        store[rowidx * 128 + colidx] = cubutterfly_hada_scale_pair(data, output_scale);')
    replace_once('store4[total_off] = bfrag_arr4[total_off];',
                 'store4[total_off] = cubutterfly_hada_scale_vector(bfrag_arr4[total_off], output_scale);')
    return '#include "shim.cuh"\n' + source


def build(upstream, output, nvcc='/usr/local/cuda-12.4/bin/nvcc'):
    upstream, output = Path(upstream).resolve(), Path(output).resolve()
    commit = subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'], text=True).strip()
    if commit != upstream_contract.COMMIT:
        raise ValueError('upstream commit mismatch')
    license_path = upstream/'LICENSE'
    if digest(license_path) != upstream_contract.LICENSE_HASH:
        raise ValueError('upstream license mismatch')
    source = (upstream/upstream_contract.SUBDIR/'hadamard_transform_cuda.cu').read_text()
    generated = device_source(source)
    output.mkdir(parents=True, exist_ok=True)
    (output/'LICENSE.HadaCore').write_bytes(license_path.read_bytes())
    for name in ('shim.cuh','tile_bridge.cuh'):
        (output/name).write_bytes((ROOT/'benchmarks/hadacore'/name).read_bytes())
    (output/'hadacore_device.cuh').write_text(generated)
    unit = output/'hadacore_tile.cu'
    unit.write_text('#include "tile_bridge.cuh"\n')
    library = output/'libhadacore_tile.so'
    command = [nvcc, '-std=c++17', '-O3', '-shared', '-Xcompiler', '-fPIC',
               '-gencode','arch=compute_80,code=sm_80', '-gencode','arch=compute_80,code=compute_80',
               '-Xptxas=-v', str(unit), '-o', str(library)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (output/'compile.log').write_text(result.stdout)
    if result.returncode:
        raise RuntimeError('compilation failed; see '+str(output/'compile.log'))
    build_report = dict(schema='cubutterfly-hadacore-tile-build-v1', gpu_execution=False,
        public_search_integrated=False, gpu_validated=False, arithmetic_contract=CONTRACT,
        upstream_commit=commit, upstream_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        command=command, configs=CONFIGS,
        artifacts={p.name:digest(p) for p in [unit,library,output/'hadacore_device.cuh',
                                             output/'tile_bridge.cuh',output/'shim.cuh',output/'LICENSE.HadaCore']},
        changes=['global entry converted to block-relative device function with caller shared scratch',
                 'same FP32 final scale fused into output stores',
                 'parameterized sequential group loop reuses scratch with a block barrier'],
        nvcc_version=subprocess.check_output([nvcc,'--version'],text=True))
    build_report['provider_sha256'] = digest(__file__)
    (output/'build.json').write_text(json.dumps(build_report,indent=2)+'\n')
    return library


class TilePlan:
    """Caller supplies complete packed local groups and retains buffers/stream.

    Construct before graph capture. No allocation, padding, order conversion or
    implicit contract change. A parent schedule owns inter-group boundaries.
    """
    def __init__(self, library, log_n, tile_count, *, arithmetic_contract,
                 chunks=None, groups_per_block=1, scale=1.0):
        self.info = descriptor(log_n,tile_count,arithmetic_contract=arithmetic_contract,
                               chunks=chunks,groups_per_block=groups_per_block,scale=scale)
        self.lib = ct.CDLL(str(Path(library).resolve()))
        prepare = self.lib.cubutterfly_hadacore_tile_prepare
        prepare.argtypes=[ct.c_int,ct.c_int]; prepare.restype=ct.c_int
        status = prepare(log_n,self.info['chunks_per_warp'])
        if status: raise RuntimeError(f'HadaCore tile preparation CUDA/status {status}')
        self.fn=self.lib.cubutterfly_hadacore_tile_execute
        self.fn.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_uint64,ct.c_int,ct.c_int,ct.c_uint32,ct.c_float,ct.c_void_p]
        self.fn.restype=ct.c_int

    def resources(self):
        if self.lib is None:
            raise RuntimeError('plan closed')
        class Resources(ct.Structure):
            _fields_ = [(name, ct.c_int) for name in (
                'registers_per_thread', 'static_shared_bytes', 'local_bytes_per_thread',
                'max_threads_per_block', 'active_blocks_per_sm', 'binary_version', 'ptx_version')]
        getter = self.lib.cubutterfly_hadacore_tile_resources
        getter.argtypes = [ct.c_int, ct.c_int, ct.POINTER(Resources)]
        getter.restype = ct.c_int
        value = Resources()
        status = getter(self.info['log_n'], self.info['chunks_per_warp'], ct.byref(value))
        if status:
            raise RuntimeError(f'HadaCore tile resource query CUDA/status {status}')
        report = {name: getattr(value, name) for name, _ in Resources._fields_}
        report.update(threads=self.info['threads'], dynamic_shared_bytes=self.info['shared_bytes'],
                      source='cudaFuncGetAttributes+cudaOccupancyMaxActiveBlocksPerMultiprocessor')
        return report

    def execute(self, source, destination, *, stream):
        if self.fn is None: raise RuntimeError('plan closed')
        if any(type(v) is not int or not 0 <= v <= 2**64-1 for v in (source,destination,stream)):
            raise ValueError('integer device pointers and explicit stream required')
        if not source or not destination or source%16 or destination%16:
            raise ValueError('nonzero 16-byte aligned device pointers required')
        m=self.info
        if source!=destination and abs(source-destination)<2*m['elements']:
            raise ValueError('partial input/output overlap is unsupported')
        status=self.fn(source,destination,m['tile_count'],m['log_n'],m['chunks_per_warp'],
                       m['groups_per_block'],m['scale'],stream)
        if status: raise RuntimeError(f'HadaCore local tile CUDA/status {status}')

    def close(self):
        # Caller must first destroy captured graphs and finish outstanding work.
        self.fn=self.lib=None


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--nvcc',default='/usr/local/cuda-12.4/bin/nvcc')
    a=p.parse_args()
    print(build(a.upstream,a.output,a.nvcc))
