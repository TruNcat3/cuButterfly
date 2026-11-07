"""Offline checks for literal local roots, including actual CUDA code generation."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_generated_header_is_reproducible():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from generate_fft_fixed_roots import render
    assert (ROOT / "src/fft_fixed_roots.cuh").read_text() == render()


def test_local_providers_compile_without_transcendental_work(tmp_path):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda-12.4/bin/nvcc"
    dump = shutil.which("cuobjdump") or "/usr/local/cuda-12.4/bin/cuobjdump"
    if not Path(nvcc).exists() or not Path(dump).exists():
        pytest.skip("CUDA compiler/disassembler unavailable")
    source = tmp_path / "local.cu"
    source.write_text('''
#include "fft_register_tile.cuh"
using namespace cuntt;
using namespace cuntt::detail::register_tile;
template<class C,bool Inv,unsigned Log,unsigned Lanes,
         template<unsigned,bool> class Core>
__global__ void probe(C* x) {
    C v[(1<<Log)/Lanes];
    #pragma unroll
    for(unsigned i=0;i<(1<<Log)/Lanes;++i)v[i]=x[threadIdx.x*((1<<Log)/Lanes)+i];
    cooperative_fft<Log,1,Lanes,Inv,Core>(v,threadIdx.x%Lanes);
    #pragma unroll
    for(unsigned i=0;i<(1<<Log)/Lanes;++i)x[threadIdx.x*((1<<Log)/Lanes)+i]=v[i];
}
template __global__ void probe<Complex32,false,4,1,NativeCodelet>(Complex32*);
template __global__ void probe<Complex64,true,5,1,Radix4Codelet>(Complex64*);
template __global__ void probe<Complex64,false,6,4,NativeCodelet>(Complex64*);
template __global__ void probe<Complex32,true,6,8,Radix4Codelet>(Complex32*);
__global__ void largest_root(Complex64* x) { x[0]=fixed_root<12,4095,true,Complex64>(); }
''')
    cubin = tmp_path / "local.cubin"
    subprocess.run([nvcc, "-std=c++17", "-O3", "-arch=sm_80", "-I"+str(ROOT/"include"),
        "-I"+str(ROOT/"src"), "-cubin", str(source), "-o", str(cubin)], check=True,
        capture_output=True, text=True)
    sass = subprocess.run([dump, "--dump-sass", str(cubin)], check=True,
                          capture_output=True, text=True).stdout
    # These roots are all static. A residual math-library expansion or runtime
    # phase conversion would reintroduce the diagnosed supply cost.
    for opcode in ("MUFU.", "FRND.", "F2I.S64.F64", "CALL."):
        assert opcode not in sass
    usage = subprocess.run([dump, "--dump-resource-usage", str(cubin)], check=True,
                           capture_output=True, text=True).stdout
    assert "GLOBAL:0" in usage
    assert "LOCAL:0" in usage
