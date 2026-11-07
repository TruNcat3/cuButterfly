"""CPU validation of the same byte-capacity rule used by selection and execution."""
from pathlib import Path
import subprocess
import shutil
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_temporal_compiled_size_and_resource_domain(tmp_path):
    nvcc = Path(shutil.which("nvcc") or "/usr/local/cuda-12.4/bin/nvcc")
    cuda_include = nvcc.resolve().parents[1] / "include"
    if not (cuda_include / "cuda_runtime_api.h").exists():
        pytest.skip("CUDA headers unavailable")
    source = tmp_path / "contract.cpp"
    source.write_text('''
#include "temporal_tile_contract.hpp"
#include <cassert>
int main() {
    using namespace cuntt;
    ButterflyConfig c;
    c.op=ButterflyOperator::Fft; c.fft_core=FftCore::Scalar;
    for(auto p : {ButterflyPrecision::Fp16,ButterflyPrecision::Bf16,ButterflyPrecision::Fp32}) {
        c.precision=p;
        for(auto log : {11U,12U}) { c.log_n=log; assert(detail::scalar_temporal_tile_supported(c,164*1024)); }
    }
    c.precision=ButterflyPrecision::Fp64; c.log_n=11;
    assert(detail::scalar_temporal_tile_supported(c,164*1024));
    c.log_n=12; assert(detail::scalar_temporal_tile_supported(c,164*1024));
    assert(!detail::scalar_temporal_tile_supported(c,48*1024));
    c.precision=ButterflyPrecision::Fp16;
    assert(!detail::scalar_temporal_tile_supported(c,8192));
    c.log_n=13; assert(detail::scalar_temporal_tile_supported(c,164*1024));
    c.precision=ButterflyPrecision::Fp32; c.log_n=13;
    assert(detail::scalar_temporal_tile_supported(c,164*1024));
    assert(!detail::scalar_temporal_tile_supported(c,48*1024));
    for(auto p : {ButterflyPrecision::Fp16,ButterflyPrecision::Bf16}) {
        c.precision=p;
        c.log_n=14; assert(detail::scalar_temporal_tile_supported(c,64*1024));
        assert(!detail::scalar_temporal_tile_supported(c,64*1024-1));
        c.log_n=15; assert(detail::scalar_temporal_tile_supported(c,128*1024));
        c.log_n=16; assert(!detail::scalar_temporal_tile_supported(c,164*1024));
    }
    c.op=ButterflyOperator::Fwht; c.precision=ButterflyPrecision::Fp16; c.log_n=16;
    assert(detail::scalar_temporal_tile_supported(c,128*1024));
    c.log_n=17; assert(!detail::scalar_temporal_tile_supported(c,1024*1024));
    // Integer zeta normalizes every non-uint64 request to uint32, even if
    // the caller provided a low floating precision enum.
    c.op=ButterflyOperator::SubsetZeta; c.precision=ButterflyPrecision::Fp16; c.log_n=15;
    assert(!detail::scalar_temporal_tile_supported(c,64*1024));
    assert(detail::scalar_temporal_tile_supported(c,128*1024));
    c.precision=ButterflyPrecision::Uint64;
    assert(!detail::scalar_temporal_tile_supported(c,128*1024));
    assert(detail::scalar_temporal_tile_supported(c,256*1024));
    c.log_n=12; c.op=ButterflyOperator::Fwht; c.precision=ButterflyPrecision::Fp64;
    assert(detail::scalar_temporal_tile_supported(c,32768));
    c.op=ButterflyOperator::Fft; c.fft_core=FftCore::CufftDxBlock;
    assert(!detail::scalar_temporal_tile_supported(c,164*1024));
}
''')
    binary = tmp_path / "contract"
    subprocess.run(["g++", "-std=c++17", "-I"+str(cuda_include), "-I"+str(ROOT/"include"), "-I"+str(ROOT/"src"),
                    str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True)
