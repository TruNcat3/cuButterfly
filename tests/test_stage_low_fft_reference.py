"""Compare actual host C++ low FFT reference with the independent Python oracle."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('low_ref',ROOT/'scripts/low_fft_cpu_reference.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)

@pytest.fixture(scope='module')
def binary(tmp_path_factory):
    folder=tmp_path_factory.mktemp('low_fft_cpp')
    source=folder/'reference.cpp';out=folder/'reference'
    source.write_text(r'''
#include "low_fft_reference.hpp"
#include <iostream>
#include <cstdlib>
using namespace cuntt;
template<class V> void run(int n,bool inverse,bool norm,bool native) {
 using S=decltype(V{}.real);std::vector<V> v(n);
 for(auto& x:v) { float a,b;std::cin.read((char*)&a,4);std::cin.read((char*)&b,4);x={detail::low_fft_narrow<S>(a),detail::low_fft_narrow<S>(b)}; }
 detail::reference_low_fft(v,inverse,norm,native);
 for(auto x:v) { float a=detail::low_fft_widen(x.real),b=detail::low_fft_widen(x.imag);std::cout.write((char*)&a,4);std::cout.write((char*)&b,4); }
}
int main(int argc,char** argv) { int n=std::atoi(argv[1]),s=std::atoi(argv[2]);if(s==4)run<Complex16>(n,std::atoi(argv[3]),std::atoi(argv[4]),std::atoi(argv[5]));else run<ComplexBf16>(n,std::atoi(argv[3]),std::atoi(argv[4]),std::atoi(argv[5])); }
''')
    cuda=Path('/usr/local/cuda-12.4/include')
    if not shutil.which('g++') or not cuda.exists():pytest.skip('compiler/CUDA headers unavailable')
    subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-I'+str(ROOT/'src'),'-I'+str(ROOT/'include'),'-I'+str(cuda),str(source),'-o',str(out)],check=True,capture_output=True)
    return out

@pytest.mark.parametrize('storage',[4,5])
@pytest.mark.parametrize('compute',[0,1])
@pytest.mark.parametrize('inverse,norm',[(0,False),(1,False),(1,True)])
def test_cpp_matches_declared_python_reference(binary,storage,compute,inverse,norm):
    for n in [8,32,128]:
        rng=np.random.default_rng(100+n)
        values=ref.rounded(rng.normal(size=(1,n,2)).astype(np.float32)/64,storage)
        source=values[...,0]+1j*values[...,1]
        expected=ref.transform(source,dict(shape=[n],batch=1,storage=storage,compute=compute,inverse=inverse,normalize_inverse=norm))
        run=subprocess.run([str(binary),str(n),str(storage),str(inverse),str(int(norm)),str(int(compute==0))],input=values.tobytes(),capture_output=True,check=True)
        pairs=np.frombuffer(run.stdout,np.float32).reshape(1,n,2)
        actual=pairs[...,0]+1j*pairs[...,1]
        np.testing.assert_array_equal(actual,expected)
