// Compile-only contract: no launch or GPU is required.
#include "fft_native_block.cuh"

using namespace cuntt;
using namespace cuntt::detail;

template<unsigned Log, unsigned Ept, unsigned Columns, bool Inverse, class Complex>
__global__ void native_block_compile_contract(Complex* data) {
    using FFT = NativeBlock<Log, Ept, Columns, Inverse, Complex>;
    static_assert(FFT::block_dim.x == (1U << Log) / Ept);
    static_assert(FFT::block_dim.y == Columns);
    static_assert(FFT::storage_size == Ept && FFT::elements_per_thread == Ept);
    static_assert(FFT::shared_memory_size == (1U << Log) * Columns * sizeof(Complex));
    static_assert(FFT::index(FFT::Rows - 1, Ept - 1, Columns - 1) <
                  (1U << Log) * Columns);
    typename FFT::value_type values[Ept];
    #pragma unroll
    for (unsigned i = 0; i < Ept; ++i) {
        const auto v = data[(threadIdx.x + i * FFT::Rows) * Columns + threadIdx.y];
        values[i] = {v.real, v.imag};
    }
    extern __shared__ __align__(16) unsigned char memory[];
    FFT{}.execute(values, memory);
    #pragma unroll
    for (unsigned i = 0; i < Ept; ++i)
        data[(threadIdx.x + i * FFT::Rows) * Columns + threadIdx.y] = values[i];
}

#define INSTANTIATE(L,E,C,I,T) \
    template __global__ void native_block_compile_contract<L,E,C,I,T>(T*);
INSTANTIATE(10,32,2,false,Complex32)
INSTANTIATE(10,32,2,true,Complex32)
INSTANTIATE(10,32,2,false,Complex64)
INSTANTIATE(10,32,2,true,Complex64)
INSTANTIATE(12,32,2,false,Complex32)
INSTANTIATE(12,32,2,true,Complex32)
INSTANTIATE(12,32,2,false,Complex64)
INSTANTIATE(12,32,2,true,Complex64)
INSTANTIATE(0,1,1,false,Complex64)
INSTANTIATE(5,32,2,true,Complex64)
INSTANTIATE(5,1,2,false,Complex32)
INSTANTIATE(5,8,4,true,Complex64)
#undef INSTANTIATE
