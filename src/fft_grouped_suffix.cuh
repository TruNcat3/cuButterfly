#pragma once
#include "cuntt/butterfly.hpp"
#include <cufftdx.hpp>
#ifndef CUBUTTERFLY_CUFFTDX_SM
#error "Define CUBUTTERFLY_CUFFTDX_SM from the selected CUDA architecture"
#endif

namespace cuntt::detail {
// Several suffix subgraphs share one CTA. Their result exchange stays in
// shared memory; the final store covers adjacent k1 values and directly
// produces natural order without a whole-array transpose kernel.
template <class FFT, unsigned Columns, class Complex = Complex32, unsigned Chunk = 0, unsigned StaticLocalLog = 0>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void grouped_suffix(const Complex* scratch, Complex* output,
                               unsigned log_n, unsigned runtime_local_log,
                               std::uint64_t distance, std::uint64_t element_stride, decltype(Complex{}.real) scale) {
    using V = typename FFT::value_type;
    constexpr unsigned Size = FFT::input_length;
    constexpr unsigned Rows = Size / FFT::elements_per_thread;
    // The standalone compiler knows the local extent. The default retains
    // the runtime extent used by precompiled and direct kernel callers.
    const unsigned local_log = StaticLocalLog ? StaticLocalLog : runtime_local_log;
    static_assert(!Chunk || (Columns>1 && Chunk<=FFT::elements_per_thread &&
                    !(Chunk&(Chunk-1)) && FFT::elements_per_thread%(Chunk ? Chunk : 1)==0));
    const unsigned column = threadIdx.y;
    const unsigned local_n = 1U << local_log;
    const std::uint64_t transform = std::uint64_t(blockIdx.x) * Columns + column;
    const unsigned k1 = transform & (local_n - 1);
    const std::uint64_t base = (transform >> local_log) * distance;
    V v[FFT::storage_size];
#pragma unroll
    for (unsigned i=0; i<FFT::storage_size; ++i) {
        const auto x = scratch[base + (std::uint64_t(k1)*Size + threadIdx.x + i*Rows)*element_stride];
        v[i] = V{x.real, x.imag};
    }
    extern __shared__ __align__(16) unsigned char storage[];
    FFT().execute(v, storage);
    __syncthreads();
    auto* tile = reinterpret_cast<V*>(storage);
    if constexpr (Columns==1) {
        // With a single column the ownership permutation is the identity.
        // Keep the FFT result in registers instead of allocating and copying
        // an entire output tile, which can exceed the core's shared footprint.
        #pragma unroll
        for (unsigned i=0;i<FFT::storage_size;++i) {
            const auto row=threadIdx.x+i*Rows;
            output[base+(std::uint64_t(row)*local_n+k1)*element_stride]={v[i].x*scale,v[i].y*scale};
        }
        return;
    }
    if constexpr(Chunk>0) {
        static_assert(FFT::storage_size==FFT::elements_per_thread && FFT::output_length==Size);
        const unsigned tid=threadIdx.y*Rows+threadIdx.x;
        // Every output iteration has the same column/batch ownership. Only
        // its row advances. Keep all global offsets in 64-bit element units.
        const unsigned out_col=tid%Columns;
        const auto out_transform=std::uint64_t(blockIdx.x)*Columns+out_col;
        const auto out_base=(out_transform>>local_log)*distance+
            (std::uint64_t(tid/Columns)*local_n+(out_transform&(local_n-1)))*element_stride;
        const auto out_step=std::uint64_t(Rows)*local_n*element_stride;
#pragma unroll
        for(unsigned first=0;first<FFT::elements_per_thread;first+=Chunk) {
#pragma unroll
            for(unsigned j=0;j<Chunk;++j) {
                const unsigned row=threadIdx.x+j*Rows;
                tile[row*Columns+(column^((row/(16/Columns))&(Columns-1)))]=v[first+j];
            }
            __syncthreads();
#pragma unroll
            for(unsigned j=0;j<Chunk;++j) {
                const unsigned i=tid+j*Rows*Columns,row=i/Columns,col=i%Columns;
                const auto x=tile[row*Columns+(col^((row/(16/Columns))&(Columns-1)))];
                output[out_base+std::uint64_t(first+j)*out_step]={x.x*scale,x.y*scale};
            }
            if(first+Chunk<FFT::elements_per_thread) __syncthreads();
        }
        return;
    }
#pragma unroll
    for (unsigned i=0; i<FFT::storage_size; ++i) {
        const unsigned row = threadIdx.x + i*Rows;
        tile[row*Columns + (column ^ ((row / (16 / Columns)) & (Columns-1)))] = v[i];
    }
    __syncthreads();
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (unsigned i=tid; i<Size*Columns; i+=blockDim.x*blockDim.y) {
        const unsigned row = i/Columns, col = i%Columns;
        const auto x = tile[row*Columns + (col ^ ((row / (16 / Columns)) & (Columns-1)))];
        const auto t = std::uint64_t(blockIdx.x)*Columns + col;
        output[(t >> local_log)*distance + (std::uint64_t(row)*local_n + (t & (local_n-1)))*element_stride] = {x.x*scale,x.y*scale};
    }
}

template <unsigned Log, unsigned Ept, unsigned Columns, bool Inverse=false, class Complex=Complex32, unsigned Chunk=0, unsigned StaticLocalLog=0>
void launch_grouped_suffix(const Complex* scratch, Complex* output,
                           unsigned log_n, std::uint64_t batch, cudaStream_t stream=nullptr,
                           std::uint64_t distance=0, std::uint64_t element_stride=1, bool normalize=false) {
    using Real = decltype(Complex{}.real);
    using FFT = decltype(cufftdx::Block() + cufftdx::Size<(1U<<Log)>() +
        cufftdx::Type<cufftdx::fft_type::c2c>() +
        cufftdx::Direction<Inverse ? cufftdx::fft_direction::inverse : cufftdx::fft_direction::forward>() +
        cufftdx::Precision<Real>() + cufftdx::ElementsPerThread<Ept>() +
        cufftdx::FFTsPerBlock<Columns>() + cufftdx::SM<CUBUTTERFLY_CUFFTDX_SM>());
    constexpr unsigned tile_bytes = (1U<<Log)*Columns*sizeof(Complex)*(Chunk?Chunk:Ept)/Ept;
    constexpr unsigned bytes = Columns==1 ? FFT::shared_memory_size :
        (tile_bytes > FFT::shared_memory_size ? tile_bytes : FFT::shared_memory_size);
    if constexpr (bytes > 48U*1024U)
        cudaFuncSetAttribute(grouped_suffix<FFT, Columns, Complex, Chunk, StaticLocalLog>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
    const unsigned local = log_n-Log;
    grouped_suffix<FFT, Columns, Complex, Chunk, StaticLocalLog><<<static_cast<unsigned>((batch << local)/Columns), FFT::block_dim, bytes, stream>>>(
        scratch,output,log_n,local,distance ? distance : (std::uint64_t{1}<<log_n),element_stride,
        Inverse && normalize ? Real{1} / Real(std::uint64_t{1}<<log_n) : Real{1});
}
} // namespace cuntt::detail
