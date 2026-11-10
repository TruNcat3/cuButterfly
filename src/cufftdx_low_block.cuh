#pragma once
#include <cubutterfly/module.h>
#include <cuda_fp16.h>
#include <cufftdx.hpp>
#include <cstdint>

namespace cuntt::detail {
struct LowBlockComplex { __half real, imag; };
static_assert(sizeof(LowBlockComplex)==4);
// Global I/O is an opaque four-byte complex value on the aligned path.
// PTX avoids type-punning a half struct through uint32_t. The fallback retains
// the original naturally half-aligned component accesses.
template<bool AlignedIO>
__device__ __forceinline__ LowBlockComplex low_block_load(const void* pointer) {
    if constexpr(AlignedIO) {
        std::uint32_t bits;
        asm volatile("ld.global.u32 %0, [%1];" : "=r"(bits) : "l"(pointer) : "memory");
        return {__ushort_as_half(static_cast<unsigned short>(bits)),
                __ushort_as_half(static_cast<unsigned short>(bits>>16))};
    } else {
        return *static_cast<const LowBlockComplex*>(pointer);
    }
}
template<bool AlignedIO>
__device__ __forceinline__ void low_block_store(void* pointer, LowBlockComplex value) {
    if constexpr(AlignedIO) {
        const std::uint32_t bits=std::uint32_t(__half_as_ushort(value.real)) |
                                 (std::uint32_t(__half_as_ushort(value.imag))<<16);
        asm volatile("st.global.u32 [%0], %1;" :: "l"(pointer), "r"(bits) : "memory");
    } else {
        *static_cast<LowBlockComplex*>(pointer)=value;
    }
}
// Each packed lane is one independent transform. Public batch remains unpadded.
// Inverse normalization is one final-store multiplication, not stagewise scaling.
template<class FFT, bool Inverse, bool AlignedIO>
__global__ void low_block_pair(const LowBlockComplex* input, LowBlockComplex* output,
                              std::uint64_t batch, std::uint64_t distance,
                              std::uint64_t stride, bool normalize) {
    constexpr unsigned N=FFT::input_length;
    constexpr unsigned Rows=FFT::block_dim.x;
    constexpr unsigned Fpb=FFT::ffts_per_block;
    const std::uint64_t t0=std::uint64_t(blockIdx.x)*Fpb+2*threadIdx.y;
    const auto t1=t0+1;
    typename FFT::value_type v[FFT::storage_size]{};
#pragma unroll
    for(unsigned i=0;i<FFT::elements_per_thread;++i) {
        const unsigned j=threadIdx.x+i*Rows;
        LowBlockComplex a{},b{};
        if(j<N && t0<batch) a=low_block_load<AlignedIO>(reinterpret_cast<const unsigned char*>(input)+4*(t0*distance+std::uint64_t(j)*stride));
        if(j<N && t1<batch) b=low_block_load<AlignedIO>(reinterpret_cast<const unsigned char*>(input)+4*(t1*distance+std::uint64_t(j)*stride));
        v[i]={__halves2half2(a.real,b.real),__halves2half2(a.imag,b.imag)};
    }
    // The public module supports in-place execution. Ensure every packed
    // transform has completed its loads before any thread can write a value
    // that is still an input of a sibling transform.
    __syncthreads();
    extern __shared__ __align__(16) unsigned char shared[];
    FFT().execute(v,shared);
    // All loads complete before any in-place output store, including sibling FFTs.
    __syncthreads();
    const auto scale=__float2half_rn(1.0f/N);
#pragma unroll
    for(unsigned i=0;i<FFT::elements_per_thread;++i) {
        const unsigned j=threadIdx.x+i*Rows;
        auto re=v[i].x,im=v[i].y;
        if constexpr(Inverse) if(normalize) {
            re=__hmul2(re,__halves2half2(scale,scale));
            im=__hmul2(im,__halves2half2(scale,scale));
        }
        if(j<N && t0<batch) low_block_store<AlignedIO>(reinterpret_cast<unsigned char*>(output)+4*(t0*distance+std::uint64_t(j)*stride),{__low2half(re),__low2half(im)});
        if(j<N && t1<batch) low_block_store<AlignedIO>(reinterpret_cast<unsigned char*>(output)+4*(t1*distance+std::uint64_t(j)*stride),{__high2half(re),__high2half(im)});
    }
}

template<class FFT,bool Inverse>
int low_block_resources(cubutterflyModuleResourcesV1* out,std::uint64_t* local=nullptr) {
    // No pointers in the resources ABI: report the conservative maximum of
    // both I/O specializations. NCU reports the actual launched specialization.
    if constexpr(FFT::shared_memory_size>48*1024) {
        auto status=cudaFuncSetAttribute(low_block_pair<FFT,Inverse,true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,FFT::shared_memory_size);
        if(status!=cudaSuccess) return status;
        status=cudaFuncSetAttribute(low_block_pair<FFT,Inverse,false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,FFT::shared_memory_size);
        if(status!=cudaSuccess) return status;
    }
    cudaFuncAttributes aligned{},unaligned{};
    auto status=cudaFuncGetAttributes(&aligned,low_block_pair<FFT,Inverse,true>);
    if(status!=cudaSuccess) return status;
    status=cudaFuncGetAttributes(&unaligned,low_block_pair<FFT,Inverse,false>);
    if(status!=cudaSuccess) return status;
    if(local) *local=aligned.localSizeBytes>unaligned.localSizeBytes ? aligned.localSizeBytes : unaligned.localSizeBytes;
    if(out) *out={static_cast<unsigned>(aligned.numRegs>unaligned.numRegs ? aligned.numRegs : unaligned.numRegs),
                 FFT::block_dim.x*FFT::block_dim.y,
                 static_cast<unsigned>(aligned.sharedSizeBytes>unaligned.sharedSizeBytes ? aligned.sharedSizeBytes : unaligned.sharedSizeBytes),
                 FFT::shared_memory_size};
    return cudaSuccess;
}
template<class FFT,bool Inverse>
int low_block_launch(const cubutterflyModuleInvocationV1* io) {
    if(!io || !io->input || !io->output || !io->batch || !io->element_stride)
        return cudaErrorInvalidValue;
    if((reinterpret_cast<std::uintptr_t>(io->input) |
        reinterpret_cast<std::uintptr_t>(io->output))&1U) return cudaErrorInvalidValue;
    constexpr unsigned N=FFT::input_length;
    const auto distance=io->batch_stride ? io->batch_stride : std::uint64_t(N)*io->element_stride;
    if(io->batch>1 && distance<(N-1)*io->element_stride+1) return cudaErrorInvalidValue;
    const auto blocks=(io->batch+FFT::ffts_per_block-1)/FFT::ffts_per_block;
    if(blocks>2147483647ULL) return cudaErrorInvalidConfiguration;
    const bool aligned=((reinterpret_cast<std::uintptr_t>(io->input) |
                         reinterpret_cast<std::uintptr_t>(io->output))&3U)==0;
    // Element and batch strides are in four-byte complex values, so base
    // alignment suffices for every address in the complete invocation.
    if(aligned)
        low_block_pair<FFT,Inverse,true><<<static_cast<unsigned>(blocks),FFT::block_dim,
            FFT::shared_memory_size,io->stream>>>(static_cast<const LowBlockComplex*>(io->input),
            static_cast<LowBlockComplex*>(io->output),io->batch,distance,io->element_stride,io->normalize_inverse);
    else
        low_block_pair<FFT,Inverse,false><<<static_cast<unsigned>(blocks),FFT::block_dim,
            FFT::shared_memory_size,io->stream>>>(static_cast<const LowBlockComplex*>(io->input),
            static_cast<LowBlockComplex*>(io->output),io->batch,distance,io->element_stride,io->normalize_inverse);
    return cudaGetLastError();
}
}
