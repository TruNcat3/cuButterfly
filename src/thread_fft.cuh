#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "cuntt/butterfly.hpp"

namespace cuntt::detail {

__device__ __forceinline__ Complex32 fft_add(Complex32 a, Complex32 b) {
    return {a.real + b.real, a.imag + b.imag};
}

__device__ __forceinline__ Complex32 fft_sub(Complex32 a, Complex32 b) {
    return {a.real - b.real, a.imag - b.imag};
}

__device__ __forceinline__ Complex32 fft_mul(Complex32 a, Complex32 b) {
    return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}

__device__ __forceinline__ void fft_pair(Complex32& a, Complex32& b, Complex32 twiddle) {
    const Complex32 left  = a;
    const Complex32 right = fft_mul(b, twiddle);
    a                     = fft_add(left, right);
    b                     = fft_sub(left, right);
}

template <std::uint32_t LogN>
__global__ void thread_fft_kernel(const Complex32* input, Complex32* output, const Complex32* twiddles, std::uint64_t transforms,
                                  std::uint64_t batch_distance, std::uint64_t element_stride, float scale) {
    constexpr std::uint32_t kN        = 1U << LogN;
    const std::uint64_t     transform = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (transform >= transforms)
        return;

    const std::uint64_t base = transform * batch_distance;
    Complex32           values[kN];
#pragma unroll
    for (std::uint32_t index = 0; index < kN; ++index) {
        const std::uint32_t reversed = __brev(index) >> (32 - LogN);
        values[index]                = input[base + static_cast<std::uint64_t>(reversed) * element_stride];
    }

#pragma unroll
    for (std::uint32_t stage = 0; stage < LogN; ++stage) {
        const std::uint32_t half = 1U << stage;
#pragma unroll
        for (std::uint32_t unit = 0; unit < kN / 2; ++unit) {
            const std::uint32_t offset = unit & (half - 1);
            const std::uint32_t left   = (unit - offset) * 2 + offset;
            const Complex32     a      = values[left];
            const Complex32     b      = fft_mul(values[left + half], twiddles[half - 1 + offset]);
            values[left]               = fft_add(a, b);
            values[left + half]        = fft_sub(a, b);
        }
    }

#pragma unroll
    for (std::uint32_t index = 0; index < kN; ++index) {
        values[index].real *= scale;
        values[index].imag *= scale;
        output[base + static_cast<std::uint64_t>(index) * element_stride] = values[index];
    }
}

template <std::uint32_t LogN>
void launch_thread_fft(const Complex32* input, Complex32* output, const Complex32* twiddles, std::uint64_t transforms, std::uint64_t batch_distance,
                       std::uint64_t element_stride, bool normalize) {
    constexpr std::uint32_t kThreads = 128;
    const auto              blocks   = static_cast<unsigned int>((transforms + kThreads - 1) / kThreads);
    const float             scale    = normalize ? 1.0F / static_cast<float>(1U << LogN) : 1.0F;
    thread_fft_kernel<LogN><<<blocks, kThreads>>>(input, output, twiddles, transforms, batch_distance, element_stride, scale);
}

template <std::uint32_t LogN, std::uint32_t Threads>
__global__ void cta_fft_kernel(const Complex32* input, Complex32* output, const Complex32* twiddles, std::uint64_t transforms,
                               std::uint64_t batch_distance, std::uint64_t element_stride, float scale) {
    static_assert(LogN >= 3 && LogN <= 10);
    static_assert(Threads == 32 || Threads == 64 || Threads == 128 || Threads == 256);
    constexpr std::uint32_t     kN                  = 1U << LogN;
    constexpr std::uint32_t     kTilePoints         = Threads * 8;
    static_assert(kTilePoints >= kN);
    constexpr std::uint32_t     kTransformsPerBlock = kTilePoints / kN;
    constexpr std::uint32_t     kGroupsPerTransform = kN / 8;
    extern __shared__ Complex32 tile[];
    const std::uint64_t         first_transform = static_cast<std::uint64_t>(blockIdx.x) * kTransformsPerBlock;

    for (std::uint32_t linear = threadIdx.x; linear < kTilePoints; linear += blockDim.x) {
        const std::uint32_t transform_in_block = linear / kN;
        const std::uint32_t item               = linear - transform_in_block * kN;
        const std::uint64_t transform          = first_transform + transform_in_block;
        tile[linear] =
            transform < transforms ? input[transform * batch_distance + static_cast<std::uint64_t>(item) * element_stride] : Complex32{0.0F, 0.0F};
    }
    __syncthreads();

    // One thread owns an eight-point DIT codelet. Global reads stay coalesced;
    // the bit-reversal is performed by shared-memory reads into registers.
    const std::uint32_t transform_in_block = threadIdx.x / kGroupsPerTransform;
    const std::uint32_t group              = threadIdx.x - transform_in_block * kGroupsPerTransform;
    const std::uint32_t tile_base          = transform_in_block * kN;
    Complex32           values[8];
#pragma unroll
    for (std::uint32_t index = 0; index < 8; ++index) {
        const std::uint32_t logical  = group * 8 + index;
        const std::uint32_t reversed = __brev(logical) >> (32 - LogN);
        values[index]                = tile[tile_base + reversed];
    }
#pragma unroll
    for (std::uint32_t stage = 0; stage < 3; ++stage) {
        const std::uint32_t half = 1U << stage;
#pragma unroll
        for (std::uint32_t unit = 0; unit < 4; ++unit) {
            const std::uint32_t offset = unit & (half - 1);
            const std::uint32_t left   = (unit - offset) * 2 + offset;
            const Complex32     a      = values[left];
            const Complex32     b      = fft_mul(values[left + half], twiddles[half - 1 + offset]);
            values[left]               = fft_add(a, b);
            values[left + half]        = fft_sub(a, b);
        }
    }
    // Above DFT8, contiguous outputs alias bit-reversed inputs owned by other
    // threads. At DFT8 each thread owns the complete transform.
    if constexpr (LogN > 3)
        __syncthreads();
#pragma unroll
    for (std::uint32_t index = 0; index < 8; ++index)
        tile[tile_base + group * 8 + index] = values[index];
    __syncthreads();

    // Compose further DFT8 codelets across the previous groups. The fixed tile
    // always contains 128 independent codelets, regardless of transform size.
    constexpr std::uint32_t kFusedStages = (LogN / 3) * 3;
#pragma unroll
    for (std::uint32_t stage = 3; stage < kFusedStages; stage += 3) {
        const std::uint32_t eighth              = 1U << stage;
        const std::uint32_t codelet_group       = group / eighth;
        const std::uint32_t offset              = group - codelet_group * eighth;
        const std::uint32_t base                = tile_base + codelet_group * 8 * eighth + offset;
        Complex32           x0                  = tile[base];
        Complex32           x1                  = tile[base + eighth];
        Complex32           x2                  = tile[base + 2 * eighth];
        Complex32           x3                  = tile[base + 3 * eighth];
        Complex32           x4                  = tile[base + 4 * eighth];
        Complex32           x5                  = tile[base + 5 * eighth];
        Complex32           x6                  = tile[base + 6 * eighth];
        Complex32           x7                  = tile[base + 7 * eighth];

        fft_pair(x0, x1, twiddles[eighth - 1 + offset]);
        fft_pair(x2, x3, twiddles[eighth - 1 + offset]);
        fft_pair(x4, x5, twiddles[eighth - 1 + offset]);
        fft_pair(x6, x7, twiddles[eighth - 1 + offset]);
        fft_pair(x0, x2, twiddles[2 * eighth - 1 + offset]);
        fft_pair(x1, x3, twiddles[2 * eighth - 1 + offset + eighth]);
        fft_pair(x4, x6, twiddles[2 * eighth - 1 + offset]);
        fft_pair(x5, x7, twiddles[2 * eighth - 1 + offset + eighth]);
        fft_pair(x0, x4, twiddles[4 * eighth - 1 + offset]);
        fft_pair(x1, x5, twiddles[4 * eighth - 1 + offset + eighth]);
        fft_pair(x2, x6, twiddles[4 * eighth - 1 + offset + 2 * eighth]);
        fft_pair(x3, x7, twiddles[4 * eighth - 1 + offset + 3 * eighth]);

        tile[base]               = x0;
        tile[base + eighth]      = x1;
        tile[base + 2 * eighth]  = x2;
        tile[base + 3 * eighth]  = x3;
        tile[base + 4 * eighth]  = x4;
        tile[base + 5 * eighth]  = x5;
        tile[base + 6 * eighth]  = x6;
        tile[base + 7 * eighth]  = x7;
        __syncthreads();
    }

    // Handle the one or two stages left after the final complete DFT8 group.
#pragma unroll
    for (std::uint32_t stage = kFusedStages; stage < LogN; ++stage) {
        const std::uint32_t half = 1U << stage;
        for (std::uint32_t linear = threadIdx.x; linear < kTilePoints / 2; linear += blockDim.x) {
            const std::uint32_t transform_slot = linear / (kN / 2);
            const std::uint32_t butterfly      = linear - transform_slot * (kN / 2);
            const std::uint32_t offset         = butterfly & (half - 1);
            const std::uint32_t left           = (butterfly - offset) * 2 + offset;
            const std::uint32_t base           = transform_slot * kN;
            const Complex32     a              = tile[base + left];
            const Complex32     b              = fft_mul(tile[base + left + half], twiddles[half - 1 + offset]);
            tile[base + left]                  = fft_add(a, b);
            tile[base + left + half]           = fft_sub(a, b);
        }
        __syncthreads();
    }

    for (std::uint32_t linear = threadIdx.x; linear < kTilePoints; linear += blockDim.x) {
        const std::uint32_t transform_in_block = linear / kN;
        const std::uint32_t item               = linear - transform_in_block * kN;
        const std::uint64_t transform          = first_transform + transform_in_block;
        if (transform < transforms) {
            Complex32 value = tile[linear];
            value.real *= scale;
            value.imag *= scale;
            output[transform * batch_distance + static_cast<std::uint64_t>(item) * element_stride] = value;
        }
    }
}

template <std::uint32_t LogN, std::uint32_t Threads>
void launch_cta_fft(const Complex32* input, Complex32* output, const Complex32* twiddles, std::uint64_t transforms, std::uint64_t batch_distance,
                    std::uint64_t element_stride, bool normalize) {
    constexpr std::uint32_t kTransformsPerBlock = (Threads * 8) / (1U << LogN);
    constexpr std::size_t   kShared              = Threads * 8 * sizeof(Complex32);
    const auto              blocks   = static_cast<unsigned int>((transforms + kTransformsPerBlock - 1) / kTransformsPerBlock);
    const float             scale    = normalize ? 1.0F / static_cast<float>(1U << LogN) : 1.0F;
    cta_fft_kernel<LogN, Threads><<<blocks, Threads, kShared>>>(input, output, twiddles, transforms, batch_distance, element_stride, scale);
}

}  // namespace cuntt::detail
