#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>

#include "cuntt/butterfly.hpp"

namespace cuntt::detail {

__global__ void wmma_dft8_kernel(const Complex32* input, Complex32* output, std::uint64_t transforms, std::uint64_t batch_distance,
                                 std::uint64_t element_stride, bool inverse, float scale) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr std::uint32_t kWarps = 8;
    __shared__ __align__(32) __half matrix_a[16 * 16];
    __shared__ __align__(32) __half matrix_b[kWarps][16 * 16];
    __shared__ __align__(32) float matrix_c[kWarps][16 * 16];

    const std::uint32_t  warp            = threadIdx.x >> 5;
    const std::uint32_t  lane            = threadIdx.x & 31U;
    const std::uint64_t  first_transform = (static_cast<std::uint64_t>(blockIdx.x) * kWarps + warp) * 16;
    const std::uint16_t* coefficients    = inverse ? generated_dft8_inverse : generated_dft8_forward;
    for (std::uint32_t index = threadIdx.x; index < 16 * 16; index += blockDim.x)
        matrix_a[index] = __ushort_as_half(coefficients[index]);

    for (std::uint32_t index = lane; index < 8 * 16; index += 32) {
        const std::uint32_t item   = index & 7U;
        const std::uint32_t column = index >> 3;
        Complex32          sample{0.0F, 0.0F};
        if (first_transform + column < transforms) {
            const std::uint64_t base   = (first_transform + column) * batch_distance;
            sample = input[base + static_cast<std::uint64_t>(item) * element_stride];
        }
        matrix_b[warp][item + column * 16]     = __float2half_rn(sample.real);
        matrix_b[warp][item + 8 + column * 16] = __float2half_rn(sample.imag);
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>                c;
    wmma::fill_fragment(c, 0.0F);
    wmma::load_matrix_sync(a, matrix_a, 16);
    wmma::load_matrix_sync(b, matrix_b[warp], 16);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(matrix_c[warp], c, 16, wmma::mem_col_major);
    __syncthreads();

    for (std::uint32_t index = lane; index < 8 * 16; index += 32) {
        const std::uint32_t item   = index & 7U;
        const std::uint32_t column = index >> 3;
        if (first_transform + column < transforms) {
            const std::uint64_t base                                         = (first_transform + column) * batch_distance;
            output[base + static_cast<std::uint64_t>(item) * element_stride] = {matrix_c[warp][item + column * 16] * scale,
                                                                                matrix_c[warp][item + 8 + column * 16] * scale};
        }
    }
#endif
}

inline void launch_wmma_dft8(const Complex32* input, Complex32* output, std::uint64_t transforms, std::uint64_t batch_distance,
                             std::uint64_t element_stride, bool inverse, bool normalize, cudaStream_t stream) {
    constexpr std::uint32_t kWarps = 8;
    const auto              blocks = static_cast<unsigned int>((transforms + kWarps * 16 - 1) / (kWarps * 16));
    const float             scale  = inverse && normalize ? 0.125F : 1.0F;
    wmma_dft8_kernel<<<blocks, kWarps * 32, 0, stream>>>(input, output, transforms, batch_distance, element_stride, inverse, scale);
}

}  // namespace cuntt::detail
