#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

template <int N>
__global__ void fused_short_fwht(const __half* input, __half* output,
                                 std::uint64_t batch, float scale) {
    const unsigned lane = threadIdx.x & 31;
    const std::uint64_t row = static_cast<std::uint64_t>(blockIdx.x) * 4 + threadIdx.x / 32;
    if (row >= batch) return;
    float low = __half2float(input[row * N + lane]);
    float high = 0.0f;
    if constexpr (N == 64) high = __half2float(input[row * N + lane + 32]);
#pragma unroll
    for (unsigned distance = 1; distance < 32; distance <<= 1) {
        const float partner = __shfl_xor_sync(0xffffffff, low, distance);
        low = (lane & distance) ? partner - low : low + partner;
        if constexpr (N == 64) {
            const float other = __shfl_xor_sync(0xffffffff, high, distance);
            high = (lane & distance) ? other - high : high + other;
        }
    }
    if constexpr (N == 64) {
        const float original = low;
        low = original + high;
        high = original - high;
    }
    output[row * N + lane] = __float2half_rn(low * scale);
    if constexpr (N == 64) output[row * N + lane + 32] = __float2half_rn(high * scale);
}

extern "C" int cubutterfly_fused_short_fwht(const void* input, void* output,
                                            std::uint64_t batch, int n, float scale,
                                            cudaStream_t stream) {
    if (!input || !output || !batch || (n != 32 && n != 64)) return cudaErrorInvalidValue;
    const unsigned blocks = static_cast<unsigned>((batch + 3) / 4);
    if (n == 32)
        fused_short_fwht<32><<<blocks, 128, 0, stream>>>(
            static_cast<const __half*>(input), static_cast<__half*>(output), batch, scale);
    else
        fused_short_fwht<64><<<blocks, 128, 0, stream>>>(
            static_cast<const __half*>(input), static_cast<__half*>(output), batch, scale);
    return cudaGetLastError();
}
