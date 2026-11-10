// Standalone regression; does not link or rebuild libcuntt.
// nvcc -std=c++17 -arch=sm_80 -Iinclude -Isrc tests/test_low_fft_specialization.cu -o /tmp/test_low_fft_specialization
#include "operator_traits.cuh"
#include <cstdio>
#include <type_traits>
#include <cstdlib>

using namespace cuntt;
using namespace cuntt::traits;

#define CHECK(call) do { const auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); std::exit(2); } } while (0)

template <typename T> __device__ bool same_bits(T a, T b) {
    const auto* x = reinterpret_cast<const unsigned char*>(&a);
    const auto* y = reinterpret_cast<const unsigned char*>(&b);
    for (unsigned i = 0; i < sizeof(T); ++i) if (x[i] != y[i]) return false;
    return true;
}

template <typename Storage, bool Native, LowFftNormalization Mode>
__global__ void compare(unsigned* failures, bool inverse, bool normalize) {
    using Value = typename LowComplexType<Storage>::type;
    // Include subnormals, signed zero, threshold neighbors, overflow-risk
    // operands, infinities and NaNs. Compare bits, including NaN propagation.
    constexpr unsigned short half_bits[] = {0, 0x8000, 1, 0x8001, 0x0400,
        0x73fe, 0x73ff, 0x7400, 0x7bff, 0xfbff, 0x7c00, 0x7e01};
    constexpr unsigned short bf_bits[] = {0, 0x8000, 1, 0x8001, 0x0080,
        0x7e7e, 0x7e7f, 0x7e80, 0x7f7f, 0xff7f, 0x7f80, 0x7fc1};
    Storage values[12];
    for (unsigned i = 0; i < 12; ++i) {
        if constexpr (std::is_same_v<Storage, Fp16>) values[i] = __ushort_as_half(half_bits[i]);
        else { __nv_bfloat16_raw raw; raw.x = bf_bits[i]; values[i] = Storage(raw); }
    }
    Value roots[15];
    for (unsigned i = 0; i < 15; ++i)
        roots[i] = {LowStorage<Storage>::narrow(i % 2 ? 0.70710678F : 1.0F),
                    LowStorage<Storage>::narrow(i % 2 ? (inverse ? 0.70710678F : -0.70710678F) : 0.0F)};
    const LowFftOperator<Storage, Native> dynamic{roots, inverse, normalize, ComplexMultiply{}};
    const LowFftOperator<Storage, Native, Mode> specialized{roots, inverse, normalize, ComplexMultiply{}};
    for (unsigned i = 0; i < 12; ++i) for (unsigned j = 0; j < 12; ++j) {
        Value a{values[i], values[j]}, b{values[(i+3)%12], values[(j+7)%12]};
        Value c = a, d = b;
        for (unsigned stage = 0; stage < 4; ++stage) {
            dynamic.apply(stage, 0, a, b);
            specialized.apply(stage, 0, c, d);
            if (!same_bits(a,c) || !same_bits(b,d)) atomicAdd(failures, 1U);
        }
        for (unsigned lane = 0; lane < 2; ++lane) {
            const bool right = lane != 0;
            const auto x = dynamic.apply_lane(2, 1, a, b, right);
            const auto y = specialized.apply_lane(2, 1, c, d, right);
            if (!same_bits(x,y)) atomicAdd(failures, 1U);
        }
    }
}

template <typename Storage, bool Native> void run(unsigned* failures) {
    compare<Storage, Native, LowFftNormalization::Unscaled><<<1,1>>>(failures, false, false);
    compare<Storage, Native, LowFftNormalization::Unscaled><<<1,1>>>(failures, false, true);
    compare<Storage, Native, LowFftNormalization::Unscaled><<<1,1>>>(failures, true, false);
    compare<Storage, Native, LowFftNormalization::Normalized><<<1,1>>>(failures, true, true);
    CHECK(cudaGetLastError());
}

int main() {
    unsigned* failures;
    CHECK(cudaMallocManaged(&failures, sizeof(unsigned)));
    *failures = 0;
    run<Fp16, true>(failures); run<Fp16, false>(failures);
    run<Bf16, true>(failures); run<Bf16, false>(failures);
    CHECK(cudaDeviceSynchronize());
    const unsigned count = *failures;
    CHECK(cudaFree(failures));
    std::printf("low FFT specialization bit mismatches: %u\n", count);
    return count ? 1 : 0;
}
