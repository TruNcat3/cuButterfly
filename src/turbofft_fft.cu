#include <cuda_runtime.h>

#include <stdexcept>

#include "external_fft_units.cuh"
#include "turbofft/macro_ops.h"
#include "code_gen/generated/float2/fft_radix_2_logN_7_upload_0.cuh"
#include "code_gen/generated/float2/fft_radix_2_logN_8_upload_0.cuh"
#include "code_gen/generated/float2/fft_radix_2_logN_9_upload_0.cuh"
#include "code_gen/generated/float2/fft_radix_2_logN_10_upload_0.cuh"

namespace cuntt::detail {
namespace {

struct TurboPoint {
    int threadblock_batch;
    int worker_fft_size;
};

constexpr TurboPoint point(std::uint32_t log_n) {
    switch (log_n) {
        case 7: return {1, 8};
        case 8: return {1, 16};
        case 9: return {1, 8};
        case 10: return {1, 16};
        default: return {0, 0};
    }
}

template <int LogN, typename Kernel>
void launch(Kernel kernel, const Complex32* input, Complex32* output, std::uint64_t transforms) {
    constexpr TurboPoint selected = point(LogN);
    constexpr int block_threads = (1 << LogN) * selected.threadblock_batch / selected.worker_fft_size;
    constexpr int shared_bytes = (1 << LogN) * selected.threadblock_batch * sizeof(float2);
    const std::uint64_t blocks = (transforms + selected.threadblock_batch - 1) / selected.threadblock_batch;
    kernel<<<static_cast<unsigned int>(blocks), block_threads, shared_bytes>>>(
        reinterpret_cast<float2*>(const_cast<Complex32*>(input)), reinterpret_cast<float2*>(output),
        selected.threadblock_batch);
}

} // namespace

bool turbofft_generated_available(std::uint32_t log_n) noexcept { return log_n >= 7 && log_n <= 10; }

void launch_turbofft_generated(std::uint32_t log_n, const Complex32* input, Complex32* output, std::uint64_t transforms,
                               std::uint64_t batch_distance, std::uint64_t element_stride) {
    if (batch_distance != (1ULL << log_n) || element_stride != 1) {
        throw std::invalid_argument("turbofft-generated currently requires contiguous batches");
    }
    switch (log_n) {
        case 7: launch<7>(fft_7, input, output, transforms); return;
        case 8: launch<8>(fft_8, input, output, transforms); return;
        case 9: launch<9>(fft_9, input, output, transforms); return;
        case 10: launch<10>(fft_10, input, output, transforms); return;
        default: throw std::invalid_argument("turbofft-generated supports logN=7..10");
    }
}

} // namespace cuntt::detail
