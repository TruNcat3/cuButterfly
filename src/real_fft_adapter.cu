#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

struct Complex32 {
    float real;
    float imag;
};

__global__ void pack_real_kernel(const float* input, Complex32* output,
                                 std::size_t elements, float scale) {
    const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index].real = input[index] * scale;
        output[index].imag = 0.0f;
    }
}

__global__ void expand_hermitian_kernel(const Complex32* half, Complex32* full,
                                        std::size_t height, std::size_t width,
                                        std::size_t batch) {
    const std::size_t total = batch * height * width;
    const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const auto column = index % width;
    const auto row = (index / width) % height;
    const auto batch_id = index / (height * width);
    const auto half_width = width / 2 + 1;
    if (column < half_width) {
        full[index] = half[batch_id * height * half_width + row * half_width + column];
        return;
    }
    const auto source_row = row == 0 ? 0 : height - row;
    const auto source_column = width - column;
    const auto source = half[batch_id * height * half_width + source_row * half_width + source_column];
    full[index] = {source.real, -source.imag};
}

}  // namespace

extern "C" int cubutterflyPackRealFp32(const float* input, void* output,
                                         std::size_t elements, float scale,
                                         std::uint64_t stream_value) {
    if (input == nullptr || output == nullptr || elements == 0) return 1;
    const auto blocks = static_cast<unsigned>((elements + 255) / 256);
    auto stream = reinterpret_cast<cudaStream_t>(stream_value);
    pack_real_kernel<<<blocks, 256, 0, stream>>>(input,
                                                  static_cast<Complex32*>(output),
                                                  elements, scale);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int cubutterflyExpandHermitianFp32(const void* half, void* full,
                                                std::uint32_t height,
                                                std::uint32_t width,
                                                std::size_t batch,
                                                std::uint64_t stream_value) {
    if (half == nullptr || full == nullptr || height == 0 || width == 0 || batch == 0) return 1;
    const std::size_t total = batch * std::size_t(height) * std::size_t(width);
    const auto blocks = static_cast<unsigned>((total + 255) / 256);
    auto stream = reinterpret_cast<cudaStream_t>(stream_value);
    expand_hermitian_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const Complex32*>(half), static_cast<Complex32*>(full),
        height, width, batch);
    return static_cast<int>(cudaGetLastError());
}
