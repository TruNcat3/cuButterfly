#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace cuntt::detail {

// Launch one CTA-local C2C transform for every contiguous rows x cols plane.
// The FP32 inventory is rows, cols in {8, 16, 32, 64}; MathDx's Thread FFT
// database limits FP64 to rows, cols in {8, 16, 32}. `planes` is the number of
// adjacent planes in natural row-major storage. Exact in-place operation
// (input == output) is supported because each plane is staged in shared memory
// before it is exported. The CUDA implementation reserves one padded shared
// slot per physical row and rejects shapes above 48 KiB.
bool xd_pair_fft_available(std::size_t rows, std::size_t cols, bool fp64) noexcept;

void launch_xd_pair_fft(const void* input, void* output, std::size_t rows,
                        std::size_t cols, std::size_t planes, bool fp64,
                        bool inverse, double scale, cudaStream_t stream);

}  // namespace cuntt::detail
