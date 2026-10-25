#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace cuntt::detail {

// Register-only cooperative FFTs for a non-contiguous natural tensor axis.
// The optional implementation is compiled for one CUDA SM and is intended
// only for axis lengths 64, 128 and 256 with a suffix stride greater than one.
bool xd_register_fft_available(std::size_t length) noexcept;

// Return whether one exact length/lane configuration is compiled and usable
// on the current SM. Length 256 intentionally exposes only lanes 8 and 16;
// callers must not silently substitute the unsupported lane-4 configuration.
bool xd_register_fft_configuration_available(std::size_t length,
                                              unsigned cooperative_lanes) noexcept;

void launch_xd_register_fft(const void* input, void* output, std::size_t length,
                            std::size_t suffix, std::size_t lines, bool fp64,
                            bool inverse, double scale, cudaStream_t stream,
                            unsigned cooperative_lanes = 4);

}  // namespace cuntt::detail
