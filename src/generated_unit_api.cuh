#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "cuntt/butterfly.hpp"

namespace cuntt::detail {

bool          generated_register_fwht_available(std::uint32_t log_n) noexcept;
std::uint32_t generated_register_fwht_threads(std::uint32_t log_n) noexcept;
void launch_generated_register_fwht(std::uint32_t log_n, const float* input, float* output, std::uint64_t transforms, std::uint64_t batch_distance,
                                    std::uint64_t element_stride, bool normalize);

bool generated_thread_dft8_available(std::uint32_t log_n) noexcept;
void launch_generated_thread_dft8(std::uint32_t log_n, const Complex32* input, Complex32* output, const Complex32* twiddles, std::uint64_t transforms,
                                  std::uint64_t batch_distance, std::uint64_t element_stride, bool normalize);

bool generated_cta_dft8_available(std::uint32_t log_n, std::uint32_t tile_threads) noexcept;
void launch_generated_cta_dft8(std::uint32_t log_n, const Complex32* input, Complex32* output, const Complex32* twiddles, std::uint64_t transforms,
                               std::uint64_t batch_distance, std::uint64_t element_stride, std::uint32_t tile_threads, bool normalize);

bool generated_wmma_dft8_available(std::uint32_t log_n) noexcept;
void launch_generated_wmma_dft8(std::uint32_t log_n, const Complex32* input, Complex32* output, std::uint64_t transforms,
                                std::uint64_t batch_distance, std::uint64_t element_stride, bool inverse, bool normalize);

}  // namespace cuntt::detail
