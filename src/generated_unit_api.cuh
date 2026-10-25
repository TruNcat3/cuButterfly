#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "cuntt/butterfly.hpp"

namespace cuntt::detail {

bool          generated_register_fwht_available(std::uint32_t log_n) noexcept;
bool generated_register_fwht_available(std::uint32_t log_n, std::uint32_t threads) noexcept;
std::vector<std::uint32_t> generated_register_fwht_thread_options(std::uint32_t log_n);
std::uint32_t generated_register_fwht_threads(std::uint32_t log_n) noexcept;
void launch_generated_register_fwht(std::uint32_t log_n, const float* input, float* output, std::uint64_t transforms, std::uint64_t batch_distance,
                                    std::uint64_t element_stride, bool normalize, cudaStream_t stream, std::uint32_t threads = 0);
void launch_generated_register_fwht_zero_extended(
    std::uint32_t log_n, const float* input, float* output,
    std::uint64_t transforms, std::uint64_t input_batch_distance,
    std::uint64_t output_batch_distance, std::uint64_t input_element_stride,
    std::uint64_t output_element_stride, std::uint32_t logical_points,
    cudaStream_t stream, std::uint32_t threads = 0);

bool          generated_register_structured_available(std::uint32_t log_n) noexcept;
bool generated_register_structured_available(std::uint32_t log_n, std::uint32_t threads) noexcept;
std::vector<std::uint32_t> generated_register_structured_thread_options(std::uint32_t log_n);
std::uint32_t generated_register_structured_threads(std::uint32_t log_n) noexcept;
void launch_generated_register_structured(std::uint32_t log_n, const float* input, float* output,
                                          const float* matrices, std::uint64_t transforms,
                                          std::uint64_t batch_distance, std::uint64_t element_stride,
                                          bool broadcast, cudaStream_t stream, std::uint32_t threads = 0);
void launch_generated_register_structured_zero_extended(
    std::uint32_t log_n, const float* input, float* output,
    const float* matrices, std::uint64_t transforms,
    std::uint64_t input_batch_distance, std::uint64_t output_batch_distance,
    std::uint64_t input_element_stride, std::uint64_t output_element_stride,
    std::uint32_t logical_points, bool broadcast, cudaStream_t stream, std::uint32_t threads = 0);

bool generated_thread_dft8_available(std::uint32_t log_n) noexcept;
void launch_generated_thread_dft8(std::uint32_t log_n, const Complex32* input, Complex32* output, const Complex32* twiddles, std::uint64_t transforms,
                                  std::uint64_t batch_distance, std::uint64_t element_stride, bool normalize, cudaStream_t stream);

bool generated_cta_dft8_available(std::uint32_t log_n, std::uint32_t tile_threads) noexcept;
void launch_generated_cta_dft8(std::uint32_t log_n, const Complex32* input, Complex32* output, const Complex32* twiddles, std::uint64_t transforms,
                               std::uint64_t batch_distance, std::uint64_t element_stride, std::uint32_t tile_threads, bool normalize,
                               cudaStream_t stream);

bool generated_wmma_dft8_available(std::uint32_t log_n) noexcept;
void launch_generated_wmma_dft8(std::uint32_t log_n, const Complex32* input, Complex32* output, std::uint64_t transforms,
                                std::uint64_t batch_distance, std::uint64_t element_stride, bool inverse, bool normalize,
                                cudaStream_t stream);

}  // namespace cuntt::detail
