#pragma once

#include <cstddef>
#include <cstdint>

#include "cuntt/butterfly.hpp"

namespace cuntt::detail {

bool cufftdx_block_available(std::uint32_t log_n) noexcept;
bool cufftdx_online_available(std::uint32_t log_n, std::uint32_t threads, std::uint32_t ept) noexcept;
void launch_cufftdx_block(std::uint32_t log_n, const Complex32* input, Complex32* output, std::uint64_t transforms,
                          std::uint64_t batch_distance, std::uint64_t element_stride, bool inverse, bool normalize);
bool cufftdx_direct_available(std::uint32_t log_n) noexcept;
void launch_cufftdx_direct(std::uint32_t log_n, const Complex32* input, Complex32* output,
                           std::uint64_t transforms, std::uint64_t batch_distance, std::uint64_t element_stride,
                           bool inverse, bool normalize, std::uint32_t tile_threads);
void launch_cufftdx_online_reorder(std::uint32_t log_n, std::uint32_t local_log_n, const Complex32* input,
                                   Complex32* output, Complex32* scratch, const Complex32* twiddles,
                                   std::uint64_t transforms, std::uint64_t batch_distance,
                                   std::uint64_t element_stride, bool inverse, bool normalize,
                                   CrossTwiddleMode cross_twiddle, std::uint32_t prefix_threads,
                                   std::uint32_t suffix_threads, std::uint32_t prefix_ept,
                                   std::uint32_t suffix_ept, DirectBoundary direct_boundary);
bool cufftdx_resident_available(std::uint32_t log_n, std::uint32_t local_log_n) noexcept;
void launch_cufftdx_resident(std::uint32_t log_n, std::uint32_t local_log_n, const Complex32* input,
                             Complex32* output, Complex32* scratch, const Complex32* twiddles, std::uint64_t transforms,
                             std::uint64_t batch_distance, std::uint64_t element_stride, bool inverse,
                             bool normalize, CrossTwiddleMode cross_twiddle, std::uint32_t tile_threads);

bool turbofft_generated_available(std::uint32_t log_n) noexcept;
void launch_turbofft_generated(std::uint32_t log_n, const Complex32* input, Complex32* output, std::uint64_t transforms,
                               std::uint64_t batch_distance, std::uint64_t element_stride);

} // namespace cuntt::detail
