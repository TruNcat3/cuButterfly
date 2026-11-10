#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace cuntt::detail {

// A CTA-local two-level FFT for one independent line per local slot. The
// implementation is compiled for the configured xD SM and supports only the
// finite local-core inventory (64, 128, and 256 points).
bool xd_local_fft_available(std::size_t length) noexcept;

// Return the actual number of independent lines assigned to one CTA. A zero
// cap selects the default eight-line configuration. N64 also accepts cap 32;
// FP64 length 256 is kept at eight lines even when cap 16 is requested so
// shared memory stays bounded.
unsigned xd_local_fft_lines_per_block(std::size_t length, bool fp64,
                                      unsigned cap = 0) noexcept;

void launch_xd_local_fft(const void* input, void* output, std::size_t length,
                         std::size_t suffix, std::size_t lines, bool fp64,
                         bool inverse, double scale, cudaStream_t stream,
                         unsigned block_cap = 0, unsigned twiddle_policy = 0);

}  // namespace cuntt::detail
