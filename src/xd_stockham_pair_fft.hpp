#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace cuntt::detail {

// Experimental physical choices; none changes the requested 32x32 C2C FFT.
enum class XdStagePairVariant {
    Radix2 = 0,
    Radix4Stage = 1,
    Radix4Register = 2,
    Radix4RegisterSplit = 3,
};

// Finite compiled inventory: one CTA owns one 32x32 plane. The register
// variant supports FP32/FP64 and is exposed through xd_pair_core; the FP32-only
// variants are diagnostics.
bool xd_stockham_pair_fft_available(std::size_t rows, std::size_t cols,
                                    bool fp64,
                                    XdStagePairVariant variant = XdStagePairVariant::Radix4Register) noexcept;

void launch_xd_stockham_pair_fft(const void* input, void* output,
                                 std::size_t rows, std::size_t cols,
                                 std::size_t planes, bool fp64, bool inverse,
                                 double scale, cudaStream_t stream,
                                 XdStagePairVariant variant = XdStagePairVariant::Radix2,
                                 void* scratch = nullptr);

// Only the split organization needs caller-owned scratch (32*32*planes
// complex FP32 elements). All allocation and preparation precede capture.

}  // namespace cuntt::detail
