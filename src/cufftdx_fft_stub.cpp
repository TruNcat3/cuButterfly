#include <stdexcept>

#include "external_fft_units.cuh"

namespace cuntt::detail {

bool cufftdx_block_available(std::uint32_t) noexcept { return false; }
bool cufftdx_online_available(std::uint32_t, std::uint32_t, std::uint32_t) noexcept { return false; }

void launch_cufftdx_block(std::uint32_t, const Complex32*, Complex32*, std::uint64_t, std::uint64_t, std::uint64_t, bool, bool,
                          cudaStream_t) {
    throw std::runtime_error("cufftdx-block was not enabled at build time");
}

bool cufftdx_fp64_block_available(std::uint32_t) noexcept { return false; }
bool cufftdx_fp64_online_available(std::uint32_t, std::uint32_t, std::uint32_t) noexcept { return false; }

void launch_cufftdx_fp64_block(std::uint32_t, const Complex64*, Complex64*, std::uint64_t, std::uint64_t,
                               std::uint64_t, bool, bool, cudaStream_t) {
    throw std::runtime_error("FP64 cufftdx-block was not enabled at build time");
}

void launch_cufftdx_fp64_online_reorder(std::uint32_t, std::uint32_t, const Complex64*, Complex64*, Complex64*,
                                        const Complex64*, std::uint64_t, std::uint64_t, std::uint64_t, bool, bool,
                                        CrossTwiddleMode, SharedLayout, std::uint32_t, std::uint32_t,
                                        std::uint32_t, std::uint32_t, cudaStream_t) {
    throw std::runtime_error("FP64 online cufftdx-block was not enabled at build time");
}

bool cufftdx_direct_available(std::uint32_t) noexcept { return false; }

void launch_cufftdx_direct(std::uint32_t, const Complex32*, Complex32*, std::uint64_t, std::uint64_t,
                           std::uint64_t, bool, bool, std::uint32_t, cudaStream_t) {
    throw std::runtime_error("cufftdx-direct was not enabled at build time");
}

void launch_cufftdx_online_reorder(std::uint32_t, std::uint32_t, const Complex32*, Complex32*, Complex32*, Complex32*,
                                   const Complex32*, std::uint64_t, std::uint64_t, std::uint64_t, bool, bool,
                                   CrossTwiddleMode, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
                                   DirectBoundary, cudaStream_t) {
    throw std::runtime_error("cufftdx-block was not enabled at build time");
}

void launch_cufftdx_multisegment(std::uint32_t, const std::vector<std::uint32_t>&,
                                 const std::vector<FftSegmentMapping>&, const std::vector<FftBoundaryMapping>&,
                                 const Complex32*, Complex32*, Complex32*, Complex32*, const Complex32*,
                                 std::uint64_t, std::uint64_t, std::uint64_t, bool, bool, cudaStream_t) {
    throw std::runtime_error("cufftdx-block was not enabled at build time");
}

bool cufftdx_resident_available(std::uint32_t, std::uint32_t) noexcept { return false; }

void launch_cufftdx_resident(std::uint32_t, std::uint32_t, const Complex32*, Complex32*, Complex32*, const Complex32*,
                             std::uint64_t, std::uint64_t, std::uint64_t, bool, bool, CrossTwiddleMode,
                             std::uint32_t, cudaStream_t) {
    throw std::runtime_error("cufftdx-resident was not enabled at build time");
}

} // namespace cuntt::detail
