#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cuntt::detail {

// One CTA-local FFT for every independent line of a natural row-major axis.
// `suffix` is the distance between adjacent axis elements and `lines` includes
// all outer prefixes and suffix positions.  The implementation is optional
// and is unavailable when cuFFTDx was not compiled into the library.
bool xd_axis_fft_available(std::size_t length) noexcept;

constexpr bool xd_axis_fft_length_supported(std::size_t length) noexcept {
    return length >= 8 && length <= 4096 && (length & (length - 1)) == 0;
}

constexpr unsigned xd_axis_fft_ept(std::size_t length, unsigned requested) noexcept {
    return length < requested ? static_cast<unsigned>(length) : requested;
}

constexpr unsigned xd_axis_fft_block_batch(std::size_t length, bool fp64,
                                           unsigned requested,
                                           unsigned ffts_per_block_cap = 0,
                                           unsigned exact_ffts_per_block = 0) noexcept {
    if (exact_ffts_per_block != 0) return exact_ffts_per_block;
    // Larger cores use one line per CTA instead of making the historical
    // 256-thread/32-KiB packing target a limit on supported axis lengths.
    const unsigned thread_values = length > 1024 ? static_cast<unsigned>(length)
                                                : 256U * requested;
    const unsigned shared_values = length > 1024 ? static_cast<unsigned>(length)
                                                : (fp64 ? 2048U : 4096U);
    const unsigned values = thread_values < shared_values ? thread_values : shared_values;
    const unsigned default_lines = length ? static_cast<unsigned>(values / length) : 0;
    const unsigned lines = default_lines < 32U ? default_lines : 32U;
    return ffts_per_block_cap != 0 && ffts_per_block_cap < lines
               ? ffts_per_block_cap
               : lines;
}

constexpr bool xd_axis_fft_configuration_supported(
    std::size_t length, bool fp64, unsigned requested,
    unsigned ffts_per_block_cap = 0,
    unsigned exact_ffts_per_block = 0) noexcept {
    if (exact_ffts_per_block != 0)
        return (length == 2048 || length == 4096) &&
               (requested == 8 || requested == 16) && ffts_per_block_cap == 0 &&
               (exact_ffts_per_block == 2 || exact_ffts_per_block == 4) &&
               length / requested * exact_ffts_per_block <= 1024;
    return xd_axis_fft_length_supported(length) &&
           (requested == 4 || requested == 8 || requested == 16) &&
           (length <= 1024 || requested >= 8) &&
           (ffts_per_block_cap == 0 || ffts_per_block_cap == 8 ||
            ffts_per_block_cap == 16) &&
           xd_axis_fft_block_batch(length, fp64, requested, ffts_per_block_cap) != 0;
}

bool xd_axis_fft_exact_configuration_available(std::size_t length, bool fp64,
    unsigned requested, unsigned exact_ffts_per_block, unsigned exchange_policy) noexcept;

inline bool xd_axis_fft_configuration_available(
    std::size_t length, bool fp64, unsigned requested,
    unsigned ffts_per_block_cap = 0, unsigned exact_ffts_per_block = 0,
    unsigned exchange_policy = 0) noexcept {
    if (!xd_axis_fft_configuration_supported(length, fp64, requested,
                                             ffts_per_block_cap, exact_ffts_per_block) ||
        !xd_axis_fft_available(length)) return false;
    if (exact_ffts_per_block != 0)
        return xd_axis_fft_exact_configuration_available(length, fp64, requested,
                                                         exact_ffts_per_block, exchange_policy);
    int device = 0;
    int shared_limit = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               device) != cudaSuccess) return false;
    const std::size_t tile_bytes = xd_axis_fft_block_batch(length, fp64, requested,
        ffts_per_block_cap) * (length + 1) * (fp64 ? 16U : 8U);
    return tile_bytes <= static_cast<std::size_t>(shared_limit);
}

// Policy 3 keeps global element-index arithmetic in uint32_t. The logical
// tensor span is length * lines; suffix is the stride used by the same
// address mapping and must independently fit in the index type.
constexpr bool xd_axis_fft_u32_index_fits(std::size_t length,
                                          std::size_t suffix,
                                          std::size_t lines) noexcept {
    constexpr std::size_t max_index =
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
    return length != 0 && suffix != 0 && suffix <= max_index && lines % suffix == 0 &&
           lines <= max_index && lines <= max_index / length;
}

void launch_xd_axis_fft(const void* input, void* output, std::size_t length,
                        std::size_t suffix, std::size_t lines, bool fp64,
                        bool inverse, double scale, cudaStream_t stream,
                        unsigned elements_per_thread = 8,
                        unsigned ffts_per_block_cap = 0,
                        unsigned exchange_policy = 0,
                        unsigned exact_ffts_per_block = 0);

}  // namespace cuntt::detail
