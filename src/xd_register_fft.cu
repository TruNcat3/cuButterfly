#include "xd_register_fft.hpp"

#include "cuntt/butterfly.hpp"
#include "xd_cooperative_codelet.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

// The xD sources are compiled for the first configured CUDA architecture by
// the build.  Keep the same macro as the other xD lowerings so availability
// rejects a device whose runtime SM differs from the compiled codelet.
#ifndef CUBUTTERFLY_XD_CUFFTDX_SM
#define CUBUTTERFLY_XD_CUFFTDX_SM 700
#endif

namespace cuntt::detail {
namespace {

constexpr unsigned kBlockThreads = 128;

template <typename Real>
struct global_complex;

template <>
struct global_complex<float> {
    using type = Complex32;
};

template <>
struct global_complex<double> {
    using type = Complex64;
};

template <typename Real>
using global_complex_t = typename global_complex<Real>::type;

inline bool valid_length(std::size_t length) noexcept {
    return length == 64 || length == 128 || length == 256;
}

inline bool valid_cooperative_lanes(unsigned cooperative_lanes) noexcept {
    return cooperative_lanes == 4U || cooperative_lanes == 8U ||
           cooperative_lanes == 16U;
}

inline bool compiled_sm_matches_device() noexcept {
    int device = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess)
        return false;
    const int runtime_sm = properties.major * 100 + properties.minor * 10;
    return runtime_sm == CUBUTTERFLY_XD_CUFFTDX_SM;
}

template <unsigned Length>
__device__ __forceinline__ std::uint64_t natural_axis_base(
    std::uint64_t line, std::uint64_t suffix, unsigned suffix_shift) {
    // `line` enumerates suffix positions inside each outer row.  A power of
    // two suffix uses a shift; arbitrary suffixes retain the exact divide.
    const std::uint64_t outer = suffix_shift < 64
                                    ? (line >> suffix_shift)
                                    : (line / suffix);
    return line + outer * static_cast<std::uint64_t>(Length - 1U) * suffix;
}

template <typename Real, unsigned Length, unsigned CooperativeLanes, bool Inverse>
__launch_bounds__(kBlockThreads) __global__ void xd_register_fft_kernel(
    const global_complex_t<Real>* input, global_complex_t<Real>* output,
    std::uint64_t suffix, std::uint64_t lines, unsigned suffix_shift, Real scale) {
    using Complex = global_complex_t<Real>;
    constexpr unsigned CooperativeColumns = 32U / CooperativeLanes;
    constexpr unsigned LinesPerWarp = CooperativeColumns;
    constexpr unsigned LinesPerBlock = 4U * LinesPerWarp;
    constexpr unsigned ValuesPerThread = Length / CooperativeLanes;
    static_assert(Length == 64 || Length == 128 || Length == 256);
    static_assert(CooperativeLanes == 4U || CooperativeLanes == 8U ||
                      CooperativeLanes == 16U);
    static_assert(CooperativeLanes * CooperativeColumns == 32U);
    static_assert(Length % CooperativeLanes == 0U);

    const unsigned tid = threadIdx.x;
    // Each warp owns Columns independent lines.  Threads for one line are
    // separated by Columns, giving the cooperative helper a contiguous
    // shuffle group without requiring a CTA-wide exchange.
    const unsigned line_in_cta = (tid / 32U) * LinesPerWarp + tid % LinesPerWarp;
    const unsigned lane = (tid % 32U) / CooperativeColumns;
    const std::uint64_t line = static_cast<std::uint64_t>(blockIdx.x) *
                                   LinesPerBlock + line_in_cta;
    const bool active = line < lines;
    const std::uint64_t base = active
                                   ? natural_axis_base<Length>(line, suffix, suffix_shift)
                                   : 0;

    Complex values[ValuesPerThread];
#pragma unroll
    for (unsigned j = 0; j < ValuesPerThread; ++j) {
        if (active) {
            const unsigned axis = j * CooperativeLanes + lane;
            const auto source = input[base + static_cast<std::uint64_t>(axis) * suffix];
            values[j] = Complex{source.real, source.imag};
        } else {
            // Tail lines still execute every shuffle in their warp.  Zero
            // values keep those inactive transforms convergent and harmless.
            values[j] = Complex{Real(0), Real(0)};
        }
    }

    // Exact in-place operation is supported.  All cooperative lane roles for
    // every line have completed their global reads before any role can store.
    // The barrier is warp-local; the kernel has no shared staging or CTA
    // barrier on this non-contiguous path.
    __syncwarp();
    if constexpr (CooperativeLanes == 4U) {
        xd_cooperative_fft4<register_tile::power_log2(Length), CooperativeColumns,
                            Inverse>(values, lane);
    } else {
        xd_cooperative_fft_lanes<register_tile::power_log2(Length),
                                 CooperativeColumns, CooperativeLanes, Inverse>(
            values, lane);
    }

#pragma unroll
    for (unsigned j = 0; j < ValuesPerThread; ++j) {
        if (active) {
            unsigned output_lane;
            if constexpr (CooperativeLanes == 4U)
                output_lane = register_tile::codelet_output_lane<4U>(lane);
            else
                output_lane = dif_output_lane<CooperativeLanes>(lane);
            const unsigned axis = j + ValuesPerThread * output_lane;
            const Complex value = values[j];
            output[base + static_cast<std::uint64_t>(axis) * suffix] =
                Complex{value.real * scale, value.imag * scale};
        }
    }
}

template <typename Real, unsigned Length, unsigned CooperativeLanes, bool Inverse>
void launch_length_impl(const void* input, void* output, std::size_t suffix,
                        std::size_t lines, Real scale, cudaStream_t stream,
                        unsigned suffix_shift) {
    const std::uint64_t line_count = static_cast<std::uint64_t>(lines);
    constexpr unsigned LinesPerBlock = 128U / CooperativeLanes;
    const std::uint64_t blocks = line_count / LinesPerBlock +
                                 (line_count % LinesPerBlock != 0 ? 1U : 0U);
    if (blocks > 0x7fffffffULL)
        throw std::invalid_argument("xD register FFT grid exceeds CUDA grid.x");

    xd_register_fft_kernel<Real, Length, CooperativeLanes, Inverse>
        <<<static_cast<unsigned>(blocks), kBlockThreads, 0, stream>>>(
            static_cast<const global_complex_t<Real>*>(input),
            static_cast<global_complex_t<Real>*>(output),
            static_cast<std::uint64_t>(suffix), line_count, suffix_shift, scale);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("launch xD register FFT: ") +
                                 cudaGetErrorString(status));
}

template <typename Real, unsigned CooperativeLanes, bool Inverse>
void dispatch_length(std::size_t length, const void* input, void* output,
                     std::size_t suffix, std::size_t lines, Real scale,
                     cudaStream_t stream, unsigned suffix_shift) {
    switch (length) {
        case 64:
            launch_length_impl<Real, 64, CooperativeLanes, Inverse>(
                input, output, suffix, lines, scale, stream, suffix_shift);
            return;
        case 128:
            launch_length_impl<Real, 128, CooperativeLanes, Inverse>(
                input, output, suffix, lines, scale, stream, suffix_shift);
            return;
        case 256:
            if constexpr (CooperativeLanes == 8U || CooperativeLanes == 16U) {
                launch_length_impl<Real, 256, CooperativeLanes, Inverse>(
                    input, output, suffix, lines, scale, stream, suffix_shift);
            } else {
                throw std::invalid_argument(
                    "xD register FFT length 256 supports cooperative lanes 8 or 16");
            }
            return;
        default:
            throw std::invalid_argument(
                "xD register FFT supports lengths 64, 128 and 256");
    }
}

template <typename Real, bool Inverse>
void dispatch_lanes(unsigned cooperative_lanes, std::size_t length,
                    const void* input, void* output, std::size_t suffix,
                    std::size_t lines, Real scale, cudaStream_t stream,
                    unsigned suffix_shift) {
    switch (cooperative_lanes) {
        case 4U:
            dispatch_length<Real, 4U, Inverse>(length, input, output, suffix,
                                               lines, scale, stream, suffix_shift);
            return;
        case 8U:
            dispatch_length<Real, 8U, Inverse>(length, input, output, suffix,
                                               lines, scale, stream, suffix_shift);
            return;
        case 16U:
            dispatch_length<Real, 16U, Inverse>(length, input, output, suffix,
                                                lines, scale, stream, suffix_shift);
            return;
        default:
            throw std::invalid_argument(
                "xD register FFT supports cooperative lanes 4, 8, or 16");
    }
}

inline unsigned power_of_two_shift(std::size_t suffix) noexcept {
    unsigned shift = 0;
    while (suffix > 1) {
        suffix >>= 1;
        ++shift;
    }
    return shift;
}

}  // namespace

bool xd_register_fft_available(std::size_t length) noexcept {
    return xd_register_fft_configuration_available(length, 4U) ||
           xd_register_fft_configuration_available(length, 8U) ||
           xd_register_fft_configuration_available(length, 16U);
}

bool xd_register_fft_configuration_available(std::size_t length,
                                              unsigned cooperative_lanes) noexcept {
    if (!valid_length(length) || !valid_cooperative_lanes(cooperative_lanes) ||
        (length == 256 && cooperative_lanes == 4U))
        return false;
    return compiled_sm_matches_device();
}

void launch_xd_register_fft(const void* input, void* output, std::size_t length,
                            std::size_t suffix, std::size_t lines, bool fp64,
                            bool inverse, double scale, cudaStream_t stream,
                            unsigned cooperative_lanes) {
    if (!valid_length(length))
        throw std::invalid_argument(
            "xD register FFT supports lengths 64, 128 and 256");
    if (suffix <= 1)
        throw std::invalid_argument("xD register FFT requires a suffix stride greater than one");
    if (!valid_cooperative_lanes(cooperative_lanes))
        throw std::invalid_argument(
            "xD register FFT supports cooperative lanes 4, 8, or 16");
    if (length == 256 && cooperative_lanes == 4U)
        throw std::invalid_argument(
            "xD register FFT length 256 supports cooperative lanes 8 or 16");
    if (lines == 0)
        return;
    if (input == nullptr || output == nullptr)
        throw std::invalid_argument("xD register FFT requires non-null input and output");

    // A power-of-two suffix needs only a shift.  The sentinel 64 selects the
    // exact integer-division path for all other suffix values.
    unsigned suffix_shift = 64;
    if ((suffix & (suffix - 1U)) == 0)
        suffix_shift = power_of_two_shift(suffix);

    if (fp64) {
        if (inverse)
            dispatch_lanes<double, true>(cooperative_lanes, length, input, output,
                                         suffix, lines, scale, stream,
                                         suffix_shift);
        else
            dispatch_lanes<double, false>(cooperative_lanes, length, input, output,
                                          suffix, lines, scale, stream,
                                          suffix_shift);
    } else {
        const float scale_fp32 = static_cast<float>(scale);
        if (inverse)
            dispatch_lanes<float, true>(cooperative_lanes, length, input, output,
                                        suffix, lines, scale_fp32, stream,
                                        suffix_shift);
        else
            dispatch_lanes<float, false>(cooperative_lanes, length, input, output,
                                         suffix, lines, scale_fp32, stream,
                                         suffix_shift);
    }
}

}  // namespace cuntt::detail
