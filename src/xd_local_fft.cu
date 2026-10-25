#include "xd_local_fft.hpp"

#include "cuntt/butterfly.hpp"

#ifndef CUBUTTERFLY_XD_CUFFTDX_SM
#define CUBUTTERFLY_XD_CUFFTDX_SM 800
#endif

#include "xd_cooperative_codelet.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cuntt::detail {
namespace {

constexpr std::size_t kShortLength = 64;
constexpr std::size_t kMinLength = 128;
constexpr std::size_t kMaxLength = 256;
constexpr unsigned kDefaultLines = 8;
constexpr unsigned kWideLines = 16;
constexpr unsigned kMaxLines = 32;
constexpr unsigned kShortRoles = 8;
constexpr unsigned kLongRoles = 16;
constexpr std::uint64_t kMaxGridX = 0x7fffffffULL;

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

constexpr bool valid_length(std::size_t length) noexcept {
    return length == kShortLength || length == kMinLength || length == kMaxLength;
}

constexpr bool valid_cap_for_length(std::size_t length, unsigned cap) noexcept {
    if (cap == 0U || cap == kDefaultLines || cap == kWideLines)
        return true;
    return length == kShortLength && cap == kMaxLines;
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
    std::uint64_t line, std::uint64_t suffix) {
    const std::uint64_t outer = line / suffix;
    return line + outer * static_cast<std::uint64_t>(Length - 1U) * suffix;
}

template <unsigned Length, unsigned Roles, unsigned Lines>
__device__ __forceinline__ unsigned shared_index(unsigned a, unsigned k,
                                                 unsigned slot) {
    constexpr unsigned K = Length / Roles;
    // XOR only permutes k inside the fixed a row.  The a*K term gives each
    // row a disjoint range and slot occupies the contiguous line dimension.
    const unsigned permuted_k = k ^ (a & (K - 1U));
    return (a * K + permuted_k) * Lines + slot;
}

template <typename Real, unsigned Length, unsigned Roles, unsigned Lines, bool Inverse,
          bool Recurrence>
__launch_bounds__(Roles * Lines) __global__ void xd_local_fft_kernel(
    const global_complex_t<Real>* input, global_complex_t<Real>* output,
    std::uint64_t suffix, std::uint64_t lines, Real scale) {
    using Complex = global_complex_t<Real>;
    constexpr unsigned K = Length / Roles;
    constexpr unsigned SharedValues = Length * Lines;

    extern __shared__ __align__(16) unsigned char storage[];
    auto* shared_real = reinterpret_cast<Real*>(storage);
    auto* shared_imag = shared_real + SharedValues;

    const unsigned tid = threadIdx.x;
    const unsigned slot = tid % Lines;
    const unsigned role = tid / Lines;
    const std::uint64_t line = static_cast<std::uint64_t>(blockIdx.x) * Lines + slot;
    const bool active = line < lines;
    const std::uint64_t base = active ? natural_axis_base<Length>(line, suffix) : 0;

    Complex values[K];
#pragma unroll
    for (unsigned b = 0; b < K; ++b) {
        if (active) {
            const auto source = input[base +
                                      static_cast<std::uint64_t>(role + Roles * b) *
                                          suffix];
            values[b] = Complex{source.real, source.imag};
        } else {
            values[b] = Complex{Real(0), Real(0)};
        }
    }

    if constexpr (K == 8U)
        xd_cooperative_detail::local_fft<3U, Inverse>(values);
    else
        xd_cooperative_detail::local_fft<4U, Inverse>(values);

    if constexpr (Recurrence) {
        Complex step;
        if constexpr (Length == 256U) {
            step = xd_cooperative_detail::table_twiddle256<Inverse, Complex>(role);
        } else if constexpr (Length == kShortLength) {
            // The shared root table is W_128; doubling the exponent gives W_64.
            step = xd_cooperative_detail::table_twiddle<Inverse, Complex>(2U * role);
        } else {
            step = xd_cooperative_detail::table_twiddle<Inverse, Complex>(role);
        }
        Complex running{Real(1), Real(0)};
#pragma unroll
        for (unsigned k = 0; k < K; ++k) {
            Complex value;
            if (k == 0U) {
                // The unity phase is already represented by the input value.
                value = values[k];
            } else {
                running = register_tile::multiply(running, step);
                value = register_tile::multiply(values[k], running);
            }
            const unsigned index = shared_index<Length, Roles, Lines>(role, k, slot);
            shared_real[index] = value.real;
            shared_imag[index] = value.imag;
        }
    } else {
#pragma unroll
        for (unsigned k = 0; k < K; ++k) {
            Complex value = values[k];
            const unsigned phase_index =
                (Length == kShortLength ? 2U : 1U) * role * k;
            if constexpr (Length == 256U) {
                value = register_tile::multiply(
                    value, xd_cooperative_detail::table_twiddle256<Inverse, Complex>(
                               phase_index));
            } else {
                value = register_tile::multiply(
                    value,
                    xd_cooperative_detail::table_twiddle<Inverse, Complex>(phase_index));
            }
            const unsigned index = shared_index<Length, Roles, Lines>(role, k, slot);
            shared_real[index] = value.real;
            shared_imag[index] = value.imag;
        }
    }

    // Every role writes its complete row before any role begins the second
    // factor.  This also makes exact in-place input/output safe.
    __syncthreads();

    const unsigned k = role;
    if (k < K) {
        Complex second_stage[Roles];
#pragma unroll
        for (unsigned a = 0; a < Roles; ++a) {
            const unsigned index = shared_index<Length, Roles, Lines>(a, k, slot);
            second_stage[a] = Complex{shared_real[index], shared_imag[index]};
        }
        if constexpr (Roles == 8U)
            xd_cooperative_detail::local_fft<3U, Inverse>(second_stage);
        else
            xd_cooperative_detail::local_fft<4U, Inverse>(second_stage);

        if (active) {
#pragma unroll
            for (unsigned q = 0; q < Roles; ++q) {
                const Complex value = second_stage[q];
                const unsigned axis = k + K * q;
                output[base + static_cast<std::uint64_t>(axis) * suffix] =
                    Complex{value.real * scale, value.imag * scale};
            }
        }
    }
}

inline std::uint64_t checked_u64(std::size_t value, const char* what) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (value > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()))
            throw std::invalid_argument(std::string("xD local FFT ") + what +
                                        " exceeds uint64 address range");
    }
    return static_cast<std::uint64_t>(value);
}

void validate_extent(std::size_t length, std::size_t suffix, std::size_t lines) {
    const std::uint64_t suffix_u64 = checked_u64(suffix, "suffix");
    const std::uint64_t lines_u64 = checked_u64(lines, "line count");
    const std::uint64_t last_line = lines_u64 - 1U;
    const std::uint64_t outer = last_line / suffix_u64;
    const std::uint64_t axis_minus_one =
        static_cast<std::uint64_t>(length - 1U);

    if (outer > std::numeric_limits<std::uint64_t>::max() / axis_minus_one)
        throw std::invalid_argument("xD local FFT axis base exceeds uint64 address range");
    const std::uint64_t row_factor = outer * axis_minus_one;
    if (row_factor > std::numeric_limits<std::uint64_t>::max() / suffix_u64)
        throw std::invalid_argument("xD local FFT axis base exceeds uint64 address range");
    const std::uint64_t row_offset = row_factor * suffix_u64;

    if (last_line > std::numeric_limits<std::uint64_t>::max() - row_offset)
        throw std::invalid_argument("xD local FFT axis base exceeds uint64 address range");
    const std::uint64_t last_base = last_line + row_offset;
    if (axis_minus_one > std::numeric_limits<std::uint64_t>::max() / suffix_u64)
        throw std::invalid_argument("xD local FFT address span exceeds uint64 range");
    const std::uint64_t axis_span = axis_minus_one * suffix_u64;
    if (last_base > std::numeric_limits<std::uint64_t>::max() - axis_span)
        throw std::invalid_argument("xD local FFT address span exceeds uint64 range");
}

template <typename Real, unsigned Length, unsigned Roles, unsigned Lines, bool Inverse,
          bool Recurrence>
void launch_impl(const void* input, void* output, std::uint64_t suffix,
                 std::uint64_t lines, Real scale, cudaStream_t stream) {
    constexpr std::size_t shared_bytes =
        2U * static_cast<std::size_t>(Length) * Lines * sizeof(Real);
    static_assert(shared_bytes <= 48U * 1024U,
                  "xD local FFT exceeds the default shared-memory budget");

    const std::uint64_t blocks = lines / Lines + (lines % Lines != 0U ? 1U : 0U);
    if (blocks > kMaxGridX)
        throw std::invalid_argument("xD local FFT grid exceeds CUDA grid.x");

    xd_local_fft_kernel<Real, Length, Roles, Lines, Inverse, Recurrence>
        <<<static_cast<unsigned>(blocks), Roles * Lines, shared_bytes, stream>>>(
            static_cast<const global_complex_t<Real>*>(input),
            static_cast<global_complex_t<Real>*>(output), suffix, lines, scale);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("launch xD local FFT: ") +
                                 cudaGetErrorString(status));
}

template <typename Real, unsigned Length, unsigned Roles, bool Inverse>
void dispatch_lines(unsigned lines_per_block, const void* input, void* output,
                    std::uint64_t suffix, std::uint64_t lines, Real scale,
                    cudaStream_t stream, unsigned twiddle_policy) {
    if (twiddle_policy > 1U)
        throw std::invalid_argument("xD local FFT twiddle policy must be 0 or 1");
    switch (lines_per_block) {
        case kDefaultLines:
            if (twiddle_policy == 0U)
                launch_impl<Real, Length, Roles, kDefaultLines, Inverse, false>(
                    input, output, suffix, lines, scale, stream);
            else
                launch_impl<Real, Length, Roles, kDefaultLines, Inverse, true>(
                    input, output, suffix, lines, scale, stream);
            return;
        case kWideLines:
            if constexpr (std::is_same_v<Real, double> && Length == 256U) {
                throw std::invalid_argument(
                    "xD local FFT FP64 length 256 supports only eight lines per block");
            } else {
                if (twiddle_policy == 0U)
                    launch_impl<Real, Length, Roles, kWideLines, Inverse, false>(
                        input, output, suffix, lines, scale, stream);
                else
                    launch_impl<Real, Length, Roles, kWideLines, Inverse, true>(
                        input, output, suffix, lines, scale, stream);
            }
            return;
        case kMaxLines:
            if constexpr (Length == kShortLength) {
                if (twiddle_policy == 0U)
                    launch_impl<Real, Length, Roles, kMaxLines, Inverse, false>(
                        input, output, suffix, lines, scale, stream);
                else
                    launch_impl<Real, Length, Roles, kMaxLines, Inverse, true>(
                        input, output, suffix, lines, scale, stream);
            } else {
                throw std::invalid_argument(
                    "xD local FFT line configuration 32 is only valid for length 64");
            }
            return;
        default:
            throw std::invalid_argument("xD local FFT has an unsupported line configuration");
    }
}

template <typename Real, bool Inverse>
void dispatch_length(std::size_t length, unsigned lines_per_block,
                     const void* input, void* output, std::uint64_t suffix,
                     std::uint64_t lines, Real scale, cudaStream_t stream,
                     unsigned twiddle_policy) {
    switch (length) {
        case 64:
            dispatch_lines<Real, 64U, kShortRoles, Inverse>(
                lines_per_block, input, output, suffix, lines, scale, stream,
                twiddle_policy);
            return;
        case 128:
            dispatch_lines<Real, 128U, kLongRoles, Inverse>(
                lines_per_block, input, output, suffix, lines, scale, stream,
                twiddle_policy);
            return;
        case 256:
            dispatch_lines<Real, 256U, kLongRoles, Inverse>(
                lines_per_block, input, output, suffix, lines, scale, stream,
                twiddle_policy);
            return;
        default:
            throw std::invalid_argument(
                "xD local FFT supports lengths 64, 128, and 256");
    }
}

}  // namespace

bool xd_local_fft_available(std::size_t length) noexcept {
    return valid_length(length) && compiled_sm_matches_device();
}

unsigned xd_local_fft_lines_per_block(std::size_t length, bool fp64,
                                      unsigned cap) noexcept {
    if (!valid_length(length) || !valid_cap_for_length(length, cap))
        return 0U;
    if (fp64 && length == 256U)
        return kDefaultLines;
    return cap == kMaxLines ? kMaxLines
                            : (cap == kWideLines ? kWideLines : kDefaultLines);
}

void launch_xd_local_fft(const void* input, void* output, std::size_t length,
                         std::size_t suffix, std::size_t lines, bool fp64,
                         bool inverse, double scale, cudaStream_t stream,
                         unsigned block_cap, unsigned twiddle_policy) {
    if (!valid_length(length))
        throw std::invalid_argument(
            "xD local FFT supports lengths 64, 128, and 256");
    if (suffix == 0U)
        throw std::invalid_argument("xD local FFT requires a nonzero suffix");
    if (!valid_cap_for_length(length, block_cap))
        throw std::invalid_argument(
            "xD local FFT line cap is invalid for the requested length");
    if (twiddle_policy > 1U)
        throw std::invalid_argument("xD local FFT twiddle policy must be 0 or 1");
    if (!std::isfinite(scale))
        throw std::invalid_argument("xD local FFT scale must be finite");
    if (lines == 0U)
        return;
    if (input == nullptr || output == nullptr)
        throw std::invalid_argument("xD local FFT requires non-null input and output");

    validate_extent(length, suffix, lines);
    const unsigned lines_per_block =
        xd_local_fft_lines_per_block(length, fp64, block_cap);
    if (lines_per_block == 0U)
        throw std::invalid_argument("xD local FFT has no valid line configuration");
    const std::uint64_t suffix_u64 = checked_u64(suffix, "suffix");
    const std::uint64_t lines_u64 = checked_u64(lines, "line count");

    if (fp64) {
        if (inverse)
            dispatch_length<double, true>(length, lines_per_block, input, output,
                                          suffix_u64, lines_u64, scale, stream,
                                          twiddle_policy);
        else
            dispatch_length<double, false>(length, lines_per_block, input, output,
                                           suffix_u64, lines_u64, scale, stream,
                                           twiddle_policy);
    } else {
        const float scale_fp32 = static_cast<float>(scale);
        if (inverse)
            dispatch_length<float, true>(length, lines_per_block, input, output,
                                         suffix_u64, lines_u64, scale_fp32, stream,
                                         twiddle_policy);
        else
            dispatch_length<float, false>(length, lines_per_block, input, output,
                                          suffix_u64, lines_u64, scale_fp32, stream,
                                          twiddle_policy);
    }
}

}  // namespace cuntt::detail
