#pragma once

#include "cuntt/butterfly.hpp"
#include "fft_register_tile.cuh"

#include <cufftdx.hpp>

// This helper is compiled together with the xD CUDA sources, which select one
// architecture for their MathDx codelets.  Keep a fallback for standalone
// syntax checks while allowing the build to override the selected SM.
#ifndef CUBUTTERFLY_XD_CUFFTDX_SM
#ifdef CUBUTTERFLY_CUFFTDX_SM
#define CUBUTTERFLY_XD_CUFFTDX_SM CUBUTTERFLY_CUFFTDX_SM
#else
#define CUBUTTERFLY_XD_CUFFTDX_SM 700
#endif
#endif

namespace cuntt::detail {

namespace xd_cooperative_detail {

// Positive-angle roots for W_128^k. All cooperative lane widths use
// these binary-rational phases instead of expanding sincospi into a large
// per-thread polynomial.  Forward selects the conjugate phase; inverse
// selects the stored positive phase.  This table costs 2 KiB of constant
// memory in a translation unit that instantiates the generic path.
static __device__ __constant__ double2 kTwiddleRoots128[128] = {
    double2{1.0, 0.0},
    double2{0.99879545620517239, 0.049067674327418014},
    double2{0.99518472667219689, 0.098017140329560602},
    double2{0.98917650996478097, 0.14673047445536175},
    double2{0.98078528040323045, 0.19509032201612827},
    double2{0.97003125319454399, 0.24298017990326389},
    double2{0.95694033573220886, 0.29028467725446237},
    double2{0.94154406518302078, 0.33688985339222005},
    double2{0.92387953251128676, 0.38268343236508977},
    double2{0.90398929312344333, 0.42755509343028209},
    double2{0.88192126434835503, 0.47139673682599765},
    double2{0.85772861000027207, 0.51410274419322173},
    double2{0.83146961230254524, 0.55557023301960222},
    double2{0.80320753148064491, 0.59569930449243334},
    double2{0.77301045336273696, 0.6343932841636455},
    double2{0.74095112535495909, 0.6715589548470184},
    double2{0.70710678118654752, 0.70710678118654752},
    double2{0.6715589548470184, 0.74095112535495909},
    double2{0.6343932841636455, 0.77301045336273696},
    double2{0.59569930449243334, 0.80320753148064491},
    double2{0.55557023301960222, 0.83146961230254524},
    double2{0.51410274419322173, 0.85772861000027207},
    double2{0.47139673682599765, 0.88192126434835503},
    double2{0.42755509343028209, 0.90398929312344333},
    double2{0.38268343236508977, 0.92387953251128676},
    double2{0.33688985339222005, 0.94154406518302078},
    double2{0.29028467725446237, 0.95694033573220886},
    double2{0.24298017990326389, 0.97003125319454399},
    double2{0.19509032201612827, 0.98078528040323045},
    double2{0.14673047445536175, 0.98917650996478097},
    double2{0.098017140329560602, 0.99518472667219689},
    double2{0.049067674327418014, 0.99879545620517239},
    double2{0.0, 1.0},
    double2{-0.049067674327418014, 0.99879545620517239},
    double2{-0.098017140329560602, 0.99518472667219689},
    double2{-0.14673047445536175, 0.98917650996478097},
    double2{-0.19509032201612827, 0.98078528040323045},
    double2{-0.24298017990326389, 0.97003125319454399},
    double2{-0.29028467725446237, 0.95694033573220886},
    double2{-0.33688985339222005, 0.94154406518302078},
    double2{-0.38268343236508977, 0.92387953251128676},
    double2{-0.42755509343028209, 0.90398929312344333},
    double2{-0.47139673682599765, 0.88192126434835503},
    double2{-0.51410274419322173, 0.85772861000027207},
    double2{-0.55557023301960222, 0.83146961230254524},
    double2{-0.59569930449243334, 0.80320753148064491},
    double2{-0.6343932841636455, 0.77301045336273696},
    double2{-0.6715589548470184, 0.74095112535495909},
    double2{-0.70710678118654752, 0.70710678118654752},
    double2{-0.74095112535495909, 0.6715589548470184},
    double2{-0.77301045336273696, 0.6343932841636455},
    double2{-0.80320753148064491, 0.59569930449243334},
    double2{-0.83146961230254524, 0.55557023301960222},
    double2{-0.85772861000027207, 0.51410274419322173},
    double2{-0.88192126434835503, 0.47139673682599765},
    double2{-0.90398929312344333, 0.42755509343028209},
    double2{-0.92387953251128676, 0.38268343236508977},
    double2{-0.94154406518302078, 0.33688985339222005},
    double2{-0.95694033573220886, 0.29028467725446237},
    double2{-0.97003125319454399, 0.24298017990326389},
    double2{-0.98078528040323045, 0.19509032201612827},
    double2{-0.98917650996478097, 0.14673047445536175},
    double2{-0.99518472667219689, 0.098017140329560602},
    double2{-0.99879545620517239, 0.049067674327418014},
    double2{-1.0, 0.0},
    double2{-0.99879545620517239, -0.049067674327418014},
    double2{-0.99518472667219689, -0.098017140329560602},
    double2{-0.98917650996478097, -0.14673047445536175},
    double2{-0.98078528040323045, -0.19509032201612827},
    double2{-0.97003125319454399, -0.24298017990326389},
    double2{-0.95694033573220886, -0.29028467725446237},
    double2{-0.94154406518302078, -0.33688985339222005},
    double2{-0.92387953251128676, -0.38268343236508977},
    double2{-0.90398929312344333, -0.42755509343028209},
    double2{-0.88192126434835503, -0.47139673682599765},
    double2{-0.85772861000027207, -0.51410274419322173},
    double2{-0.83146961230254524, -0.55557023301960222},
    double2{-0.80320753148064491, -0.59569930449243334},
    double2{-0.77301045336273696, -0.6343932841636455},
    double2{-0.74095112535495909, -0.6715589548470184},
    double2{-0.70710678118654752, -0.70710678118654752},
    double2{-0.6715589548470184, -0.74095112535495909},
    double2{-0.6343932841636455, -0.77301045336273696},
    double2{-0.59569930449243334, -0.80320753148064491},
    double2{-0.55557023301960222, -0.83146961230254524},
    double2{-0.51410274419322173, -0.85772861000027207},
    double2{-0.47139673682599765, -0.88192126434835503},
    double2{-0.42755509343028209, -0.90398929312344333},
    double2{-0.38268343236508977, -0.92387953251128676},
    double2{-0.33688985339222005, -0.94154406518302078},
    double2{-0.29028467725446237, -0.95694033573220886},
    double2{-0.24298017990326389, -0.97003125319454399},
    double2{-0.19509032201612827, -0.98078528040323045},
    double2{-0.14673047445536175, -0.98917650996478097},
    double2{-0.098017140329560602, -0.99518472667219689},
    double2{-0.049067674327418014, -0.99879545620517239},
    double2{0.0, -1.0},
    double2{0.049067674327418014, -0.99879545620517239},
    double2{0.098017140329560602, -0.99518472667219689},
    double2{0.14673047445536175, -0.98917650996478097},
    double2{0.19509032201612827, -0.98078528040323045},
    double2{0.24298017990326389, -0.97003125319454399},
    double2{0.29028467725446237, -0.95694033573220886},
    double2{0.33688985339222005, -0.94154406518302078},
    double2{0.38268343236508977, -0.92387953251128676},
    double2{0.42755509343028209, -0.90398929312344333},
    double2{0.47139673682599765, -0.88192126434835503},
    double2{0.51410274419322173, -0.85772861000027207},
    double2{0.55557023301960222, -0.83146961230254524},
    double2{0.59569930449243334, -0.80320753148064491},
    double2{0.6343932841636455, -0.77301045336273696},
    double2{0.6715589548470184, -0.74095112535495909},
    double2{0.70710678118654752, -0.70710678118654752},
    double2{0.74095112535495909, -0.6715589548470184},
    double2{0.77301045336273696, -0.6343932841636455},
    double2{0.80320753148064491, -0.59569930449243334},
    double2{0.83146961230254524, -0.55557023301960222},
    double2{0.85772861000027207, -0.51410274419322173},
    double2{0.88192126434835503, -0.47139673682599765},
    double2{0.90398929312344333, -0.42755509343028209},
    double2{0.92387953251128676, -0.38268343236508977},
    double2{0.94154406518302078, -0.33688985339222005},
    double2{0.95694033573220886, -0.29028467725446237},
    double2{0.97003125319454399, -0.24298017990326389},
    double2{0.98078528040323045, -0.19509032201612827},
    double2{0.98917650996478097, -0.14673047445536175},
    double2{0.99518472667219689, -0.098017140329560602},
    double2{0.99879545620517239, -0.049067674327418014},
};

#include "xd_cooperative_roots256.cuh"

template <bool Inverse, class Complex>
__device__ __forceinline__ Complex table_twiddle(unsigned index) {
    using Real = decltype(Complex{}.real);
    const double2 root = kTwiddleRoots128[index & 127U];
    const Real real = static_cast<Real>(root.x);
    const Real imag = static_cast<Real>(root.y);
    return Inverse ? Complex{real, imag} : Complex{real, -imag};
}

template <typename Real, unsigned Length, bool Inverse>
using thread_fft = decltype(
    cufftdx::Thread() + cufftdx::Size<Length>() +
    cufftdx::Type<cufftdx::fft_type::c2c>() +
    cufftdx::Direction<Inverse ? cufftdx::fft_direction::inverse
                               : cufftdx::fft_direction::forward>() +
    cufftdx::Precision<Real>() + cufftdx::SM<CUBUTTERFLY_XD_CUFFTDX_SM>());

template <unsigned LocalLog, bool Inverse, class Complex>
__device__ __forceinline__ void local_fft(Complex (&v)[1U << LocalLog]) {
    using Real = decltype(Complex{}.real);
    constexpr unsigned Length = 1U << LocalLog;
    using FFT = thread_fft<Real, Length, Inverse>;
    static_assert(FFT::storage_size == Length,
                  "xD cooperative Thread FFT must expose one value per local point");
    typename FFT::value_type data[FFT::storage_size];
#pragma unroll
    for (unsigned i = 0; i < Length; ++i)
        data[i] = typename FFT::value_type{v[i].real, v[i].imag};
    FFT().execute(data);
#pragma unroll
    for (unsigned i = 0; i < Length; ++i)
        v[i] = Complex{data[i].x, data[i].y};
}

}  // namespace xd_cooperative_detail

// Complete an N-point FFT from four interleaved local K=N/4 transforms.  The
// lane exchange and output permutation match register_tile::cooperative_fft's
// four-lane path, while the local codelet is supplied by cuFFTDx Thread FFT.
template <unsigned Log, unsigned Columns, bool Inverse, class Complex>
__device__ __forceinline__ void xd_cooperative_fft4(
    Complex (&v)[(1U << Log) / 4U], unsigned lane) {
    static_assert(Log >= 2, "xD cooperative FFT needs four lane roles");
    static_assert(Columns && !(Columns & (Columns - 1U)) && Columns * 4U <= 32U,
                  "xD cooperative shuffle group must fit a warp");
    static_assert((1U << Log) == 64U || (1U << Log) == 128U,
                  "xD cooperative FFT supports only lengths 64 and 128");

    using Real = decltype(Complex{}.real);
    constexpr unsigned K = (1U << Log) / 4U;
    xd_cooperative_detail::local_fft<Log - 2U, Inverse>(v);

    const unsigned mask = __activemask();
#pragma unroll
    for (unsigned k = 0; k < K; ++k) {
        auto value = v[k];
        if (k && lane) {
            const auto phase = xd_cooperative_detail::table_twiddle<Inverse, Complex>(
                lane * k * (128U / (1U << Log)));
            value = register_tile::multiply(value, phase);
        }

        const Complex peer_radix2{
            __shfl_xor_sync(mask, value.real, 2U * Columns),
            __shfl_xor_sync(mask, value.imag, 2U * Columns)};
        const bool upper_radix2 = (lane & 2U) != 0;
        value = {(upper_radix2 ? -value.real : value.real) + peer_radix2.real,
                 (upper_radix2 ? -value.imag : value.imag) + peer_radix2.imag};
        if (lane == 3U)
            value = Inverse ? Complex{-value.imag, value.real}
                            : Complex{value.imag, -value.real};

        const Complex peer_final{
            __shfl_xor_sync(mask, value.real, Columns),
            __shfl_xor_sync(mask, value.imag, Columns)};
        const bool upper_final = (lane & 1U) != 0;
        v[k] = {(upper_final ? -value.real : value.real) + peer_final.real,
                (upper_final ? -value.imag : value.imag) + peer_final.imag};
    }
}

template <unsigned Lanes>
__device__ __forceinline__ unsigned dif_output_lane(unsigned lane) {
    static_assert(Lanes && !(Lanes & (Lanes - 1U)) && Lanes <= 32U,
                  "DIF lane count must be a power of two no larger than a warp");
    constexpr unsigned LaneLog = register_tile::power_log2(Lanes);
    return __brev(lane) >> (32U - LaneLog);
}

template <bool Inverse, class Complex>
__device__ __forceinline__ Complex xd_quarter_turn(Complex value) {
    return Inverse ? Complex{-value.imag, value.real}
                   : Complex{value.imag, -value.real};
}

template <bool Inverse, class Complex>
__device__ __forceinline__ Complex xd_eighth_turn(Complex value) {
    using Real = decltype(Complex{}.real);
    constexpr Real root_half = Real(0.70710678118654752440084436210485);
    if constexpr (Inverse)
        return {(value.real - value.imag) * root_half,
                (value.real + value.imag) * root_half};
    else
        return {(value.real + value.imag) * root_half,
                (value.imag - value.real) * root_half};
}

// The lane FFT needs only the first half of W_Radix. Keep its unity and
// quarter turns exact, and express the remaining small radix rotations with
// uniform literal coefficients. Dynamic constant-table loads serialize
// different lane addresses even though this coefficient set is very small.
template <unsigned Radix, bool Inverse, class Complex>
__device__ __forceinline__ Complex xd_lane_twiddle(Complex value, unsigned digit) {
    static_assert(Radix == 2U || Radix == 4U || Radix == 8U || Radix == 16U);
    if constexpr (Radix == 4U) {
        if (digit & 1U) value = xd_quarter_turn<Inverse>(value);
    } else if constexpr (Radix == 8U) {
        if (digit & 1U) value = xd_eighth_turn<Inverse>(value);
        if (digit & 2U) value = xd_quarter_turn<Inverse>(value);
    } else if constexpr (Radix == 16U) {
        using Real = decltype(Complex{}.real);
        if (digit & 1U) {
            constexpr Real cosine = Real(0.92387953251128675612818318939679);
            constexpr Real sine = Real(0.38268343236508977172845998403040);
            const Real real = (digit & 2U) ? sine : cosine;
            const Real imag = (digit & 2U) ? cosine : sine;
            value = register_tile::multiply(value, Complex{real, Inverse ? imag : -imag});
        } else if (digit & 2U) {
            value = xd_eighth_turn<Inverse>(value);
        }
        if (digit & 4U) value = xd_quarter_turn<Inverse>(value);
    }
    return value;
}

template <unsigned Half, unsigned Columns, unsigned Radix, bool Inverse,
          class Complex>
__device__ __forceinline__ void dif_stage(Complex& value, unsigned lane,
                                          unsigned mask) {
    const Complex peer{
        __shfl_xor_sync(mask, value.real, Half * Columns),
        __shfl_xor_sync(mask, value.imag, Half * Columns)};
    if ((lane & Half) == 0U) {
        value = {value.real + peer.real, value.imag + peer.imag};
        return;
    }

    const Complex difference{peer.real - value.real,
                             peer.imag - value.imag};
    const unsigned digit = lane & (Half - 1U);
    static_assert(Radix == 2U || Radix == 4U || Radix == 8U || Radix == 16U,
                  "xD DIF radix must be a supported power of two");
    value = xd_lane_twiddle<Radix, Inverse>(difference, digit);
}

// Complete an N-point FFT from Lanes interleaved local K=N/Lanes transforms.
// The lane pass is radix-2 decimation-in-frequency: stages proceed from the
// full Lanes-point butterfly down to radix two, so each lane retains the
// corresponding bit-reversed output.  This output order is intentionally
// exposed through dif_output_lane rather than register_tile's legacy helper.
template <unsigned Log, unsigned Columns, unsigned Lanes, bool Inverse,
          class Complex>
__device__ __forceinline__ void xd_cooperative_fft_lanes(
    Complex (&v)[(1U << Log) / Lanes], unsigned lane) {
    static_assert(Log >= 2U, "xD cooperative FFT needs four lane roles");
    static_assert(Lanes && !(Lanes & (Lanes - 1U)) && Lanes <= 32U,
                  "xD cooperative FFT lanes must be a warp power of two");
    static_assert(Columns && !(Columns & (Columns - 1U)) &&
                      Columns * Lanes <= 32U,
                  "xD cooperative shuffle group must fit a warp");
    static_assert((1U << Log) == 64U || (1U << Log) == 128U ||
                      (1U << Log) == 256U,
                  "xD cooperative FFT supports only lengths 64, 128 and 256");
    static_assert(Lanes >= 8U,
                  "four-lane xD cooperative FFT uses the dedicated wrapper");
    static_assert((1U << Log) != 256U || Lanes == 8U || Lanes == 16U,
                  "256-point xD cooperative FFT supports only lanes 8 and 16");
    constexpr unsigned LaneLog = register_tile::power_log2(Lanes);
    constexpr unsigned LocalLog = Log - LaneLog;
    xd_cooperative_detail::local_fft<LocalLog, Inverse>(v);

    const unsigned mask = __activemask();
#pragma unroll
    for (unsigned k = 0; k < (1U << LocalLog); ++k) {
        auto value = v[k];
        if constexpr ((1U << Log) == 256U) {
            constexpr unsigned phase_stride = 256U / (1U << Log);
            const unsigned phase_index = (lane * k) * phase_stride;
            if (k && lane)
                value = register_tile::multiply(
                    value, xd_cooperative_detail::table_twiddle256<Inverse, Complex>(
                               phase_index));
            if constexpr (Lanes == 8U) {
                dif_stage<4U, Columns, 8U, Inverse>(value, lane, mask);
                dif_stage<2U, Columns, 4U, Inverse>(value, lane, mask);
                dif_stage<1U, Columns, 2U, Inverse>(value, lane, mask);
            } else {
                dif_stage<8U, Columns, 16U, Inverse>(value, lane, mask);
                dif_stage<4U, Columns, 8U, Inverse>(value, lane, mask);
                dif_stage<2U, Columns, 4U, Inverse>(value, lane, mask);
                dif_stage<1U, Columns, 2U, Inverse>(value, lane, mask);
            }
        } else {
            constexpr unsigned phase_stride = 128U / (1U << Log);
            const unsigned phase_index = (lane * k) * phase_stride;
            if (k && lane)
                value = register_tile::multiply(
                    value, xd_cooperative_detail::table_twiddle<Inverse, Complex>(phase_index));
            if constexpr (Lanes == 8U) {
                dif_stage<4U, Columns, 8U, Inverse>(value, lane, mask);
                dif_stage<2U, Columns, 4U, Inverse>(value, lane, mask);
                dif_stage<1U, Columns, 2U, Inverse>(value, lane, mask);
            } else {
                dif_stage<8U, Columns, 16U, Inverse>(value, lane, mask);
                dif_stage<4U, Columns, 8U, Inverse>(value, lane, mask);
                dif_stage<2U, Columns, 4U, Inverse>(value, lane, mask);
                dif_stage<1U, Columns, 2U, Inverse>(value, lane, mask);
            }
        }
        v[k] = value;
    }
}

}  // namespace cuntt::detail
