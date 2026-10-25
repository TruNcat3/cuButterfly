#include "../src/xd_axis_fft.hpp"

#include <cufftdx.hpp>

#ifndef CUBUTTERFLY_XD_CUFFTDX_SM
#define CUBUTTERFLY_XD_CUFFTDX_SM 800
#endif

using namespace cuntt::detail;

static_assert(xd_axis_fft_length_supported(2048));
static_assert(xd_axis_fft_length_supported(4096));
static_assert(!xd_axis_fft_length_supported(8192));
static_assert(!xd_axis_fft_length_supported(3072));
static_assert(!xd_axis_fft_configuration_supported(2048, false, 4));
static_assert(!xd_axis_fft_configuration_supported(4096, true, 4));
static_assert(xd_axis_fft_block_batch(1024, false, 16) == 4);
static_assert(xd_axis_fft_block_batch(1024, true, 16) == 2);

template <typename Real, unsigned Length, unsigned Ept,
          cufftdx::fft_direction Direction>
constexpr bool check_descriptor() {
    constexpr bool fp64 = sizeof(Real) == sizeof(double);
    static_assert(xd_axis_fft_configuration_supported(Length, fp64, Ept));
    static_assert(xd_axis_fft_configuration_supported(Length, fp64, Ept, 8));
    static_assert(xd_axis_fft_configuration_supported(Length, fp64, Ept, 16));
    constexpr unsigned batch = xd_axis_fft_block_batch(Length, fp64, Ept);
    static_assert(batch == 1);
    using FFT = decltype(cufftdx::Block() + cufftdx::Size<Length>() +
                         cufftdx::Type<cufftdx::fft_type::c2c>() +
                         cufftdx::Direction<Direction>() + cufftdx::Precision<Real>() +
                         cufftdx::ElementsPerThread<Ept>() +
                         cufftdx::FFTsPerBlock<batch>() +
                         cufftdx::SM<CUBUTTERFLY_XD_CUFFTDX_SM>());
    static_assert(FFT::block_dim.z == 1 && FFT::block_dim.y == batch);
    static_assert(FFT::block_dim.x == FFT::stride);
    static_assert(FFT::block_dim.x * FFT::block_dim.y <= 1024);
    static_assert(FFT::storage_size * FFT::stride >= Length);
    // Candidate resource filtering uses the padded exchange tile as its bound.
    static_assert(FFT::shared_memory_size <=
                  batch * (Length + 1) * sizeof(typename FFT::value_type));
    return true;
}

template <typename Real, unsigned Length, unsigned Ept>
constexpr bool check_directions() {
    return check_descriptor<Real, Length, Ept, cufftdx::fft_direction::forward>() &&
           check_descriptor<Real, Length, Ept, cufftdx::fft_direction::inverse>();
}

static_assert(check_directions<float, 2048, 8>());
static_assert(check_directions<float, 2048, 16>());
static_assert(check_directions<float, 4096, 8>());
static_assert(check_directions<float, 4096, 16>());
static_assert(check_directions<double, 2048, 8>());
static_assert(check_directions<double, 2048, 16>());
static_assert(check_directions<double, 4096, 8>());
static_assert(check_directions<double, 4096, 16>());

// Compilation is the check: this executable never initializes CUDA or a GPU.
int main() { return 0; }
