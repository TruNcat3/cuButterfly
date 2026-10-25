#include "xd_axis_fft.hpp"

#include <stdexcept>

namespace cuntt::detail {

bool xd_axis_fft_available(std::size_t) noexcept { return false; }

bool xd_axis_fft_exact_configuration_available(std::size_t, bool, unsigned,
                                               unsigned, unsigned) noexcept { return false; }

void launch_xd_axis_fft(const void*, void*, std::size_t, std::size_t, std::size_t,
                        bool, bool, double, cudaStream_t, unsigned, unsigned,
                        unsigned, unsigned) {
    throw std::runtime_error("xD axis cuFFTDx was not enabled at build time");
}

}  // namespace cuntt::detail
