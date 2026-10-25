#include "xd_register_fft.hpp"

#include <stdexcept>

namespace cuntt::detail {

bool xd_register_fft_available(std::size_t) noexcept { return false; }

bool xd_register_fft_configuration_available(std::size_t, unsigned) noexcept {
    return false;
}

void launch_xd_register_fft(const void*, void*, std::size_t, std::size_t,
                            std::size_t, bool, bool, double, cudaStream_t,
                            unsigned) {
    throw std::runtime_error("xD register FFT was not enabled at build time");
}

}  // namespace cuntt::detail
