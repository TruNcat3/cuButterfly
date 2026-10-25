#include "xd_pair_fft.hpp"

#include <stdexcept>

namespace cuntt::detail {

bool xd_pair_fft_available(std::size_t, std::size_t, bool) noexcept { return false; }

void launch_xd_pair_fft(const void*, void*, std::size_t, std::size_t, std::size_t,
                        bool, bool, double, cudaStream_t) {
    throw std::runtime_error("xD pair FFT was not enabled at build time");
}

}  // namespace cuntt::detail
