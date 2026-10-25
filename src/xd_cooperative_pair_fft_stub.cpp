#include "xd_cooperative_pair_fft.hpp"

#include <stdexcept>

namespace cuntt::detail {

bool xd_cooperative_pair_fft_available(std::size_t, std::size_t, bool,
                                       unsigned) noexcept {
    return false;
}

void prepare_xd_cooperative_pair_fft(std::size_t, std::size_t, bool, bool,
                                     unsigned) {
    throw std::runtime_error("xD cooperative pair FFT was not enabled at build time");
}

void launch_xd_cooperative_pair_fft(const void*, void*, std::size_t, std::size_t,
                                    std::size_t, bool, bool, double, cudaStream_t,
                                    unsigned) {
    throw std::runtime_error("xD cooperative pair FFT was not enabled at build time");
}

}  // namespace cuntt::detail
