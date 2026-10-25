#include "xd_local_fft.hpp"

#include <stdexcept>

namespace cuntt::detail {

bool xd_local_fft_available(std::size_t) noexcept { return false; }

unsigned xd_local_fft_lines_per_block(std::size_t length, bool fp64,
                                      unsigned cap) noexcept {
    if ((length != 64U && length != 128U && length != 256U) ||
        (cap != 0U && cap != 8U && cap != 16U &&
         (length != 64U || cap != 32U)))
        return 0U;
    if (fp64 && length == 256U)
        return 8U;
    return cap == 32U ? 32U : (cap == 16U ? 16U : 8U);
}

void launch_xd_local_fft(const void*, void*, std::size_t, std::size_t,
                         std::size_t, bool, bool, double, cudaStream_t,
                         unsigned, unsigned twiddle_policy) {
    if (twiddle_policy > 1U)
        throw std::invalid_argument("xD local FFT twiddle policy must be 0 or 1");
    throw std::runtime_error("xD local FFT was not enabled at build time");
}

}  // namespace cuntt::detail
