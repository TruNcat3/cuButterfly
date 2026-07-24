#include <stdexcept>

#include "external_fft_units.cuh"

namespace cuntt::detail {

bool turbofft_generated_available(std::uint32_t) noexcept { return false; }

void launch_turbofft_generated(std::uint32_t, const Complex32*, Complex32*, std::uint64_t, std::uint64_t, std::uint64_t) {
    throw std::runtime_error("turbofft-generated was not enabled at build time");
}

} // namespace cuntt::detail
