#pragma once

#include "cuntt/butterfly.hpp"
#include <algorithm>
#include <cstdint>

namespace cuntt::detail {
// dispatch_temporal_length instantiates sizes 1..16. A single shared array
// uses the device opt-in dynamic shared-memory budget when needed.
inline bool scalar_temporal_tile_supported(const ButterflyConfig& config,
                                          std::uint64_t hardware_shared_bytes) {
    if (config.log_n < 1 || config.log_n > 16 ||
        (config.op == ButterflyOperator::Fft && config.fft_core != FftCore::Scalar))
        return false;
    const bool integer_zeta = config.op == ButterflyOperator::SubsetZeta ||
        config.op == ButterflyOperator::SupersetZeta || config.op == ButterflyOperator::XorZeta;
    const std::uint64_t scalar_bytes = integer_zeta ?
        (config.precision == ButterflyPrecision::Uint64 ? 8 : 4) :
        config.precision == ButterflyPrecision::Fp64 || config.precision == ButterflyPrecision::Uint64 ? 8 :
        config.precision == ButterflyPrecision::Fp16 || config.precision == ButterflyPrecision::Bf16 ? 2 : 4;
    const auto value_bytes = scalar_bytes * (config.op == ButterflyOperator::Fft ? 2 : 1);
    return (std::uint64_t{1} << config.log_n) * value_bytes <=
        hardware_shared_bytes;
}
}
