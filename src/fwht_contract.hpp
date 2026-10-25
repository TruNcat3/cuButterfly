#pragma once

#include "cuntt/butterfly.hpp"

namespace cuntt::detail {
inline bool short_fp16_fwht_supported(const ButterflyConfig& c) noexcept {
    return c.op == ButterflyOperator::Fwht && c.precision == ButterflyPrecision::Fp16 &&
        c.accumulation == ButterflyAccumulation::Fp32 && c.final_storage_rounding &&
        (c.log_n == 5 || c.log_n == 6) && !c.inverse &&
        c.placement == ButterflyPlacement::OutOfPlace && c.element_stride == 1 &&
        (c.batch_stride == 0 || c.batch_stride == (std::size_t{1} << c.log_n));
}
}  // namespace cuntt::detail
