#pragma once
#include "cuntt/ntt.hpp"

namespace cuntt::detail {
// Limits of the compiled CompactStage kernel, shared by validation and the
// portable inventory. These are executable capabilities, not search-space limits.
inline const char* compact_stage_constraint(const PlanConfig& config) noexcept {
    if (config.log_n != 20 || config.word_bits != 64 || config.inverse)
        return "compact-stage currently requires a forward, 64-bit logN=20 transform";
    if (config.modulus >= (1ULL << 62))
        return "compact-stage supports moduli of at most 62 bits";
    if (config.batch > 65535)
        return "compact-stage batch exceeds the CUDA grid-z limit";
    if (config.input_order != InputOrder::Natural ||
        (config.output_order != OutputOrder::Natural && config.output_order != OutputOrder::BitReversed))
        return "compact-stage requires natural input and natural or bit-reversed output";
    return nullptr;
}
} // namespace cuntt::detail
