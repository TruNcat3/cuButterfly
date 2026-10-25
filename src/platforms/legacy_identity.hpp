#pragma once
#include <string_view>

namespace cuntt::detail {
// The historical tables describe this model/capacity, not the SM70 ISA.
// Other Volta devices and V100 variants use local rules or portable defaults.
inline bool has_legacy_v100_prior(std::string_view name, int major, int minor) noexcept {
    return major == 7 && minor == 0 && name == "Tesla V100-SXM2-16GB";
}
}
