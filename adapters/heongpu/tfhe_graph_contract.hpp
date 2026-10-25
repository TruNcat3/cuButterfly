#pragma once
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace cubutterfly::heongpu_tfhe {
template<std::size_t N>
std::size_t identity_index(const void* function, const std::array<const void*, N>& known) {
    for (std::size_t i = 0; i < N; ++i) if (function == known[i]) return i;
    return N;
}
inline std::array<std::size_t, 4> expected_bootstrap_counts(std::size_t bootstraps) {
    if (bootstraps > std::numeric_limits<std::size_t>::max() / 511)
        throw std::overflow_error("TFHE bootstrap count overflow");
    return {bootstraps, bootstraps * 511, bootstraps, bootstraps * 511};
}
inline void validate_bootstrap_counts(const std::array<std::size_t, 4>& counts,
                                      std::size_t bootstraps) {
    if (counts != expected_bootstrap_counts(bootstraps))
        throw std::runtime_error("TFHE Graph bootstrap identity counts do not match pinned n=512 contract");
}
}  // namespace cubutterfly::heongpu_tfhe
