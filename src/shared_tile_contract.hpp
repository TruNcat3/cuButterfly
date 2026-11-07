#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cuntt::detail {

// Independent graph tiles share a CTA, with their columns interleaved across
// lanes. This is data-space expansion, separate from serial external batches.
inline std::uint64_t shared_tile_bytes(unsigned stages, unsigned tiles, unsigned value_bytes) {
    return (2ULL << stages) * tiles * value_bytes;
}

inline void validate_shared_tiles(unsigned log_n, const std::vector<unsigned>& groups,
                                  unsigned tiles) {
    if (!tiles || (tiles & (tiles - 1)))
        throw std::invalid_argument("shared data_tiles_per_cta must be a positive power of two");
    for (auto stages : groups)
        if (!stages || stages > log_n || log_n > 30 || tiles > (1U << (log_n - stages)))
            throw std::invalid_argument("shared data tiles exceed the independent tiles in an execution group");
}

inline std::vector<unsigned> shared_tile_candidates(unsigned log_n,
        const std::vector<unsigned>& groups, unsigned value_bytes, std::uint64_t capacity) {
    std::vector<unsigned> result;
    if (groups.empty() || log_n > 30 || !value_bytes) return result;
    for (auto stages : groups) if (!stages || stages > log_n) return result;
    const auto largest = *std::max_element(groups.begin(), groups.end());
    const auto independent = 1U << (log_n - largest);
    for (unsigned tiles = 1; tiles <= independent; tiles *= 2) {
        if (shared_tile_bytes(largest, tiles, value_bytes) > capacity) break;
        result.push_back(tiles);
    }
    return result;
}

} // namespace cuntt::detail
