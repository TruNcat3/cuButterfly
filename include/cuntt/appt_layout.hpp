#pragma once

#include <cstddef>
#include <cstdint>

namespace cuntt {

// Linearizes the APPT bank/offset arrangement for one tile-local data domain.
// `total_elements` must be the complete domain covered by the tile.
constexpr std::uint64_t appt_physical_index(std::uint64_t logical,
                                            std::uint32_t total_elements,
                                            std::uint32_t n_part,
                                            std::uint32_t parallelism) {
    const std::uint64_t bank_count = 2ULL * parallelism;
    const std::uint64_t bank = (logical ^ (logical / n_part)) % bank_count;
    const std::uint64_t offset = logical / bank_count;
    return bank * (static_cast<std::uint64_t>(total_elements) / bank_count) + offset;
}

constexpr std::uint64_t appt_logical_index(std::uint64_t physical,
                                           std::uint32_t total_elements,
                                           std::uint32_t n_part,
                                           std::uint32_t parallelism) {
    const std::uint64_t bank_count = 2ULL * parallelism;
    const std::uint64_t bank_span = static_cast<std::uint64_t>(total_elements) / bank_count;
    const std::uint64_t bank = physical / bank_span;
    const std::uint64_t offset = physical % bank_span;
    const std::uint64_t base = offset * bank_count;
    for (std::uint64_t low = 0; low < bank_count && base + low < total_elements; ++low) {
        const std::uint64_t logical = base + low;
        if (((logical ^ (logical / n_part)) % bank_count) == bank) return logical;
    }
    return total_elements;
}

}  // namespace cuntt
