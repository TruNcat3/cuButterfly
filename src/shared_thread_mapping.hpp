#pragma once

#include "cuntt/butterfly.hpp"
#include "cuntt/ntt.hpp"
#include <stdexcept>
#include <vector>

namespace cuntt::detail {

inline void validate_shared_threads(const std::vector<std::uint32_t>& threads) {
    for (const auto count : threads)
        if (count < 32 || count > 1024 || count % 32)
            throw std::invalid_argument("shared physical-group threads must be warp multiples in [32,1024]");
}

// These descriptors name physical groups, after fused logical boundaries.
// Other core/scheduler fields are not consumed by this lowering and must not
// silently change the meaning of an explicitly replayed mapping.
inline std::vector<std::uint32_t> shared_group_threads(const ButterflyConfig& config,
                                                      std::size_t groups) {
    if (!config.segment_mappings.empty())
        throw std::invalid_argument("shared-iterative thread mappings describe physical execution groups, not logical segments");
    std::vector<std::uint32_t> result;
    if (config.execution_group_mappings.empty()) {
        result.assign(groups, config.tile_threads);
    } else {
        if (config.execution_group_mappings.size() != groups)
            throw std::invalid_argument("shared thread mappings must have one entry per physical execution group");
        for (const auto& mapping : config.execution_group_mappings) {
            if (mapping.core != FftCore::Scalar || mapping.exchange != LocalExchange::SharedMemory ||
                mapping.ept != 0 || mapping.codelet != "native" || mapping.io_policy != "dynamic")
                throw std::invalid_argument("shared thread mappings require scalar/shared/native/dynamic with ept=0");
            result.push_back(mapping.threads);
        }
    }
    validate_shared_threads(result);
    return result;
}

inline std::vector<std::uint32_t> shared_group_threads(const PlanConfig& config,
                                                      std::size_t groups) {
    if (!config.subgraph_mappings.empty())
        throw std::invalid_argument("shared-iterative thread mappings describe physical execution groups, not logical subgraphs");
    std::vector<std::uint32_t> result;
    if (config.execution_group_mappings.empty()) {
        result.assign(groups, config.threads_per_block ? config.threads_per_block : 128);
    } else {
        if (config.execution_group_mappings.size() != groups)
            throw std::invalid_argument("shared thread mappings must have one entry per physical execution group");
        for (const auto& mapping : config.execution_group_mappings) {
            if (mapping.core != NttSubgraphCore::Radix2 || mapping.units_per_cta != 1 ||
                mapping.data_space != 1 || mapping.data_time != 1 || mapping.role_stages != 1 ||
                mapping.token_interleave != 1 || mapping.coefficient_reuse_stages != 0 || mapping.cta_weight != 0)
                throw std::invalid_argument("shared NTT thread mappings require radix2-shoup with default scheduler fields");
            result.push_back(mapping.threads_per_block);
        }
    }
    validate_shared_threads(result);
    return result;
}
} // namespace cuntt::detail
