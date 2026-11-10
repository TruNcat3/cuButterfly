#include "mapping_selector.hpp"
#include "plan_registry.hpp"
#include <stdexcept>
#include <utility>

namespace cuntt::detail {
ButterflySelectionResult select_butterfly_mapping(ButterflyConfig config) {
    if (!config.stage_partition.empty() || !config.segment_mappings.empty() ||
        !config.boundaries.empty() || !config.execution_group_mappings.empty())
        throw std::invalid_argument("auto-select owns butterfly decomposition and physical mappings");
    if (auto selected=registry_butterfly_mapping(config)) return std::move(*selected);
    return select_portable_butterfly(config);
}

NttSelectionResult select_ntt_mapping(PlanConfig config) {
    if (auto selected=registry_ntt_mapping(config)) return std::move(*selected);
    return select_portable_ntt(config);
}
} // namespace cuntt::detail
