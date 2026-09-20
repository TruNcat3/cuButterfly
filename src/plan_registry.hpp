#pragma once
#include "mapping_selector.hpp"
#include <optional>
#include <string>
#include <vector>

namespace cuntt::detail {
const char* runtime_fingerprint() noexcept;
std::optional<ButterflySelectionResult> registry_butterfly_mapping(const ButterflyConfig& config);
std::optional<NttSelectionResult> registry_ntt_mapping(const PlanConfig& config);
std::vector<ButterflyConfig> portable_butterfly_candidates(const ButterflyConfig& config);
std::vector<PlanConfig> portable_ntt_candidates(const PlanConfig& config);
ButterflySelectionResult select_portable_butterfly(const ButterflyConfig& config);
NttSelectionResult select_portable_ntt(const PlanConfig& config);
std::string serialize_mapping(const ButterflyConfig& config);
std::string serialize_mapping(const PlanConfig& config);
void apply_serialized_mapping(const std::string& value, ButterflyConfig& config);
void apply_serialized_mapping(const std::string& value, PlanConfig& config);
} // namespace cuntt::detail
