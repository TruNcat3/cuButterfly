#pragma once
#include "mapping_selector.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace cuntt::detail {
// Platform rules supply candidate priors only. Public planners still validate
// executable capabilities; only the exact measured registry is calibrated.
std::vector<PlanConfig> platform_candidates(const PlanConfig& config);
std::vector<ButterflyConfig> platform_candidates(const ButterflyConfig& config);
std::string platform_tuning_identity();

// Apply matching rule mappings to a supplied semantic/configuration pair.
// This pure entry point keeps malformed individual mappings isolated from the
// rest of the candidate inventory and is also used by the CPU contract test.
std::vector<PlanConfig> platform_rule_candidates(const PlanConfig& config,
    const nlohmann::json& document, const nlohmann::json& hardware);
std::vector<ButterflyConfig> platform_rule_candidates(const ButterflyConfig& config,
    const nlohmann::json& document, const nlohmann::json& hardware);

// Pure matching interface also used by CPU contract tests.
std::vector<nlohmann::json> match_platform_rules(const nlohmann::json& document,
    const nlohmann::json& hardware, const nlohmann::json& semantics);

// Compatibility priors live outside the generic search/selection framework.
ButterflySelectionResult select_legacy_butterfly_mapping(ButterflyConfig config);
NttSelectionResult select_legacy_ntt_mapping(PlanConfig config);
} // namespace cuntt::detail
