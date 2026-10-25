#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime_api.h>

#include "cuntt/ntt.hpp"
#include "generated_hybrid_dataflow_config.hpp"
#include "legacy_identity.hpp"

namespace cuntt::detail {

// These are measured/default preferences for a platform profile. They are
// intentionally separate from executable capability checks: the caller still
// validates shared-memory, occupancy, launch, and generated-kernel legality.
struct NttPlatformPreferences {
    std::string target;

    bool use_generated_hybrid_dataflow_defaults = false;
    bool has_hierarchical_dataflow_profile = false;

    ComputeUnit hybrid2d_compute_unit = ComputeUnit::Radix2;
    std::uint32_t hybrid2d_threads = 256;
    bool hybrid2d_prefers_two_rows = false;

    bool apply_generated_hybrid_dataflow_default(
        PlanConfig& config, std::uint32_t sm_count) const noexcept {
        if (!use_generated_hybrid_dataflow_defaults) {
            return false;
        }
        return ::cuntt::detail::apply_generated_hybrid_dataflow_default(
            config, sm_count);
    }

    bool hierarchical_default_10x10(std::uint32_t log_n) const noexcept {
        return has_hierarchical_dataflow_profile && log_n == 20;
    }

    bool hierarchical_10x10(const PlanConfig& config) const noexcept {
        return has_hierarchical_dataflow_profile && config.log_n == 20 &&
               config.stage_partition.size() == 2 &&
               config.stage_partition[0] == 10 &&
               config.stage_partition[1] == 10;
    }

    bool hierarchical_7x7x6(const PlanConfig& config) const noexcept {
        return has_hierarchical_dataflow_profile && config.log_n == 20 &&
               config.stage_partition.size() == 3 &&
               config.stage_partition[0] == 7 &&
               config.stage_partition[1] == 7 &&
               config.stage_partition[2] == 6;
    }

    bool hierarchical_three_level(const PlanConfig& config) const noexcept {
        if (!has_hierarchical_dataflow_profile || config.log_n != 20 ||
            config.stage_partition.size() != 3) {
            return false;
        }
        const auto& partition = config.stage_partition;
        return (partition[0] == 6 && partition[1] == 7 && partition[2] == 7) ||
               (partition[0] == 7 && partition[1] == 6 && partition[2] == 7) ||
               (partition[0] == 7 && partition[1] == 7 && partition[2] == 6) ||
               (partition[0] == 6 && partition[1] == 6 && partition[2] == 8) ||
               (partition[0] == 6 && partition[1] == 8 && partition[2] == 6) ||
               (partition[0] == 8 && partition[1] == 6 && partition[2] == 6);
    }

    std::uint32_t hierarchical_units_for_generated_mapping(
        const PlanConfig& config) const noexcept {
        return hierarchical_three_level(config) ? 32U : 1U;
    }

    std::uint32_t hierarchical_units_for_missing_mapping(
        const PlanConfig& config) const noexcept {
        return hierarchical_7x7x6(config) ? 32U : 1U;
    }

    std::uint32_t hierarchical_non_appt_target_ctas(
        const PlanConfig& config) const noexcept {
        if (hierarchical_7x7x6(config) && config.word_bits == 32) {
            return 3U;
        }
        if (has_hierarchical_dataflow_profile && config.log_n == 20 &&
            config.word_bits == 32) {
            return 4U;
        }
        return 2U;
    }

    std::uint32_t hierarchical_cta_weight(const PlanConfig& config,
                                           std::size_t segment,
                                           bool appt_online) const noexcept {
        if (hierarchical_7x7x6(config) && appt_online) {
            const auto core = config.subgraph_mappings.front().core;
            const bool register_tail =
                core == NttSubgraphCore::ApptOnlineRegisterTail ||
                core == NttSubgraphCore::ApptOnlineRegisterTailWarp ||
                core == NttSubgraphCore::ApptOnlineRegisterTailColumnWarp ||
                core == NttSubgraphCore::ApptOnlineRegisterTailRadix8 ||
                core == NttSubgraphCore::ApptOnlineRegisterTailGrouped ||
                core == NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinal ||
                core == NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinalDataTime ||
                core == NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinalResident ||
                core == NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter;
            if (register_tail) {
                constexpr std::uint32_t u32_batch1[] = {8, 6, 6};
                constexpr std::uint32_t u32_batched[] = {7, 7, 6};
                constexpr std::uint32_t u64_weights[] = {10, 5, 5};
                return config.word_bits == 32
                           ? (config.batch == 1 ? u32_batch1[segment]
                                                : u32_batched[segment])
                           : u64_weights[segment];
            }
            if (core == NttSubgraphCore::ApptOnlineSplitTail) {
                constexpr std::uint32_t split_weights[] = {7, 6, 7};
                return split_weights[segment];
            }
            if (core == NttSubgraphCore::ApptOnlineFusedTail) {
                constexpr std::uint32_t fused_weights[] = {7, 7, 6};
                return fused_weights[segment];
            }
            const bool cta_radix4 = core == NttSubgraphCore::ApptOnlineRadix4;
            if (config.word_bits == 32) {
                constexpr std::uint32_t warp_batch1[] = {6, 6, 8};
                constexpr std::uint32_t warp_batched[] = {7, 7, 6};
                constexpr std::uint32_t cta[] = {7, 6, 7};
                return cta_radix4 ? cta[segment]
                                  : config.batch == 1 ? warp_batch1[segment]
                                                      : warp_batched[segment];
            }
            constexpr std::uint32_t batch1[] = {4, 8, 8};
            constexpr std::uint32_t warp_batched[] = {6, 8, 6};
            constexpr std::uint32_t cta_batched[] = {5, 7, 8};
            return config.batch == 1
                       ? batch1[segment]
                       : cta_radix4 ? cta_batched[segment]
                                    : warp_batched[segment];
        }
        if (hierarchical_10x10(config)) {
            return segment == 0 ? 9U : 11U;
        }
        if (hierarchical_7x7x6(config)) {
            return segment < 2 ? 6U : 8U;
        }
        return config.stage_partition[segment];
    }
};

inline NttPlatformPreferences query_ntt_platform_preferences(
    const cudaDeviceProp& properties, const PlanConfig& config) {
    const bool legacy_prior = has_legacy_v100_prior(properties.name, properties.major, properties.minor);
    const bool fused = config.cross_twiddle_placement == CrossTwiddlePlacement::Fused;

    NttPlatformPreferences preferences;
    preferences.target = std::string(properties.name);
    preferences.use_generated_hybrid_dataflow_defaults = legacy_prior;
    preferences.has_hierarchical_dataflow_profile = legacy_prior;
    preferences.hybrid2d_compute_unit =
        legacy_prior && !(config.word_bits == 64 && config.log_n == 12)
            ? ComputeUnit::Radix4
            : ComputeUnit::Radix2;
    preferences.hybrid2d_threads =
        legacy_prior && fused && config.log_n >= 18 ? 512U : 256U;
    preferences.hybrid2d_prefers_two_rows =
        legacy_prior && fused && config.log_n == 20;
    return preferences;
}

}  // namespace cuntt::detail
