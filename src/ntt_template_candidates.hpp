#pragma once

#include "cuntt/ntt.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace cuntt::detail {

// These are executable-template domains, not the complete NttSubgraphCore
// enum.  The launch predicates in ntt.cu currently specialize the following
// logN=20, Shoup, natural-order points.  The primary APPT register-tail core
// below is emitted only with its complete role/boundary descriptor; the other
// APPT variants remain omitted until they have independent descriptors.
inline const char* ntt_template_support_reason(const PlanConfig& config) noexcept {
    if (config.log_n != 20) return "specialized NTT templates require logN=20";
    if (config.batch == 0) return "specialized NTT templates require a positive batch";
    if (config.modulus < 3 || config.modulus >= (1ULL << 63))
        return "specialized NTT templates require a modulus in [3, 2^63)";
    if (!is_prime(config.modulus)) return "specialized NTT templates require a prime modulus";
    if ((config.modulus - 1) % (1ULL << config.log_n) != 0)
        return "specialized NTT templates require modulus-1 divisible by N";
    if (config.modular_multiply != ModularMultiply::Shoup)
        return "specialized NTT templates require Shoup multiplication";
    if (config.input_order != InputOrder::Natural || config.output_order != OutputOrder::Natural)
        return "specialized NTT templates require natural input and output order";
    if (config.word_bits != 32 && config.word_bits != 64)
        return "specialized NTT templates require a 32- or 64-bit word";
    if (config.word_bits == 32 && config.modulus >= (1ULL << 31))
        return "32-bit specialized NTT templates require modulus < 2^31";
    return nullptr;
}

inline const char* ntt_template_core_name(NttSubgraphCore core) noexcept {
    switch (core) {
    case NttSubgraphCore::HomogeneousRadix4: return "homogeneous-radix4-10x10";
    case NttSubgraphCore::HomogeneousWarpRadix2: return "homogeneous-warp-radix2-10x10";
    case NttSubgraphCore::HomogeneousWarp256Radix2: return "homogeneous-warp256-radix2-10x10";
    case NttSubgraphCore::HomogeneousWarp256StaticRadix2: return "homogeneous-warp256-static-radix2-10x10";
    case NttSubgraphCore::HomogeneousWarp256StaticIoRadix2: return "homogeneous-warp256-static-io-radix2-10x10";
    case NttSubgraphCore::HomogeneousWarp256CoefficientReuseStaticIoRadix2: return "homogeneous-warp256-coefficient-reuse-static-io-radix2-10x10";
    case NttSubgraphCore::HomogeneousWarp128StaticIoRadix2: return "homogeneous-warp128-static-io-radix2-10x10";
    case NttSubgraphCore::HomogeneousWarp128CoefficientReuseStaticIoRadix2: return "homogeneous-warp128-coefficient-reuse-static-io-radix2-10x10";
    case NttSubgraphCore::HomogeneousWarp128VectorRadix4StaticIo: return "homogeneous-warp128-vector-radix4-static-io-10x10";
    case NttSubgraphCore::HomogeneousWarp128VectorRadix4PackedStage6StaticIo: return "homogeneous-warp128-vector-radix4-packed-stage6-static-io-10x10";
    case NttSubgraphCore::HomogeneousWarp128VectorRadix4PackedStage6DistributedStaticIo: return "homogeneous-warp128-vector-radix4-packed-stage6-distributed-static-io-10x10";
    case NttSubgraphCore::HomogeneousWarp64StaticIoRadix2: return "homogeneous-warp64-static-io-radix2-10x10";
    case NttSubgraphCore::HomogeneousWarp128PipelineStaticIoRadix2: return "homogeneous-warp128-pipeline-static-io-10x10";
    case NttSubgraphCore::HomogeneousWarp128CooperativeStaticIoRadix2: return "homogeneous-warp128-cooperative-static-io-10x10";
    case NttSubgraphCore::DataflowRadix4: return "dataflow-radix4";
    case NttSubgraphCore::ApptOnlineRegisterTail: return "appt-online-register-tail";
    default: return nullptr;
    }
}

inline const char* ntt_template_core_omission_reason(NttSubgraphCore core) noexcept {
    switch (core) {
    case NttSubgraphCore::ApptPipeline:
        return "APPT pipeline has a launcher but no portable role/boundary descriptor";
    case NttSubgraphCore::ApptOnline:
        return "APPT online has a launcher but no portable role/boundary descriptor";
    case NttSubgraphCore::ApptOnlineRadix4:
        return "APPT radix4 requires an independent physical-core descriptor";
    case NttSubgraphCore::ApptOnlineFusedTail:
        return "APPT fused-tail requires an independent tail-liveness descriptor";
    case NttSubgraphCore::ApptOnlineSplitTail:
        return "APPT split-tail requires an independent tail-liveness descriptor";
    case NttSubgraphCore::ApptOnlineRegisterTailWarp:
        return "APPT register-tail warp requires an independent codelet descriptor";
    case NttSubgraphCore::ApptOnlineRegisterTailColumnWarp:
        return "APPT register-tail column-warp requires an independent codelet descriptor";
    case NttSubgraphCore::ApptOnlineRegisterTailRadix8:
        return "APPT register-tail radix8 requires an independent codelet descriptor";
    case NttSubgraphCore::ApptOnlineRegisterTailGrouped:
        return "APPT grouped register-tail requires a grouped producer descriptor";
    case NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinal:
        return "APPT grouped writer-final requires a natural-output descriptor";
    case NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinalDataTime:
        return "APPT grouped data-time variant requires a role-mask descriptor";
    case NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinalResident:
        return "APPT grouped resident variant requires a shared-memory liveness descriptor";
    case NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter:
        return "APPT grouped resident-quarter requires a shared-memory liveness descriptor";
    default:
        return nullptr;
    }
}

// The register-tail template is the only APPT variant currently safe to put
// in the portable inventory.  Keep its physical protocol in one contract so
// a future candidate edit cannot silently drop a role, token, or ring-buffer
// requirement.  This checks the serializable plan fields; device-specific
// limits (shared memory and max threads) remain Plan::validate_config's job.
inline const char* ntt_template_descriptor_reason(const PlanConfig& config) noexcept {
    const auto core = NttSubgraphCore::ApptOnlineRegisterTail;
    if (config.stage_partition != std::vector<std::uint32_t>{7, 7, 6})
        return "register-tail descriptor requires stage partition 7+7+6";
    if (config.pipeline_buffers != 2)
        return "register-tail descriptor requires two pipeline buffers";
    if (config.boundary_mappings.size() != 2)
        return "register-tail descriptor requires two boundaries";
    for (const auto& boundary : config.boundary_mappings) {
        if (boundary.storage != BoundaryStorage::Ring || boundary.buffers != 2)
            return "register-tail descriptor requires two-slot ring boundaries";
    }
    if (config.subgraph_mappings.size() != 3)
        return "register-tail descriptor requires three physical role mappings";
    for (const auto& mapping : config.subgraph_mappings) {
        if (mapping.core != core || mapping.threads_per_block != 256 ||
            mapping.units_per_cta != 8 || mapping.data_space != 16 ||
            mapping.data_time != 8 || mapping.role_stages != 1 ||
            mapping.token_interleave != 1)
            return "register-tail descriptor has an incomplete physical role mapping";
    }
    const auto& roles = config.appt_role_mapping;
    if (roles.producer_weight == 0 || roles.tail_weight == 0 || roles.writer_weight == 0 ||
        roles.fragment_width != 16 || roles.writer_tiles_per_cta != 1 ||
        roles.data_time_role_mask != 7)
        return "register-tail descriptor has incomplete producer/tail/writer roles";
    if (config.input_order != InputOrder::Natural || config.output_order != OutputOrder::Natural)
        return "register-tail descriptor requires natural input and output order";
    return nullptr;
}

inline bool ntt_template_descriptor_complete(const PlanConfig& config) noexcept {
    return ntt_template_descriptor_reason(config) == nullptr;
}

inline bool ntt_template_core_supported(const PlanConfig& config, NttSubgraphCore core) noexcept {
    if (ntt_template_support_reason(config) != nullptr) return false;
    if (core == NttSubgraphCore::DataflowRadix4) return true;
    if (core == NttSubgraphCore::HomogeneousWarp128StaticIoRadix2 ||
        core == NttSubgraphCore::HomogeneousWarp128CoefficientReuseStaticIoRadix2 ||
        core == NttSubgraphCore::HomogeneousWarp128VectorRadix4StaticIo ||
        core == NttSubgraphCore::HomogeneousWarp128VectorRadix4PackedStage6StaticIo ||
        core == NttSubgraphCore::HomogeneousWarp128VectorRadix4PackedStage6DistributedStaticIo ||
        core == NttSubgraphCore::HomogeneousWarp64StaticIoRadix2 ||
        core == NttSubgraphCore::HomogeneousWarp128PipelineStaticIoRadix2 ||
        core == NttSubgraphCore::HomogeneousWarp128CooperativeStaticIoRadix2)
        return config.word_bits == 32;
    if (core == NttSubgraphCore::HomogeneousWarp256CoefficientReuseStaticIoRadix2)
        return config.word_bits == 32;
    return true;
}

inline PlanConfig ntt_template_base(const PlanConfig& original,
                                    std::vector<std::uint32_t> partition) {
    auto config = original;
    config.auto_select = false;
    config.backend = Backend::HierarchicalDataflow;
    config.compute_unit = ComputeUnit::Radix4;
    config.stage_partition = std::move(partition);
    config.boundary_mappings.assign(config.stage_partition.size() - 1,
                                    NttBoundaryMapping{BoundaryStorage::FullScratch, 1});
    config.execution_group_mappings.clear();
    config.local_stage_partitions.clear();
    config.exchange_chunks.clear();
    config.stage_overlap = false;
    config.batch_tile_count = 1;
    config.input_order = InputOrder::Natural;
    config.output_order = OutputOrder::Natural;
    config.modular_multiply = ModularMultiply::Shoup;
    // Let Plan construction resolve occupancy and platform CTA weights.  A
    // candidate is a public mapping proposal, not a GPU-specific calibration.
    config.target_ctas_per_sm = 0;
    return config;
}

inline NttSubgraphMapping ntt_template_mapping(NttSubgraphCore core,
                                                std::uint32_t threads,
                                                std::uint32_t units,
                                                std::uint32_t data_space,
                                                std::uint32_t data_time,
                                                std::uint32_t reuse = 0) {
    NttSubgraphMapping mapping;
    mapping.core = core;
    mapping.threads_per_block = threads;
    mapping.units_per_cta = units;
    mapping.data_space = data_space;
    mapping.data_time = data_time;
    mapping.role_stages = 1;
    mapping.token_interleave = 1;
    mapping.coefficient_reuse_stages = reuse;
    // Zero delegates service-rate weighting to the existing platform policy.
    mapping.cta_weight = 0;
    return mapping;
}

inline std::vector<PlanConfig> ntt_template_candidates(const PlanConfig& original) {
    std::vector<PlanConfig> candidates;
    if (ntt_template_support_reason(original) != nullptr) return candidates;

    auto add_two = [&](NttSubgraphCore core, std::uint32_t threads,
                       std::uint32_t units, std::uint32_t data_space,
                       std::uint32_t data_time, std::uint32_t reuse = 0) {
        if (!ntt_template_core_supported(original, core)) return;
        auto config = ntt_template_base(original, {10, 10});
        config.subgraph_mappings = {
            ntt_template_mapping(core, threads, units, data_space, data_time, reuse),
            ntt_template_mapping(core, threads, units, data_space, data_time, reuse)};
        if (core == NttSubgraphCore::HomogeneousWarp128PipelineStaticIoRadix2)
            config.pipeline_buffers = 1;
        candidates.push_back(std::move(config));
    };

    add_two(NttSubgraphCore::HomogeneousRadix4, 256, 4, 1, 2);
    add_two(NttSubgraphCore::HomogeneousWarpRadix2, 256, 8, 1, 2);
    add_two(NttSubgraphCore::HomogeneousWarp256Radix2, 128, 4, 1, 2);
    add_two(NttSubgraphCore::HomogeneousWarp256Radix2, 256, 8, 1, 2);
    const std::uint32_t half_rows = original.word_bits == 32 ? 4U : 2U;
    add_two(NttSubgraphCore::HomogeneousWarp256StaticRadix2, 128, 4, half_rows, 2);
    add_two(NttSubgraphCore::HomogeneousWarp256StaticRadix2, 256, 8, half_rows, 2);
    add_two(NttSubgraphCore::HomogeneousWarp256StaticIoRadix2, 128, 4, half_rows, 2);
    add_two(NttSubgraphCore::HomogeneousWarp256StaticIoRadix2, 256, 8, half_rows, 2);
    add_two(NttSubgraphCore::HomogeneousWarp256CoefficientReuseStaticIoRadix2, 128, 4, half_rows, 2, 1);
    add_two(NttSubgraphCore::HomogeneousWarp128StaticIoRadix2, 128, 4, 4, 2);
    add_two(NttSubgraphCore::HomogeneousWarp128StaticIoRadix2, 256, 8, 4, 2);
    add_two(NttSubgraphCore::HomogeneousWarp128CoefficientReuseStaticIoRadix2, 128, 4, 4, 2, 4);
    add_two(NttSubgraphCore::HomogeneousWarp128VectorRadix4StaticIo, 128, 4, 4, 2);
    add_two(NttSubgraphCore::HomogeneousWarp128VectorRadix4PackedStage6StaticIo, 128, 4, 4, 2, 6);
    add_two(NttSubgraphCore::HomogeneousWarp128VectorRadix4PackedStage6DistributedStaticIo, 128, 4, 4, 2, 6);
    add_two(NttSubgraphCore::HomogeneousWarp64StaticIoRadix2, 128, 4, 4, 2);
    add_two(NttSubgraphCore::HomogeneousWarp128PipelineStaticIoRadix2, 128, 4, 4, 2);
    add_two(NttSubgraphCore::HomogeneousWarp128CooperativeStaticIoRadix2, 128, 4, 4, 2);
    // This is the existing two-level dataflow launcher, kept alongside the
    // homogeneous families so the public inventory covers both mature
    // implementations without conflating their local cores.
    add_two(NttSubgraphCore::DataflowRadix4, 256, 8, 1, 1);

    // The base register-tail APPT kernel is a complete public template: its
    // ring boundary, fixed 256-thread physical CTA and producer/tail/writer
    // service roles are all validated by Plan::validate_config.  Keep the
    // role weights semantic and let the platform resolver choose CTA weights.
    {
        auto config = ntt_template_base(original, {7, 7, 6});
        config.pipeline_buffers = 2;
        config.boundary_mappings.assign(
            2, NttBoundaryMapping{BoundaryStorage::Ring, 2});
        config.subgraph_mappings.assign(
            3, ntt_template_mapping(NttSubgraphCore::ApptOnlineRegisterTail,
                                     256, 8, 16, 8));
        config.appt_role_mapping.producer_weight =
            original.word_bits == 32 ? 7U : 6U;
        config.appt_role_mapping.tail_weight =
            original.word_bits == 32 ? 7U : 8U;
        config.appt_role_mapping.writer_weight =
            original.word_bits == 32 ? 6U : 6U;
        config.appt_role_mapping.fragment_width = 16;
        config.appt_role_mapping.writer_tiles_per_cta = 1;
        config.appt_role_mapping.data_time_role_mask = 7;
        if (ntt_template_descriptor_complete(config))
            candidates.push_back(std::move(config));
    }

    if (original.word_bits == 32 || original.word_bits == 64) {
        for (const auto partition : {std::vector<std::uint32_t>{6, 7, 7},
                                     std::vector<std::uint32_t>{7, 6, 7},
                                     std::vector<std::uint32_t>{7, 7, 6},
                                     std::vector<std::uint32_t>{6, 6, 8},
                                     std::vector<std::uint32_t>{6, 8, 6},
                                     std::vector<std::uint32_t>{8, 6, 6}}) {
            auto config = ntt_template_base(original, partition);
            const std::uint32_t threads = 128;
            config.subgraph_mappings.assign(partition.size(),
                ntt_template_mapping(NttSubgraphCore::DataflowRadix4, threads, 4, 1, 1));
            candidates.push_back(std::move(config));
        }
    }
    return candidates;
}

} // namespace cuntt::detail
