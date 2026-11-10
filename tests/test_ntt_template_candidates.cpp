#include "ntt_template_candidates.hpp"
#include "cubutterfly/mapping.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

int main() try {
    using namespace cuntt;
    using cuntt::detail::ntt_template_candidates;
    using cuntt::detail::ntt_template_core_name;
    using cuntt::detail::ntt_template_core_omission_reason;
    using cuntt::detail::ntt_template_descriptor_complete;
    using cuntt::detail::ntt_template_descriptor_reason;
    using cuntt::detail::ntt_template_support_reason;

    PlanConfig base;
    base.log_n = 20;
    base.batch = 4;
    base.modulus = 576460756061519873ULL;
    base.word_bits = 64;
    base.inverse = true;
    base.input_order = InputOrder::Natural;
    base.output_order = OutputOrder::Natural;
    const auto candidates = ntt_template_candidates(base);
    if (candidates.size() != 16) throw std::runtime_error("unexpected word64 specialized candidate count");
    bool saw_three_level = false, saw_static_io = false;
    bool saw_appt_register_tail = false;
    for (const auto& candidate : candidates) {
        if (candidate.modulus != base.modulus || candidate.word_bits != base.word_bits ||
            candidate.inverse != base.inverse || candidate.batch != base.batch ||
            candidate.input_order != base.input_order || candidate.output_order != base.output_order)
            throw std::runtime_error("candidate changed NTT semantics");
        if (candidate.backend != Backend::HierarchicalDataflow ||
            candidate.subgraph_mappings.size() != candidate.stage_partition.size() ||
            candidate.boundary_mappings.size() + 1 != candidate.stage_partition.size())
            throw std::runtime_error("candidate has incomplete executable descriptor");
        if (candidate.stage_partition.size() == 3) saw_three_level = true;
        if (candidate.stage_partition == std::vector<std::uint32_t>{10, 10} &&
            candidate.subgraph_mappings.front().data_space > 1) saw_static_io = true;
        if (!candidate.subgraph_mappings.empty() &&
            candidate.subgraph_mappings.front().core == NttSubgraphCore::ApptOnlineRegisterTail) {
            saw_appt_register_tail = true;
            if (candidate.stage_partition != std::vector<std::uint32_t>{7, 7, 6} ||
                candidate.boundary_mappings.size() != 2 ||
                candidate.boundary_mappings.front().storage != BoundaryStorage::Ring ||
                candidate.boundary_mappings.front().buffers != 2 ||
                candidate.appt_role_mapping.fragment_width != 16 ||
                candidate.appt_role_mapping.producer_weight == 0 ||
                candidate.appt_role_mapping.tail_weight == 0 ||
                candidate.appt_role_mapping.writer_weight == 0)
                throw std::runtime_error("APPT candidate has incomplete role/boundary contract");
            if (!ntt_template_descriptor_complete(candidate))
                throw std::runtime_error("APPT candidate failed descriptor contract");

            cuntt::PlanConfig replay = base;
            cuntt::detail::apply_serialized_mapping(cuntt::detail::serialize_mapping(candidate), replay);
            if (!ntt_template_descriptor_complete(replay))
                throw std::runtime_error("APPT descriptor was lost during serialization roundtrip");
            auto broken = replay;
            broken.boundary_mappings.front().buffers = 1;
            if (ntt_template_descriptor_reason(broken) == nullptr)
                throw std::runtime_error("incomplete APPT ring descriptor was accepted");
        }
    }
    if (!saw_three_level || !saw_static_io || !saw_appt_register_tail)
        throw std::runtime_error("missing specialized family");
    if (ntt_template_support_reason(base) != nullptr ||
        ntt_template_core_name(NttSubgraphCore::HomogeneousWarp128StaticIoRadix2) == nullptr ||
        ntt_template_core_name(NttSubgraphCore::ApptOnlineRegisterTail) == nullptr ||
        ntt_template_core_name(NttSubgraphCore::ApptOnline) != nullptr ||
        ntt_template_core_omission_reason(NttSubgraphCore::ApptOnlineRegisterTail) != nullptr ||
        ntt_template_core_omission_reason(NttSubgraphCore::ApptOnline) == nullptr)
        throw std::runtime_error("support-domain contract mismatch");

    const NttSubgraphCore omitted_appt_variants[] = {
        NttSubgraphCore::ApptPipeline,
        NttSubgraphCore::ApptOnline,
        NttSubgraphCore::ApptOnlineRadix4,
        NttSubgraphCore::ApptOnlineFusedTail,
        NttSubgraphCore::ApptOnlineSplitTail,
        NttSubgraphCore::ApptOnlineRegisterTailWarp,
        NttSubgraphCore::ApptOnlineRegisterTailColumnWarp,
        NttSubgraphCore::ApptOnlineRegisterTailRadix8,
        NttSubgraphCore::ApptOnlineRegisterTailGrouped,
        NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinal,
        NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinalDataTime,
        NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinalResident,
        NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter};
    for (const auto core : omitted_appt_variants) {
        if (ntt_template_core_name(core) != nullptr ||
            ntt_template_core_omission_reason(core) == nullptr)
            throw std::runtime_error("APPT variant lacks an explicit omission contract");
    }

    auto unsupported = base;
    unsupported.log_n = 19;
    if (ntt_template_support_reason(unsupported) == nullptr || !ntt_template_candidates(unsupported).empty())
        throw std::runtime_error("unsupported length was emitted");
    unsupported = base;
    unsupported.modular_multiply = ModularMultiply::Barrett;
    if (ntt_template_support_reason(unsupported) == nullptr || !ntt_template_candidates(unsupported).empty())
        throw std::runtime_error("unsupported arithmetic was emitted");
    unsupported = base;
    unsupported.output_order = OutputOrder::BitReversed;
    if (ntt_template_support_reason(unsupported) == nullptr || !ntt_template_candidates(unsupported).empty())
        throw std::runtime_error("unsupported order was emitted");
    unsupported = base;
    unsupported.batch = 0;
    if (ntt_template_support_reason(unsupported) == nullptr || !ntt_template_candidates(unsupported).empty())
        throw std::runtime_error("zero-batch request was emitted");
    unsupported = base;
    unsupported.modulus = 576460756061519871ULL;
    if (ntt_template_support_reason(unsupported) == nullptr || !ntt_template_candidates(unsupported).empty())
        throw std::runtime_error("non-prime modulus was emitted");
    unsupported = base;
    unsupported.modulus = 576460756061519872ULL;
    if (ntt_template_support_reason(unsupported) == nullptr || !ntt_template_candidates(unsupported).empty())
        throw std::runtime_error("wrong-root modulus was emitted");
    auto word32 = base;
    word32.word_bits = 32;
    word32.modulus = 2013265921ULL;
    const auto word32_count = ntt_template_candidates(word32).size();
    if (word32_count != 26)
        throw std::runtime_error("unexpected word32 specialized candidate count");
    word32.modulus = 3221225473ULL;
    if (ntt_template_support_reason(word32) == nullptr || !ntt_template_candidates(word32).empty())
        throw std::runtime_error("out-of-range 32-bit modulus was emitted");

    std::cout << "NTT specialized template candidate contract passed: " << candidates.size()
              << " word64 and " << word32_count
              << " word32 candidates\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
