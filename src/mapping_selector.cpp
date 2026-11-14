#include "mapping_selector.hpp"
#include "plan_registry.hpp"
#include "hadacore_provider.hpp"
#include <limits>
#include <stdexcept>
#include <utility>

namespace cuntt::detail {
ButterflySelectionResult select_butterfly_mapping(ButterflyConfig config) {
    if (!config.stage_partition.empty() || !config.segment_mappings.empty() ||
        !config.boundaries.empty() || !config.execution_group_mappings.empty())
        throw std::invalid_argument("auto-select owns butterfly decomposition and physical mappings");
    if(config.allow_fp16_mma) {
        const auto profile=hada_profile(config);
        auto mma=hada_candidates(config,profile);
        if(!mma.empty() && profile.contains("native_rows")) {
            for(const auto& native:profile.at("native_rows")) {
                if(native.at("log_n")!=config.log_n||native.at("batch")!=config.batch)continue;
                if(!hada_evidence(native))throw std::invalid_argument("native control evidence hash mismatch");
                hada_validate_native(native,profile,hada_read(native.at("evidence_path")));
                Dl_info library_info{};
                if(!dladdr(reinterpret_cast<void*>(&select_butterfly_mapping),&library_info)||
                   hada_hash(library_info.dli_fname)!=native.at("library_sha256"))continue;
                double best=native.at("median_ms");if(!(best>0&&std::isfinite(best)))continue;
                ButterflyConfig chosen=config;
                auto mapping=native.at("mapping");mapping["arithmetic_policy"]="allow-fp16-mma";
                apply_serialized_mapping(mapping.dump(),chosen);
                for(const auto& c:mma) {
                    const auto row=hada_row(c,profile);
                    if(row.at("timing_protocol")!=native.at("timing_protocol"))continue;
                    const double ms=row.at("median_ms");if(ms<best){best=ms;chosen=c;}
                }
                ButterflyPlan checked(chosen);
                return {std::move(chosen),{true,true,profile.value("gpu_name", "profile-bound GPU"),
                  "calibrated-local-core", "calibrated-exact-request",
                  "same-protocol measured native and validated HadaCore local-core candidates",best}};
            }
        }
        auto strict=config;strict.allow_fp16_mma=false;strict.local_core="native";
        auto result=select_butterfly_mapping(strict);result.config.allow_fp16_mma=true;
        result.info.reason += "; FP16 MMA unavailable: no matching calibrated native/MMA pair; retained native selection";
        return result;
    }
    if (auto selected=registry_butterfly_mapping(config)) return std::move(*selected);
    return select_portable_butterfly(config);
}

NttSelectionResult select_ntt_mapping(PlanConfig config) {
    if (auto selected=registry_ntt_mapping(config)) return std::move(*selected);
    return select_portable_ntt(config);
}
} // namespace cuntt::detail
