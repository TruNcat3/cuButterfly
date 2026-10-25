#pragma once

#include <cuntt/rns_ntt.hpp>
#include <string>

namespace cuntt {
enum class RnsCachePolicy { Prefer, Require, Refresh };
struct RnsPrepareOptions {
    // Empty disables persistence under Prefer/Refresh; Require rejects it.
    std::string cache_directory;
    RnsCachePolicy policy = RnsCachePolicy::Prefer;
    unsigned repetitions = 5;
};
struct RnsPreparation {
    bool cache_hit = false;
    std::string cache_key;
    std::string source;
    double kernel_ms = 0;
    std::string request_json;
    std::string mapping_json;
};

// Strict, versioned JSON. Integer moduli and roots retain all 64 bits.
std::string rns_config_json(const RnsPlanConfig&);
RnsPlanConfig rns_config_from_json(const std::string&);
std::string rns_mapping_json(const RnsMapping&);
RnsMapping rns_mapping_from_json(const std::string&);

// Replays compatible measured mappings without timing. New measurements use
// RnsPlan's bounded timing-only search, not mathematical validation or exhaustive
// installation calibration. Corrupt records are errors, including under Prefer.
RnsPreparation prepare_rns_plan(RnsPlan&, RnsPlacement,
                                const RnsPrepareOptions&, cudaStream_t = nullptr);
} // namespace cuntt
