#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace cuntt::detail {

enum class XdAxisCore { CufftdxBlock, CooperativeRegister, ThreadShared, Vkfft };
inline const char* xd_axis_core_name(XdAxisCore core) {
    switch (core) {
        case XdAxisCore::CufftdxBlock: return "cufftdx-block";
        case XdAxisCore::CooperativeRegister: return "cooperative-register";
        case XdAxisCore::ThreadShared: return "thread-shared";
        case XdAxisCore::Vkfft: return "vkfft";
    }
    throw std::invalid_argument("unknown xD axis core");
}
inline XdAxisCore xd_axis_core_from_name(const std::string& name) {
    if (name == "cufftdx-block") return XdAxisCore::CufftdxBlock;
    if (name == "cooperative-register") return XdAxisCore::CooperativeRegister;
    if (name == "thread-shared") return XdAxisCore::ThreadShared;
    if (name == "vkfft") return XdAxisCore::Vkfft;
    throw std::invalid_argument("unknown xD axis core: " + name);
}
enum class XdPairCore { Thread, CooperativeRegister, Radix4Register };

inline const char* xd_pair_core_name(XdPairCore core) {
    switch (core) {
        case XdPairCore::Thread: return "thread";
        case XdPairCore::CooperativeRegister: return "cooperative-register";
        case XdPairCore::Radix4Register: return "radix4-register";
    }
    throw std::invalid_argument("unknown xD pair core");
}

inline XdPairCore xd_pair_core_from_name(const std::string& name) {
    if (name == "thread") return XdPairCore::Thread;
    if (name == "cooperative-register") return XdPairCore::CooperativeRegister;
    if (name == "radix4-register") return XdPairCore::Radix4Register;
    throw std::invalid_argument("unknown xD pair core: " + name);
}

// Per-logical-axis resources for an explicit packed/fused xD lowering.
// The vector is indexed by the logical complex axis, independently of the
// current retained physical layout or execution order.
struct XdAxisResources {
    XdAxisCore core = XdAxisCore::CufftdxBlock;
    unsigned ept = 8;
    unsigned register_lanes = 4;
    unsigned block_cap = 0;
    // 0..3 select axis-core exchange; 4 retains an axis-last permutation
    // and invokes the cuFFTDx core with contiguous exchange (policy 0).
    unsigned exchange = 0;
    // ThreadShared only: 0 selects the constant-root path, 1 recurrence roots.
    unsigned local_twiddle_policy = 0;
    // Exact CTA line count; zero retains the legacy resource policy.
    unsigned ffts_per_block = 0;
};

// Validate the core parameter contract; shape and device support belong to lowering.
inline void validate_xd_axis_resource(const XdAxisResources& resource) {
    bool valid = false;
    switch (resource.core) {
        case XdAxisCore::CufftdxBlock:
            valid = (resource.ept == 4 || resource.ept == 8 || resource.ept == 16) &&
                    resource.register_lanes == 4 &&
                    (resource.block_cap == 0 || resource.block_cap == 8 || resource.block_cap == 16) &&
                    resource.exchange <= 4 && resource.local_twiddle_policy == 0 &&
                    (resource.ffts_per_block == 0 || resource.ffts_per_block == 2 || resource.ffts_per_block == 4) &&
                    (resource.ffts_per_block == 0 ||
                     (resource.block_cap == 0 && resource.exchange != 1));
            break;
        case XdAxisCore::CooperativeRegister:
            valid = resource.ept == 8 &&
                    (resource.register_lanes == 4 || resource.register_lanes == 8 || resource.register_lanes == 16) &&
                    resource.block_cap == 0 && resource.exchange == 0 &&
                    resource.local_twiddle_policy == 0 && resource.ffts_per_block == 0;
            break;
        case XdAxisCore::ThreadShared:
            valid = resource.ept == 8 && resource.register_lanes == 4 &&
                    (resource.block_cap == 0 || resource.block_cap == 8 ||
                     resource.block_cap == 16 || resource.block_cap == 32) &&
                    resource.exchange == 0 && resource.local_twiddle_policy <= 1 &&
                    resource.ffts_per_block == 0;
            break;
        case XdAxisCore::Vkfft:
            valid = resource.ept == 8 && resource.register_lanes == 4 &&
                    resource.block_cap == 0 && resource.exchange == 0 &&
                    resource.local_twiddle_policy == 0 && resource.ffts_per_block == 0;
            break;
        default:
            throw std::invalid_argument("unknown xD axis core");
    }
    if (!valid)
        throw std::invalid_argument(std::string("invalid xD axis resources for ") +
                                    xd_axis_core_name(resource.core));
}

// Compiled candidates are an installation/search choice, not a theoretical limit.
// Keep the historical 55 candidates (and optional provider) in their original order.
inline std::vector<XdAxisResources> xd_axis_resource_candidates(bool include_vkfft) {
    std::vector<XdAxisResources> options;
    auto append = [&](XdAxisResources resource) {
        validate_xd_axis_resource(resource);
        options.push_back(resource);
    };
    for (unsigned ept : {4U,8U,16U})
        for (unsigned cap : {0U,8U,16U})
            for (unsigned exchange : {0U,1U,2U,3U})
                append({XdAxisCore::CufftdxBlock,ept,4,cap,exchange});
    for (unsigned ept : {8U,16U})
        for (unsigned exact : {2U,4U})
            for (unsigned exchange : {0U,2U,3U})
                append({XdAxisCore::CufftdxBlock,ept,4,0,exchange,0,exact});
    for (unsigned lanes : {4U,8U,16U})
        append({XdAxisCore::CooperativeRegister,8,lanes,0,0});
    for (unsigned cap : {0U,16U})
        append({XdAxisCore::ThreadShared,8,4,cap,0});
    for (unsigned cap : {0U,16U})
        append({XdAxisCore::ThreadShared,8,4,cap,0,1});
    if (include_vkfft)
        append({XdAxisCore::Vkfft,8,4,0,0});
    for (unsigned ept : {4U,8U,16U})
        for (unsigned cap : {0U,8U,16U})
            append({XdAxisCore::CufftdxBlock,ept,4,cap,4});
    for (unsigned ept : {8U,16U})
        for (unsigned exact : {2U,4U})
            append({XdAxisCore::CufftdxBlock,ept,4,0,4,0,exact});
    for (unsigned roots : {0U,1U})
        append({XdAxisCore::ThreadShared,8,4,32,0,roots});
    return options;
}

}  // namespace cuntt::detail
