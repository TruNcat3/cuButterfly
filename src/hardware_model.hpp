#pragma once

#include "cuntt/mixed_dataflow.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace cuntt::detail {

struct HardwareModelLoad {
    HardwareResourceModel model;
    std::string source;
};

inline std::string hardware_model_tag(const HardwareResourceModel& identity) {
    std::string name;
    bool separator = false;
    for (unsigned char c : identity.device_name) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '.' || c == '-') {
            if (separator && !name.empty()) name += '-';
            name += static_cast<char>(c);
            separator = false;
        } else separator = true;
    }
    if (name.empty()) name = "unknown";
    return name + "-sm" + std::to_string(identity.compute_major) +
        std::to_string(identity.compute_minor) + "-" + std::to_string(identity.memory_bytes) + "B";
}

inline HardwareResourceModel parse_hardware_model(const nlohmann::json& document,
                                                 const HardwareResourceModel& identity) {
    using Json = nlohmann::json;
    try {
        if (!document.is_object() || document.at("schema") != "cubutterfly-hardware-model-v1" ||
            !document.at("provenance").is_object())
            throw std::invalid_argument("unsupported hardware model schema or provenance");
        const auto& h = document.at("hardware");
        if (!h.is_object() || !h.at("device").is_string() || !h.at("compute_capability").is_string())
            throw std::invalid_argument("hardware model identity must contain strings");
        const auto positive = [&](const char* field, std::uint64_t maximum) {
            const auto& value = h.at(field);
            if ((!value.is_number_unsigned() && !value.is_number_integer()) ||
                (value.is_number_integer() && !value.is_number_unsigned() && value.get<std::int64_t>() <= 0))
                throw std::invalid_argument(std::string("hardware model needs positive integer ") + field);
            const auto result = value.get<std::uint64_t>();
            if (!result || result > maximum)
                throw std::invalid_argument(std::string("hardware model integer outside range: ") + field);
            return result;
        };
        const auto memory = positive("memory_bytes", std::numeric_limits<std::uint64_t>::max());
        const auto cc = std::to_string(identity.compute_major) + "." + std::to_string(identity.compute_minor);
        if (h.at("device") != identity.device_name || h.at("compute_capability") != cc || memory != identity.memory_bytes)
            throw std::invalid_argument("hardware model device/compute capability/memory identity mismatch");
        HardwareResourceModel result;
        result.device_name = identity.device_name;
        result.compute_major = identity.compute_major;
        result.compute_minor = identity.compute_minor;
        result.memory_bytes = memory;
#define CUB_LOAD_CAP(field) result.field = static_cast<std::uint32_t>(positive(#field, UINT32_MAX))
        CUB_LOAD_CAP(sm_count);
        CUB_LOAD_CAP(max_threads_per_block);
        CUB_LOAD_CAP(max_threads_per_sm);
        CUB_LOAD_CAP(max_blocks_per_sm);
        CUB_LOAD_CAP(registers_per_sm);
        CUB_LOAD_CAP(registers_per_block);
        CUB_LOAD_CAP(shared_bytes_per_sm);
        CUB_LOAD_CAP(shared_bytes_per_block);
#undef CUB_LOAD_CAP
        if (!h.at("cooperative_launch").is_boolean())
            throw std::invalid_argument("hardware model cooperative_launch must be boolean");
        result.cooperative_launch = h.at("cooperative_launch").get<bool>();
        if (result.max_threads_per_block > result.max_threads_per_sm ||
            result.registers_per_block > result.registers_per_sm ||
            result.shared_bytes_per_block > result.shared_bytes_per_sm)
            throw std::invalid_argument("hardware model CTA capacity exceeds SM capacity");
        return result;
    } catch (const Json::exception& error) {
        throw std::invalid_argument(std::string("invalid hardware model: ") + error.what());
    }
}

inline HardwareModelLoad load_hardware_model(const std::filesystem::path& path,
                                            const HardwareResourceModel& identity) {
    std::ifstream input(path);
    if (!input) throw std::invalid_argument("cannot read hardware model: " + path.string());
    try {
        nlohmann::json document;
        input >> document;
        return {parse_hardware_model(document, identity), "static-hardware-model:" + path.string()};
    } catch (const std::exception& error) {
        throw std::invalid_argument("hardware model " + path.string() + ": " + error.what());
    }
}

// Missing automatic candidates permit fallback; an explicitly requested or
// present-but-invalid artifact is never silently replaced by live properties.
inline HardwareModelLoad resolve_hardware_model(
    const HardwareResourceModel& live, const std::optional<std::string>& explicit_file,
    const std::filesystem::path& profile_root, const std::filesystem::path& bundled_root) {
    if (explicit_file) return load_hardware_model(*explicit_file, live);
    for (const auto& root : {profile_root, bundled_root}) {
        if (root.empty()) continue;
        const auto path = root / hardware_model_tag(live) / "hardware_model.json";
        if (std::filesystem::exists(path) || std::filesystem::is_symlink(path))
            return load_hardware_model(path, live);
    }
    return {live, "live-cuda-properties:no-compatible-static-artifact"};
}

// Describes the last common hardware query on this thread, not a calibration claim.
std::string hardware_resource_model_source();
// Explicit hardware collection only: never consult static artifacts.
HardwareResourceModel query_live_hardware_resource_model();

} // namespace cuntt::detail
