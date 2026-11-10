#include <cuntt/rns_c_api.h>
#include <cuntt/rns_tuning.hpp>
#include <nlohmann/json.hpp>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

struct cubutterflyRnsPlanOpaque {
    explicit cubutterflyRnsPlanOpaque(const cuntt::RnsPlanConfig& config) : plan(config) {}
    cuntt::RnsPlan plan;
    cuntt::RnsPreparation preparation;
    cuntt::RnsPlacement placement = cuntt::RnsPlacement::Separate;
    bool prepared = false;
};

namespace {
thread_local std::string last_error;
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
template<class F> int guarded(F&& operation) noexcept {
    try {
        last_error.clear();
        operation();
        return 0;
    } catch (const std::invalid_argument& error) {
        try { last_error = error.what(); } catch (...) {}
        return 1;
    } catch (const std::exception& error) {
        try { last_error = error.what(); } catch (...) {}
        return 2;
    } catch (...) {
        try { last_error = "unknown RNS failure"; } catch (...) {}
        return 2;
    }
}
}

extern "C" {
const char* cubutterflyRnsLastError(void) { return last_error.c_str(); }

int cubutterflyRnsCreate(const char* config_json, cubutterflyRnsPlan* result) {
    return guarded([&] {
        require(result != nullptr, "result must be non-null");
        *result = nullptr;
        require(config_json != nullptr, "config JSON must be non-null");
        auto plan = std::make_unique<cubutterflyRnsPlanOpaque>(
            cuntt::rns_config_from_json(config_json));
        *result = plan.release();
    });
}

int cubutterflyRnsPrepare(cubutterflyRnsPlan plan, int placement,
                         const char* cache_directory, int policy,
                         unsigned repetitions, void* stream) {
    return guarded([&] {
        require(plan != nullptr, "plan must be non-null");
        require(!plan->prepared, "plan is already prepared; create another plan to retune");
        require(placement == 0 || placement == 1, "invalid placement");
        require(policy >= 0 && policy <= 2, "invalid cache policy");
        require(repetitions > 0, "repetitions must be positive");
        cuntt::RnsPrepareOptions options;
        options.cache_directory = cache_directory ? cache_directory : "";
        options.policy = static_cast<cuntt::RnsCachePolicy>(policy);
        options.repetitions = repetitions;
        const auto selected = static_cast<cuntt::RnsPlacement>(placement);
        plan->preparation = cuntt::prepare_rns_plan(plan->plan, selected, options,
                                                  reinterpret_cast<cudaStream_t>(stream));
        plan->placement = selected;
        plan->prepared = true;
    });
}

int cubutterflyRnsExecute(cubutterflyRnsPlan plan, const uint64_t* input,
                         uint64_t* output, void* stream) {
    return guarded([&] {
        require(plan != nullptr && plan->prepared, "plan must be prepared");
        require(input != nullptr && output != nullptr, "device pointers must be non-null");
        require(reinterpret_cast<uintptr_t>(input) % alignof(uint64_t) == 0 &&
                reinterpret_cast<uintptr_t>(output) % alignof(uint64_t) == 0,
                "device pointers must be uint64 aligned");
        require((input == output) == (plan->placement == cuntt::RnsPlacement::InPlace),
                "pointer aliasing does not match prepared placement");
        plan->plan.execute_async(input, output, reinterpret_cast<cudaStream_t>(stream));
    });
}

int cubutterflyRnsInfoJson(cubutterflyRnsPlan plan, char* buffer, size_t* bytes) {
    return guarded([&] {
        require(plan != nullptr && bytes != nullptr, "plan and bytes must be non-null");
        const auto& p = plan->preparation;
        nlohmann::json info{
            {"prepared", plan->prepared}, {"workspace_bytes", plan->plan.workspace_bytes()},
            {"request", nlohmann::json::parse(cuntt::rns_config_json(plan->plan.config()))},
            {"mapping", nlohmann::json::parse(cuntt::rns_mapping_json(plan->plan.mapping()))},
            {"placement", plan->placement == cuntt::RnsPlacement::InPlace ? "in_place" : "separate"},
            {"cache_hit", p.cache_hit}, {"cache_key", p.cache_key},
            {"source", p.source}, {"kernel_ms", p.kernel_ms}};
        const auto text = info.dump();
        const auto capacity = *bytes;
        *bytes = text.size() + 1;
        if (!buffer) return;
        require(capacity >= *bytes, "info JSON buffer is too small");
        std::memcpy(buffer, text.c_str(), *bytes);
    });
}

int cubutterflyRnsDestroy(cubutterflyRnsPlan plan) {
    return guarded([&] { delete plan; });
}
}
