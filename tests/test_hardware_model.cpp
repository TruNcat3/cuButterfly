#include "hardware_model.hpp"
#include <chrono>
#include <iostream>

namespace {
using Json = nlohmann::json;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F operation) {
    try { operation(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid hardware model accepted");
}
struct TemporaryDirectory {
    std::filesystem::path path;
    TemporaryDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (unsigned attempt = 0; attempt < 100; ++attempt) {
            path = std::filesystem::temp_directory_path() /
                ("cubutterfly-hardware-model-test-" + std::to_string(stamp) + "-" + std::to_string(attempt));
            if (std::filesystem::create_directory(path)) return;
        }
        throw std::runtime_error("cannot create test directory");
    }
    ~TemporaryDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};
void save(const std::filesystem::path& path, const Json& document) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    output << document;
    if (!output) throw std::runtime_error("cannot write test fixture");
}
}

int main() try {
    using namespace cuntt::detail;
    cuntt::HardwareResourceModel live;
    live.device_name = "NVIDIA A100 80GB PCIe";
    live.compute_major = 8; live.compute_minor = 0; live.memory_bytes = 84974239744ULL;
    live.sm_count = 1;
    const Json valid = {{"schema", "cubutterfly-hardware-model-v1"},
        {"hardware", {{"device", live.device_name}, {"compute_capability", "8.0"},
            {"memory_bytes", live.memory_bytes}, {"sm_count", 108},
            {"max_threads_per_block", 1024}, {"max_threads_per_sm", 2048}, {"max_blocks_per_sm", 32},
            {"registers_per_sm", 65536}, {"registers_per_block", 65536},
            {"shared_bytes_per_sm", 167936}, {"shared_bytes_per_block", 166912},
            {"cooperative_launch", true}}},
        {"provenance", {{"source", "CPU test fixture, not measured"}}}};
    const auto parsed = parse_hardware_model(valid, live);
    require(parsed.sm_count == 108 && parsed.shared_bytes_per_block == 166912 &&
            parsed.registers_per_block == 65536 && parsed.cooperative_launch, "static capacities not loaded");
    require(parsed.memory_bandwidth_gbps == 0 && parsed.launch_overhead_us == 0,
            "unmeasured performance was fabricated");
    require(hardware_model_tag(live) == "NVIDIA-A100-80GB-PCIe-sm80-84974239744B", "profile tag mismatch");
    for (const auto* field : {"memory_bytes", "sm_count", "max_threads_per_block", "max_threads_per_sm",
            "max_blocks_per_sm", "registers_per_sm", "registers_per_block", "shared_bytes_per_sm", "shared_bytes_per_block"}) {
        auto bad = valid; bad["hardware"].erase(field);
        rejects([&] { parse_hardware_model(bad, live); });
        for (const Json value : {Json(0), Json(-1), Json(1.5), Json(true), Json("108")}) {
            bad = valid; bad["hardware"][field] = value;
            rejects([&] { parse_hardware_model(bad, live); });
        }
    }
    for (const auto* field : {"sm_count", "max_threads_per_block", "max_threads_per_sm", "max_blocks_per_sm",
            "registers_per_sm", "registers_per_block", "shared_bytes_per_sm", "shared_bytes_per_block"}) {
        auto bad = valid; bad["hardware"][field] = 1ULL << 32;
        rejects([&] { parse_hardware_model(bad, live); });
    }
    for (const auto* field : {"device", "compute_capability", "memory_bytes"}) {
        auto bad = valid;
        bad["hardware"][field] = std::string(field) == "memory_bytes" ? Json(1) : Json("wrong");
        rejects([&] { parse_hardware_model(bad, live); });
    }
    for (const Json value : {Json(0), Json(1), Json("true"), Json(nullptr)}) {
        auto bad = valid; bad["hardware"]["cooperative_launch"] = value;
        rejects([&] { parse_hardware_model(bad, live); });
    }
    auto bad = valid; bad["schema"] = "stale-v0";
    rejects([&] { parse_hardware_model(bad, live); });
    bad = valid; bad.erase("provenance");
    rejects([&] { parse_hardware_model(bad, live); });
    bad = valid; bad["hardware"]["shared_bytes_per_block"] = 200000;
    rejects([&] { parse_hardware_model(bad, live); });
    auto no_cooperative = valid; no_cooperative["hardware"]["cooperative_launch"] = false;
    require(!parse_hardware_model(no_cooperative, live).cooperative_launch, "false boolean not preserved");

    TemporaryDirectory temporary;
    const auto local = temporary.path / "local";
    const auto bundled = temporary.path / "bundled";
    const auto relative = std::filesystem::path(hardware_model_tag(live)) / "hardware_model.json";
    const auto fallback = resolve_hardware_model(live, std::nullopt, local, bundled);
    require(fallback.model.sm_count == 1 && fallback.source.find("live-cuda-properties:") == 0,
            "missing artifact fallback is not explicit");
    rejects([&] { resolve_hardware_model(live, (temporary.path / "missing.json").string(), local, bundled); });
    rejects([&] { resolve_hardware_model(live, std::string(), local, bundled); });
    save(bundled / relative, valid);
    require(resolve_hardware_model(live, std::nullopt, local, bundled).model.sm_count == 108,
            "bundled static artifact not used");
    auto local_document = valid; local_document["hardware"]["sm_count"] = 107;
    save(local / relative, local_document);
    require(resolve_hardware_model(live, std::nullopt, local, bundled).model.sm_count == 107,
            "local artifact did not win over bundled");
    const auto explicit_path = temporary.path / "explicit.json";
    save(explicit_path, valid);
    require(resolve_hardware_model(live, explicit_path.string(), local, bundled).model.sm_count == 108,
            "explicit artifact did not win over local");
    save(local / relative, bad);
    rejects([&] { resolve_hardware_model(live, std::nullopt, local, bundled); });
    save(explicit_path, Json{{"not", "a hardware model"}});
    rejects([&] { resolve_hardware_model(live, explicit_path.string(), local, bundled); });
    { std::ofstream output(explicit_path); output << "{"; }
    rejects([&] { load_hardware_model(explicit_path, live); });
    std::cout << "hardware model CPU tests passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
