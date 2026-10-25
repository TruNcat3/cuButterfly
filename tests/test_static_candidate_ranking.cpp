#include "static_candidate_ranking.hpp"

#include <iostream>
#include <limits>

namespace {
using Json = nlohmann::json;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class F> void rejects(F operation, const char* message) {
    try { operation(); }
    catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}

struct Fixture {
    std::filesystem::path directory;
    Fixture() {
        auto pattern = (std::filesystem::temp_directory_path() / "cubutterfly-rank-test-XXXXXX").string();
        const auto* created = mkdtemp(pattern.data());
        if (!created) throw std::runtime_error("cannot create ranking fixture");
        directory = created;
    }
    ~Fixture() { std::error_code error; std::filesystem::remove_all(directory, error); }
    void write(const Json& profile) const {
        std::ofstream output(directory / "hardware_profile.json");
        output << profile.dump();
        if (!output) throw std::runtime_error("cannot write fixture profile");
    }
};

Json hardware() {
    return {{"device", "synthetic CPU fixture"}, {"compute_capability", "8.0"}, {"memory_bytes", 1024},
        {"sm_count", 10}, {"max_threads_per_block", 1024}, {"max_threads_per_sm", 2048},
        {"max_blocks_per_sm", 32}, {"registers_per_sm", 65536}, {"registers_per_block", 65536},
        {"shared_bytes_per_sm", 163840}, {"shared_bytes_per_block", 163840}, {"cooperative_launch", true}};
}

Json profile() {
    return {{"schema", "cubutterfly-hardware-profile-v1"}, {"device", "synthetic CPU fixture"},
        {"compute_capability", "8.0"}, {"global_memory_bytes", 1024}, {"sm_count", 10},
        {"provenance", {{"fixture", true}}}, {"capabilities", {
            {"global_feedback_bytes_per_second", 1e12}, {"equivalent_butterflies_per_second", 1e11},
            {"interstage_shared_bytes_per_second", 1e13}, {"cta_barriers_per_second", 1e9}}}};
}

Json sample(std::initializer_list<unsigned> stages) {
    Json groups = Json::array();
    for (const auto count : stages)
        groups.push_back({{"stage_count", count}, {"grid_ctas", 1024}, {"threads", 64}, {"core", "scalar"},
            {"live_shared_bytes", (1u << count) * 16}, {"compiler_resources_known", false},
            {"compiler_registers_per_thread", 32}});
    return {{"operator", "fft"}, {"precision", "fp32"}, {"accumulation", "native"},
        {"logN", 12}, {"batch", 4}, {"direction", "forward"}, {"normalization", "none"},
        {"mapping_json", Json{{"backend", "shared-iterative"}, {"stage_overlap", false}}.dump()},
        {"execution_groups_json", groups.dump()}};
}
} // namespace

int main(int argc, char** argv) try {
    using namespace cuntt::detail;
#ifdef CUBUTTERFLY_SOURCE_ROOT
    (void)argc;
    (void)argv;
    const auto root = std::filesystem::path(CUBUTTERFLY_SOURCE_ROOT);
#else
    const auto root = argc > 1 ? std::filesystem::path(argv[1])
        : std::filesystem::absolute(__FILE__).parent_path().parent_path();
#endif
    const auto* selected_python = std::getenv("CUBUTTERFLY_TEST_PYTHON");
    const std::string python = selected_python ? selected_python : "python3";
    const auto script = (root / "scripts/rank_static_candidates.py").string();
    Fixture fixture;
    const auto full_profile = profile();
    fixture.write(full_profile);
    Json request = {{"schema", "cubutterfly-static-ranking-request-v1"}, {"hardware", hardware()},
        {"runtime_fingerprint", "fixture-build:research"}, {"profile_dir", fixture.directory.string()},
        {"candidates", Json::array({{{"index", 0}, {"sample", sample({4, 8})}},
                                    {{"index", 1}, {"sample", sample({4})}},
                                    {{"index", 2}, {"sample", sample({4})}}})}};
    const std::set<std::size_t> submitted{0, 1, 2};
    const auto run = [&](const Json& input) { return run_static_ranker(input, python, script); };
    const auto ranked = run(request);
    validate_static_ranking(ranked, submitted);
    require(ranked.at("mode") == "hardware-rate-prior", "fixture did not exercise hardware priors");
    require(ranked.at("ranked").size() == 3 && ranked.at("ranked")[0].at("index") == 1 &&
            ranked.at("ranked")[1].at("index") == 2 && ranked.at("ranked")[2].at("index") == 0,
            "real helper did not sort candidates stably");
    require(ranked.at("unranked").empty() && ranked.at("rejected").empty(), "valid inventory was partitioned incorrectly");
    Json retained_profile;
    std::ifstream(fixture.directory / "hardware_profile.json") >> retained_profile;
    require(retained_profile == full_profile, "helper changed its input hardware profile");

    auto broken = ranked;
    broken["ranked"].erase(broken["ranked"].begin());
    rejects([&] { validate_static_ranking(broken, submitted); }, "missing candidate accepted");
    broken = ranked;
    broken["unranked"].push_back({{"index", 1}});
    rejects([&] { validate_static_ranking(broken, submitted); }, "duplicate across categories accepted");
    broken = ranked;
    broken["ranked"][0]["index"] = 999;
    rejects([&] { validate_static_ranking(broken, submitted); }, "unknown candidate accepted");
    broken = ranked;
    broken["ranked"][0]["index"] = -1;
    rejects([&] { validate_static_ranking(broken, submitted); }, "negative candidate index accepted");
    for (const auto score : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
        broken = ranked;
        broken["ranked"][0]["predicted_kernel_ms"] = score;
        rejects([&] { validate_static_ranking(broken, submitted); }, "invalid candidate score accepted");
    }
    broken = ranked;
    broken["ranked"][0]["predicted_kernel_ms"] = "fast";
    rejects([&] { validate_static_ranking(broken, submitted); }, "nonnumeric candidate score accepted");
    broken = ranked;
    std::swap(broken["ranked"][0], broken["ranked"][2]);
    rejects([&] { validate_static_ranking(broken, submitted); }, "unordered scores accepted");

    auto partial_profile = full_profile;
    partial_profile["capabilities"].erase("cta_barriers_per_second");
    fixture.write(partial_profile);
    const auto partial = run(request);
    validate_static_ranking(partial, submitted);
    require(partial.at("mode") == "unavailable" && partial.at("ranked").empty() &&
            partial.at("unranked").size() == 3 && partial.at("rejected").empty(),
            "partial profile invented a usable ranking");
    require(partial.at("reason").get<std::string>().find("missing or invalid") != std::string::npos,
            "partial profile did not explain unavailability");

    request["profile_dir"] = nullptr;
    auto bad_sample = sample({4});
    auto groups = Json::parse(bad_sample.at("execution_groups_json").get<std::string>());
    groups[0]["threads"] = 2048;
    bad_sample["execution_groups_json"] = groups;
    request["candidates"][2]["sample"] = bad_sample;
    const auto unavailable = run(request);
    validate_static_ranking(unavailable, submitted);
    require(unavailable.at("mode") == "unavailable" && unavailable.at("unranked").size() == 2 &&
            unavailable.at("rejected").size() == 1 && unavailable.at("rejected")[0].at("index") == 2,
            "unavailable performance model lost a known resource rejection");
    require(!std::filesystem::exists(fixture.directory / "cost_model.json"), "helper wrote a cost model");
    rejects([&] { run_static_ranker(request, (fixture.directory / "missing-python").string(), script); },
            "failed helper spawn accepted");
    std::cout << "PASS CPU static-ranking process bridge, response partition and partial-profile tests\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
