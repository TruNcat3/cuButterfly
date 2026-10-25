#include <cuntt/rns_tuning.hpp>
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using Json = nlohmann::json;
using Word = std::uint64_t;
namespace fs = std::filesystem;
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }
template <class F> void must_throw(F action, const std::string& label) {
    bool threw = false;
    try { action(); } catch (const std::exception&) { threw = true; }
    require(threw, "expected failure: " + label);
}
Word mul(Word a, Word b, Word q) { return static_cast<unsigned __int128>(a) * b % q; }
Word power(Word a, Word n, Word q) {
    Word result = 1;
    for (; n; n >>= 1, a = mul(a, a, q)) if (n & 1) result = mul(result, a, q);
    return result;
}
std::size_t reverse(std::size_t x, unsigned bits) {
    std::size_t y = 0;
    for (unsigned i = 0; i < bits; ++i, x >>= 1) y = 2 * y + (x & 1);
    return y;
}
cuntt::RnsPlanConfig config() {
    cuntt::RnsPlanConfig c;
    c.log_n = 12;
    const Word q = 2013265921;
    c.domains = {{q, power(31, (q - 1) / (2ULL << c.log_n), q)}};
    c.polynomials = {{0, 0, 0}, {0, 1ULL << c.log_n, 1ULL << c.log_n}};
    return c;
}
void json_contract_tests() {
    const auto request = Json::parse(cuntt::rns_config_json(config()));
    const auto mapping = Json::parse(cuntt::rns_mapping_json(cuntt::RnsMapping{}));
    require(Json::parse(cuntt::rns_mapping_json(cuntt::rns_mapping_from_json(mapping.dump()))) == mapping, "mapping roundtrip");
    require(mapping.at("exchange") == "block", "default exchange is block");
    auto legacy = mapping;
    legacy.erase("exchange");
    const auto legacy_mapping = cuntt::rns_mapping_from_json(legacy.dump());
    require(legacy_mapping.exchange == cuntt::RnsExchange::Block, "legacy schema-v1 exchange is explicitly block");
    require(Json::parse(cuntt::rns_mapping_json(legacy_mapping)) == mapping, "legacy mapping exports explicit block exchange");
    auto dependency = mapping;
    dependency["exchange"] = "dependency_scoped";
    require(cuntt::rns_mapping_from_json(dependency.dump()).exchange == cuntt::RnsExchange::DependencyScoped,
            "dependency-scoped exchange parses");
    require(Json::parse(cuntt::rns_mapping_json(cuntt::rns_mapping_from_json(dependency.dump()))) == dependency,
            "dependency-scoped mapping roundtrip");
    for (const auto& value : Json::array({"unknown", "Block", "", 0, true, nullptr, Json::array(), Json::object()})) {
        auto bad_exchange = mapping;
        bad_exchange["exchange"] = value;
        must_throw([&] { cuntt::rns_mapping_from_json(bad_exchange.dump()); }, "invalid exchange rejected");
    }
    auto invalid_enum = cuntt::RnsMapping{};
    invalid_enum.exchange = static_cast<cuntt::RnsExchange>(255);
    must_throw([&] { cuntt::rns_mapping_json(invalid_enum); }, "invalid exchange cannot serialize");
    for (const auto* field : {"unknown", "normalization", "normalize", "batch"}) {
        auto invalid = request;
        invalid[field] = true;
        must_throw([&] { cuntt::rns_config_from_json(invalid.dump()); }, std::string("unknown config field ") + field);
    }
    auto invalid = request;
    invalid["domains"][0]["root_order"] = "forward";
    must_throw([&] { cuntt::rns_config_from_json(invalid.dump()); }, "unknown domain field");
    invalid = request;
    invalid["polynomials"][0]["stride"] = 1;
    must_throw([&] { cuntt::rns_config_from_json(invalid.dump()); }, "unknown polynomial field");
    invalid = mapping;
    invalid["normalize"] = true;
    must_throw([&] { cuntt::rns_mapping_from_json(invalid.dump()); }, "unknown mapping field");
    for (const auto& value : Json::array({0, 1, 0.5, "true", nullptr, Json::array(), Json::object()})) {
        invalid = request;
        invalid["inverse"] = value;
        must_throw([&] { cuntt::rns_config_from_json(invalid.dump()); }, "inverse requires JSON boolean");
        invalid = mapping;
        invalid["physical_boundary_traversal"] = value;
        must_throw([&] { cuntt::rns_mapping_from_json(invalid.dump()); }, "physical traversal requires JSON boolean");
    }
    must_throw([&] { cuntt::rns_config_from_json(request.dump() + " {}"); }, "config trailing JSON");
    must_throw([&] { cuntt::rns_mapping_from_json(mapping.dump() + " {}"); }, "mapping trailing JSON");
}
// Independent CPU radix-2 NTT, with explicit twist and output permutation.
void exact(cuntt::RnsPlan& plan) {
    const auto& c = plan.config();
    const std::size_t n = 1ULL << c.log_n;
    std::vector<Word> input(2 * n), expected(2 * n), output(2 * n);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = (i * 137 + 29) % c.domains[0].modulus;
    for (const auto& p : c.polynomials) {
        const auto& d = c.domains[p.domain];
        std::vector<Word> values(n);
        Word twist = 1;
        for (std::size_t j = 0; j < n; ++j) {
            values[reverse(j, c.log_n)] = mul(input[p.input_offset + j], twist, d.modulus);
            twist = mul(twist, d.root, d.modulus);
        }
        const Word omega = mul(d.root, d.root, d.modulus);
        for (std::size_t length = 2; length <= n; length *= 2) {
            const Word step = power(omega, n / length, d.modulus);
            for (std::size_t start = 0; start < n; start += length) {
                Word factor = 1;
                for (std::size_t j = 0; j < length / 2; ++j) {
                    const Word a = values[start + j], b = mul(values[start + j + length / 2], factor, d.modulus);
                    values[start + j] = (a + b) % d.modulus;
                    values[start + j + length / 2] = (a + d.modulus - b) % d.modulus;
                    factor = mul(factor, step, d.modulus);
                }
            }
        }
        for (std::size_t j = 0; j < n; ++j) expected[p.output_offset + reverse(j, c.log_n)] = values[j];
    }
    Word* device = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&device), input.size() * sizeof(Word)));
    try {
        check(cudaMemcpy(device, input.data(), input.size() * sizeof(Word), cudaMemcpyHostToDevice));
        plan.execute_async(device, device);
        check(cudaMemcpy(output.data(), device, output.size() * sizeof(Word), cudaMemcpyDeviceToHost));
        require(output == expected, "cached mapping must match exact CPU negacyclic NTT");
    } catch (...) { cudaFree(device); throw; }
    check(cudaFree(device));
}
void write_record(const fs::path& path, const Json& record) {
    std::ofstream output(path, std::ios::trunc);
    output << record.dump(2);
    require(bool(output), "write test cache fixture");
}
struct Temporary {
    fs::path path;
    Temporary() {
        char pattern[] = "/tmp/cuntt-rns-cache-test-XXXXXX";
        const auto created = mkdtemp(pattern);
        require(created != nullptr, "mkdtemp");
        path = created;
    }
    ~Temporary() { std::error_code error; fs::remove_all(path, error); }
};
} // namespace

int main(int argc, char** argv) {
    try {
        json_contract_tests();
        if (argc == 2 && std::string(argv[1]) == "--json-only") {
            std::cout << "RNS strict JSON contract tests passed\n";
            return 0;
        }
        auto c = config();
        require(cuntt::rns_config_json(cuntt::rns_config_from_json(cuntt::rns_config_json(c))) == cuntt::rns_config_json(c), "request roundtrip");
        if (argc == 4 && std::string(argv[1]) == "--require-child") {
            cuntt::RnsPlan plan(c);
            const auto result = cuntt::prepare_rns_plan(plan, cuntt::RnsPlacement::InPlace,
                {argv[2], cuntt::RnsCachePolicy::Require, 1});
            require(result.cache_hit && result.cache_key == argv[3], "fresh process equivalent identity cache hit");
            require(plan.measurements().empty(), "cache replay does not benchmark");
            if (plan.mapping().core_family != cuntt::RnsCoreFamily::Hybrid2D)
                require(plan.workspace_bytes() == 0, "in-place native fused replay releases scratch");
            exact(plan);
            return 0;
        }
        require(argc == 1, "unexpected test arguments");
        Temporary temporary;
        const auto directory = temporary.path.string();
        cuntt::RnsPrepareOptions options{directory, cuntt::RnsCachePolicy::Require, 1};
        cuntt::RnsPlan missing(c);
        must_throw([&] { cuntt::prepare_rns_plan(missing, cuntt::RnsPlacement::InPlace,
            {"", cuntt::RnsCachePolicy::Require, 1}); }, "Require empty directory");
        must_throw([&] { cuntt::prepare_rns_plan(missing, cuntt::RnsPlacement::InPlace, options); }, "Require missing cache");
        options.policy = cuntt::RnsCachePolicy::Prefer;
        const auto prepared = cuntt::prepare_rns_plan(missing, cuntt::RnsPlacement::InPlace, options);
        require(!prepared.cache_hit && !missing.measurements().empty(), "Prefer measures missing request");
        exact(missing);
        const auto path = temporary.path / (prepared.cache_key + ".json");
        Json record;
        { std::ifstream input(path); input >> record; }
        require(record.at("validation") == "timing_only_not_mathematically_validated", "honest measurement status");
        const pid_t child = fork();
        require(child >= 0, "fork");
        if (child == 0) {
            execl("/proc/self/exe", argv[0], "--require-child", directory.c_str(), prepared.cache_key.c_str(), static_cast<char*>(nullptr));
            _exit(127);
        }
        int status = 0;
        require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0, "fresh-process cache replay");
        options.policy = cuntt::RnsCachePolicy::Require;
        cudaStream_t capturing = nullptr;
        check(cudaStreamCreateWithFlags(&capturing, cudaStreamNonBlocking));
        check(cudaStreamBeginCapture(capturing, cudaStreamCaptureModeThreadLocal));
        must_throw([&] { cuntt::prepare_rns_plan(missing, cuntt::RnsPlacement::InPlace, options, capturing); }, "cache hit during active capture");
        cudaGraph_t empty_graph = nullptr;
        check(cudaStreamEndCapture(capturing, &empty_graph));
        if (empty_graph) check(cudaGraphDestroy(empty_graph));
        check(cudaStreamDestroy(capturing));
        must_throw([&] { cuntt::prepare_rns_plan(missing, cuntt::RnsPlacement::Separate, options); }, "placement identity");
        auto changed = c;
        changed.domains[0].root = power(changed.domains[0].root, 3, changed.domains[0].modulus);
        cuntt::RnsPlan changed_root(changed);
        must_throw([&] { cuntt::prepare_rns_plan(changed_root, cuntt::RnsPlacement::InPlace, options); }, "root identity");
        changed = c;
        changed.domains.push_back({469762049, power(3, (469762049 - 1) / 8192, 469762049)});
        changed.polynomials[1].domain = 1;
        cuntt::RnsPlan changed_domain(changed);
        must_throw([&] { cuntt::prepare_rns_plan(changed_domain, cuntt::RnsPlacement::InPlace, options); }, "domain and per-polynomial domain identity");
        changed = c;
        std::swap(changed.polynomials[0], changed.polynomials[1]);
        cuntt::RnsPlan changed_offsets(changed);
        must_throw([&] { cuntt::prepare_rns_plan(changed_offsets, cuntt::RnsPlacement::InPlace, options); }, "polynomial offsets and order identity");
        auto mismatch = record;
        for (const auto* field : {"name", "total_global_memory", "cc_major", "cc_minor", "multiprocessor_count"}) {
            mismatch = record;
            mismatch["identity"]["hardware"][field] = "incompatible GPU fixture";
            write_record(path, mismatch);
            must_throw([&] { cuntt::prepare_rns_plan(missing, cuntt::RnsPlacement::InPlace, options); }, "full hardware identity collision check");
        }
        mismatch = record;
        mismatch["identity"]["runtime"]["fingerprint"] = "different build fixture";
        write_record(path, mismatch);
        must_throw([&] { cuntt::prepare_rns_plan(missing, cuntt::RnsPlacement::InPlace, options); }, "build identity");
        mismatch = record;
        mismatch["identity"]["runtime"]["driver_version"] = 0;
        write_record(path, mismatch);
        must_throw([&] { cuntt::prepare_rns_plan(missing, cuntt::RnsPlacement::InPlace, options); }, "driver identity");
        mismatch = record;
        mismatch["provenance"]["device_uuid"] = "another identical GPU fixture";
        write_record(path, mismatch);
        require(cuntt::prepare_rns_plan(missing, cuntt::RnsPlacement::InPlace, options).cache_hit, "UUID is provenance, identical GPU capabilities remain compatible");
        mismatch = record;
        mismatch["mapping"]["threads_per_block"] = 3;
        write_record(path, mismatch);
        must_throw([&] { cuntt::prepare_rns_plan(missing, cuntt::RnsPlacement::InPlace, options); }, "illegal mapping");
        { std::ofstream output(path); output << "{incomplete"; }
        must_throw([&] { cuntt::prepare_rns_plan(missing, cuntt::RnsPlacement::InPlace, options); }, "Require malformed cache");
        options.policy = cuntt::RnsCachePolicy::Prefer;
        must_throw([&] { cuntt::prepare_rns_plan(missing, cuntt::RnsPlacement::InPlace, options); }, "Prefer must expose malformed cache");
        write_record(path, record);
        { std::ofstream output(path, std::ios::app); output << " trailing"; }
        must_throw([&] { cuntt::prepare_rns_plan(missing, cuntt::RnsPlacement::InPlace, options); }, "trailing cache corruption");
        auto invalid = Json::parse(cuntt::rns_config_json(c));
        invalid["domains"][0]["root"] = -1;
        must_throw([&] { cuntt::rns_config_from_json(invalid.dump()); }, "negative unsigned root");
        auto wide = c;
        wide.domains[0] = {18446744073709551557ULL, 18446744073709551556ULL};
        require(cuntt::rns_config_json(cuntt::rns_config_from_json(cuntt::rns_config_json(wide))) == cuntt::rns_config_json(wide), "full-width uint64 JSON roundtrip");
        options.policy = cuntt::RnsCachePolicy::Refresh;
        const auto refreshed = cuntt::prepare_rns_plan(missing, cuntt::RnsPlacement::InPlace, options);
        require(!refreshed.cache_hit && refreshed.cache_key == prepared.cache_key, "Refresh measures and atomically replaces corrupt record");
        exact(missing);
        std::cout << "RNS preparation cache: exact CPU acceptance and cache policy/identity/corruption tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RNS preparation cache test failed: " << error.what() << '\n';
        return 1;
    }
}
