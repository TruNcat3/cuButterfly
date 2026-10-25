#include <cuntt/rns_tuning.hpp>
#include "plan_registry.hpp"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unistd.h>

namespace cuntt {
namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;

void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("RNS preparation CUDA error: ") + cudaGetErrorString(status));
}
template <class T> T integer(const Json& value) {
    static_assert(std::is_unsigned<T>::value, "unsigned field required");
    if (!value.is_number_unsigned() && !value.is_number_integer())
        throw std::invalid_argument("RNS JSON integer field has invalid type");
    if (value.is_number_integer() && !value.is_number_unsigned() && value.get<std::int64_t>() < 0)
        throw std::invalid_argument("RNS JSON integer field must be nonnegative");
    const auto number = value.get<std::uint64_t>();
    if (number > std::numeric_limits<T>::max())
        throw std::invalid_argument("RNS JSON integer field overflows");
    return static_cast<T>(number);
}
void version(const Json& value) {
    if (!value.is_object() || integer<unsigned>(value.at("schema_version")) != 1)
        throw std::invalid_argument("unsupported RNS JSON schema_version");
}
void fields(const Json& value, std::initializer_list<const char*> allowed, const char* object) {
    if (!value.is_object()) throw std::invalid_argument(std::string("RNS JSON ") + object + " must be an object");
    for (auto item = value.begin(); item != value.end(); ++item)
        if (std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end())
            throw std::invalid_argument(std::string("unknown RNS JSON ") + object + " field: " + item.key());
}
bool boolean(const Json& value, const char* field) {
    if (!value.is_boolean()) throw std::invalid_argument(std::string("RNS JSON ") + field + " must be boolean");
    return value.get<bool>();
}
template <class E> const char* enum_name(E value, std::initializer_list<std::pair<E, const char*>> choices) {
    for (const auto& item : choices) if (value == item.first) return item.second;
    throw std::invalid_argument("invalid RNS enum value");
}
template <class E> E enum_value(const Json& value, std::initializer_list<std::pair<E, const char*>> choices) {
    const auto name = value.get<std::string>();
    for (const auto& item : choices) if (name == item.second) return item.first;
    throw std::invalid_argument("invalid RNS JSON enum: " + name);
}
#define RING_CHOICES {{RnsRing::Cyclic, "cyclic"}, {RnsRing::Negacyclic, "negacyclic"}}
#define ORDER_CHOICES {{RnsOrder::Natural, "natural"}, {RnsOrder::BitReversed, "bit_reversed"}}
#define CORE_CHOICES {{RnsCoreFamily::Hybrid2D, "hybrid2d"}, {RnsCoreFamily::FusedCTGS, "fused_ctgs"}, {RnsCoreFamily::GpuNttMerge, "gpu_ntt_merge"}}
#define UNIT_CHOICES {{ComputeUnit::Radix2, "radix2"}, {ComputeUnit::Radix4, "radix4"}, {ComputeUnit::Radix8, "radix8"}}
#define CROSS_CHOICES {{CrossTwiddlePlacement::FirstPass, "first_pass"}, {CrossTwiddlePlacement::SecondPass, "second_pass"}}
#define ARITH_CHOICES {{RnsArithmetic::Shoup, "shoup"}, {RnsArithmetic::Barrett, "barrett"}}
#define EXCHANGE_CHOICES {{RnsExchange::Block, "block"}, {RnsExchange::DependencyScoped, "dependency_scoped"}}

std::string key_for(const Json& identity) {
    // Stable filename only: the complete identity is compared after loading, so
    // a hash collision cannot authorize replay of another request.
    std::uint64_t hash = 14695981039346656037ULL;
    for (unsigned char byte : identity.dump()) { hash ^= byte; hash *= 1099511628211ULL; }
    std::ostringstream output;
    output << "rns-v1-" << std::hex << std::setfill('0') << std::setw(16) << hash;
    return output.str();
}
std::string device_uuid(const cudaDeviceProp& properties) {
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (unsigned char byte : properties.uuid.bytes) result << std::setw(2) << unsigned(byte);
    return result.str();
}
void atomic_write(const fs::path& path, const Json& record) {
    fs::create_directories(path.parent_path());
    auto temporary = path.string() + ".tmp.XXXXXX";
    const int fd = mkstemp(temporary.data());
    if (fd < 0) throw std::runtime_error("cannot create RNS cache temporary file: " + std::string(std::strerror(errno)));
    FILE* file = fdopen(fd, "wb");
    if (!file) {
        const int error = errno;
        close(fd);
        std::remove(temporary.c_str());
        throw std::runtime_error("cannot open RNS cache temporary file: " + std::string(std::strerror(error)));
    }
    const auto contents = record.dump(2) + "\n";
    const bool wrote = std::fwrite(contents.data(), 1, contents.size(), file) == contents.size();
    const bool flushed = std::fflush(file) == 0;
    const bool synced = fsync(fd) == 0;
    const bool closed = std::fclose(file) == 0;
    if (!wrote || !flushed || !synced || !closed) {
        std::remove(temporary.c_str());
        throw std::runtime_error("cannot write complete RNS cache record");
    }
    std::error_code error;
    fs::rename(temporary, path, error);
    if (error) {
        std::remove(temporary.c_str());
        throw std::runtime_error("cannot publish RNS cache record: " + error.message());
    }
}
} // namespace

std::string rns_config_json(const RnsPlanConfig& c) {
    Json domains = Json::array(), polynomials = Json::array();
    for (const auto& d : c.domains) domains.push_back({{"modulus", d.modulus}, {"root", d.root}});
    for (const auto& p : c.polynomials)
        polynomials.push_back({{"domain", p.domain}, {"input_offset", p.input_offset}, {"output_offset", p.output_offset}});
    return Json{{"schema_version", 1}, {"log_n", c.log_n}, {"inverse", c.inverse},
                {"ring", enum_name<RnsRing>(c.ring, RING_CHOICES)},
                {"input_order", enum_name<RnsOrder>(c.input_order, ORDER_CHOICES)},
                {"output_order", enum_name<RnsOrder>(c.output_order, ORDER_CHOICES)},
                {"domains", domains}, {"polynomials", polynomials}}.dump();
}
RnsPlanConfig rns_config_from_json(const std::string& text) {
    const auto j = Json::parse(text);
    version(j);
    fields(j, {"schema_version", "log_n", "inverse", "ring", "input_order", "output_order", "domains", "polynomials"}, "config");
    RnsPlanConfig c;
    c.log_n = integer<std::uint32_t>(j.at("log_n"));
    c.inverse = boolean(j.at("inverse"), "inverse");
    c.ring = enum_value<RnsRing>(j.at("ring"), RING_CHOICES);
    c.input_order = enum_value<RnsOrder>(j.at("input_order"), ORDER_CHOICES);
    c.output_order = enum_value<RnsOrder>(j.at("output_order"), ORDER_CHOICES);
    if (!j.at("domains").is_array() || !j.at("polynomials").is_array())
        throw std::invalid_argument("RNS domains and polynomials must be arrays");
    for (const auto& d : j.at("domains")) {
        fields(d, {"modulus", "root"}, "domain");
        c.domains.push_back({integer<std::uint64_t>(d.at("modulus")), integer<std::uint64_t>(d.at("root"))});
    }
    for (const auto& p : j.at("polynomials")) {
        fields(p, {"domain", "input_offset", "output_offset"}, "polynomial");
        c.polynomials.push_back({integer<std::uint32_t>(p.at("domain")), integer<std::size_t>(p.at("input_offset")), integer<std::size_t>(p.at("output_offset"))});
    }
    return c;
}
std::string rns_mapping_json(const RnsMapping& m) {
    return Json{{"schema_version", 1}, {"n1_log", m.n1_log}, {"rows_per_block", m.rows_per_block},
                {"threads_per_block", m.threads_per_block}, {"compute_unit", enum_name<ComputeUnit>(m.compute_unit, UNIT_CHOICES)},
                {"cross_twiddle", enum_name<CrossTwiddlePlacement>(m.cross_twiddle, CROSS_CHOICES)},
                {"physical_boundary_traversal", m.physical_boundary_traversal},
                {"core_family", enum_name<RnsCoreFamily>(m.core_family, CORE_CHOICES)},
                {"second_rows_per_block", m.second_rows_per_block}, {"second_threads_per_block", m.second_threads_per_block},
                {"arithmetic", enum_name<RnsArithmetic>(m.arithmetic, ARITH_CHOICES)},
                {"exchange", enum_name<RnsExchange>(m.exchange, EXCHANGE_CHOICES)}}.dump();
}
RnsMapping rns_mapping_from_json(const std::string& text) {
    const auto j = Json::parse(text);
    version(j);
    fields(j, {"schema_version", "n1_log", "rows_per_block", "threads_per_block", "compute_unit", "cross_twiddle",
               "physical_boundary_traversal", "core_family", "second_rows_per_block", "second_threads_per_block", "arithmetic", "exchange"}, "mapping");
    RnsMapping m;
    m.n1_log = integer<std::uint32_t>(j.at("n1_log"));
    m.rows_per_block = integer<std::uint32_t>(j.at("rows_per_block"));
    m.threads_per_block = integer<std::uint32_t>(j.at("threads_per_block"));
    m.compute_unit = enum_value<ComputeUnit>(j.at("compute_unit"), UNIT_CHOICES);
    m.cross_twiddle = enum_value<CrossTwiddlePlacement>(j.at("cross_twiddle"), CROSS_CHOICES);
    m.physical_boundary_traversal = boolean(j.at("physical_boundary_traversal"), "physical_boundary_traversal");
    m.core_family = enum_value<RnsCoreFamily>(j.at("core_family"), CORE_CHOICES);
    m.second_rows_per_block = integer<std::uint32_t>(j.at("second_rows_per_block"));
    m.second_threads_per_block = integer<std::uint32_t>(j.at("second_threads_per_block"));
    m.arithmetic = enum_value<RnsArithmetic>(j.at("arithmetic"), ARITH_CHOICES);
    // Schema-v1 records predating exchange always used block-wide exchange.
    m.exchange = j.contains("exchange") ? enum_value<RnsExchange>(j.at("exchange"), EXCHANGE_CHOICES) : RnsExchange::Block;
    return m;
}

RnsPreparation prepare_rns_plan(RnsPlan& plan, RnsPlacement placement,
                                const RnsPrepareOptions& options, cudaStream_t stream) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    check(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
        throw std::invalid_argument("RNS preparation must run outside CUDA Graph capture");
    if (options.cache_directory.empty() && options.policy == RnsCachePolicy::Require)
        throw std::invalid_argument("RNS Require policy needs a nonempty cache_directory");
    if (!options.repetitions || options.repetitions > std::numeric_limits<unsigned>::max() / 3)
        throw std::invalid_argument("RNS repetitions must be positive and leave room for finalist confirmation");
    if (options.policy != RnsCachePolicy::Prefer && options.policy != RnsCachePolicy::Require && options.policy != RnsCachePolicy::Refresh)
        throw std::invalid_argument("invalid RNS cache policy");
    const auto placement_name = enum_name<RnsPlacement>(placement, {{RnsPlacement::Separate, "separate"}, {RnsPlacement::InPlace, "in_place"}});
    int device = 0, driver = 0, runtime = 0;
    cudaDeviceProp properties{};
    check(cudaGetDevice(&device));
    check(cudaGetDeviceProperties(&properties, device));
    check(cudaDriverGetVersion(&driver));
    check(cudaRuntimeGetVersion(&runtime));
    const auto request = Json::parse(rns_config_json(plan.config()));
    const Json identity{{"request", request}, {"placement", placement_name},
        {"hardware", {{"name", properties.name}, {"total_global_memory", properties.totalGlobalMem},
                      {"cc_major", properties.major}, {"cc_minor", properties.minor},
                      {"multiprocessor_count", properties.multiProcessorCount}}},
        {"runtime", {{"fingerprint", detail::runtime_fingerprint()}, {"driver_version", driver}, {"runtime_version", runtime}}},
        {"protocol", {{"name", "rns_bounded_cuda_graph_timing_only_v1"}, {"repetitions", options.repetitions}}}};
    RnsPreparation result;
    result.cache_key = key_for(identity);
    result.request_json = request.dump();
    const fs::path path = fs::path(options.cache_directory) / (result.cache_key + ".json");
    if (!options.cache_directory.empty() && options.policy != RnsCachePolicy::Refresh && fs::exists(path)) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("cannot read RNS cache record: " + path.string());
        Json record;
        input >> record;
        input >> std::ws;
        if (!input.eof()) throw std::runtime_error("RNS cache record has trailing data: " + path.string());
        version(record);
        if (record.at("identity") != identity)
            throw std::runtime_error("RNS cache identity mismatch (incompatible record or hash collision): " + path.string());
        if (record.at("source") != "bounded_tune_timing_only" ||
            record.at("validation") != "timing_only_not_mathematically_validated")
            throw std::runtime_error("unsupported RNS cache measurement provenance");
        result.kernel_ms = record.at("kernel_ms").get<double>();
        if (!std::isfinite(result.kernel_ms) || result.kernel_ms <= 0)
            throw std::runtime_error("invalid RNS cache kernel_ms");
        const auto mapping = rns_mapping_from_json(record.at("mapping").dump());
        plan.set_mapping(mapping, placement);
        result.mapping_json = rns_mapping_json(plan.mapping());
        result.cache_hit = true;
        result.source = "prepared_cache_timing_only";
        return result;
    }
    if (options.policy == RnsCachePolicy::Require)
        throw std::runtime_error("required RNS prepared cache record is missing: " + path.string());
    plan.tune(placement, stream, options.repetitions);
    result.kernel_ms = plan.selection().predicted_kernel_ms;
    if (!std::isfinite(result.kernel_ms) || result.kernel_ms <= 0)
        throw std::runtime_error("RNS bounded tuning did not produce a positive finite measurement");
    result.source = "bounded_tune_timing_only";
    result.mapping_json = rns_mapping_json(plan.mapping());
    if (options.cache_directory.empty()) {
        result.source = "uncached_bounded_tune_timing_only";
        return result;
    }
    const Json record{{"schema_version", 1}, {"identity", identity}, {"mapping", Json::parse(result.mapping_json)},
        {"kernel_ms", result.kernel_ms}, {"source", result.source}, {"validation", "timing_only_not_mathematically_validated"},
        {"search", {{"measured_candidates", plan.measurements().size()}, {"exhaustive", false},
                    {"installation_calibration", false}, {"finalists_confirmed", std::min<std::size_t>(3, plan.measurements().size())}}},
        {"provenance", {{"device_uuid", device_uuid(properties)}, {"device_index", device}}}};
    atomic_write(path, record);
    return result;
}
} // namespace cuntt
