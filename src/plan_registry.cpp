#include "plan_registry.hpp"
#include "fft_register_dispatch.hpp"
#include "jit_module.hpp"
#include "build_fingerprint.hpp"
#include "partition_space.hpp"
#include "resident_mapping.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace cuntt::detail {
const char* runtime_fingerprint() noexcept {
    const auto* mode=std::getenv("CUBUTTERFLY_COMPILE_MODE");
    return mode && std::string_view(mode)=="research" ? CUBUTTERFLY_RUNTIME_FINGERPRINT ":research" : CUBUTTERFLY_RUNTIME_FINGERPRINT ":default";
}
namespace {
using Json = nlohmann::json;

std::vector<std::vector<std::uint32_t>> capacity_partitions(unsigned log_n,unsigned value_bytes) {
    const auto hardware=query_hardware_resource_model();
    unsigned local=0;
    while(local<log_n && (2ULL<<(local+1))*value_bytes<=hardware.shared_bytes_per_block) ++local;
    const auto* value=std::getenv("CUBUTTERFLY_PARTITION_BUDGET");
    if(value && *value=='-') throw std::invalid_argument("CUBUTTERFLY_PARTITION_BUDGET must be nonnegative");
    return shared_partitions(log_n,local,value ? std::stoull(value) : 128);
}

std::string string_value(const Json& j) { return j.is_string() ? j.get<std::string>() : j.dump(); }
std::string field(const Json& j, const char* name, const std::string& fallback = "") {
    const auto it = j.find(name);
    return it == j.end() ? fallback : string_value(*it);
}
std::uint32_t integer(const Json& j, const char* name, std::uint32_t fallback) {
    if (!j.contains(name)) return fallback;
    const auto value = string_value(j.at(name));
    std::size_t used = 0;
    if (value.empty() || value.front() == '-') throw std::invalid_argument("negative mapping field");
    const auto number = std::stoull(value, &used);
    if (used != value.size() || number > UINT32_MAX) throw std::invalid_argument("invalid mapping integer");
    return static_cast<std::uint32_t>(number);
}
bool boolean(const Json& j, const char* name, bool fallback = false) {
    if (!j.contains(name)) return fallback;
    const auto value = field(j, name);
    if (value == "true" || value == "1") return true;
    if (value == "false" || value == "0") return false;
    throw std::invalid_argument("invalid mapping boolean");
}
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> out;
    std::istringstream stream(s);
    for (std::string value; std::getline(stream, value, delimiter);) out.push_back(value);
    return out;
}
std::vector<std::string> strings(const Json& j, const char* name, char delimiter = 'x') {
    if (!j.contains(name)) return {};
    if (j.at(name).is_array()) return j.at(name).get<std::vector<std::string>>();
    return split(field(j, name), delimiter);
}
std::vector<std::uint32_t> numbers(const Json& j, const char* name, char delimiter = 'x') {
    if (!j.contains(name)) return {};
    if (j.at(name).is_array()) return j.at(name).get<std::vector<std::uint32_t>>();
    std::vector<std::uint32_t> out;
    for (const auto& token : split(field(j, name), delimiter))
        out.push_back(integer(Json{{"value", token}}, "value", 0));
    return out;
}
Json mapping_object(const std::string& value) {
    auto j = Json::parse(value);
    if (j.contains("mapping_json")) j = Json::parse(j.at("mapping_json").get<std::string>());
    if (!j.is_object() || (j.contains("schema_version") && j.at("schema_version") != 1))
        throw std::invalid_argument("unsupported mapping schema");
    return j;
}

LocalPartitions local_partitions(const Json& j) {
    LocalPartitions out;
    if(!j.contains("local_stage_partitions")) return out;
    const auto& entries=j.at("local_stage_partitions");
    if(!entries.is_array()) throw std::invalid_argument("local_stage_partitions must be an array of arrays");
    for(const auto& entry:entries) {
        if(!entry.is_array()) throw std::invalid_argument("local stage partition must be an array");
        std::vector<std::uint32_t> part;
        for(const auto& value:entry) part.push_back(integer(Json{{"value",value}},"value",0));
        out.push_back(std::move(part));
    }
    return out;
}
std::vector<std::uint32_t> exchange_chunks(const Json& j) {
    std::vector<std::uint32_t> out;
    if(!j.contains("exchange_chunks")) return out;
    if(!j.at("exchange_chunks").is_array()) throw std::invalid_argument("exchange_chunks must be an array");
    for(const auto& value:j.at("exchange_chunks")) out.push_back(integer(Json{{"value",value}},"value",0));
    return out;
}

std::string registry_path() {
    if (const auto* path = std::getenv("CUBUTTERFLY_REGISTRY")) return path;
    if (const auto* path = std::getenv("XDG_CACHE_HOME")) return std::string(path) + "/cubutterfly/registry.json";
    if (const auto* path = std::getenv("HOME")) return std::string(path) + "/.cache/cubutterfly/registry.json";
    return {};
}
Json registry_records() {
    const auto path = registry_path();
    std::ifstream input(path);
    if (!input) return Json::array();
    Json document;
    input >> document;
    if (document.value("schema", "") != "cubutterfly-registry-v1")
        throw std::invalid_argument("unsupported cuButterfly registry schema: " + path);
    const auto device = current_device_info();
    const auto sm = std::to_string(device.compute_major) + "." + std::to_string(device.compute_minor);
    for (const auto& target : document.at("targets")) {
        const auto& hardware = target.at("hardware");
        if (hardware.at("device") == device.name && hardware.at("compute_capability") == sm &&
            hardware.at("global_memory_bytes").get<std::uint64_t>() == device.global_memory_bytes)
            return target.at("records");
    }
    return Json::array();
}
bool matches(const Json& key, const char* name, const std::string& value) {
    return field(key, name) == value;
}
bool matches(const Json& key, const char* name, std::uint64_t value) {
    return matches(key, name, std::to_string(value));
}
bool matches_semantics(const Json& key, const ButterflyConfig& c) {
    if (c.log_n > 30 || c.element_stride == 0) return false;
    const auto distance = c.batch_stride ? c.batch_stride : (((std::size_t{1} << c.log_n) - 1) * c.element_stride + 1);
    if (!matches(key, "operator", butterfly_operator_name(c.op)) ||
        !matches(key, "precision", butterfly_precision_name(c.precision)) ||
        !matches(key, "placement", butterfly_placement_name(c.placement)) ||
        !matches(key, "logN", c.log_n) || !matches(key, "batch", c.batch) ||
        !matches(key, "direction", c.inverse ? "inverse" : "forward") ||
        !matches(key, "accumulation", butterfly_accumulation_name(c.accumulation)) ||
        !matches(key, "element_stride", c.element_stride) || !matches(key, "batch_stride", distance)) return false;
    const bool inverse_scale=c.normalize_inverse && (c.op==ButterflyOperator::Fft || c.op==ButterflyOperator::Fwht);
    if (c.inverse && !matches(key, "normalization", inverse_scale ? "inverse" : "none")) return false;
    const auto matrices = split(field(key, "stage_matrices"), 'x');
    if (matrices.size() != c.stage_matrices.size()) return false;
    for (std::size_t i = 0; i < matrices.size(); ++i) {
        const auto entries = split(matrices[i], ':');
        const auto& m = c.stage_matrices[i];
        if (entries.size() != 4 || std::stod(entries[0]) != m.m00 || std::stod(entries[1]) != m.m01 ||
            std::stod(entries[2]) != m.m10 || std::stod(entries[3]) != m.m11) return false;
    }
    return true;
}
SelectionInfo information(const Json& record) {
    const auto device = current_device_info();
    return {true, true, device.name, record.at("candidate_id").get<std::string>(), "calibrated-registry",
            "exact hardware/semantic match; complete serialized mapping", record.at("kernel_ms").get<double>()};
}
std::string mapping_id(const std::string& mapping) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : mapping) hash = (hash ^ byte) * 1099511628211ULL;
    std::ostringstream text;
    text << "mapping-" << std::hex << hash;
    return text.str();
}

#define BUTTERFLY_ENUM_FIELDS(X) \
 X(backend, butterfly_backend) X(compute_unit, compute_unit) X(fft_core, fft_core) \
 X(complex_multiply, complex_multiply) X(cross_twiddle, cross_twiddle_mode) X(direct_boundary, direct_boundary) \
 X(local_exchange, local_exchange) X(shared_layout, shared_layout) X(stage_handoff, stage_handoff)
#define BUTTERFLY_INT_FIELDS(X) \
 X(tile_threads) X(local_stages) X(reorder_columns) X(stage_space) X(warp_stages) X(pipeline_warps) \
 X(prefix_threads) X(suffix_threads) X(prefix_ept) X(suffix_ept) X(prefix_units_per_cta) X(suffix_units_per_cta) X(batch_tile_count) \
 X(factor_ept) X(factor_columns) X(data_tiles_per_cta) X(prefetch_depth) X(factor_slices)
#define NTT_ENUM_FIELDS(X) \
 X(backend, backend) X(compute_unit, compute_unit) X(stage_handoff, stage_handoff) \
 X(hierarchical_core, hierarchical_core) X(dataflow_layout, dataflow_layout) X(dataflow_state_mode, dataflow_state_mode) \
 X(cross_twiddle_placement, cross_twiddle_placement) X(modular_multiply, modular_multiply) \
 X(packet_readiness_mode, packet_readiness_mode) X(packet_compute_layout, packet_compute_layout)
#define NTT_INT_FIELDS(X) \
 X(stage_space) X(flow_tile_log_n) X(data_space) X(role_stages) X(target_ctas_per_sm) X(data_time) \
 X(token_interleave) X(pipeline_buffers) X(n1_log) X(rows_per_block) X(threads_per_block) X(ready_window) X(batch_tile_count)
#define NTT_MAPPING_INTS(X) \
 X(threads_per_block) X(units_per_cta) X(data_space) X(data_time) X(role_stages) X(token_interleave) X(coefficient_reuse_stages) X(cta_weight)
#define NTT_ROLE_INTS(X) X(producer_weight) X(tail_weight) X(writer_weight) X(fragment_width) X(writer_tiles_per_cta) X(data_time_role_mask)
} // namespace

std::string serialize_mapping(const ButterflyConfig& c) {
    Json j{{"schema_version", 1}, {"kind", "butterfly"}, {"stage_partition", c.stage_partition}, {"stage_overlap", c.stage_overlap}};
    j["local_stage_partitions"]=c.local_stage_partitions;
    j["exchange_chunks"]=c.exchange_chunks;
#define ENUM_WRITE(member, name) j[#member] = name##_name(c.member);
#define INT_WRITE(member) j[#member] = c.member;
    BUTTERFLY_ENUM_FIELDS(ENUM_WRITE)
    BUTTERFLY_INT_FIELDS(INT_WRITE)
#undef ENUM_WRITE
#undef INT_WRITE
    // These axes are code-generation identities rather than workload
    // semantics.  Emit them explicitly so a replay cannot silently fall back
    // to the old native/linear implementation.
    j["prefix_codelet_lanes"] = c.prefix_codelet_lanes;
    j["prefix_codelet"] = c.prefix_codelet;
    j["prefix_shared_layout"] = c.prefix_shared_layout;
    auto factor_policies = c.factor_io_policies;
    if (c.backend == ButterflyBackend::FactorStreamed && factor_policies.empty() && !c.factor_partition.empty())
        factor_policies.assign(c.factor_partition.size(), "dynamic");
    j["factor_io_policies"] = factor_policies;
    auto mappings = [](const auto& values) {
        auto result = Json::array();
        for (const auto& m : values) {
            Json entry{{"core", fft_core_name(m.core)}, {"exchange", local_exchange_name(m.exchange)},
                       {"threads", m.threads}, {"ept", m.ept}};
            if (m.codelet != "native") entry["codelet"] = m.codelet;
            if (m.io_policy != "dynamic") entry["io_policy"] = m.io_policy;
            result.push_back(std::move(entry));
        }
        return result;
    };
    j["segment_mappings"] = mappings(c.segment_mappings);
    j["factor_partition"] = c.factor_partition;
    j["factor_overlap"] = c.factor_overlap;
    j["execution_group_mappings"] = mappings(c.execution_group_mappings);
    j["boundaries"] = Json::array();
    for (const auto& b : c.boundaries) j["boundaries"].push_back({{"twiddle", cross_twiddle_mode_name(b.cross_twiddle)},
        {"layout", direct_boundary_name(b.layout)}, {"residency", fft_boundary_residency_name(b.residency)}});
    return j.dump();
}

void apply_serialized_mapping(const std::string& value, ButterflyConfig& c) {
    const auto j = mapping_object(value);
    if (j.contains("kind") && j.at("kind") != "butterfly") throw std::invalid_argument("wrong mapping kind");
    c.auto_select = false;
#define ENUM_READ(member, name) if (j.contains(#member)) c.member = parse_##name(field(j, #member));
#define INT_READ(member) c.member = integer(j, #member, c.member);
    BUTTERFLY_ENUM_FIELDS(ENUM_READ)
    BUTTERFLY_INT_FIELDS(INT_READ)
#undef ENUM_READ
#undef INT_READ
    // A legacy mapping has no cooperative-lane axis and therefore denotes
    // the historical one-thread-per-prefix-register geometry.  Do not retain
    // a caller's non-default value when replaying such a record.
    c.prefix_codelet_lanes = integer(j, "prefix_codelet_lanes", 1);
    c.local_stage_partitions=local_partitions(j);
    c.exchange_chunks=exchange_chunks(j);
    if (j.contains("prefix_codelet")) c.prefix_codelet = field(j, "prefix_codelet");
    if (j.contains("prefix_shared_layout")) c.prefix_shared_layout = field(j, "prefix_shared_layout");
    if (j.contains("factor_io_policies")) c.factor_io_policies = strings(j, "factor_io_policies");
    c.stage_overlap = boolean(j, "stage_overlap");
    c.factor_overlap = boolean(j, "factor_overlap");
    c.stage_partition = numbers(j, j.contains("stage_partition") ? "stage_partition" : "stages_per_decomposition");
    c.factor_partition = numbers(j, "factor_partition");
    c.segment_mappings.clear(); c.execution_group_mappings.clear(); c.boundaries.clear();
    auto read_mappings = [&](const char* name, auto& out) {
        if (!j.contains(name)) return;
        for (const auto& m : j.at(name)) {
            FftSegmentMapping mapping{parse_fft_core(field(m, "core")),
                parse_local_exchange(field(m, "exchange")), integer(m, "threads", 0), integer(m, "ept", 0)};
            mapping.codelet = field(m, "codelet", "native");
            mapping.io_policy = field(m, "io_policy", "dynamic");
            out.push_back(std::move(mapping));
        }
    };
    read_mappings("segment_mappings", c.segment_mappings);
    read_mappings("execution_group_mappings", c.execution_group_mappings);
    // Older records and compact user mappings omit derived physical metadata.
    // Fill only omitted entries from the new top-level strategy axes; an
    // explicitly supplied conflicting value remains visible and is rejected
    // by the backend normalizer instead of being silently merged.
    if (c.backend == ButterflyBackend::OnlineReorder && c.fft_core == FftCore::RegisterTile &&
        c.stage_partition.size() == 2) {
        auto fill_codelet = [&](const char* name, auto& mappings) {
            if (!j.contains(name) || mappings.size() != c.stage_partition.size()) return;
            const auto& entries = j.at(name);
            for (std::size_t i = 0; i < mappings.size(); ++i)
                if (!entries.at(i).contains("codelet")) mappings[i].codelet = i == 0 ? c.prefix_codelet : "native";
        };
        fill_codelet("segment_mappings", c.segment_mappings);
        fill_codelet("execution_group_mappings", c.execution_group_mappings);
    }
    if (c.backend == ButterflyBackend::FactorStreamed && !c.factor_partition.empty()) {
        if (j.contains("factor_io_policies")) {
            auto policies = c.factor_io_policies;
            if (policies.empty()) policies.assign(c.factor_partition.size(), "dynamic");
            auto fill_io_policy = [&](const char* name, auto& mappings) {
                if (!j.contains(name) || mappings.size() != c.factor_partition.size()) return;
                const auto& entries = j.at(name);
                for (std::size_t i = 0; i < mappings.size(); ++i)
                    if (!entries.at(i).contains("io_policy")) mappings[i].io_policy = policies[i];
            };
            fill_io_policy("execution_group_mappings", c.execution_group_mappings);
        } else if (c.factor_io_policies.empty() && c.execution_group_mappings.size() == c.factor_partition.size()) {
            c.factor_io_policies.reserve(c.execution_group_mappings.size());
            for (const auto& mapping : c.execution_group_mappings) c.factor_io_policies.push_back(mapping.io_policy);
        }
    }
    if (j.contains("boundaries")) {
        for (const auto& b : j.at("boundaries")) c.boundaries.push_back({parse_cross_twiddle_mode(field(b, "twiddle")),
            parse_direct_boundary(field(b, "layout")), parse_fft_boundary_residency(field(b, "residency"))});
    } else if (c.backend == ButterflyBackend::OnlineReorder && c.stage_partition.size() > 1) {
        // Import legacy measured CSV records without candidate-name dispatch.
        auto legacy_mappings = [&](const char* tk, const char* ek, const char* ck, auto& out) {
            const auto threads = numbers(j, tk), ept = numbers(j, ek);
            const auto cores = split(field(j, ck), ':');
            if (threads.size() != ept.size() || (!cores.empty() && cores.size() != threads.size()))
                throw std::invalid_argument("incomplete legacy unit mapping");
            for (std::size_t i = 0; i < threads.size(); ++i) {
                FftSegmentMapping mapping{cores.empty() ? c.fft_core : parse_fft_core(cores[i]),
                    c.local_exchange, threads[i], ept[i]};
                out.push_back(std::move(mapping));
            }
        };
        legacy_mappings("segment_threads", "segment_ept", "segment_cores", c.segment_mappings);
        legacy_mappings("group_threads", "group_ept", "group_cores", c.execution_group_mappings);
        const auto twiddles = split(field(j, "boundary_twiddles"), 'x');
        const auto residences = split(field(j, "boundary_residencies"), 'x');
        const auto layouts = split(field(j, "boundary_layouts"), 'x');
        for (std::size_t i = 0; i + 1 < c.stage_partition.size(); ++i)
            c.boundaries.push_back({i < twiddles.size() ? parse_cross_twiddle_mode(twiddles[i]) : c.cross_twiddle,
                i < layouts.size() ? parse_direct_boundary(layouts[i]) : c.direct_boundary,
                i < residences.size() ? parse_fft_boundary_residency(residences[i]) : FftBoundaryResidency::GlobalScratch});
    }
}

std::string serialize_mapping(const PlanConfig& c) {
    Json j{{"schema_version", 1}, {"kind", "ntt"}, {"stage_partition", c.stage_partition},
           {"packet_fold_wave_barriers", c.packet_fold_wave_barriers}, {"profile_appt_roles", c.profile_appt_roles},
           {"stage_overlap", c.stage_overlap}};
    j["local_stage_partitions"]=c.local_stage_partitions;
    j["exchange_chunks"]=c.exchange_chunks;
#define ENUM_WRITE(member, name) j[#member] = name##_name(c.member);
#define INT_WRITE(member) j[#member] = c.member;
    NTT_ENUM_FIELDS(ENUM_WRITE)
    NTT_INT_FIELDS(INT_WRITE)
#undef ENUM_WRITE
#undef INT_WRITE
    auto mappings = [](const auto& values) {
        auto result = Json::array();
        for (const auto& m : values) {
            Json entry{{"core", ntt_subgraph_core_name(m.core)}};
#define MAP_WRITE(member) entry[#member] = m.member;
            NTT_MAPPING_INTS(MAP_WRITE)
#undef MAP_WRITE
            result.push_back(entry);
        }
        return result;
    };
    j["subgraph_mappings"] = mappings(c.subgraph_mappings);
    j["execution_group_mappings"] = mappings(c.execution_group_mappings);
    j["boundary_mappings"] = Json::array();
    for (const auto& b : c.boundary_mappings) j["boundary_mappings"].push_back({{"storage", boundary_storage_name(b.storage)}, {"buffers", b.buffers}});
#define ROLE_WRITE(member) j["appt_role_mapping"][#member] = c.appt_role_mapping.member;
    NTT_ROLE_INTS(ROLE_WRITE)
#undef ROLE_WRITE
    return j.dump();
}

void apply_serialized_mapping(const std::string& value, PlanConfig& c) {
    auto j = mapping_object(value);
    if (j.contains("kind") && j.at("kind") != "ntt") throw std::invalid_argument("wrong mapping kind");
    c.auto_select = false;
    for (const auto& alias : {std::pair<const char*, const char*>{"flow_tile_log", "flow_tile_log_n"},
                             {"cross_twiddle", "cross_twiddle_placement"}, {"dataflow_state", "dataflow_state_mode"}})
        if (j.contains(alias.first) && !j.contains(alias.second)) j[alias.second] = j[alias.first];
#define ENUM_READ(member, name) if (j.contains(#member)) c.member = parse_##name(field(j, #member));
#define INT_READ(member) c.member = integer(j, #member, c.member);
    NTT_ENUM_FIELDS(ENUM_READ)
    NTT_INT_FIELDS(INT_READ)
#undef ENUM_READ
#undef INT_READ
    c.stage_partition = numbers(j, "stage_partition", '+');
    c.local_stage_partitions=local_partitions(j);
    c.exchange_chunks=exchange_chunks(j);
    c.packet_fold_wave_barriers = boolean(j, "packet_fold_wave_barriers");
    c.stage_overlap = boolean(j, "stage_overlap");
    c.profile_appt_roles = boolean(j, "profile_appt_roles");
    c.subgraph_mappings.clear(); c.execution_group_mappings.clear(); c.boundary_mappings.clear();
    auto mappings = [&](const char* name, auto& result) {
        if (!j.contains(name)) return;
        for (const auto& entry : j.at(name)) {
            NttSubgraphMapping m;
            m.core = parse_ntt_subgraph_core(field(entry, "core"));
#define MAP_READ(member) m.member = integer(entry, #member, m.member);
            NTT_MAPPING_INTS(MAP_READ)
#undef MAP_READ
            result.push_back(m);
        }
    };
    mappings("subgraph_mappings", c.subgraph_mappings);
    mappings("execution_group_mappings", c.execution_group_mappings);
    if (j.contains("boundary_mappings")) for (const auto& entry : j.at("boundary_mappings"))
        c.boundary_mappings.push_back({parse_boundary_storage(field(entry, "storage")), integer(entry, "buffers", 1)});
    if (j.contains("appt_role_mapping")) {
#define ROLE_READ(member) c.appt_role_mapping.member = integer(j.at("appt_role_mapping"), #member, c.appt_role_mapping.member);
        NTT_ROLE_INTS(ROLE_READ)
#undef ROLE_READ
    }
    if (!j.contains("schema_version") && !c.stage_partition.empty())
        throw std::invalid_argument("legacy NTT subgraph records require full mapping revalidation");
}

std::optional<ButterflySelectionResult> registry_butterfly_mapping(const ButterflyConfig& config) {
    Json best;
    for (const auto& record : registry_records()) {
        if (record.value("status", "") != "measured" || record.value("runtime_fingerprint", "") != runtime_fingerprint() ||
            !matches_semantics(record.at("key"), config)) continue;
        const auto latency = record.at("kernel_ms").get<double>();
        if (latency > 0 && std::isfinite(latency) && (best.is_null() || latency < best.at("kernel_ms").get<double>())) best = record;
    }
    if (best.is_null()) return std::nullopt;
    auto selected = config;
    apply_serialized_mapping(best.at("mapping").dump(), selected);
    if (selected.backend == ButterflyBackend::CuFft) throw std::invalid_argument("external baseline in internal registry");
    return ButterflySelectionResult{std::move(selected), information(best)};
}
std::optional<NttSelectionResult> registry_ntt_mapping(const PlanConfig& config) {
    Json best;
    for (const auto& record : registry_records()) {
        if (record.value("status", "") != "measured" || record.value("runtime_fingerprint", "") != runtime_fingerprint()) continue;
        const auto& key = record.at("key");
        if (!matches(key, "operator", "ntt") || !matches(key, "precision", "word" + std::to_string(config.word_bits)) ||
            !matches(key, "logN", config.log_n) || !matches(key, "batch", config.batch) ||
            !matches(key, "modulus", config.modulus) || !matches(key, "direction", config.inverse ? "inverse" : "forward") ||
            !matches(key, "input_order", input_order_name(config.input_order)) ||
            !matches(key, "output_order", output_order_name(config.output_order))) continue;
        const auto latency = record.at("kernel_ms").get<double>();
        if (latency > 0 && std::isfinite(latency) && (best.is_null() || latency < best.at("kernel_ms").get<double>())) best = record;
    }
    if (best.is_null()) return std::nullopt;
    auto selected = config;
    apply_serialized_mapping(best.at("mapping").dump(), selected);
    return NttSelectionResult{std::move(selected), information(best)};
}

std::vector<ButterflyConfig> portable_butterfly_candidates(const ButterflyConfig& original) {
    auto base = original;
    base.auto_select = false;
    base.stage_partition.clear(); base.segment_mappings.clear(); base.execution_group_mappings.clear(); base.boundaries.clear();
    base.local_stage_partitions.clear(); base.exchange_chunks.clear();
    base.factor_partition.clear(); base.data_tiles_per_cta=1; base.prefetch_depth=0;
    base.factor_slices=1; base.factor_overlap=false;
    base.stage_overlap = false; base.batch_tile_count = 1;
    base.fft_core = FftCore::Scalar; base.local_exchange = LocalExchange::SharedMemory;
    base.shared_layout = SharedLayout::Linear; base.cross_twiddle = CrossTwiddleMode::Table;
    base.direct_boundary = DirectBoundary::Strided;
    std::vector<ButterflyConfig> candidates, specializations;
    if (base.log_n == 0 || base.log_n > 30) return candidates;
    if ((base.op == ButterflyOperator::Fwht || base.op == ButterflyOperator::Structured2x2) &&
        base.precision == ButterflyPrecision::Fp32 && base.log_n >= 3 && base.log_n <= 15) {
        auto c = base;
        c.backend = ButterflyBackend::TemporalTile; c.local_exchange = LocalExchange::WarpRegister;
        c.compute_unit = ComputeUnit::Radix2; c.tile_threads = 256;
        candidates.push_back(c);
    }
    if (base.op == ButterflyOperator::Fft) {
        if (base.precision == ButterflyPrecision::Fp32) {
            for (const auto& m : register_tile_mappings()) {
                if (m.log_n != base.log_n) continue;
                // Keep the register codelet/layout axes adjacent to the
                // existing executable points.  This makes the JIT variants
                // reachable under a bounded calibration budget instead of
                // appending a large Cartesian product after unrelated points.
                for (const auto& codelet : {std::string("native"), std::string("cufftdx-thread")})
                    for (const auto& layout : {std::string("linear"), std::string("xor")}) {
                        auto c = base;
                        apply_serialized_mapping(serialize_mapping(m), c);
                        c.prefix_codelet = codelet;
                        c.prefix_shared_layout = layout;
                        candidates.push_back(std::move(c));
                    }
            }
        }
        if (on_demand_compilation_enabled() &&
            (base.precision == ButterflyPrecision::Fp32 || base.precision == ButterflyPrecision::Fp64)) {
            const auto hardware = query_hardware_resource_model();
            const auto width = base.precision == ButterflyPrecision::Fp64 ? 16ULL : 8ULL;
            // Enumerate physical factor partitions independently of macro
            // grouping and per-CTA temporal data folding. No fixed group count.
            const auto* partition_budget=std::getenv("CUBUTTERFLY_PARTITION_BUDGET");
            if(partition_budget && *partition_budget=='-') throw std::invalid_argument("CUBUTTERFLY_PARTITION_BUDGET must be nonnegative");
            const auto budget=partition_budget ? std::stoull(partition_budget) : 128;
            for(unsigned columns:{1U,2U,4U,8U}) for(unsigned ept:{8U,16U})
                for(unsigned depth:{0U,1U,2U}) {
                    if(depth && hardware.compute_major<8) continue;
                    unsigned min_local=0,max_local=0;
                    while((1U<<min_local)<std::max(ept,32*ept/columns)) ++min_local;
                    while(max_local<14 && max_local<base.log_n &&
                          (1ULL<<(max_local+1))*columns*width*(depth+1)<=hardware.shared_bytes_per_block &&
                          (1ULL<<(max_local+1))/ept*columns<=hardware.max_threads_per_block) ++max_local;
                    for(const auto& partition:shared_partitions(base.log_n,max_local,budget,min_local,true))
                    for(unsigned tiles:{1U,4U,16U}) {
                        bool legal=!partition.empty(); unsigned done=0;
                        for(auto s:partition) {
                            if(s>14) { legal=false; break; }
                            const auto f=1U<<s;
                            const auto threads=f/ept*columns;
                            if(f<ept || threads<32 || threads>hardware.max_threads_per_block ||
                               threads%32 || columns>(1ULL<<(base.log_n-s)) || (done && columns>(1ULL<<done)) ||
                               f*columns*width*(depth+1)>hardware.shared_bytes_per_block) legal=false;
                            done+=s;
                        }
                        if(!legal || (depth && hardware.compute_major<8) || (tiles==1 && depth)) continue;
                        auto c=base; c.backend=ButterflyBackend::FactorStreamed; c.fft_core=FftCore::CufftDxBlock;
                        c.compute_unit=ComputeUnit::Auto; c.shared_layout=SharedLayout::WriterAligned;
                        c.cross_twiddle=CrossTwiddleMode::Recurrence; c.stage_partition={base.log_n};
                        c.factor_partition=partition; c.factor_ept=ept; c.factor_columns=columns;
                        c.data_tiles_per_cta=tiles; c.prefetch_depth=depth;
                        auto add_factor_schedules=[&](const auto& bulk) {
                            specializations.push_back(bulk);
                            if(partition.size()<3) return;
                            for(unsigned slices=2;slices<=(1U<<partition.back())/columns;slices*=2) {
                                auto streamed=bulk; streamed.factor_slices=slices; streamed.factor_overlap=true;
                                specializations.push_back(streamed);
                            }
                        };
                        // The factor I/O axis is physical and intentionally
                        // bounded: dynamic (empty means all dynamic), all
                        // static, one static factor at each position, and a
                        // few representative mixed assignments.  Arbitrary
                        // explicit vectors remain accepted by the public
                        // mapping parser without enumerating 2^K points.
                        std::vector<std::vector<std::string>> io_variants;
                        io_variants.push_back({});
                        io_variants.push_back(std::vector<std::string>(partition.size(), "static-unrolled"));
                        for (std::size_t static_group=0; static_group<partition.size(); ++static_group) {
                            std::vector<std::string> policies(partition.size(), "dynamic");
                            policies[static_group] = "static-unrolled";
                            io_variants.push_back(std::move(policies));
                        }
                        if (partition.size() > 1) {
                            std::vector<std::string> first_last(partition.size(), "dynamic");
                            first_last.front() = first_last.back() = "static-unrolled";
                            io_variants.push_back(std::move(first_last));
                        }
                        for (const auto& policies : io_variants) {
                            auto policy_candidate=c;
                            policy_candidate.factor_io_policies=policies;
                            add_factor_schedules(policy_candidate);
                        }
                        if(std::all_of(partition.begin(),partition.end(),[&](unsigned s) { return !(s%2) && ept==(1U<<(s/2)); })) {
                            c.fft_core=FftCore::RegisterTile;
                            for (const auto& policies : io_variants) {
                                auto policy_candidate=c;
                                policy_candidate.factor_io_policies=policies;
                                add_factor_schedules(policy_candidate);
                            }
                        }
                    }
                }
            for (unsigned local=2; local<=12; ++local) {
                if(base.log_n<=local || base.log_n-local<3 || base.log_n-local>14) continue;
                const auto suffix=base.log_n-local,pn=1U<<local,sn=1U<<suffix;
                std::vector<unsigned> factors;
                for(unsigned a=1;a<local;++a) factors.push_back(a);
                std::stable_sort(factors.begin(),factors.end(),[local](unsigned a,unsigned b) {
                    return std::abs(int(2*a)-int(local))<std::abs(int(2*b)-int(local));
                });
                for(unsigned log_a:factors) {
                    const auto log_b=local-log_a,a=1U<<log_a,b=1U<<log_b;
                    for(unsigned pc : {4U,2U,1U,8U,16U})
                    for(unsigned k=std::min(a,b);k;k/=2)
                    for(unsigned ept : {16U,32U,8U,4U}) for(unsigned sc : {4U,2U,1U}) {
                        const auto lanes=a/k,other_lanes=b/k,threads=pn/k*pc;
                        if(pc>sn || sc>pn || ept>sn || sn%ept ||
                           (lanes>1 && lanes*pc>32) || (other_lanes>1 && other_lanes*pc>32) ||
                           threads>hardware.max_threads_per_block || threads<32 ||
                           (sn/ept)*sc>hardware.max_threads_per_block || (sn/ept)*sc<32 ||
                           pn*pc*width>hardware.shared_bytes_per_block) continue;
                        // Whole exchange remains a control. All proper power-
                        // of-two chunks are reachable; resource checks below
                        // account for exchange storage, then JIT checks core storage.
                        std::vector<unsigned> chunks{0};
                        if(sc>1) for(unsigned chunk=1;chunk<ept;chunk*=2) chunks.push_back(chunk);
                        for(unsigned chunk:chunks) {
                            if(sc>1 && std::uint64_t(sn)*sc*width*(chunk?chunk:ept)/ept>hardware.shared_bytes_per_block) continue;
                            for(const auto& codelet:{std::string("native"),std::string("cufftdx-thread")})
                            for(const auto& layout:{std::string("linear"),std::string("xor")}) {
                            if(codelet=="cufftdx-thread" && base.precision!=ButterflyPrecision::Fp32) continue;
                            if(log_a!=log_b && layout=="xor" &&
                               std::max(lanes,other_lanes)>std::min<std::uint64_t>(b,128/(width*pc))) continue;
                            auto c=base;
                            c.backend=ButterflyBackend::OnlineReorder; c.fft_core=FftCore::RegisterTile;
                            c.compute_unit=ComputeUnit::Auto; c.local_stages=local; c.stage_partition={local,suffix};
                            c.prefix_codelet_lanes=lanes;
                            c.prefix_threads=threads; c.prefix_ept=k;
                            c.prefix_codelet=codelet; c.prefix_shared_layout=layout;
                            if(log_a!=log_b) c.local_stage_partitions={{log_a,log_b},{}};
                            if(chunk) c.exchange_chunks={0,chunk};
                            c.prefix_units_per_cta=pc;
                            c.suffix_threads=(sn/ept)*sc; c.suffix_ept=ept;
                            c.shared_layout=SharedLayout::WriterAligned; c.cross_twiddle=CrossTwiddleMode::Recurrence;
                            c.reorder_columns=1;
                            specializations.push_back(c);
                            }
                        }
                    }
                }
            }
        }
        if (base.log_n <= 14 && (base.precision == ButterflyPrecision::Fp32 || base.precision == ButterflyPrecision::Fp64)) {
            for (const auto threads : {128U, 256U, 512U, 1024U, 64U, 32U}) {
                auto c = base; c.backend = ButterflyBackend::TemporalTile;
                c.fft_core = base.log_n <= 10 ? FftCore::CufftDxBlock : FftCore::CufftDxDirect;
                c.tile_threads = threads; c.compute_unit = ComputeUnit::Auto;
                candidates.push_back(c);
            }
        }
        const auto units = fft_online_processing_units();
        for (const auto& prefix : units) for (const auto& suffix : units) {
            if (prefix.precision != base.precision || suffix.precision != base.precision || prefix.log_n + suffix.log_n != base.log_n) continue;
            auto c = base; c.backend = ButterflyBackend::OnlineReorder; c.fft_core = FftCore::CufftDxBlock;
            c.local_stages = prefix.log_n; c.prefix_threads = prefix.threads; c.prefix_ept = prefix.ept;
            c.suffix_threads = suffix.threads; c.suffix_ept = suffix.ept; c.compute_unit = ComputeUnit::Auto;
            for (auto twiddle : {CrossTwiddleMode::Recurrence, CrossTwiddleMode::Table})
                for (auto layout : {SharedLayout::Linear, SharedLayout::XorSwizzle})
                    for (auto boundary : {DirectBoundary::Strided, DirectBoundary::TiledTranspose, DirectBoundary::PrefixTiledTranspose}) {
                        if (layout==SharedLayout::XorSwizzle && base.precision==ButterflyPrecision::Fp32 && prefix.log_n>10) continue;
                        if (boundary!=DirectBoundary::Strided && base.precision!=ButterflyPrecision::Fp32) continue;
                        if (boundary==DirectBoundary::TiledTranspose && suffix.log_n<11) continue;
                        if (boundary==DirectBoundary::PrefixTiledTranspose && prefix.log_n<11) continue;
                        c.cross_twiddle=twiddle; c.shared_layout=layout; c.direct_boundary=boundary;
                        c.stage_overlap=false; c.batch_tile_count=1;
                        candidates.push_back(c);
                        for(std::size_t tile=1; tile<base.batch && tile<=UINT32_MAX; tile*=2) {
                            c.stage_overlap=true; c.batch_tile_count=tile;
                            candidates.push_back(c);
                        }
                    }
        }
    }
    for (const auto threads : {128U, 256U, 64U, 32U}) {
        if (base.log_n <= 10) {
            auto c = base; c.backend = ButterflyBackend::TemporalTile; c.tile_threads = threads;
            for(auto unit : {ComputeUnit::Radix4, ComputeUnit::Radix2, ComputeUnit::Radix8}) {
                c.compute_unit=unit; candidates.push_back(c);
            }
        } else {
            for (std::uint32_t local = std::min(10U, base.log_n - 1); local >= 5; --local) {
                auto c = base; c.backend = ButterflyBackend::Hierarchical; c.local_stages = local;
                c.tile_threads = threads;
                for(auto unit : {ComputeUnit::Radix4, ComputeUnit::Radix2, ComputeUnit::Radix8}) {
                    c.compute_unit=unit; c.backend=ButterflyBackend::Hierarchical; candidates.push_back(c);
                    c.backend=ButterflyBackend::OnlineReorder; c.reorder_columns=1; candidates.push_back(c);
                }
            }
        }
        for (std::uint32_t local = std::min(12U, base.log_n); local >= 1; --local) {
            auto c = base; c.backend = ButterflyBackend::SharedIterative; c.local_stages = local;
            c.tile_threads = threads; c.compute_unit = ComputeUnit::Radix2; c.shared_layout = SharedLayout::WriterAligned;
            candidates.push_back(c);
            c.shared_layout=SharedLayout::Linear; candidates.push_back(c);
        }
    }
    if(base.log_n==8) for(auto stages : {1U,2U,4U,8U}) {
        auto c=base; c.backend=ButterflyBackend::StagePipeline; c.compute_unit=ComputeUnit::Radix2;
        c.stage_space=stages; c.pipeline_warps=8; candidates.push_back(c);
    }
    const unsigned scalar_bytes=base.precision==ButterflyPrecision::Fp64 ? 8 :
        (base.precision==ButterflyPrecision::Fp16 || base.precision==ButterflyPrecision::Bf16 ? 2 : 4);
    for(const auto& partition : capacity_partitions(base.log_n,scalar_bytes*(base.op==ButterflyOperator::Fft ? 2 : 1)))
        for(auto threads : {64U,128U,256U}) for(auto layout : {SharedLayout::WriterAligned,SharedLayout::Linear}) {
            auto c=base; c.backend=ButterflyBackend::SharedIterative; c.stage_partition=partition;
            c.local_stages=0; c.compute_unit=ComputeUnit::Radix2; c.tile_threads=threads; c.shared_layout=layout;
            candidates.push_back(c);
            if(on_demand_compilation_enabled()) for(unsigned unit_stages:{2U,3U,4U,5U}) {
                auto resident=c;
                resident.local_stage_partitions=resident_partition_seed(partition,unit_stages);
                candidates.push_back(std::move(resident));
            }
        }
    candidates.insert(candidates.end(), specializations.begin(), specializations.end());
    const auto bulk_count=candidates.size();
    for (std::size_t i=0;i<bulk_count;++i) {
        const auto bulk=candidates[i];
        const bool shared=bulk.backend==ButterflyBackend::SharedIterative &&
            (bulk.stage_partition.empty() ? base.log_n>bulk.local_stages : bulk.stage_partition.size()>1);
        const bool register_fft=bulk.backend==ButterflyBackend::OnlineReorder && bulk.fft_core==FftCore::RegisterTile;
        if (!shared && !register_fft) continue;
        for (std::size_t tile=1;tile<base.batch && tile<=UINT32_MAX;tile*=2) {
            auto c=bulk; c.stage_overlap=true; c.batch_tile_count=tile;
            candidates.push_back(c);
        }
    }
    return candidates;
}

std::vector<PlanConfig> portable_ntt_candidates(const PlanConfig& original) {
    auto base = original; base.auto_select = false;
    base.local_stage_partitions.clear(); base.exchange_chunks.clear();
    base.stage_overlap=false; base.batch_tile_count=1;
    std::vector<PlanConfig> out;
    if(!base.log_n || base.log_n>30 || (base.word_bits!=32 && base.word_bits!=64)) return out;
    // These legacy families have no bit-reversed terminal writeback. Keep
    // them out of the executable search inventory for that semantic cell.
    if (base.output_order == OutputOrder::Natural && base.log_n >= 12 && base.log_n <= 20) {
        for (const auto unit : {ComputeUnit::Radix4, ComputeUnit::Radix2})
            for (std::uint32_t local = 6; local <= 10; ++local)
                for (const auto rows : {4U, 2U, 1U}) {
                    auto c = base; c.backend = Backend::Hybrid2D; c.compute_unit = unit;
                    c.n1_log = local; c.rows_per_block = rows; c.threads_per_block = 256;
                    out.push_back(c);
                }
    }
    if (base.output_order == OutputOrder::Natural)
        for (const auto backend : {Backend::Tile256, Backend::Baseline}) {
            auto c = base; c.backend = backend; c.compute_unit = ComputeUnit::Radix2; out.push_back(c);
        }
    for (std::uint32_t local=1; local<=std::min(12U, base.log_n); ++local)
        for (auto threads : {32U,64U,128U,256U})
            for (auto layout : {DataflowLayout::HermesXor, DataflowLayout::Linear}) {
                auto c=base; c.backend=Backend::SharedIterative; c.compute_unit=ComputeUnit::Radix2;
                c.flow_tile_log_n=local; c.threads_per_block=threads; c.dataflow_layout=layout;
                c.stage_partition.clear(); c.subgraph_mappings.clear(); c.boundary_mappings.clear();
                c.execution_group_mappings.clear(); out.push_back(c);
            }
    for(const auto& partition : capacity_partitions(base.log_n,base.word_bits/8))
        for(auto threads : {64U,128U,256U}) for(auto layout : {DataflowLayout::HermesXor,DataflowLayout::Linear}) {
            auto c=base; c.backend=Backend::SharedIterative; c.stage_partition=partition;
            c.compute_unit=ComputeUnit::Radix2; c.threads_per_block=threads; c.dataflow_layout=layout;
            c.subgraph_mappings.clear(); c.boundary_mappings.clear(); c.execution_group_mappings.clear();
            out.push_back(c);
            if(on_demand_compilation_enabled()) for(unsigned unit_stages:{2U,3U,4U,5U}) {
                auto resident=c;
                resident.local_stage_partitions=resident_partition_seed(partition,unit_stages);
                out.push_back(std::move(resident));
            }
        }
    const auto bulk_count=out.size();
    for (std::size_t i=0;i<bulk_count;++i) {
        const auto bulk=out[i];
        if (bulk.backend!=Backend::SharedIterative ||
            (bulk.stage_partition.empty() ? base.log_n<=bulk.flow_tile_log_n : bulk.stage_partition.size()<2)) continue;
        for (std::size_t tile=1;tile<base.batch && tile<=UINT32_MAX;tile*=2) {
            auto c=bulk; c.stage_overlap=true; c.batch_tile_count=tile;
            out.push_back(c);
        }
    }
    return out;
}

ButterflySelectionResult select_portable_butterfly(const ButterflyConfig& config) {
    std::string last_error;
    for (auto c : portable_butterfly_candidates(config)) try {
        const auto allocate = c.auto_allocate_workspace;
        c.auto_allocate_workspace = false;
        ButterflyPlan probe(c);
        c = probe.config(); c.auto_allocate_workspace = allocate;
        const auto id = mapping_id(serialize_mapping(c));
        return {std::move(c), {true, false, current_device_info().name, id, "unmeasured-feasible",
                              "compiled executable candidate; run calibration to optimize this workload", 0.0}};
    } catch (const std::invalid_argument& e) { last_error = e.what(); }
    throw std::invalid_argument("no executable butterfly mapping: " + last_error);
}
NttSelectionResult select_portable_ntt(const PlanConfig& config) {
    std::string last_error;
    for (auto c : portable_ntt_candidates(config)) try {
        const auto allocate = c.auto_allocate_workspace; c.auto_allocate_workspace = false;
        Plan probe(c); c = probe.config(); c.auto_allocate_workspace = allocate;
        const auto id = mapping_id(serialize_mapping(c));
        return {std::move(c), {true, false, current_device_info().name, id, "unmeasured-feasible",
                              "compiled executable candidate; run calibration to optimize this workload", 0.0}};
    } catch (const std::invalid_argument& e) { last_error = e.what(); }
    throw std::invalid_argument("no executable NTT mapping: " + last_error);
}
} // namespace cuntt::detail
