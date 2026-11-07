#include "plan_registry.hpp"
#include "fwht_contract.hpp"
#include "temporal_tile_contract.hpp"
#include "online_reorder_contract.hpp"
#include "shared_tile_contract.hpp"
#include "fft_register_dispatch.hpp"
#include "generated_unit_api.cuh"
#include "jit_module.hpp"
#include "build_fingerprint.hpp"
#include "partition_space.hpp"
#include "resident_mapping.hpp"
#include "shared_organization_candidates.hpp"
#include "ntt_capabilities.hpp"
#include "ntt_template_candidates.hpp"
#include "platform_tuning.hpp"
#include "cufftdx_provider.hpp"
#include "fft_local_core_contract.hpp"
#include "hardware_model.hpp"
#include "static_candidate_ranking.hpp"
#include "stage_probe.hpp"
#include "plan_search_trace.hpp"
#include "jit_config.hpp"
#include <cubutterfly/mapping.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace cuntt::detail {
const char* local_selector_fingerprint() noexcept {
    return CUBUTTERFLY_LOCAL_SELECTOR_FINGERPRINT;
}
const char* runtime_fingerprint() noexcept {
    switch (compile_policy()) {
    case CompilePolicy::Precompiled: return CUBUTTERFLY_RUNTIME_FINGERPRINT ":precompiled";
    case CompilePolicy::Auto: return CUBUTTERFLY_RUNTIME_FINGERPRINT ":auto";
    case CompilePolicy::OnDemand: return CUBUTTERFLY_RUNTIME_FINGERPRINT ":on-demand";
    case CompilePolicy::Research: return CUBUTTERFLY_RUNTIME_FINGERPRINT ":research";
    case CompilePolicy::Invalid: return CUBUTTERFLY_RUNTIME_FINGERPRINT ":invalid";
    }
    return CUBUTTERFLY_RUNTIME_FINGERPRINT ":invalid";
}
namespace {
using Json = nlohmann::json;

std::vector<std::vector<std::uint32_t>> capacity_partitions(unsigned log_n,unsigned value_bytes) {
    const auto hardware=query_hardware_resource_model();
    unsigned local=0;
    while(local<log_n && (2ULL<<(local+1))*value_bytes<=hardware.shared_bytes_per_block) ++local;
    const auto* value=std::getenv("CUBUTTERFLY_PARTITION_BUDGET");
    if(value && *value=='-') throw std::invalid_argument("CUBUTTERFLY_PARTITION_BUDGET must be nonnegative");
    return shared_partitions(log_n,local,value ? std::stoull(value) : 128,1,true);
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
    const bool explicit_epilogue = c.output_scale != 1.0 || c.final_storage_rounding;
    if (explicit_epilogue && (!key.contains("output_scale") || !key.contains("output_rounding"))) return false;
    if (key.contains("output_scale")) {
        if (!key.at("output_scale").is_number() || key.at("output_scale").get<double>() != c.output_scale) return false;
    }
    if (key.contains("output_rounding") &&
        (!key.at("output_rounding").is_number_integer() ||
         key.at("output_rounding").get<int>() != (c.final_storage_rounding ? 1 : 0))) return false;
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
    if (c.output_scale != 1.0 || c.final_storage_rounding)
        j["output_epilogue"] = {{"scale", c.output_scale}, {"final_storage_rounding", c.final_storage_rounding}};
    if (c.final_storage_rounding)
        j["numeric_lowering"] = {{"kind", c.local_exchange == LocalExchange::WarpRegister
            ? "short-fwht-fp32-final-storage" : "promoted-full-fp32-final-storage"},
            {"arithmetic", "fp32"}, {"storage_conversion_kernels", c.local_exchange == LocalExchange::WarpRegister ? 0 : 2},
            {"scale_position", "fp32-before-final-fp16-narrow"}};
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
    j["register_threads"] = c.register_threads;
    j["prefix_codelet"] = c.prefix_codelet;
    j["prefix_shared_layout"] = c.prefix_shared_layout;
    j["prefix_twiddle"] = c.prefix_twiddle;
    if (fp16_local_block(c)) {
        const auto shape = fp16_local_block_shape(c);
        j["local_core_contract"] = {{"packing", "half2-transform-pair-v1"},
            {"accumulation", "native"}, {"normalization", "final-store-fp16"},
            {"logical_transforms_per_cta", shape.transforms_per_cta}};
    }
    if (c.suffix_core != "cufftdx-block") j["suffix_core"] = c.suffix_core;
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
    if (j.contains("output_epilogue")) {
        const auto& e = j.at("output_epilogue");
        if (!e.is_object() || e.size() != 2 || !e.contains("scale") || !e.at("scale").is_number() ||
            !e.contains("final_storage_rounding") || !e.at("final_storage_rounding").is_boolean() ||
            e.at("scale").get<double>() != c.output_scale ||
            e.at("final_storage_rounding").get<bool>() != c.final_storage_rounding)
            throw std::invalid_argument("mapping output epilogue contract mismatch");
    }
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
    c.suffix_core = field(j, "suffix_core", "cufftdx-block");
    if (c.suffix_core != "cufftdx-block" && c.suffix_core != "native")
        throw std::invalid_argument("suffix_core must be cufftdx-block or native");
    if (c.suffix_core != "cufftdx-block" &&
        !(c.op == ButterflyOperator::Fft && c.backend == ButterflyBackend::OnlineReorder && c.fft_core == FftCore::RegisterTile))
        throw std::invalid_argument("suffix_core applies only to online-reorder register-tile FFT");
    // Legacy register mappings select the historical compiled shape. Do not
    // inherit a different request from the config being replayed into.
    c.register_threads = integer(j, "register_threads", 0);
    c.local_stage_partitions=local_partitions(j);
    c.exchange_chunks=exchange_chunks(j);
    c.prefix_codelet = field(j, "prefix_codelet", "native");
    c.prefix_shared_layout = field(j, "prefix_shared_layout", "linear");
    c.prefix_twiddle = field(j, "prefix_twiddle", "runtime");
    if (c.prefix_twiddle != "runtime" && c.prefix_twiddle != "table")
        throw std::invalid_argument("prefix_twiddle must be runtime or table");
    if (c.prefix_twiddle != "runtime" &&
        !(c.op == ButterflyOperator::Fft && c.backend == ButterflyBackend::OnlineReorder && c.fft_core == FftCore::RegisterTile))
        throw std::invalid_argument("prefix_twiddle applies only to online-reorder register-tile FFT");
    if (j.contains("factor_io_policies")) c.factor_io_policies = strings(j, "factor_io_policies");
    c.stage_overlap = boolean(j, "stage_overlap");
    c.factor_overlap = boolean(j, "factor_overlap");
    c.stage_partition = numbers(j, j.contains("stage_partition") ? "stage_partition" : "stages_per_decomposition");
    c.factor_partition = numbers(j, "factor_partition");
    c.segment_mappings.clear(); c.execution_group_mappings.clear(); c.boundaries.clear();
    auto read_mappings = [&](const char* name, auto& out) {
        if (!j.contains(name)) return;
        for (const auto& m : j.at(name)) {
            const bool reg = c.backend == ButterflyBackend::OnlineReorder && c.fft_core == FftCore::RegisterTile;
            const auto default_core = reg ? (out.empty() || c.suffix_core == "native" ? "register-tile" : "cufftdx-block") : "";
            FftSegmentMapping mapping{parse_fft_core(field(m, "core", default_core)),
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
            for (std::size_t i = 0; i < mappings.size(); ++i) {
                const auto expected_core = i == 0 || c.suffix_core == "native" ? FftCore::RegisterTile : FftCore::CufftDxBlock;
                if (mappings[i].core != expected_core || mappings[i].codelet != (i == 0 ? c.prefix_codelet : "native"))
                    throw std::invalid_argument("register-tile physical mapping disagrees with prefix_codelet or suffix_core");
            }
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
                const auto default_core = c.fft_core == FftCore::RegisterTile && i != 0 && c.suffix_core != "native" ? FftCore::CufftDxBlock : c.fft_core;
                FftSegmentMapping mapping{cores.empty() ? default_core : parse_fft_core(cores[i]),
                    c.local_exchange, threads[i], ept[i]};
                if(c.fft_core == FftCore::RegisterTile && i == 0) mapping.codelet = c.prefix_codelet;
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
    if(c.backend == ButterflyBackend::OnlineReorder && c.fft_core == FftCore::RegisterTile) {
        for (const auto* mappings : {&c.segment_mappings, &c.execution_group_mappings}) {
            if(mappings->empty()) continue;
            if(mappings->size()!=2) throw std::invalid_argument("register-tile physical mapping requires two groups");
            for(std::size_t i=0;i<2;++i) {
                const auto expected = i==0 || c.suffix_core=="native" ? FftCore::RegisterTile : FftCore::CufftDxBlock;
                if(mappings->at(i).core!=expected || mappings->at(i).codelet!=(i==0 ? c.prefix_codelet : "native"))
                    throw std::invalid_argument("register-tile physical mapping disagrees with prefix_codelet or suffix_core");
            }
        }
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
        if (config.precision == ButterflyPrecision::Fp16 && !config.final_storage_rounding &&
            field(record.at("mapping"), "local_exchange") == "warp-register") continue;
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
    return portable_butterfly_candidates(original, nullptr);
}

std::vector<ButterflyConfig> portable_butterfly_candidates(const ButterflyConfig& original,
                                                         std::string* provider_audit_json) {
    Json provider_checks = Json::object();
    const bool provider_table_matches = cufftdx_provider_matches_runtime();
    auto provider_available = [&](unsigned sm, unsigned size, unsigned ept) {
        const auto arithmetic_bits = cufftdx_arithmetic_precision_bits(original);
        if (!arithmetic_bits) {
            if (provider_audit_json)
                provider_checks["arithmetic-contract"] = {
                    {"provider", "cufftdx"}, {"precision", butterfly_precision_name(original.precision)},
                    {"accumulation", butterfly_accumulation_name(original.accumulation)},
                    {"classification", "local-arithmetic-not-implemented"},
                    {"invalid_transform", false}, {"action", "exclude-provider-combination"}};
            return false;
        }
        const auto bits = *arithmetic_bits;
        const auto status = provider_table_matches ? cufftdx_provider_support_both(sm, bits, size, ept)
                                                  : CufftDxSupport::Unknown;
        if (provider_audit_json && status != CufftDxSupport::Supported) {
            const auto key = std::to_string(sm)+":"+std::to_string(bits)+":"+
                std::to_string(size)+":"+std::to_string(ept);
            provider_checks[key] = {{"provider", "cufftdx"}, {"sm", sm}, {"precision_bits", bits},
                {"local_size", size}, {"elements_per_thread", ept}, {"directions", {"forward", "inverse"}},
                {"classification", cufftdx_support_name(status)}, {"invalid_transform", false},
                {"action", status==CufftDxSupport::Unsupported ? "exclude-provider-combination" : "defer-to-compilation"}};
        }
        return status != CufftDxSupport::Unsupported;
    };
    auto finish_audit = [&]() {
        if (provider_audit_json) *provider_audit_json = Json{
            {"schema", "cubutterfly-provider-candidate-audit-v1"},
            {"provider_fingerprint", cufftdx_provider_fingerprint},
            {"provider_table_matches_runtime", provider_table_matches},
            {"scope", "dynamic-cuFFTDx-block-configurations; geometry and resources are checked separately"},
            {"checks", provider_checks}}.dump();
    };
    auto base = original;
    base.auto_select = false;
    base.stage_partition.clear(); base.segment_mappings.clear(); base.execution_group_mappings.clear(); base.boundaries.clear();
    base.local_stage_partitions.clear(); base.exchange_chunks.clear();
    base.factor_partition.clear(); base.data_tiles_per_cta=1; base.prefetch_depth=0;
    base.factor_slices=1; base.factor_overlap=false;
    base.stage_overlap = false; base.batch_tile_count = 1;
    base.register_threads = 0;
    base.fft_core = FftCore::Scalar; base.local_exchange = LocalExchange::SharedMemory;
    base.suffix_core = "cufftdx-block";
    base.shared_layout = SharedLayout::Linear; base.cross_twiddle = CrossTwiddleMode::Table;
    base.direct_boundary = DirectBoundary::Strided;
    std::vector<ButterflyConfig> candidates, specializations;
    if (base.log_n == 0 || base.log_n > 30) { finish_audit(); return candidates; }
    candidates=platform_candidates(base);
    if ((base.op == ButterflyOperator::Fwht || base.op == ButterflyOperator::Structured2x2) &&
        base.precision == ButterflyPrecision::Fp32 && base.log_n >= 3 && base.log_n <= 15) {
        const auto threads = base.op == ButterflyOperator::Fwht
            ? generated_register_fwht_thread_options(base.log_n)
            : generated_register_structured_thread_options(base.log_n);
        // Use the generated inventory, including reduced/custom installations.
        // Its legacy shape stays first for the unmeasured AUTO fallback.
        for (const auto count : threads) {
            auto c = base;
            c.backend = ButterflyBackend::TemporalTile; c.local_exchange = LocalExchange::WarpRegister;
            c.compute_unit = ComputeUnit::Radix2;
            c.register_threads = count; c.tile_threads = count;
            candidates.push_back(std::move(c));
        }
    }
    // Compiled short FP16 FWHT provider.  The semantic contract is narrower
    // than the generic low-precision inventory because it uses one warp
    // register exchange and one FP16 narrowing store.  Unsupported direction,
    // placement, or length requests retain the normal fallback candidates.
    if (short_fp16_fwht_supported(base)) {
        auto c = base;
        c.backend = ButterflyBackend::TemporalTile;
        c.accumulation = ButterflyAccumulation::Fp32;
        c.local_exchange = LocalExchange::WarpRegister;
        c.tile_threads = 128;
        // This exact compiled contract must be tried before the generic
        // portable priors.  Registration alone is insufficient: AUTO must
        // reach the public short kernel when the request matches it.
        candidates.insert(candidates.begin(), std::move(c));
    }
    if (base.op == ButterflyOperator::Fft) {
        if (base.precision == ButterflyPrecision::Fp16 &&
            base.accumulation == ButterflyAccumulation::Native && base.log_n <= 20 &&
            on_demand_compilation_enabled()) {
            const auto device = current_device_info();
            const auto sm = device.compute_major * 100 + device.compute_minor * 10;
            const auto hardware = query_hardware_resource_model();
            for (auto ept : {8U, 16U, 32U, 4U, 2U})
                for (auto threads : {128U, 256U, 64U, 32U, 512U, 1024U}) {
                    if (threads > hardware.max_threads_per_block) continue;
                    auto c = base;
                    c.backend = ButterflyBackend::TemporalTile;
                    c.fft_core = FftCore::CufftDxBlock;
                    c.compute_unit = ComputeUnit::Auto;
                    c.tile_threads = threads;
                    c.prefix_ept = ept;
                    try { fp16_local_block_shape(c); }
                    catch (const std::invalid_argument&) { continue; }
                    if (provider_available(sm, 1U << c.log_n, ept))
                        candidates.push_back(std::move(c));
                }
        }
        if (base.precision == ButterflyPrecision::Fp32) {
            for (const auto& m : register_tile_mappings()) {
                if (m.log_n != base.log_n) continue;
                auto add_register_mapping = [&](const auto& mapping) {
                    auto c = base;
                    apply_serialized_mapping(serialize_mapping(mapping), c);
                    candidates.push_back(std::move(c));
                };
                add_register_mapping(m);
                // Preserve each prebuilt provider's identity and support domain.
                // Only the native inventory seeds additional JIT strategies.
                if (!on_demand_compilation_enabled() || m.prefix_codelet != "native") continue;
                auto table_variant = m;
                table_variant.prefix_twiddle = "table";
                add_register_mapping(table_variant);
                for (const auto& codelet : {std::string("native"), std::string("cufftdx-thread"), std::string("radix4")})
                    for (const auto& layout : {std::string("linear"), std::string("xor")}) {
                        if (codelet == m.prefix_codelet && layout == m.prefix_shared_layout) continue;
                        // The linked radix-4 inventory already supplies this
                        // exact identity; all other radix-4 variants use JIT.
                        if (codelet == "radix4" && layout == "linear" && m.local_stages == 8) continue;
                        auto variant = m;
                        variant.prefix_codelet = codelet;
                        variant.prefix_shared_layout = layout;
                        for (auto* mappings : {&variant.segment_mappings, &variant.execution_group_mappings})
                            if (!mappings->empty()) mappings->front().codelet = codelet;
                        add_register_mapping(variant);
                    }
            }
        }
        if (on_demand_compilation_enabled() &&
            (base.precision == ButterflyPrecision::Fp32 || base.precision == ButterflyPrecision::Fp64 ||
             (base.precision == ButterflyPrecision::Fp16 && base.accumulation == ButterflyAccumulation::Native &&
              !(base.inverse && base.normalize_inverse)))) {
            const auto hardware = query_hardware_resource_model();
            const auto provider_sm = (hardware.compute_major*10+hardware.compute_minor)*10;
            const bool half = base.precision == ButterflyPrecision::Fp16;
            const auto width = half ? 4ULL : base.precision == ButterflyPrecision::Fp64 ? 16ULL : 8ULL;
            // Enumerate physical factor partitions independently of macro
            // grouping and per-CTA temporal data folding. No fixed group count.
            const auto* partition_budget=std::getenv("CUBUTTERFLY_PARTITION_BUDGET");
            if(partition_budget && *partition_budget=='-') throw std::invalid_argument("CUBUTTERFLY_PARTITION_BUDGET must be nonnegative");
            const auto budget=partition_budget ? std::stoull(partition_budget) : 128;
            const auto column_values = half ? std::vector<unsigned>{2U,4U,8U,16U,32U} : std::vector<unsigned>{1U,2U,4U,8U};
            for(unsigned columns:column_values) for(unsigned ept:{8U,16U})
                for(unsigned depth:{0U,1U,2U}) {
                    if(depth && hardware.compute_major<8) continue;
                    const auto physical_columns = columns / (half ? 2U : 1U);
                    unsigned min_local=0,max_local=0;
                    while((1U<<min_local)<std::max(ept,32*ept/physical_columns)) ++min_local;
                    // Keep the synchronous envelope for t=1.  A one-tile CTA
                    // has no successor to prefetch, so depth is lowered to
                    // zero for that schedule; the per-tile legality check
                    // below applies the buffered envelope to t>1.
                    while(max_local<14 && max_local<base.log_n &&
                          (1ULL<<(max_local+1))*columns*width<=hardware.shared_bytes_per_block &&
                          (1ULL<<(max_local+1))/ept*physical_columns<=hardware.max_threads_per_block) ++max_local;
                    for(const auto& partition:shared_partitions(base.log_n,max_local,budget,min_local,true))
                    for(unsigned tiles:{1U,4U,16U}) {
                        const auto effective_depth=tiles>1 ? depth : 0;
                        bool legal=!partition.empty(); unsigned done=0;
                        for(auto s:partition) {
                            if(s>14) { legal=false; break; }
                            const auto f=1U<<s;
                            const auto threads=f/ept*physical_columns;
                            if(f<ept || threads<32 || threads>hardware.max_threads_per_block ||
                               threads%32 || columns>(1ULL<<(base.log_n-s)) || (done && columns>(1ULL<<done)) ||
                               f*columns*width*(effective_depth ? effective_depth+2 : 1)>hardware.shared_bytes_per_block) legal=false;
                            done+=s;
                        }
                        if(!legal || (depth && hardware.compute_major<8)) continue;
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
                        bool provider_supported = true;
                        for (auto s : partition)
                            if (!provider_available(provider_sm, 1U<<s, ept)) provider_supported = false;
                        if (provider_supported) for (const auto& policies : io_variants) {
                            auto policy_candidate=c;
                            policy_candidate.factor_io_policies=policies;
                            add_factor_schedules(policy_candidate);
                        }
                        if(!half && std::all_of(partition.begin(),partition.end(),[&](unsigned s) { return !(s%2) && ept==(1U<<(s/2)); })) {
                            c.fft_core=FftCore::RegisterTile;
                            for (const auto& policies : io_variants) {
                                auto policy_candidate=c;
                                policy_candidate.factor_io_policies=policies;
                                add_factor_schedules(policy_candidate);
                            }
                        }
                    }
                }
            if (!half) for (unsigned local=2; local<=12; ++local) {
                if(base.log_n<=local || base.log_n-local<3 || base.log_n-local>14) continue;
                const auto suffix=base.log_n-local,pn=1U<<local,sn=1U<<suffix;
                // Preserve the historical priorities, then complete the local
                // provider's EPT domain. Unknown capability still reaches JIT.
                std::vector<unsigned> suffix_epts;
                for(unsigned ept : {16U,32U,8U,4U})
                    if(ept<=sn) suffix_epts.push_back(ept);
                for(unsigned ept=1;ept<=sn;ept*=2)
                    if(std::find(suffix_epts.begin(),suffix_epts.end(),ept)==suffix_epts.end())
                        suffix_epts.push_back(ept);
                suffix_epts.erase(std::remove_if(suffix_epts.begin(),suffix_epts.end(),[&](unsigned ept) {
                    return !provider_available(provider_sm,sn,ept);
                }),suffix_epts.end());
                std::vector<unsigned> factors;
                for(unsigned a=1;a<local;++a) factors.push_back(a);
                std::stable_sort(factors.begin(),factors.end(),[local](unsigned a,unsigned b) {
                    return std::abs(int(2*a)-int(local))<std::abs(int(2*b)-int(local));
                });
                for(unsigned log_a:factors) {
                    const auto log_b=local-log_a,a=1U<<log_a,b=1U<<log_b;
                    for(unsigned pc : {4U,2U,1U,8U,16U})
                    for(unsigned k=std::min(a,b);k;k/=2)
                    for(unsigned ept : suffix_epts) for(unsigned sc : {4U,2U,1U}) {
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
                            for(const auto& codelet:{std::string("native"),std::string("cufftdx-thread"),std::string("radix4")})
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
                            c.suffix_core="cufftdx-block";
                            if(log_a!=log_b) c.local_stage_partitions={{log_a,log_b},{}};
                            if(chunk) c.exchange_chunks={0,chunk};
                            c.prefix_units_per_cta=pc;
                            c.suffix_threads=(sn/ept)*sc; c.suffix_ept=ept;
                            c.shared_layout=SharedLayout::WriterAligned; c.cross_twiddle=CrossTwiddleMode::Recurrence;
                            c.reorder_columns=1;
                            specializations.push_back(c);
                            // A small orthogonal basis exposes twiddle supply
                            // for both precisions and rectangular factor shapes.
                            if (codelet == "native" && layout == "linear" && !chunk && k == std::min(a,b)) {
                                c.prefix_twiddle = "table";
                                specializations.push_back(c);
                            }
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
                        if (base.batch > 1 && base.batch <= UINT32_MAX) {
                            c.stage_overlap=true; c.batch_tile_count=static_cast<std::uint32_t>(base.batch);
                            candidates.push_back(c);
                        }
                    }
        }
    }
    const auto temporal_hardware = query_hardware_resource_model();
    const auto append_shared = [&](const ButterflyConfig& c) {
        candidates.push_back(c);
        if (!on_demand_compilation_enabled() || !c.boundaries.empty() ||
            c.direct_boundary != DirectBoundary::Strided ||
            resident_axes_requested(c.local_stage_partitions, c.exchange_chunks)) return;
        auto groups = c.stage_partition;
        if (groups.empty()) {
            for (auto remaining = c.log_n; remaining;) {
                const auto stages = std::min(remaining, c.local_stages);
                groups.push_back(stages); remaining -= stages;
            }
        }
        for (auto tiles : shared_tile_candidates(c.log_n, groups, online_reorder_value_bytes(c),
                                                 temporal_hardware.shared_bytes_per_block)) {
            if (tiles == 1) continue;
            auto grouped = c; grouped.data_tiles_per_cta = tiles;
            candidates.push_back(std::move(grouped));
        }
    };
    for (const auto threads : {128U, 256U, 64U, 32U}) {
        if (scalar_temporal_tile_supported(base, temporal_hardware.shared_bytes_per_block)) {
            auto c = base; c.backend = ButterflyBackend::TemporalTile; c.tile_threads = threads;
            for(auto unit : {ComputeUnit::Radix4, ComputeUnit::Radix2, ComputeUnit::Radix8}) {
                c.compute_unit=unit; candidates.push_back(c);
            }
        }
        // The larger single tile is another candidate, not a replacement for
        // the already available multistage organizations.
        if (base.log_n > 10) {
            for (std::uint32_t local = std::min(10U, base.log_n - 1); local >= 5; --local) {
                auto c = base; c.backend = ButterflyBackend::Hierarchical; c.local_stages = local;
                c.tile_threads = threads;
                for(auto unit : {ComputeUnit::Radix4, ComputeUnit::Radix2, ComputeUnit::Radix8}) {
                    c.compute_unit=unit; c.backend=ButterflyBackend::Hierarchical;
                    c.data_tiles_per_cta=1; c.reorder_columns=1; candidates.push_back(c);
                    c.backend=ButterflyBackend::OnlineReorder;
                    for (const auto& grouping : detail::online_reorder_group_candidates(base.log_n, local,
                            detail::online_reorder_value_bytes(base), temporal_hardware.shared_bytes_per_block)) {
                        c.data_tiles_per_cta=grouping.first; c.reorder_columns=grouping.second;
                        candidates.push_back(c);
                    }
                }
            }
        }
        for (std::uint32_t local = std::min(12U, base.log_n); local >= 1; --local) {
            auto c = base; c.backend = ButterflyBackend::SharedIterative; c.local_stages = local;
            c.tile_threads = threads; c.compute_unit = ComputeUnit::Radix2; c.shared_layout = SharedLayout::WriterAligned;
            append_shared(c);
            c.shared_layout=SharedLayout::Linear; append_shared(c);
        }
    }
    if(base.log_n==8) for(auto stages : {1U,2U,4U,8U}) {
        auto c=base; c.backend=ButterflyBackend::StagePipeline; c.compute_unit=ComputeUnit::Radix2;
        c.stage_space=stages; c.pipeline_warps=8; candidates.push_back(c);
    }
    const unsigned scalar_bytes=(base.precision==ButterflyPrecision::Fp64 || base.precision==ButterflyPrecision::Uint64) ? 8 :
        (base.precision==ButterflyPrecision::Fp16 || base.precision==ButterflyPrecision::Bf16 ? 2 : 4);
    const auto shared_hardware=query_hardware_resource_model();
    for(const auto& organization : shared_organization_candidates(
            capacity_partitions(base.log_n,scalar_bytes*(base.op==ButterflyOperator::Fft ? 2 : 1)),
            {{64U,128U,256U,32U,512U,1024U},shared_hardware.max_threads_per_block,
                on_demand_compilation_enabled(),4,true},base.batch)) {
        auto c=base; c.backend=ButterflyBackend::SharedIterative;
        c.stage_partition=organization.stage_partition; c.local_stages=0;
        c.compute_unit=ComputeUnit::Radix2; c.tile_threads=organization.threads;
        c.shared_layout=organization.writer_aligned ? SharedLayout::WriterAligned : SharedLayout::Linear;
        c.local_stage_partitions=organization.local_stage_partitions;
        c.batch_tile_count=organization.batch_tile_count;
        c.execution_group_mappings.clear();
        for(auto threads:organization.group_threads) {
            FftSegmentMapping mapping;
            mapping.core=FftCore::Scalar; mapping.ept=0; mapping.threads=threads;
            c.execution_group_mappings.push_back(mapping);
        }
        if (!organization.logical_stage_partition.empty()) {
            c.stage_partition=organization.logical_stage_partition;
            for (auto fused : organization.fused_edges) {
                FftBoundaryMapping boundary;
                if (fused) boundary.residency=FftBoundaryResidency::Fused;
                c.boundaries.push_back(boundary);
            }
        }
        append_shared(c);
    }
    candidates.insert(candidates.end(), specializations.begin(), specializations.end());
    const auto bulk_count=candidates.size();
    for (std::size_t i=0;i<bulk_count;++i) {
        const auto bulk=candidates[i];
        const bool shared=bulk.backend==ButterflyBackend::SharedIterative &&
            (bulk.stage_partition.empty() ? base.log_n>bulk.local_stages : bulk.stage_partition.size()>1);
        const bool register_fft=bulk.backend==ButterflyBackend::OnlineReorder && bulk.fft_core==FftCore::RegisterTile;
        if (!shared && !register_fft) continue;
        if (bulk.batch_tile_count != 1 && !bulk.stage_overlap) continue;
        const auto physical_groups=bulk.boundaries.empty() ?
            (bulk.stage_partition.empty() ? 2 : bulk.stage_partition.size()) :
            1+std::count_if(bulk.boundaries.begin(),bulk.boundaries.end(),[](const auto& boundary) {
                return boundary.residency!=FftBoundaryResidency::Fused;
            });
        for (auto tile : shared_overlap_tiles(base.batch,physical_groups)) {
            auto c=bulk; c.stage_overlap=true; c.batch_tile_count=tile;
            candidates.push_back(c);
        }
    }
    finish_audit();
    return candidates;
}

std::vector<PlanConfig> portable_ntt_candidates(const PlanConfig& original) {
    auto base = original; base.auto_select = false;
    base.local_stage_partitions.clear(); base.exchange_chunks.clear();
    base.stage_overlap=false; base.batch_tile_count=1;
    std::vector<PlanConfig> out;
    if(!base.log_n || base.log_n>30 || (base.word_bits!=32 && base.word_bits!=64)) return out;
    out=platform_candidates(base);
    if (!compact_stage_constraint(base)) {
        auto c=base; c.backend=Backend::CompactStage; c.compute_unit=ComputeUnit::Radix2;
        c.stage_partition.clear(); c.subgraph_mappings.clear(); c.boundary_mappings.clear();
        c.execution_group_mappings.clear(); out.push_back(c);
    }
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
    // Enumerate balanced, capacity-derived groups as well as small groups.
    // Static ranking orders this inventory; enumeration order is only a
    // deterministic fallback when no estimate is available.
    const auto shared_hardware=query_hardware_resource_model();
    for(const auto& organization : shared_organization_candidates(
            capacity_partitions(base.log_n,base.word_bits/8),
            {{128U,256U,64U,32U,512U,1024U},shared_hardware.max_threads_per_block,
                on_demand_compilation_enabled(),4,true},base.batch)) {
        auto c=base; c.backend=Backend::SharedIterative; c.stage_partition=organization.stage_partition;
        c.compute_unit=ComputeUnit::Radix2; c.threads_per_block=organization.threads;
        c.dataflow_layout=organization.writer_aligned ? DataflowLayout::HermesXor : DataflowLayout::Linear;
        c.local_stage_partitions=organization.local_stage_partitions;
        c.subgraph_mappings.clear(); c.boundary_mappings.clear(); c.execution_group_mappings.clear();
        c.batch_tile_count=organization.batch_tile_count;
        for(auto threads:organization.group_threads) {
            NttSubgraphMapping mapping;
            mapping.threads_per_block=threads;
            c.execution_group_mappings.push_back(mapping);
        }
        if (!organization.logical_stage_partition.empty()) {
            c.stage_partition=organization.logical_stage_partition;
            for (auto fused : organization.fused_edges) {
                NttBoundaryMapping boundary;
                if (fused) boundary.storage=BoundaryStorage::ResidentFused;
                c.boundary_mappings.push_back(boundary);
            }
        }
        out.push_back(std::move(c));
    }
    for (std::uint32_t local=1; local<=std::min(12U, base.log_n); ++local)
        for (auto threads : {32U,64U,128U,256U})
            for (auto layout : {DataflowLayout::HermesXor, DataflowLayout::Linear}) {
                auto c=base; c.backend=Backend::SharedIterative; c.compute_unit=ComputeUnit::Radix2;
                c.flow_tile_log_n=local; c.threads_per_block=threads; c.dataflow_layout=layout;
                c.stage_partition.clear(); c.subgraph_mappings.clear(); c.boundary_mappings.clear();
                c.execution_group_mappings.clear(); out.push_back(c);
            }
    const auto bulk_count=out.size();
    for (std::size_t i=0;i<bulk_count;++i) {
        const auto bulk=out[i];
        if (bulk.backend!=Backend::SharedIterative ||
            (bulk.stage_partition.empty() ? base.log_n<=bulk.flow_tile_log_n : bulk.stage_partition.size()<2)) continue;
        if (bulk.batch_tile_count != 1) continue;
        const auto physical_groups=bulk.boundary_mappings.empty() ?
            (bulk.stage_partition.empty() ? 2 : bulk.stage_partition.size()) :
            1+std::count_if(bulk.boundary_mappings.begin(),bulk.boundary_mappings.end(),[](const auto& boundary) {
                return boundary.storage!=BoundaryStorage::ResidentFused;
            });
        for (auto tile : shared_overlap_tiles(base.batch,physical_groups)) {
            auto c=bulk; c.stage_overlap=true; c.batch_tile_count=tile;
            out.push_back(c);
        }
    }
    // Existing specialized NTT cores are part of the executable template
    // inventory only through their explicit support-domain factory. MEASURE
    // can compare them, while
    // unsupported domains produce no candidate rather than a guessed mapping.
    for (auto candidate : ntt_template_candidates(base))
        out.push_back(std::move(candidate));
    return out;
}

namespace {
Json ranking_hardware(const HardwareResourceModel& h) {
    return {{"device",h.device_name},{"compute_capability",std::to_string(h.compute_major)+"."+std::to_string(h.compute_minor)},
        {"memory_bytes",h.memory_bytes},{"sm_count",h.sm_count},{"max_threads_per_block",h.max_threads_per_block},
        {"max_threads_per_sm",h.max_threads_per_sm},{"max_blocks_per_sm",h.max_blocks_per_sm},
        {"registers_per_sm",h.registers_per_sm},{"registers_per_block",h.registers_per_block},
        {"shared_bytes_per_sm",h.shared_bytes_per_sm},{"shared_bytes_per_block",h.shared_bytes_per_block},
        {"cooperative_launch",h.cooperative_launch}};
}
Json ranking_sample(const ButterflyConfig& c, const MixedDataflowPlan& plan) {
    auto sample=Json::parse(serialize_mapping(c));
    sample.update({{"operator",butterfly_operator_name(c.op)},{"precision",butterfly_precision_name(c.precision)},
        {"accumulation",butterfly_accumulation_name(c.accumulation)},{"placement",butterfly_placement_name(c.placement)},
        {"logN",c.log_n},{"batch",c.batch},{"direction",c.inverse ? "inverse":"forward"},
        {"normalization",c.normalize_inverse ? "inverse":"none"},
        {"element_stride",c.element_stride},{"batch_stride",c.batch_stride},
        {"mapping_json",serialize_mapping(c)},{"execution_groups_json",cubutterfly::execution_groups_json(plan)}});
    if (!c.stage_matrices.empty()) {
        // Match the benchmark/calibration semantic encoding exactly.
        std::ostringstream matrices;
        matrices << std::setprecision(17);
        for (std::size_t i=0;i<c.stage_matrices.size();++i) {
            if (i) matrices << 'x';
            const auto& m=c.stage_matrices[i];
            matrices << m.m00 << ':' << m.m01 << ':' << m.m10 << ':' << m.m11;
        }
        sample["stage_matrices"]=matrices.str();
    }
    return sample;
}
Json ranking_sample(const PlanConfig& c, const MixedDataflowPlan& plan) {
    auto sample=Json::parse(serialize_mapping(c));
    sample.update({{"operator","ntt"},{"precision","word"+std::to_string(c.word_bits)},
        {"logN",c.log_n},{"batch",c.batch},{"modulus",c.modulus},
        {"direction",c.inverse ? "inverse":"forward"},{"normalization",c.inverse ? "inverse":"none"},
        {"input_order",input_order_name(c.input_order)},{"output_order",output_order_name(c.output_order)},
        {"mapping_json",serialize_mapping(c)},{"execution_groups_json",cubutterfly::execution_groups_json(plan)}});
    if (c.backend != Backend::SharedIterative) {
        // Legacy logical groups do not necessarily describe physical launches:
        // copy/scale and cooperative multi-role kernels need their own contract.
        // Keep these cores executable and available to measured search without
        // assigning a falsely complete whole-plan cost to their logical IR.
        sample["physical_descriptor_status"]="unavailable";
        sample["physical_descriptor_reason"]=
            "legacy NTT logical groups lack a complete physical launch cost descriptor";
    }
    return sample;
}
Json resolved_ranking_sample(const ButterflyConfig& c, ButterflyPlan& plan) {
    auto sample=ranking_sample(c,plan.dataflow_plan());
    sample["physical_descriptor_status"]="resolved";
    return sample;
}
Json resolved_ranking_sample(const PlanConfig& c, Plan& plan) {
    auto physical=plan.dataflow_plan();
    auto sample=ranking_sample(c,physical);
    if (c.backend == Backend::SharedIterative) {
        sample["physical_descriptor_status"]="resolved";
    } else if (c.backend == Backend::CompactStage) {
        // Reuse the existing actual-launch adapter on this same constructed
        // plan. Reading it launches no kernels and performs no measurements.
        StageProbeAdapter adapter(plan);
        physical.execution_groups.clear();
        for (const auto& group:adapter.groups()) {
            if (!group.independent || !group.execution.compiler_resources_known)
                return sample;
            physical.execution_groups.push_back(group.execution);
        }
        if (!physical.execution_groups.empty()) {
            sample["execution_groups_json"]=cubutterfly::execution_groups_json(physical);
            sample["physical_descriptor_status"]="resolved";
            sample.erase("physical_descriptor_reason");
        }
    }
    return sample;
}
Json rank_samples(const Json& request) {
    const auto* override_script=std::getenv("CUBUTTERFLY_STATIC_RANKER");
    std::filesystem::path script=override_script ? override_script :
        std::string(CUBUTTERFLY_SOURCE_ROOT)+"/scripts/rank_static_candidates.py";
    if (!override_script && !std::filesystem::is_regular_file(script))
        script=CUBUTTERFLY_INSTALLED_RANKER;
    if (!std::filesystem::is_regular_file(script)) {
        if (override_script) throw std::invalid_argument("CUBUTTERFLY_STATIC_RANKER does not exist");
        auto result=Json{{"schema","cubutterfly-static-ranking-v1"},{"mode","unavailable"},
            {"reason","CPU ranking helper is not installed"},{"ranked",Json::array()},
            {"rejected",Json::array()},{"unranked",Json::array()}};
        for (const auto& item:request.at("candidates")) result["unranked"].push_back({{"index",item.at("index")}});
        return result;
    }
    const auto* python=std::getenv("CUBUTTERFLY_JIT_PYTHON");
    auto result=run_static_ranker(request,python ? python :
        (std::filesystem::exists(CUBUTTERFLY_JIT_PYTHON) ? CUBUTTERFLY_JIT_PYTHON : "python3"),script.string());
    std::set<std::size_t> indices;
    for (const auto& item:request.at("candidates")) indices.insert(item.at("index").get<std::size_t>());
    validate_static_ranking(result,indices);
    return result;
}

template<class Config, class PlanType, class Result>
Result select_static_inventory(std::vector<Config> candidates) {
    PlanSearchTrace trace("static-candidate-ranking");
    const auto hardware=query_hardware_resource_model();
    const auto source=hardware_resource_model_source();
    const std::string prefix="static-hardware-model:";
    Json request={{"schema","cubutterfly-static-ranking-request-v1"},{"hardware",ranking_hardware(hardware)},
        {"runtime_fingerprint",runtime_fingerprint()},{"profile_dir",nullptr},{"candidates",Json::array()}};
    if (source.rfind(prefix,0)==0)
        request["profile_dir"]=std::filesystem::path(source.substr(prefix.size())).parent_path().string();
    if (const auto* protocol=std::getenv("CUBUTTERFLY_MODEL_TIMING_PROTOCOL"))
        request["timing_protocol"]=Json::parse(protocol);
    std::string last_error;
    for (std::size_t i=0;i<candidates.size();++i) {
        try {
            auto plan=make_dataflow_plan(candidates[i]);
            if (!validate_plan(plan,hardware)) throw std::invalid_argument(plan.rejection_reason);
            auto sample=ranking_sample(candidates[i],plan);
            sample["descriptor_source"]="projected-plan";
            request["candidates"].push_back({{"index",i},{"sample",std::move(sample)}});
        } catch (const std::invalid_argument& e) {
            last_error=e.what();
            trace.event("projection-rejected",[&]{return Json{{"index",i},{"reason",last_error}};});
        }
    }
    const Json ranking=rank_samples(request);
    trace.event("projected-ranking",[&]{return ranking;});
    std::vector<std::size_t> order;
    for (const auto* category:{"ranked","unranked"})
        for (const auto& row:ranking.at(category)) order.push_back(row.at("index").template get<std::size_t>());
    std::size_t budget=4;
    if (const auto* value=std::getenv("CUBUTTERFLY_MODEL_FINALISTS")) {
        const std::string token=value;
        if (token.empty() || token.find_first_not_of("0123456789")!=std::string::npos)
            throw std::invalid_argument("CUBUTTERFLY_MODEL_FINALISTS must be a positive integer");
        budget=std::stoull(token);
        if (!budget) throw std::invalid_argument("CUBUTTERFLY_MODEL_FINALISTS must be positive");
    }
    if (ranking.at("ranked").empty()) budget=1;
    // Do not duplicate the entire projected inventory just to replace it with
    // a handful of finalists. Large inventories otherwise multiply RAM use.
    Json compiled=Json::object();
    for (const auto& item:request.items())
        if (item.key()!="candidates") compiled[item.key()]=item.value();
    std::vector<std::size_t> feasible;
    Json resolved;
    std::size_t cursor=0, probes=0, resource_replacements=0;
    while (cursor<order.size()) {
      compiled["candidates"]=Json::array();
      feasible.clear();
      while (cursor<order.size() && feasible.size()<budget) {
        const auto i=order[cursor++];
        ++probes;
        bool accepted=false;
        try {
            auto& c=candidates[i]; const auto allocate=c.auto_allocate_workspace;
            c.auto_allocate_workspace=false;
            PlanType probe(c);
            c=probe.config(); c.auto_allocate_workspace=allocate;
            auto sample=resolved_ranking_sample(c,probe);
            sample["descriptor_source"]="resolved-plan";
            compiled["candidates"].push_back({{"index",i},{"sample",std::move(sample)}});
            feasible.push_back(i);
            accepted=true;
        } catch (const std::bad_alloc&) { throw; }
          catch (const std::invalid_argument& e) { last_error=e.what(); }
          catch (const std::exception& e) {
            (void)cudaGetLastError();
            if (cudaDeviceSynchronize()!=cudaSuccess) throw;
            last_error=e.what();
        }
        trace.event("candidate-probe",[&]{return Json{{"index",i},{"accepted",accepted},{"last_error",accepted ? "" : last_error}};});
      }
      if (feasible.empty()) break;
      resolved=rank_samples(compiled);
      trace.event("resolved-ranking",[&]{return resolved;});
      if (!resolved.at("ranked").empty() || !resolved.at("unranked").empty()) break;
      // Resource-rejected finalists cannot be selected. Replace that batch
      // instead of incorrectly claiming that the remaining inventory is empty.
      resource_replacements+=feasible.size();
      feasible.clear();
    }
    if (feasible.empty()) throw std::invalid_argument("no executable mapping after static ranking: "+last_error);
    std::size_t chosen=feasible.front(); double predicted=0;
    std::string confidence="unmeasured-feasible", chosen_reason;
    if (!resolved.at("ranked").empty()) {
        const auto& first=resolved.at("ranked").front();
        chosen=first.at("index"); predicted=first.at("predicted_kernel_ms");
        confidence=first.value("confidence","unmeasured-static-model");
        chosen_reason=first.value("reason","");
    } else {
        if (resolved.at("unranked").empty())
            throw std::invalid_argument("compiled candidates rejected by the static resource model");
        chosen=resolved.at("unranked").front().at("index");
        chosen_reason=resolved.at("unranked").front().value("reason","");
    }
    std::ostringstream reason;
    reason << (predicted>0 ? "static cost ranking of executable finalists" : "unmeasured executable fallback; no local timing estimate")
        << "; mode=" << resolved.value("mode","unknown") << "; profile=" << request["profile_dir"].dump()
        << "; inventory=" << candidates.size() << "; projected=" << request["candidates"].size()
        << "; ranked=" << ranking.at("ranked").size() << "; unranked=" << ranking.at("unranked").size()
        << "; rejected=" << ranking.at("rejected").size() << "; finalists=" << feasible.size()
        << "; finalist_budget=" << budget << "; probes=" << probes
        << "; resource_replacements=" << resource_replacements << "; " << resolved.value("reason","")
        << "; selected_reason=" << chosen_reason
        << "; estimated_kernel_ms=" << predicted;
    const auto id=mapping_id(serialize_mapping(candidates[chosen]));
    trace.event("selected",[&]{return Json{{"index",chosen},{"reason",reason.str()},{"mapping",serialize_mapping(candidates[chosen])}};});
    return {std::move(candidates[chosen]),{true,false,hardware.device_name,id,confidence,reason.str(),predicted}};
}
} // namespace

ButterflySelectionResult select_portable_butterfly(const ButterflyConfig& config) {
    return select_static_inventory<ButterflyConfig,ButterflyPlan,ButterflySelectionResult>(portable_butterfly_candidates(config));
}
NttSelectionResult select_portable_ntt(const PlanConfig& config) {
    return select_static_inventory<PlanConfig,Plan,NttSelectionResult>(portable_ntt_candidates(config));
}

std::vector<std::string> search_disabled_mechanisms() {
    const char* value = std::getenv("CUBUTTERFLY_SEARCH_DISABLE");
    if (!value || !*value) return {};
    const std::set<std::string> allowed{
        "stage-overlap", "resident-fusion", "successor-layout"};
    std::vector<std::string> result;
    for (auto token : split(value, ',')) {
        while (!token.empty() && std::isspace(static_cast<unsigned char>(token.front()))) token.erase(token.begin());
        while (!token.empty() && std::isspace(static_cast<unsigned char>(token.back()))) token.pop_back();
        if (token.empty() || !allowed.count(token))
            throw std::invalid_argument("CUBUTTERFLY_SEARCH_DISABLE contains unknown mechanism: " + token);
        if (std::find(result.begin(), result.end(), token) == result.end()) result.push_back(std::move(token));
    }
    return result;
}

bool mapping_uses_disabled_mechanism(const std::string& mapping_json,
                                     const std::vector<std::string>& disabled) {
    if (disabled.empty()) return false;
    const auto mapping = mapping_object(mapping_json);
    const auto kind = field(mapping, "kind");
    for (const auto& mechanism : disabled) {
        if (mechanism == "stage-overlap" &&
            (boolean(mapping, "stage_overlap") || boolean(mapping, "factor_overlap")))
            return true;
        if (mechanism == "resident-fusion") {
            if (kind == "butterfly" && mapping.contains("boundaries")) {
                for (const auto& boundary : mapping.at("boundaries"))
                    if (field(boundary, "residency") == "fused") return true;
            }
            if (kind == "ntt" && mapping.contains("boundary_mappings")) {
                for (const auto& boundary : mapping.at("boundary_mappings"))
                    if (field(boundary, "storage") == "resident-fused") return true;
            }
        }
        if (mechanism == "successor-layout") {
            if (kind == "butterfly" &&
                (field(mapping, "shared_layout") == "writer-aligned" ||
                 field(mapping, "shared_layout") == "xor-swizzle" ||
                 field(mapping, "prefix_shared_layout") == "xor" ||
                 field(mapping, "prefix_shared_layout") == "xor-swizzle")) return true;
            if (kind == "ntt" && field(mapping, "dataflow_layout") == "hermes-xor") return true;
        }
    }
    return false;
}
} // namespace cuntt::detail
