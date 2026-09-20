#pragma once

#include <cuntt/butterfly.hpp>
#include <cuntt/ntt.hpp>
#include <string>
#include <sstream>
#include <iomanip>

namespace cuntt::detail {
const char* runtime_fingerprint() noexcept;
std::string serialize_mapping(const ButterflyConfig& config);
std::string serialize_mapping(const PlanConfig& config);
void apply_serialized_mapping(const std::string& json, ButterflyConfig& config);
void apply_serialized_mapping(const std::string& json, PlanConfig& config);
std::vector<ButterflyConfig> portable_butterfly_candidates(const ButterflyConfig& config);
std::vector<PlanConfig> portable_ntt_candidates(const PlanConfig& config);
}

namespace cubutterfly {
using cuntt::detail::runtime_fingerprint;
// A mapping contains execution choices, separate from workload semantics.
// Reapplying it preserves shape, direction, numeric contract and workspace policy.
using cuntt::detail::serialize_mapping;
using cuntt::detail::apply_serialized_mapping;
using cuntt::detail::portable_butterfly_candidates;
using cuntt::detail::portable_ntt_candidates;

inline std::string csv_field(const std::string& value) {
    std::string out = "\"";
    for (const auto ch : value) { if (ch == '"') out += '"'; out += ch; }
    return out + '"';
}

inline std::string execution_groups_json(const cuntt::MixedDataflowPlan& plan) {
    std::ostringstream json; json << '[';
    for(std::size_t i=0;i<plan.execution_groups.size();++i) {
        const auto& g=plan.execution_groups[i];
        json << (i ? "," : "") << "{\"first_stage\":" << g.first_stage << ",\"stage_count\":" << g.stage_count
             << ",\"stage_space\":" << g.stage_space << ",\"stage_time\":" << g.stage_time
             << ",\"data_space\":" << g.data_space << ",\"data_time\":" << g.data_time
             << ",\"batch_space\":" << g.batch_space << ",\"batch_time\":" << g.batch_time
             << ",\"threads\":" << g.threads << ",\"elements_per_thread\":" << g.elements_per_thread
             << ",\"core\":" << std::quoted(g.core) << ",\"shared_layout\":" << std::quoted(g.shared_layout)
             << ",\"codelet\":" << std::quoted(g.codelet) << ",\"io_policy\":" << std::quoted(g.io_policy)
             << ",\"exchange_chunk\":" << g.exchange_chunk << ",\"local_stage_partition\":[";
        for(std::size_t j=0;j<g.local_stage_partition.size();++j)
            json << (j ? "," : "") << g.local_stage_partition[j];
        json << ']'
             << ",\"live_shared_bytes\":" << g.live_shared_bytes << ",\"grid_ctas\":" << g.grid_ctas
             << ",\"compiler_registers_per_thread\":" << g.compiler_registers_per_thread
             << ",\"compiler_local_bytes_per_thread\":" << g.compiler_local_bytes_per_thread
             << ",\"compiler_local_resources_known\":" << (g.compiler_local_resources_known ? "true" : "false")
             << ",\"data_tiles_per_cta\":" << g.data_tiles_per_cta << ",\"prefetch_depth\":" << g.prefetch_depth
             << ",\"logical_macro_group\":" << g.logical_macro_group
             << ",\"launch_count\":" << g.launch_count
             << ",\"partial_dependency_ready\":" << (g.partial_dependency_ready ? "true" : "false")
             << ",\"compiler_resources_known\":" << (g.compiler_resources_known ? "true" : "false") << '}';
    }
    return json.str()+']';
}
}
