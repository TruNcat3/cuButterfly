#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuntt/ntt.hpp"

namespace {

std::string take_arg(int& index, int argc, char** argv) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + argv[index]);
    }
    return argv[++index];
}

std::vector<std::string> split_list(const std::string& value) {
    std::vector<std::string> result;
    std::size_t begin = 0;
    while (begin <= value.size()) {
        const std::size_t end = value.find(',', begin);
        result.push_back(value.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return result;
}

std::vector<std::uint32_t> parse_u32_list(const std::string& value) {
    std::vector<std::uint32_t> result;
    for (const auto& item : split_list(value)) result.push_back(static_cast<std::uint32_t>(std::stoul(item)));
    return result;
}

std::string join_u32(const std::vector<std::uint32_t>& values) {
    std::string result;
    for (const auto value : values) {
        if (!result.empty()) result += '+';
        result += std::to_string(value);
    }
    return result;
}

std::string join_subgraph_cores(const std::vector<cuntt::NttSubgraphMapping>& mappings) {
    std::string result;
    for (const auto& mapping : mappings) {
        if (!result.empty()) result += '+';
        result += cuntt::ntt_subgraph_core_name(mapping.core);
    }
    return result;
}

std::string join_mapping_field(const std::vector<cuntt::NttSubgraphMapping>& mappings,
                               std::uint32_t cuntt::NttSubgraphMapping::*field) {
    std::vector<std::uint32_t> values;
    for (const auto& mapping : mappings) values.push_back(mapping.*field);
    return join_u32(values);
}

std::string join_boundary_storage(const std::vector<cuntt::NttBoundaryMapping>& mappings) {
    std::string result;
    for (const auto& mapping : mappings) {
        if (!result.empty()) result += '+';
        result += cuntt::boundary_storage_name(mapping.storage);
    }
    return result;
}

std::string join_boundary_buffers(const std::vector<cuntt::NttBoundaryMapping>& mappings) {
    std::vector<std::uint32_t> values;
    for (const auto& mapping : mappings) values.push_back(mapping.buffers);
    return join_u32(values);
}

std::vector<std::uint32_t> execution_stage_partition(
    const cuntt::PlanConfig& config) {
    if (config.stage_partition.empty()) return {};
    std::vector<std::uint32_t> result;
    std::uint32_t stages = config.stage_partition.front();
    for (std::size_t boundary = 0; boundary < config.boundary_mappings.size();
         ++boundary) {
        if (config.boundary_mappings[boundary].storage ==
            cuntt::BoundaryStorage::ResidentFused) {
            stages += config.stage_partition[boundary + 1];
        } else {
            result.push_back(stages);
            stages = config.stage_partition[boundary + 1];
        }
    }
    result.push_back(stages);
    return result;
}

void print_usage() {
    std::cout << "cuntt_bench [--logN 16] [--batch 1] [--backend baseline|tile256|hybrid2d|compact-stage|stage-pipeline|hybrid-dataflow|hierarchical-barrier|hierarchical-dataflow]\n"
              << "            [--stage-space K]\n"
              << "            [--stage-handoff atomic|named-barrier]\n"
              << "            [--flow-tile-log 5|6|7|8] [--data-space 8|16|32] [--role-stages K]\n"
              << "            [--target-ctas-per-sm K]\n"
              << "            [--data-time K] [--token-interleave 1|2]\n"
              << "            [--pipeline-buffers 1|2|3]\n"
              << "            [--dataflow-layout hermes-xor|linear] [--dataflow-state inplace|ping-pong]\n"
              << "            [--n1-log 8] [--rows-per-block 4] [--threads-per-block 256]\n"
              << "            [--compute-unit auto|radix2|radix4|radix8]\n"
              << "            [--hierarchical-core dataflow-radix4|hybrid2d-radix4]\n"
              << "            [--stage-partition 8,8,4] [--segment-cores dataflow-radix4|homogeneous-radix4|homogeneous-warp-radix2|homogeneous-warp256-radix2|homogeneous-warp256-static-radix2|homogeneous-warp256-static-io-radix2|homogeneous-warp256-coefficient-reuse-static-io-radix2|homogeneous-warp128-static-io-radix2|homogeneous-warp128-coefficient-reuse-static-io-radix2|homogeneous-warp128-vector-radix4-static-io|homogeneous-warp128-vector-radix4-packed-stage6-static-io|homogeneous-warp128-vector-radix4-packed-stage6-distributed-static-io|homogeneous-warp128-packet-shared-radix4-static-io|homogeneous-warp64-static-io-radix2|homogeneous-warp128-pipeline-static-io-radix2|homogeneous-warp128-cooperative-static-io-radix2|appt-pipeline|appt-online|appt-online-radix4|appt-online-fused-tail|appt-online-split-tail|appt-online-register-tail|appt-online-register-tail-warp|appt-online-register-tail-column-warp|appt-online-register-tail-radix8|appt-online-register-tail-grouped|appt-online-register-tail-grouped-writer-final|appt-online-register-tail-grouped-writer-final-data-time|appt-online-register-tail-grouped-writer-final-resident|appt-online-register-tail-grouped-writer-final-resident-quarter]\n"
              << "            [--segment-threads 256] [--segment-units 8|16|32]\n"
              << "            [--segment-data-space K] [--segment-data-time K]\n"
              << "            [--segment-role-stages K] [--segment-token-interleave 1|2]\n"
              << "            [--segment-coefficient-reuse-stages 0|1|2|3|4|5|6|7]\n"
              << "            [--segment-cta-weights 7,7,6]\n"
              << "            [--execution-group-cores radix2|radix4|radix8]\n"
              << "            [--execution-group-threads K] [--execution-group-units K]\n"
              << "            [--execution-group-data-space K] [--execution-group-data-time K]\n"
              << "            [--execution-group-cta-weights K,...]\n"
              << "            [--appt-role-weights producer,tail,writer]\n"
              << "            [--appt-fragment-width 8|16|32] [--appt-writer-tiles 1|2|4]\n"
              << "            [--appt-data-time-roles MASK] (producer=1, tail=2, writer=4)\n"
              << "            [--boundary-storage full-scratch|ring|resident-fused] [--boundary-buffers 1] [--ready-window K]\n"
              << "            [--packet-readiness per-packet|wave-bitmap]\n"
              << "            [--packet-compute-layout interleaved-rows|warp-rows]\n"
              << "            [--packet-fold-wave-barriers]\n"
              << "            [--cross-twiddle first|second|fused|fused-barrett]\n"
              << "            [--mod-multiply shoup|barrett]\n"
              << "            [--word-bits 32|64]\n"
              << "            [--input-order natural|appt-static]\n"
              << "            [--output-order natural|bit-reversed|appt-static]\n"
              << "            [--auto-select] [--trace-pipeline] [--profile-appt-roles] [--warmup 5] [--repeat 20] [--modulus Q] [--inverse] [--verify] [--csv]\n";
}

std::uint32_t reverse_bits(std::uint32_t value, std::uint32_t bits) {
    std::uint32_t reversed = 0;
    for (std::uint32_t bit = 0; bit < bits; ++bit) {
        reversed = (reversed << 1) | ((value >> bit) & 1U);
    }
    return reversed;
}

bool verify_output(const std::vector<std::uint64_t>& input, const std::vector<std::uint64_t>& output, const cuntt::PlanConfig& config,
                   std::size_t& mismatch_index) {
    const std::size_t n = 1ULL << config.log_n;
    cuntt::ApptLayoutInfo layout;
    layout.log_n = config.log_n;
    layout.stage_partition = config.stage_partition;
    layout.fragment_width = config.appt_role_mapping.fragment_width;
    layout.xor_permutation = true;
    for (std::size_t batch_index = 0; batch_index < config.batch; ++batch_index) {
        std::vector<std::uint64_t> expected(n);
        for (std::size_t index = 0; index < n; ++index) {
            const std::size_t source = config.input_order == cuntt::InputOrder::ApptStatic
                                           ? cuntt::appt_static_index(index, layout)
                                           : index;
            expected[index] = input[batch_index * n + source];
        }
        cuntt::reference_ntt(expected, config.modulus, config.inverse);
        if (config.output_order == cuntt::OutputOrder::BitReversed) {
            std::vector<std::uint64_t> reordered(n);
            for (std::size_t index = 0; index < n; ++index) {
                const std::size_t reversed = reverse_bits(static_cast<std::uint32_t>(index), config.log_n);
                reordered[index]           = expected[reversed];
            }
            expected.swap(reordered);
        } else if (config.output_order == cuntt::OutputOrder::ApptStatic) {
            std::vector<std::uint64_t> reordered(n);
            for (std::size_t index = 0; index < n; ++index) {
                reordered[cuntt::appt_static_index(index, layout)] = expected[index];
            }
            expected.swap(reordered);
        }
        const auto output_begin = output.begin() + batch_index * n;
        const auto mismatch     = std::mismatch(expected.begin(), expected.end(), output_begin);
        if (mismatch.first != expected.end()) {
            mismatch_index = batch_index * n + static_cast<std::size_t>(mismatch.first - expected.begin());
            std::cerr << "expected=" << *mismatch.first
                      << ", actual=" << *(output_begin +
                          (mismatch.first - expected.begin())) << '\n';
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    cuntt::PlanConfig config;
    std::uint32_t     warmup = 5;
    std::uint32_t     repeat = 20;
    bool              verify = false;
    bool              csv    = false;
    bool              trace_pipeline = false;
    std::vector<cuntt::NttSubgraphCore> segment_cores;
    std::vector<std::uint32_t> segment_threads;
    std::vector<std::uint32_t> segment_units;
    std::vector<std::uint32_t> segment_data_space;
    std::vector<std::uint32_t> segment_data_time;
    std::vector<std::uint32_t> segment_role_stages;
    std::vector<std::uint32_t> segment_token_interleave;
    std::vector<std::uint32_t> segment_coefficient_reuse_stages;
    std::vector<std::uint32_t> segment_weights;
    std::vector<cuntt::NttSubgraphCore> execution_group_cores;
    std::vector<std::uint32_t> execution_group_threads;
    std::vector<std::uint32_t> execution_group_units;
    std::vector<std::uint32_t> execution_group_data_space;
    std::vector<std::uint32_t> execution_group_data_time;
    std::vector<std::uint32_t> execution_group_weights;
    std::vector<cuntt::BoundaryStorage> boundary_storage;
    std::vector<std::uint32_t> boundary_buffers;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_usage();
                return 0;
            }
            if (arg == "--logN") {
                config.log_n = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--batch") {
                config.batch = static_cast<std::size_t>(std::stoull(take_arg(i, argc, argv)));
            } else if (arg == "--backend") {
                config.backend = cuntt::parse_backend(take_arg(i, argc, argv));
            } else if (arg == "--stage-space") {
                config.stage_space = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--stage-handoff") {
                config.stage_handoff = cuntt::parse_stage_handoff(take_arg(i, argc, argv));
            } else if (arg == "--flow-tile-log") {
                config.flow_tile_log_n = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--data-space") {
                config.data_space = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--role-stages") {
                config.role_stages = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--target-ctas-per-sm") {
                config.target_ctas_per_sm = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--data-time") {
                config.data_time = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--token-interleave") {
                config.token_interleave = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--pipeline-buffers") {
                config.pipeline_buffers = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--dataflow-layout") {
                config.dataflow_layout = cuntt::parse_dataflow_layout(take_arg(i, argc, argv));
            } else if (arg == "--dataflow-state") {
                config.dataflow_state_mode = cuntt::parse_dataflow_state_mode(take_arg(i, argc, argv));
            } else if (arg == "--n1-log") {
                config.n1_log = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--rows-per-block") {
                config.rows_per_block = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--threads-per-block") {
                config.threads_per_block = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--compute-unit") {
                config.compute_unit = cuntt::parse_compute_unit(take_arg(i, argc, argv));
            } else if (arg == "--hierarchical-core") {
                config.hierarchical_core = cuntt::parse_hierarchical_core(take_arg(i, argc, argv));
            } else if (arg == "--stage-partition") {
                config.stage_partition = parse_u32_list(take_arg(i, argc, argv));
            } else if (arg == "--segment-cores") {
                for (const auto& item : split_list(take_arg(i, argc, argv)))
                    segment_cores.push_back(cuntt::parse_ntt_subgraph_core(item));
            } else if (arg == "--segment-threads") {
                segment_threads = parse_u32_list(take_arg(i, argc, argv));
            } else if (arg == "--segment-units") {
                segment_units = parse_u32_list(take_arg(i, argc, argv));
            } else if (arg == "--segment-data-space") {
                segment_data_space = parse_u32_list(take_arg(i, argc, argv));
            } else if (arg == "--segment-data-time") {
                segment_data_time = parse_u32_list(take_arg(i, argc, argv));
            } else if (arg == "--segment-role-stages") {
                segment_role_stages = parse_u32_list(take_arg(i, argc, argv));
            } else if (arg == "--segment-token-interleave") {
                segment_token_interleave = parse_u32_list(take_arg(i, argc, argv));
            } else if (arg == "--segment-coefficient-reuse-stages") {
                segment_coefficient_reuse_stages =
                    parse_u32_list(take_arg(i, argc, argv));
            } else if (arg == "--segment-cta-weights") {
                segment_weights = parse_u32_list(take_arg(i, argc, argv));
            } else if (arg == "--execution-group-cores") {
                for (const auto& item : split_list(take_arg(i, argc, argv)))
                    execution_group_cores.push_back(
                        cuntt::parse_ntt_subgraph_core(item));
            } else if (arg == "--execution-group-threads") {
                execution_group_threads = parse_u32_list(take_arg(i, argc, argv));
            } else if (arg == "--execution-group-units") {
                execution_group_units = parse_u32_list(take_arg(i, argc, argv));
            } else if (arg == "--execution-group-data-space") {
                execution_group_data_space = parse_u32_list(take_arg(i, argc, argv));
            } else if (arg == "--execution-group-data-time") {
                execution_group_data_time = parse_u32_list(take_arg(i, argc, argv));
            } else if (arg == "--execution-group-cta-weights") {
                execution_group_weights = parse_u32_list(take_arg(i, argc, argv));
            } else if (arg == "--appt-role-weights") {
                const auto weights = parse_u32_list(take_arg(i, argc, argv));
                if (weights.size() != 3) {
                    throw std::invalid_argument(
                        "--appt-role-weights requires producer,tail,writer");
                }
                config.appt_role_mapping.producer_weight = weights[0];
                config.appt_role_mapping.tail_weight = weights[1];
                config.appt_role_mapping.writer_weight = weights[2];
            } else if (arg == "--appt-fragment-width") {
                config.appt_role_mapping.fragment_width =
                    static_cast<std::uint32_t>(
                        std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--appt-writer-tiles") {
                config.appt_role_mapping.writer_tiles_per_cta =
                    static_cast<std::uint32_t>(
                        std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--appt-data-time-roles") {
                config.appt_role_mapping.data_time_role_mask =
                    static_cast<std::uint32_t>(
                        std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--boundary-storage") {
                for (const auto& item : split_list(take_arg(i, argc, argv)))
                    boundary_storage.push_back(cuntt::parse_boundary_storage(item));
            } else if (arg == "--boundary-buffers") {
                boundary_buffers = parse_u32_list(take_arg(i, argc, argv));
            } else if (arg == "--ready-window") {
                config.ready_window = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--packet-readiness") {
                config.packet_readiness_mode =
                    cuntt::parse_packet_readiness_mode(take_arg(i, argc, argv));
            } else if (arg == "--packet-compute-layout") {
                config.packet_compute_layout =
                    cuntt::parse_packet_compute_layout(take_arg(i, argc, argv));
            } else if (arg == "--packet-fold-wave-barriers") {
                config.packet_fold_wave_barriers = true;
            } else if (arg == "--cross-twiddle") {
                config.cross_twiddle_placement = cuntt::parse_cross_twiddle_placement(take_arg(i, argc, argv));
            } else if (arg == "--mod-multiply") {
                config.modular_multiply = cuntt::parse_modular_multiply(take_arg(i, argc, argv));
            } else if (arg == "--word-bits") {
                config.word_bits = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--input-order") {
                config.input_order = cuntt::parse_input_order(take_arg(i, argc, argv));
            } else if (arg == "--output-order") {
                config.output_order = cuntt::parse_output_order(take_arg(i, argc, argv));
            } else if (arg == "--warmup") {
                warmup = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--repeat") {
                repeat = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--modulus") {
                config.modulus = std::stoull(take_arg(i, argc, argv));
            } else if (arg == "--inverse") {
                config.inverse = true;
            } else if (arg == "--auto-select") {
                config.auto_select = true;
            } else if (arg == "--verify") {
                verify = true;
            } else if (arg == "--trace-pipeline") {
                trace_pipeline = true;
            } else if (arg == "--profile-appt-roles") {
                config.profile_appt_roles = true;
                trace_pipeline = true;
            } else if (arg == "--csv") {
                csv = true;
            } else {
                throw std::invalid_argument("unknown argument: " + arg);
            }
        }

        const auto expand_index = [](std::size_t size, std::size_t index, std::size_t expected) {
            if (size != 1 && size != expected) throw std::invalid_argument("mapping list must contain one value or one per descriptor");
            return size == 1 ? std::size_t{0} : index;
        };
        if (!segment_cores.empty() || !segment_threads.empty() ||
            !segment_units.empty() || !segment_data_space.empty() ||
            !segment_data_time.empty() || !segment_role_stages.empty() ||
            !segment_token_interleave.empty() ||
            !segment_coefficient_reuse_stages.empty() ||
            !segment_weights.empty()) {
            if (config.stage_partition.empty()) throw std::invalid_argument("segment mappings require --stage-partition");
            config.subgraph_mappings.resize(config.stage_partition.size());
            for (std::size_t index = 0; index < config.subgraph_mappings.size(); ++index) {
                auto& mapping = config.subgraph_mappings[index];
                // Preserve generated defaults for mapping dimensions that the
                // CLI did not override. A zero CTA weight means auto-map.
                mapping.core = cuntt::NttSubgraphCore::DataflowRadix4;
                mapping.threads_per_block = 256;
                mapping.units_per_cta = 0;
                mapping.data_space = 0;
                mapping.data_time = 0;
                mapping.role_stages = 0;
                mapping.token_interleave = 0;
                mapping.coefficient_reuse_stages = 0;
                mapping.cta_weight = 0;
                if (!segment_cores.empty()) mapping.core = segment_cores[expand_index(segment_cores.size(), index, config.subgraph_mappings.size())];
                if (!segment_threads.empty()) mapping.threads_per_block = segment_threads[expand_index(segment_threads.size(), index, config.subgraph_mappings.size())];
                if (!segment_units.empty()) mapping.units_per_cta = segment_units[expand_index(segment_units.size(), index, config.subgraph_mappings.size())];
                if (!segment_data_space.empty()) mapping.data_space = segment_data_space[expand_index(segment_data_space.size(), index, config.subgraph_mappings.size())];
                if (!segment_data_time.empty()) mapping.data_time = segment_data_time[expand_index(segment_data_time.size(), index, config.subgraph_mappings.size())];
                if (!segment_role_stages.empty()) mapping.role_stages = segment_role_stages[expand_index(segment_role_stages.size(), index, config.subgraph_mappings.size())];
                if (!segment_token_interleave.empty()) mapping.token_interleave = segment_token_interleave[expand_index(segment_token_interleave.size(), index, config.subgraph_mappings.size())];
                if (!segment_coefficient_reuse_stages.empty()) mapping.coefficient_reuse_stages = segment_coefficient_reuse_stages[expand_index(segment_coefficient_reuse_stages.size(), index, config.subgraph_mappings.size())];
                if (!segment_weights.empty()) mapping.cta_weight = segment_weights[expand_index(segment_weights.size(), index, config.subgraph_mappings.size())];
            }
        }
        if (!boundary_storage.empty() || !boundary_buffers.empty()) {
            if (config.stage_partition.size() < 2) throw std::invalid_argument("boundary mappings require --stage-partition");
            config.boundary_mappings.resize(config.stage_partition.size() - 1);
            for (std::size_t index = 0; index < config.boundary_mappings.size(); ++index) {
                auto& mapping = config.boundary_mappings[index];
                if (!boundary_storage.empty()) mapping.storage = boundary_storage[expand_index(boundary_storage.size(), index, config.boundary_mappings.size())];
                if (!boundary_buffers.empty()) mapping.buffers = boundary_buffers[expand_index(boundary_buffers.size(), index, config.boundary_mappings.size())];
            }
        }
        if (!execution_group_cores.empty() || !execution_group_threads.empty() ||
            !execution_group_units.empty() || !execution_group_data_space.empty() ||
            !execution_group_data_time.empty() || !execution_group_weights.empty()) {
            if (config.stage_partition.size() < 2 ||
                config.boundary_mappings.size() + 1 != config.stage_partition.size()) {
                throw std::invalid_argument(
                    "execution group mappings require explicit stage and boundary mappings");
            }
            const auto groups = execution_stage_partition(config);
            config.execution_group_mappings.resize(groups.size());
            for (std::size_t index = 0;
                 index < config.execution_group_mappings.size(); ++index) {
                auto& mapping = config.execution_group_mappings[index];
                mapping.core = cuntt::NttSubgraphCore::DataflowRadix4;
                mapping.threads_per_block = 256;
                mapping.units_per_cta = 1;
                mapping.data_space = 1;
                mapping.data_time = 1;
                mapping.role_stages = 1;
                mapping.token_interleave = 1;
                mapping.coefficient_reuse_stages = 0;
                mapping.cta_weight = groups[index];
                if (!execution_group_cores.empty())
                    mapping.core = execution_group_cores[expand_index(
                        execution_group_cores.size(), index, groups.size())];
                if (!execution_group_threads.empty())
                    mapping.threads_per_block = execution_group_threads[expand_index(
                        execution_group_threads.size(), index, groups.size())];
                if (!execution_group_units.empty())
                    mapping.units_per_cta = execution_group_units[expand_index(
                        execution_group_units.size(), index, groups.size())];
                if (!execution_group_data_space.empty())
                    mapping.data_space = execution_group_data_space[expand_index(
                        execution_group_data_space.size(), index, groups.size())];
                if (!execution_group_data_time.empty())
                    mapping.data_time = execution_group_data_time[expand_index(
                        execution_group_data_time.size(), index, groups.size())];
                if (!execution_group_weights.empty())
                    mapping.cta_weight = execution_group_weights[expand_index(
                        execution_group_weights.size(), index, groups.size())];
            }
        }

        const auto  device = cuntt::current_device_info();
        cuntt::Plan plan(config);
        config                       = plan.config();
        const auto execution_partition = execution_stage_partition(config);
        const auto& selection        = plan.selection();
        std::string appt_layout_id;
        if (!config.subgraph_mappings.empty() &&
            (config.subgraph_mappings.front().core ==
                 cuntt::NttSubgraphCore::ApptOnlineRegisterTail ||
             config.subgraph_mappings.front().core ==
                 cuntt::NttSubgraphCore::ApptOnlineRegisterTailWarp ||
             config.subgraph_mappings.front().core ==
                 cuntt::NttSubgraphCore::ApptOnlineRegisterTailColumnWarp ||
             config.subgraph_mappings.front().core ==
                 cuntt::NttSubgraphCore::ApptOnlineRegisterTailRadix8 ||
             config.subgraph_mappings.front().core ==
                 cuntt::NttSubgraphCore::ApptOnlineRegisterTailGrouped ||
             config.subgraph_mappings.front().core == cuntt::NttSubgraphCore::
                 ApptOnlineRegisterTailGroupedWriterFinal ||
             config.subgraph_mappings.front().core == cuntt::NttSubgraphCore::
                 ApptOnlineRegisterTailGroupedWriterFinalDataTime ||
             config.subgraph_mappings.front().core == cuntt::NttSubgraphCore::
                 ApptOnlineRegisterTailGroupedWriterFinalResident ||
             config.subgraph_mappings.front().core == cuntt::NttSubgraphCore::
                 ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter)) {
            appt_layout_id = plan.appt_layout_info().compatibility_id;
        }
        const std::size_t          n = plan.points_per_transform();
        std::mt19937_64            random(0x43554e5454ULL + config.log_n + config.batch);
        std::vector<std::uint64_t> input(n * config.batch);
        for (auto& value : input) {
            value = random() % config.modulus;
        }

        std::vector<std::uint64_t> output;
        const auto                 stats = plan.execute(input, output, warmup, repeat);
        const auto pipeline_trace = trace_pipeline ? plan.pipeline_trace()
                                                   : std::vector<cuntt::PipelineSegmentTrace>{};
        const auto role_metrics = config.profile_appt_roles
                                      ? plan.pipeline_role_metrics()
                                      : std::vector<cuntt::PipelineRoleMetrics>{};

        bool        correct        = true;
        std::size_t mismatch_index = 0;
        if (verify) {
            correct = verify_output(input, output, config, mismatch_index);
        }

        std::cout << std::fixed << std::setprecision(6);
        if (csv) {
            std::cout << "device,compute_capability,auto_select,selection_target,selected_implementation,selection_confidence,predicted_kernel_ms,selection_reason,backend,stage_space,stage_handoff,flow_tile_log,data_space,role_stages,target_ctas_per_sm,data_time,token_interleave,pipeline_buffers,dataflow_layout,dataflow_state,compute_unit,word_bits,logN,N,batch,modulus,modulus_bits,n1_"
                         "log,rows_per_block,threads_"
                         "per_block,hierarchical_core,stage_partition,logical_subgraphs,execution_stage_partition,execution_groups,materialized_boundaries,segment_cores,segment_threads,segment_units,segment_data_space,segment_data_time,segment_role_stages,segment_token_interleave,segment_coefficient_reuse_stages,segment_cta_weights,execution_group_cores,boundary_storage,boundary_buffers,ready_window,packet_readiness,packet_compute_layout,packet_fold_wave_barriers,inverse,warmup,repeat,h2d_ms,kernel_ms,timed_region_wall_ms,host_timing_overhead_ms,"
                         "appt_producer_weight,appt_tail_weight,appt_writer_weight,appt_fragment_width,appt_writer_tiles,appt_data_time_roles,input_order,output_order,appt_layout_id,d2h_ms,kernel_ntt_s,end_to_end_ntt_s,kernel_points_s,cross_twiddle,mod_multiply,correct\n";
            std::cout << '"' << device.name << "\"," << device.compute_major << '.' << device.compute_minor << ','
                      << static_cast<int>(selection.automatic) << ",\"" << selection.target << "\",\"" << selection.implementation
                      << "\",\"" << selection.confidence << "\"," << selection.predicted_kernel_ms << ",\"" << selection.reason << "\","
                      << cuntt::backend_name(config.backend) << ',' << config.stage_space << ',' << cuntt::stage_handoff_name(config.stage_handoff)
                      << ',' << config.flow_tile_log_n << ',' << config.data_space << ',' << config.role_stages << ','
                      << config.target_ctas_per_sm << ',' << config.data_time << ',' << config.token_interleave << ','
                      << config.pipeline_buffers << ','
                      << cuntt::dataflow_layout_name(config.dataflow_layout) << ','
                      << cuntt::dataflow_state_mode_name(config.dataflow_state_mode)
                      << ',' << cuntt::compute_unit_name(config.compute_unit) << ',' << config.word_bits << ',' << config.log_n << ',' << n << ','
                      << config.batch << ',' << config.modulus << ',' << (64U - static_cast<std::uint32_t>(__builtin_clzll(config.modulus))) << ','
                      << config.n1_log << ',' << config.rows_per_block << ',' << config.threads_per_block << ','
                      << cuntt::hierarchical_core_name(config.hierarchical_core) << ",\"" << join_u32(config.stage_partition)
                      << "\"," << config.stage_partition.size() << ",\""
                      << join_u32(execution_partition) << "\"," << execution_partition.size()
                      << ',' << (execution_partition.empty() ? 0 : execution_partition.size() - 1)
                      << ",\"" << join_subgraph_cores(config.subgraph_mappings) << "\",\""
                      << join_mapping_field(config.subgraph_mappings, &cuntt::NttSubgraphMapping::threads_per_block)
                      << "\",\"" << join_mapping_field(config.subgraph_mappings, &cuntt::NttSubgraphMapping::units_per_cta)
                      << "\",\"" << join_mapping_field(config.subgraph_mappings, &cuntt::NttSubgraphMapping::data_space)
                      << "\",\"" << join_mapping_field(config.subgraph_mappings, &cuntt::NttSubgraphMapping::data_time)
                      << "\",\"" << join_mapping_field(config.subgraph_mappings, &cuntt::NttSubgraphMapping::role_stages)
                      << "\",\"" << join_mapping_field(config.subgraph_mappings, &cuntt::NttSubgraphMapping::token_interleave)
                      << "\",\"" << join_mapping_field(config.subgraph_mappings, &cuntt::NttSubgraphMapping::coefficient_reuse_stages)
                      << "\",\"" << join_mapping_field(config.subgraph_mappings, &cuntt::NttSubgraphMapping::cta_weight)
                      << "\",\"" << join_subgraph_cores(config.execution_group_mappings)
                      << "\",\"" << join_boundary_storage(config.boundary_mappings) << "\",\""
                      << join_boundary_buffers(config.boundary_mappings) << "\"," << config.ready_window << ','
                      << cuntt::packet_readiness_mode_name(
                             config.packet_readiness_mode)
                      << ','
                      << cuntt::packet_compute_layout_name(
                             config.packet_compute_layout)
                      << ','
                      << static_cast<int>(config.packet_fold_wave_barriers)
                      << ','
                      << static_cast<int>(config.inverse)
                      << ',' << warmup << ',' << repeat << ',' << stats.h2d_ms << ',' << stats.kernel_ms << ','
                      << stats.timed_region_wall_ms << ',' << stats.host_timing_overhead_ms << ','
                      << config.appt_role_mapping.producer_weight << ','
                      << config.appt_role_mapping.tail_weight << ','
                      << config.appt_role_mapping.writer_weight << ','
                      << config.appt_role_mapping.fragment_width << ','
                      << config.appt_role_mapping.writer_tiles_per_cta << ','
                      << config.appt_role_mapping.data_time_role_mask << ','
                      << cuntt::input_order_name(config.input_order) << ','
                      << cuntt::output_order_name(config.output_order) << ",\""
                      << appt_layout_id << "\"," << stats.d2h_ms << ','
                      << stats.kernel_ntt_per_second << ',' << stats.end_to_end_ntt_per_second << ',' << stats.kernel_points_per_second << ','
                      << cuntt::cross_twiddle_placement_name(config.cross_twiddle_placement) << ','
                      << cuntt::modular_multiply_name(config.modular_multiply) << ','
                      << (verify ? static_cast<int>(correct) : -1) << '\n';
        } else {
            std::cout << "device: " << device.name << " (sm_" << device.compute_major << device.compute_minor << ")\n"
                      << "backend: " << cuntt::backend_name(config.backend) << "\n"
                      << "auto_select: " << (selection.automatic ? "yes" : "no") << "\n";
            if (selection.automatic) {
                std::cout << "selected_implementation: " << selection.implementation << "\n"
                          << "selection_target: " << selection.target << "\n"
                          << "selection_confidence: " << selection.confidence << "\n"
                          << "predicted_kernel_ms: " << selection.predicted_kernel_ms << "\n"
                          << "selection_reason: " << selection.reason << "\n";
            }
            if (config.backend == cuntt::Backend::HybridDataflow && !selection.implementation.empty()) {
                std::cout << "hybrid_dataflow_mapping_source: " << selection.implementation << "\n"
                          << "hybrid_dataflow_mapping_confidence: " << selection.confidence << "\n"
                          << "hybrid_dataflow_mapping_reason: " << selection.reason << "\n";
            }
            std::cout
                      << "stage_space: " << config.stage_space << "\n"
                      << "stage_handoff: " << cuntt::stage_handoff_name(config.stage_handoff) << "\n"
                      << "hybrid_dataflow: tile_log=" << config.flow_tile_log_n << ", data_space=" << config.data_space
                      << ", role_stages=" << config.role_stages
                      << ", target_ctas_per_sm=" << config.target_ctas_per_sm
                      << ", data_time=" << config.data_time
                      << ", token_interleave=" << config.token_interleave
                      << ", buffers=" << config.pipeline_buffers
                      << ", layout=" << cuntt::dataflow_layout_name(config.dataflow_layout)
                      << ", state=" << cuntt::dataflow_state_mode_name(config.dataflow_state_mode) << "\n"
                      << "compute_unit: " << cuntt::compute_unit_name(config.compute_unit) << "\n"
                      << "hierarchical_core: " << cuntt::hierarchical_core_name(config.hierarchical_core) << "\n"
                      << "stage_partition: " << join_u32(config.stage_partition) << "\n"
                      << "logical_subgraphs: " << config.stage_partition.size() << "\n"
                      << "execution_stage_partition: " << join_u32(execution_partition) << "\n"
                      << "execution_groups: " << execution_partition.size() << "\n"
                      << "materialized_boundaries: "
                      << (execution_partition.empty() ? 0 : execution_partition.size() - 1) << "\n"
                      << "ready_window: " << config.ready_window << "\n"
                      << "packet_readiness: "
                      << cuntt::packet_readiness_mode_name(
                             config.packet_readiness_mode)
                      << "\n"
                      << "packet_compute_layout: "
                      << cuntt::packet_compute_layout_name(
                             config.packet_compute_layout)
                      << "\n"
                      << "packet_fold_wave_barriers: "
                      << static_cast<int>(config.packet_fold_wave_barriers)
                      << "\n"
                      << "word_bits: " << config.word_bits << "\n"
                      << "shape: logN=" << config.log_n << ", N=" << n << ", batch=" << config.batch << "\n"
                      << "mapping: n1_log=" << config.n1_log << ", rows_per_block=" << config.rows_per_block
                      << ", threads_per_block=" << config.threads_per_block
                      << ", cross_twiddle=" << cuntt::cross_twiddle_placement_name(config.cross_twiddle_placement) << "\n"
                      << "mod_multiply: " << cuntt::modular_multiply_name(config.modular_multiply) << "\n"
                      << "input_order: " << cuntt::input_order_name(config.input_order) << "\n"
                      << "output_order: " << cuntt::output_order_name(config.output_order) << "\n"
                      << "appt_layout_id: " << appt_layout_id << "\n"
                      << "appt_roles: producer=" << config.appt_role_mapping.producer_weight
                      << ", tail=" << config.appt_role_mapping.tail_weight
                      << ", writer=" << config.appt_role_mapping.writer_weight
                      << ", fragment_width=" << config.appt_role_mapping.fragment_width
                      << ", writer_tiles=" << config.appt_role_mapping.writer_tiles_per_cta
                      << ", data_time_roles=" << config.appt_role_mapping.data_time_role_mask << "\n"
                      << "direction: " << (config.inverse ? "inverse" : "forward") << "\n"
                      << "h2d_ms: " << stats.h2d_ms << "\n"
                      << "kernel_ms: " << stats.kernel_ms << "\n"
                      << "timed_region_wall_ms: " << stats.timed_region_wall_ms << "\n"
                      << "host_timing_overhead_ms: " << stats.host_timing_overhead_ms << "\n"
                      << "d2h_ms: " << stats.d2h_ms << "\n"
                      << "kernel_ntt_s: " << stats.kernel_ntt_per_second << "\n"
                      << "end_to_end_ntt_s: " << stats.end_to_end_ntt_per_second << "\n"
                      << "kernel_points_s: " << stats.kernel_points_per_second << "\n"
                      << "correct: " << (verify ? (correct ? "yes" : "no") : "not checked") << "\n";
            for (std::size_t segment = 0; segment < pipeline_trace.size(); ++segment) {
                std::cout << "pipeline_segment_" << segment << ": start="
                          << pipeline_trace[segment].start_globaltimer << ", end="
                          << pipeline_trace[segment].end_globaltimer << "\n";
            }
            for (std::size_t role = 0; role < role_metrics.size(); ++role) {
                const auto& metric = role_metrics[role];
                std::cout << "pipeline_role_" << role
                          << ": wait=" << metric.wait_globaltimer
                          << ", compute=" << metric.compute_globaltimer
                          << ", boundary=" << metric.boundary_globaltimer
                          << ", tasks=" << metric.tasks << "\n";
            }
        }

        if (verify && !correct) {
            std::cerr << "verification mismatch at flat index " << mismatch_index << '\n';
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        print_usage();
        return 1;
    }
    return 0;
}
