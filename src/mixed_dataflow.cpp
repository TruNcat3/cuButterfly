#include "cuntt/mixed_dataflow.hpp"
#include "cuntt/butterfly.hpp"
#include "fft_factor_plan.hpp"
#include "resident_mapping.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace cuntt {

namespace {

DataflowOperator to_dataflow_operator(ButterflyOperator op) {
    switch (op) {
        case ButterflyOperator::Fft: return DataflowOperator::Fft;
        case ButterflyOperator::Fwht: return DataflowOperator::Fwht;
        case ButterflyOperator::SubsetZeta: return DataflowOperator::SubsetZeta;
        case ButterflyOperator::SupersetZeta: return DataflowOperator::SupersetZeta;
        case ButterflyOperator::Structured2x2: return DataflowOperator::Structured2x2;
        case ButterflyOperator::XorZeta: return DataflowOperator::XorZeta;
    }
    throw std::invalid_argument("unknown butterfly operator");
}

BoundaryPolicy to_boundary_policy(const ButterflyConfig& config) {
    if (config.boundaries.empty()) {
        if (config.backend == ButterflyBackend::TemporalTile && config.local_exchange == LocalExchange::WarpRegister)
            return BoundaryPolicy::Register;
        return config.backend == ButterflyBackend::Hierarchical || config.backend == ButterflyBackend::OnlineReorder
                   ? BoundaryPolicy::GlobalScratch : BoundaryPolicy::None;
    }
    switch (config.boundaries.front().residency) {
        case FftBoundaryResidency::Fused: return BoundaryPolicy::ResidentFused;
        case FftBoundaryResidency::GlobalScratch: return BoundaryPolicy::GlobalScratch;
    }
    return BoundaryPolicy::GlobalScratch;
}

BoundaryPolicy to_boundary_policy(const NttBoundaryMapping& boundary) {
    switch (boundary.storage) {
        case BoundaryStorage::FullScratch: return BoundaryPolicy::GlobalScratch;
        case BoundaryStorage::Ring: return BoundaryPolicy::RingBuffer;
        case BoundaryStorage::ResidentFused: return BoundaryPolicy::ResidentFused;
    }
    return BoundaryPolicy::GlobalScratch;
}

ExchangePolicy to_exchange_policy(const ButterflyConfig& config) {
    return config.local_exchange == LocalExchange::WarpRegister ? ExchangePolicy::WarpShuffle
                                                                  : ExchangePolicy::SharedMemory;
}

std::uint32_t generic_data_space(std::uint64_t butterflies, std::uint32_t threads) {
    const auto active = std::max<std::uint64_t>(1, std::min<std::uint64_t>(butterflies, threads));
    return static_cast<std::uint32_t>(active);
}

void set_generic_group_shape(ExecutionGroup& group, std::uint32_t threads, std::uint64_t grid_ctas,
                             std::uint64_t live_shared_bytes, ExchangePolicy exchange,
                             std::uint64_t butterflies_per_cta) {
    group.core = "scalar";
    group.threads = threads;
    group.grid_ctas = grid_ctas;
    group.live_shared_bytes = live_shared_bytes;
    group.exchange = exchange;
    group.stage_space = 1;
    group.elements_per_thread = 0;
    group.data_space = generic_data_space(butterflies_per_cta, threads);
    group.data_time = static_cast<std::uint32_t>((butterflies_per_cta + group.data_space - 1) / group.data_space);
}

void project_resident_axes(ExecutionGroup& group,const detail::LocalPartitions& parts,
                          const std::vector<std::uint32_t>& chunks,std::size_t index,bool shared) {
    if(!parts.empty()) group.local_stage_partition=parts.at(index);
    if(!chunks.empty()) group.exchange_chunk=chunks.at(index);
    if(shared && !group.local_stage_partition.empty()) {
        group.codelet="native-register-subgraph";
        group.elements_per_thread=1U<<*std::max_element(group.local_stage_partition.begin(),group.local_stage_partition.end());
    }
}

void project_generic_butterfly_groups(MixedDataflowPlan& plan, const ButterflyConfig& config) {
    // Non-FFT operators and scalar FFTs are executed by the typed generic
    // kernels.  Their launch shapes are different from imported FFT units,
    // so do not let an unused fft_core or config mapping leak into the IR.
    const bool generic = config.op != ButterflyOperator::Fft || config.fft_core == FftCore::Scalar;
    if (!generic || (config.backend != ButterflyBackend::Hierarchical && config.backend != ButterflyBackend::OnlineReorder) ||
        (plan.dispatch != DataflowDispatch::Hierarchical && plan.dispatch != DataflowDispatch::OnlineReorder))
        return;

    const auto value_bytes = (plan.graph.word_bits / 8) * (plan.graph.complex_values ? 2ULL : 1ULL);
    const auto local_stages = plan.stage_partition.front();
    const auto local_points = std::uint64_t{1} << local_stages;
    const auto prefix_grid = static_cast<std::uint64_t>(plan.graph.batch) << (plan.graph.log_n - local_stages);
    const auto prefix_butterflies = local_points / 2;
    const auto prefix_shared = local_points * value_bytes;

    if (plan.dispatch == DataflowDispatch::Hierarchical) {
        const auto total_butterflies = static_cast<std::uint64_t>(plan.graph.batch) << (plan.graph.log_n - 1);
        const auto suffix_grid = (total_butterflies + 256 - 1) / 256;
        const auto suffix_butterflies = std::min<std::uint64_t>(total_butterflies, 256);
        for (std::size_t index = 0; index < plan.execution_groups.size(); ++index) {
            auto& group = plan.execution_groups[index];
            if (index == 0) {
                set_generic_group_shape(group, config.tile_threads, prefix_grid, prefix_shared,
                                         ExchangePolicy::SharedMemory, prefix_butterflies);
            } else {
                // hierarchical_stage_kernel is a one-stage, global-memory
                // radix-2 kernel launched with a fixed 256-thread block.
                set_generic_group_shape(group, 256, suffix_grid, 0, ExchangePolicy::GlobalMemory, suffix_butterflies);
            }
        }
        return;
    }

    if (plan.stage_partition.size() != 2 || plan.execution_groups.size() != 2)
        return;
    const auto remaining_stages = plan.stage_partition.back();
    const auto remaining_points = std::uint64_t{1} << remaining_stages;
    const auto columns = std::max(1U, config.reorder_columns);
    const auto column_tiles = (local_points + columns - 1) / columns;
    const auto suffix_grid = static_cast<std::uint64_t>(plan.graph.batch) * column_tiles;
    const auto suffix_butterflies = static_cast<std::uint64_t>(columns) * (remaining_points / 2);
    set_generic_group_shape(plan.execution_groups[0], config.tile_threads, prefix_grid, prefix_shared,
                            ExchangePolicy::SharedMemory, prefix_butterflies);
    set_generic_group_shape(plan.execution_groups[1], config.tile_threads,
                            suffix_grid, remaining_points * columns * value_bytes,
                            ExchangePolicy::SharedMemory, suffix_butterflies);
}

void project_execution_groups(MixedDataflowPlan& plan) {
    const auto value_bytes = (plan.graph.word_bits / 8) * (plan.graph.complex_values ? 2ULL : 1ULL);
    std::uint32_t first = 0, count = 0;
    for (std::size_t edge = 0; edge < plan.stage_partition.size(); ++edge) {
        count += plan.stage_partition[edge];
        if (edge + 1 < plan.stage_partition.size() && plan.boundary_policies[edge] == BoundaryPolicy::ResidentFused)
            continue;
        ExecutionGroup group;
        group.first_stage = first; group.stage_count = count;
        group.stage_time = count;
        group.threads = plan.threads_per_block;
        group.core = plan.processing_unit; group.exchange = plan.exchange;
        group.batch_space = plan.batch_tile_count;
        group.batch_time = static_cast<std::uint32_t>((plan.graph.batch + group.batch_space - 1) / group.batch_space);
        group.grid_ctas = static_cast<std::uint64_t>(group.batch_time) << (plan.graph.log_n - count);
        if (plan.dispatch == DataflowDispatch::BatchPipeline)
            group.grid_ctas = std::min<std::uint64_t>(plan.graph.batch, plan.batch_tile_count) << (plan.graph.log_n - count);
        plan.execution_groups.push_back(group);
        if (edge + 1 < plan.stage_partition.size()) {
            ExecutionBoundary boundary;
            boundary.producer_group = plan.execution_groups.size() - 1;
            boundary.consumer_group = boundary.producer_group + 1;
            boundary.storage = plan.boundary_policies[edge];
            boundary.bytes = (1ULL << plan.graph.log_n) * plan.graph.batch * value_bytes;
            if (plan.dispatch == DataflowDispatch::BatchPipeline)
                boundary.bytes = (plan.graph.batch_stride ? plan.graph.batch_stride : (1ULL << plan.graph.log_n)) *
                    plan.batch_tile_count * value_bytes;
            boundary.buffers = plan.steady_state_buffers;
            plan.execution_boundaries.push_back(boundary);
        }
        first += count; count = 0;
    }
    plan.state = CandidateState::NeedsLowering;
}

}  // namespace

HardwareResourceModel query_hardware_resource_model() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) throw std::runtime_error("unable to query the active CUDA device");
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess)
        throw std::runtime_error("unable to query CUDA device properties");
    HardwareResourceModel model;
    model.device_name = properties.name;
    model.compute_major = properties.major;
    model.compute_minor = properties.minor;
    model.sm_count = static_cast<std::uint32_t>(properties.multiProcessorCount);
    model.max_threads_per_block = static_cast<std::uint32_t>(properties.maxThreadsPerBlock);
    model.max_threads_per_sm = static_cast<std::uint32_t>(properties.maxThreadsPerMultiProcessor);
    model.registers_per_sm = static_cast<std::uint32_t>(properties.regsPerMultiprocessor);
    model.shared_bytes_per_sm = static_cast<std::uint32_t>(properties.sharedMemPerMultiprocessor);
    model.memory_bytes = properties.totalGlobalMem;
    model.max_blocks_per_sm = static_cast<std::uint32_t>(properties.maxBlocksPerMultiProcessor);
    model.shared_bytes_per_block = static_cast<std::uint32_t>(properties.sharedMemPerBlockOptin);
    model.registers_per_block = static_cast<std::uint32_t>(properties.regsPerBlock);
    int cooperative = 0;
    if (cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device) == cudaSuccess)
        model.cooperative_launch = cooperative != 0;
    return model;
}

const char* dataflow_operator_name(DataflowOperator op) noexcept {
    switch (op) {
        case DataflowOperator::Fft: return "fft";
        case DataflowOperator::Ntt: return "ntt";
        case DataflowOperator::Fwht: return "fwht";
        case DataflowOperator::SubsetZeta: return "subset-zeta";
        case DataflowOperator::SupersetZeta: return "superset-zeta";
        case DataflowOperator::XorZeta: return "xor-zeta";
        case DataflowOperator::Structured2x2: return "structured-2x2";
    }
    return "unknown";
}

DataflowOperator parse_dataflow_operator(const std::string& name) {
    for (const auto op : {DataflowOperator::Fft, DataflowOperator::Ntt, DataflowOperator::Fwht,
                          DataflowOperator::SubsetZeta, DataflowOperator::SupersetZeta,
                          DataflowOperator::XorZeta, DataflowOperator::Structured2x2}) {
        if (name == dataflow_operator_name(op)) return op;
    }
    throw std::invalid_argument("unknown dataflow operator: " + name);
}

const char* boundary_policy_name(BoundaryPolicy policy) noexcept {
    switch (policy) {
        case BoundaryPolicy::None: return "none";
        case BoundaryPolicy::Register: return "register";
        case BoundaryPolicy::SharedResident: return "shared-resident";
        case BoundaryPolicy::ResidentFused: return "resident-fused";
        case BoundaryPolicy::GlobalScratch: return "global-scratch";
        case BoundaryPolicy::RingBuffer: return "ring-buffer";
        case BoundaryPolicy::CooperativeGrid: return "cooperative-grid";
    }
    return "unknown";
}

const char* exchange_policy_name(ExchangePolicy policy) noexcept {
    switch (policy) {
        case ExchangePolicy::Register: return "register";
        case ExchangePolicy::WarpShuffle: return "warp-shuffle";
        case ExchangePolicy::SharedMemory: return "shared-memory";
        case ExchangePolicy::GlobalMemory: return "global-memory";
        case ExchangePolicy::CooperativeGrid: return "cooperative-grid";
    }
    return "unknown";
}

const char* lowering_status_name(LoweringStatus status) noexcept {
    switch (status) {
        case LoweringStatus::Supported: return "supported";
        case LoweringStatus::UnsupportedOperator: return "unsupported-operator";
        case LoweringStatus::UnsupportedSize: return "unsupported-size";
        case LoweringStatus::UnsupportedBoundary: return "unsupported-boundary";
        case LoweringStatus::UnsupportedExchange: return "unsupported-exchange";
        case LoweringStatus::MissingBackend: return "missing-backend";
        case LoweringStatus::RequiresBackendValidation: return "requires-backend-validation";
    }
    return "unknown";
}

const char* dataflow_dispatch_name(DataflowDispatch dispatch) noexcept {
    switch (dispatch) {
        case DataflowDispatch::SharedIterative: return "shared-iterative";
        case DataflowDispatch::BatchPipeline: return "batch-pipeline";
        case DataflowDispatch::TemporalTile: return "temporal-tile";
        case DataflowDispatch::WarpRegister: return "warp-register";
        case DataflowDispatch::Hierarchical: return "hierarchical";
        case DataflowDispatch::OnlineReorder: return "online-reorder";
        case DataflowDispatch::WarpHybrid: return "warp-hybrid";
        case DataflowDispatch::StagePipeline: return "stage-pipeline";
        case DataflowDispatch::ExternalFft: return "external-fft";
        case DataflowDispatch::Unresolved: return "unresolved";
    }
    return "unresolved";
}

OperatorGraph make_operator_graph(DataflowOperator op, std::uint32_t log_n, std::size_t batch) {
    if (log_n == 0 || log_n >= 31) throw std::invalid_argument("log_n must be in [1, 30]");
    if (batch == 0) throw std::invalid_argument("batch must be positive");
    OperatorGraph graph;
    graph.op = op;
    graph.log_n = log_n;
    graph.stage_count = log_n;
    graph.batch = batch;
    graph.stage_arities.assign(log_n, 2);
    graph.complex_values = op == DataflowOperator::Fft;
    graph.coefficient_stages.assign(log_n, op == DataflowOperator::Fft || op == DataflowOperator::Ntt ||
                                                   op == DataflowOperator::Structured2x2 ? 1U : 0U);
    return graph;
}

OperatorGraph make_operator_graph(const ButterflyConfig& config) {
    auto graph = make_operator_graph(to_dataflow_operator(config.op), config.log_n, config.batch);
    graph.word_bits = config.precision == ButterflyPrecision::Fp64 ? 64U :
                      config.precision == ButterflyPrecision::Fp16 || config.precision == ButterflyPrecision::Bf16 ? 16U : 32U;
    graph.precision = butterfly_precision_name(config.precision);
    graph.element_stride = config.element_stride;
    graph.batch_stride = config.batch_stride;
    graph.accumulator_bits = config.accumulation == ButterflyAccumulation::Fp32 ? 32U : graph.word_bits;
    graph.inverse = config.inverse;
    graph.in_place = config.placement == ButterflyPlacement::InPlace;
    graph.normalized_inverse = config.normalize_inverse;
    // Processing radix fuses binary graph stages; it does not change the graph.
    return graph;
}

OperatorGraph make_operator_graph(const PlanConfig& config) {
    auto graph = make_operator_graph(DataflowOperator::Ntt, config.log_n, config.batch);
    graph.word_bits = config.word_bits;
    graph.accumulator_bits = config.word_bits;
    graph.inverse = config.inverse;
    graph.in_place = false;
    graph.normalized_inverse = false;
    graph.precision = std::string("word") + std::to_string(config.word_bits);
    graph.stage_arities.assign(config.log_n, 2U);
    return graph;
}

MixedDataflowPlan make_dataflow_plan(const ButterflyConfig& config) {
    if(config.backend==ButterflyBackend::FactorStreamed) return detail::make_factor_dataflow_plan(config);
    MixedDataflowPlan plan;
    plan.graph = make_operator_graph(config);
    plan.stage_partition = config.stage_partition;
    if (plan.stage_partition.empty()) plan.stage_partition = {config.log_n};
    if (config.backend == ButterflyBackend::OnlineReorder && config.stage_partition.empty()) {
        if (config.local_stages == 0 || config.local_stages >= config.log_n)
            throw std::invalid_argument("online-reorder requires a proper local stage partition");
        plan.stage_partition = {config.local_stages, config.log_n - config.local_stages};
    }
    if (config.backend == ButterflyBackend::Hierarchical) {
        if (config.local_stages == 0 || config.local_stages >= config.log_n)
            throw std::invalid_argument("hierarchical requires a proper local stage partition");
        plan.stage_partition = {config.local_stages};
        plan.stage_partition.insert(plan.stage_partition.end(), config.log_n - config.local_stages, 1U);
    }
    std::uint32_t stage_sum = 0;
    for (const auto stages : plan.stage_partition) {
        if (stages == 0 || stages > config.log_n - stage_sum)
            throw std::invalid_argument("stage partition must contain positive stages summing to log_n");
        stage_sum += stages;
    }
    if (stage_sum != config.log_n)
        throw std::invalid_argument("stage partition must cover log_n");
    plan.subgraph_log_n = plan.stage_partition.front();
    // These legacy backends do not expose data-time or a residency target.
    plan.target_resident_ctas = 0;
    plan.boundary = to_boundary_policy(config);
    plan.exchange = to_exchange_policy(config);
    plan.threads_per_block = config.tile_threads;
    plan.processing_radix = config.compute_unit == ComputeUnit::Radix4 ? 4U :
                            config.compute_unit == ComputeUnit::Radix8 ? 8U : 2U;
    plan.processing_unit = config.op == ButterflyOperator::Fft ? fft_core_name(config.fft_core) : "scalar";
    plan.fused_twiddle = config.op == ButterflyOperator::Fft && config.backend == ButterflyBackend::OnlineReorder;
    plan.cross_subgraph_dependencies = plan.stage_partition.size() > 1;
    plan.backend = butterfly_backend_name(config.backend);
    for (const auto& boundary : config.boundaries)
        plan.boundary_policies.push_back(boundary.residency == FftBoundaryResidency::Fused
                                            ? BoundaryPolicy::ResidentFused : BoundaryPolicy::GlobalScratch);
    if (plan.boundary_policies.empty() && plan.stage_partition.size() > 1)
        plan.boundary_policies.assign(plan.stage_partition.size() - 1, BoundaryPolicy::GlobalScratch);
    if (config.op == ButterflyOperator::Fft &&
        (config.backend == ButterflyBackend::CuFft || config.fft_core != FftCore::Scalar)) {
        plan.dispatch = DataflowDispatch::ExternalFft;
    } else if (config.local_exchange == LocalExchange::WarpRegister) {
        plan.dispatch = DataflowDispatch::WarpRegister;
    } else {
        switch (config.backend) {
            case ButterflyBackend::TemporalTile: plan.dispatch = DataflowDispatch::TemporalTile; break;
            case ButterflyBackend::Hierarchical: plan.dispatch = DataflowDispatch::Hierarchical; break;
            case ButterflyBackend::OnlineReorder: plan.dispatch = DataflowDispatch::OnlineReorder; break;
            case ButterflyBackend::WarpHybrid: plan.dispatch = DataflowDispatch::WarpHybrid; break;
            case ButterflyBackend::StagePipeline: plan.dispatch = DataflowDispatch::StagePipeline; break;
            case ButterflyBackend::CuFft: break;
            case ButterflyBackend::SharedIterative: plan.dispatch = DataflowDispatch::SharedIterative; break;
            case ButterflyBackend::FactorStreamed: break; // handled by its common physical projection above
        }
    }
    if (plan.dispatch == DataflowDispatch::StagePipeline) plan.threads_per_block = 32 * config.pipeline_warps;
    if (plan.dispatch == DataflowDispatch::WarpHybrid) plan.threads_per_block = 256;
    // External codelets have per-segment launch shapes validated by their backend.
    if (plan.dispatch != DataflowDispatch::ExternalFft) {
        plan.grid_ctas = static_cast<std::uint64_t>(config.batch) << (config.log_n - plan.subgraph_log_n);
        if (plan.dispatch == DataflowDispatch::TemporalTile || plan.dispatch == DataflowDispatch::Hierarchical ||
            plan.dispatch == DataflowDispatch::OnlineReorder) {
            auto state_points = std::uint64_t{1} << plan.subgraph_log_n;
            if (plan.dispatch == DataflowDispatch::OnlineReorder && plan.stage_partition.size() == 2)
                state_points = std::max(state_points, (std::uint64_t{1} << plan.stage_partition.back()) *
                                                        std::max(1U, config.reorder_columns));
            plan.shared_bytes_per_subgraph = static_cast<std::uint32_t>(state_points * (plan.graph.word_bits / 8) *
                                                                         (plan.graph.complex_values ? 2 : 1));
        }
    }
    if (config.backend == ButterflyBackend::SharedIterative) {
        std::uint32_t group = 0, largest = 0;
        for (std::size_t i = 0; i < plan.stage_partition.size(); ++i) {
            group += plan.stage_partition[i];
            if (i + 1 == plan.stage_partition.size() || plan.boundary_policies[i] != BoundaryPolicy::ResidentFused) {
                largest = std::max(largest, group); group = 0;
            }
        }
        plan.subgraph_log_n = largest;
        plan.pipeline_buffers = 2;
        plan.shared_bytes_per_subgraph = (2ULL << largest) * (plan.graph.word_bits / 8) * (plan.graph.complex_values ? 2 : 1);
        plan.grid_ctas = static_cast<std::uint64_t>(config.batch) << (config.log_n - largest);
        plan.data_time = std::max(1U, (1U << largest) / (2 * std::max(1U, config.tile_threads)));
        plan.batch_tile_count = 1;
        plan.steady_state_buffers = 1;
    }
    // Default to bulk boundaries; the optional batch pipeline below changes
    // the schedule without changing the logical graph or claiming residency.
    if (config.backend == ButterflyBackend::OnlineReorder) {
        plan.batch_tile_count = 1;
        plan.steady_state_buffers = 1;
        plan.inter_stage_global_bytes = static_cast<std::uint64_t>(config.batch) *
                                        (std::uint64_t{1} << config.log_n) *
                                        (plan.graph.word_bits / 8) * (plan.graph.complex_values ? 2 : 1);
    }
    if (config.stage_overlap) {
        if (config.batch_tile_count == 0)
            throw std::invalid_argument("batch tile count must be positive");
        plan.dispatch = DataflowDispatch::BatchPipeline;
        plan.batch_tile_count = config.batch_tile_count;
        plan.steady_state_buffers = 2;
        plan.grid_ctas = std::min<std::uint64_t>(config.batch,config.batch_tile_count) << (config.log_n-plan.subgraph_log_n);
        plan.boundary = BoundaryPolicy::RingBuffer;
        for (auto& boundary : plan.boundary_policies)
            if (boundary == BoundaryPolicy::GlobalScratch) boundary = BoundaryPolicy::RingBuffer;
        plan.inter_stage_global_bytes = static_cast<std::uint64_t>(config.batch) *
            (1ULL << config.log_n) * (plan.graph.word_bits / 8) * (plan.graph.complex_values ? 2 : 1) *
            std::count(plan.boundary_policies.begin(),plan.boundary_policies.end(),BoundaryPolicy::RingBuffer);
        // This is concurrent kernel scheduling with global ring buffers;
        // it must not be labeled as CTA shared residency or persistent CTAs.
        plan.persistent = false;
    }
    project_execution_groups(plan);
    for (std::size_t i = 0; i < plan.execution_groups.size(); ++i) {
        auto& group = plan.execution_groups[i];
        group.shared_layout = shared_layout_name(config.shared_layout);
        if (i < config.execution_group_mappings.size()) {
            const auto& mapping = config.execution_group_mappings[i];
            group.core = fft_core_name(mapping.core);
            group.threads = mapping.threads;
            group.elements_per_thread = mapping.ept;
            group.exchange = mapping.exchange == LocalExchange::WarpRegister ? ExchangePolicy::WarpShuffle : ExchangePolicy::SharedMemory;
            group.codelet = mapping.codelet;
            group.io_policy = mapping.io_policy;
        }
        if (config.backend == ButterflyBackend::SharedIterative) {
            group.data_space = std::max(1U, std::min(group.threads, 1U << (group.stage_count - 1)));
            group.data_time = ((1U << (group.stage_count - 1)) + group.data_space - 1) / group.data_space;
            group.live_shared_bytes = (2ULL << group.stage_count) * (plan.graph.word_bits / 8) *
                                      (plan.graph.complex_values ? 2 : 1);
        }
        if (config.op == ButterflyOperator::Fft && config.fft_core==FftCore::RegisterTile && plan.execution_groups.size()==2) {
            group.threads=i==0 ? config.prefix_threads : config.suffix_threads;
            group.elements_per_thread=i==0 ? config.prefix_ept : config.suffix_ept;
            // Each cooperative lane owns fewer values. Both groups therefore
            // derive independent columns from actual thread/EPT ownership;
            // threads/R alone would multiply prefix columns by the lane count.
            const auto columns=(std::uint64_t(group.threads)*group.elements_per_thread) /
                (1U << group.stage_count);
            if (!columns) throw std::invalid_argument("register FFT requires positive columns per CTA");
            const auto launch_batch=config.stage_overlap ? std::min<std::uint64_t>(config.batch,config.batch_tile_count) : config.batch;
            group.grid_ctas=(launch_batch << (config.log_n-group.stage_count))/columns;
            group.live_shared_bytes=(1ULL << group.stage_count)*columns*(plan.graph.word_bits/8)*2;
            group.data_space=group.threads;
            group.data_time=((1ULL << (group.stage_count-1))*columns+group.threads-1)/group.threads;
            group.core=i==0 ? "register-tile" : "cufftdx-block";
            if (i == 0) {
                group.codelet = config.prefix_codelet;
                group.shared_layout = config.prefix_shared_layout;
            }
        }
    }
    detail::validate_resident_axis_counts(config.local_stage_partitions,config.exchange_chunks,plan.execution_groups.size());
    for(std::size_t i=0;i<plan.execution_groups.size();++i)
        project_resident_axes(plan.execution_groups[i],config.local_stage_partitions,config.exchange_chunks,i,
                              config.backend==ButterflyBackend::SharedIterative);
    project_generic_butterfly_groups(plan, config);
    if (config.op == ButterflyOperator::Fft && config.fft_core==FftCore::RegisterTile) {
        plan.grid_ctas=0;
        for (const auto& group : plan.execution_groups) plan.grid_ctas=std::max(plan.grid_ctas,group.grid_ctas);
    }
    std::size_t physical = 0;
    for (const auto& boundary : config.boundaries) if (boundary.residency != FftBoundaryResidency::Fused) {
        auto& edge = plan.execution_boundaries.at(physical++);
        edge.producer_layout = edge.consumer_layout = direct_boundary_name(boundary.layout);
    }
    return plan;
}

MixedDataflowPlan make_dataflow_plan(const PlanConfig& config) {
    MixedDataflowPlan plan;
    plan.graph = make_operator_graph(config);
    plan.stage_partition = config.stage_partition;
    if (plan.stage_partition.empty()) plan.stage_partition = {config.log_n};
    std::uint32_t stage_sum = 0;
    for (const auto stages : plan.stage_partition) stage_sum += stages;
    if (stage_sum != config.log_n || std::any_of(plan.stage_partition.begin(), plan.stage_partition.end(), [](std::uint32_t n) { return n == 0; }))
        throw std::invalid_argument("NTT stage partition must contain positive stages summing to log_n");
    plan.subgraph_log_n = plan.stage_partition.front();
    plan.data_space = std::max(1U, config.data_space);
    plan.data_time = std::max(1U, config.data_time);
    plan.target_resident_ctas = config.target_ctas_per_sm;
    plan.pipeline_buffers = std::max(1U, config.pipeline_buffers);
    plan.threads_per_block = config.threads_per_block;
    if (plan.threads_per_block == 0 && !config.subgraph_mappings.empty())
        plan.threads_per_block = config.subgraph_mappings.front().threads_per_block;
    if (plan.threads_per_block == 0) plan.threads_per_block = 256;
    plan.elements_per_thread = 1;
    plan.processing_radix = config.compute_unit == ComputeUnit::Radix4 ? 4U :
                            config.compute_unit == ComputeUnit::Radix8 ? 8U : 2U;
    plan.processing_unit = hierarchical_core_name(config.hierarchical_core);
    plan.backend = backend_name(config.backend);
    plan.persistent = config.backend == Backend::HybridDataflow || config.backend == Backend::HierarchicalBarrier ||
                      config.backend == Backend::HierarchicalDataflow;
    plan.fused_twiddle = config.cross_twiddle_placement == CrossTwiddlePlacement::Fused ||
                         config.cross_twiddle_placement == CrossTwiddlePlacement::FusedBarrett;
    plan.cross_subgraph_dependencies = plan.stage_partition.size() > 1;
    plan.dispatch = DataflowDispatch::Unresolved;
    switch (config.backend) {
        case Backend::Baseline: plan.dispatch = DataflowDispatch::TemporalTile; break;
        case Backend::Tile256: plan.dispatch = DataflowDispatch::TemporalTile; break;
        case Backend::Hybrid2D: plan.dispatch = DataflowDispatch::Hierarchical; break;
        case Backend::CompactStage: plan.dispatch = DataflowDispatch::TemporalTile; break;
        case Backend::StagePipeline: plan.dispatch = DataflowDispatch::StagePipeline; break;
        case Backend::HybridDataflow: plan.dispatch = DataflowDispatch::Hierarchical; break;
        case Backend::HierarchicalBarrier: plan.dispatch = DataflowDispatch::Hierarchical; break;
        case Backend::HierarchicalDataflow: plan.dispatch = DataflowDispatch::Hierarchical; break;
        case Backend::SharedIterative: plan.dispatch = DataflowDispatch::SharedIterative; break;
    }
    plan.boundary_policies.reserve(config.boundary_mappings.size());
    for (const auto& boundary : config.boundary_mappings)
        plan.boundary_policies.push_back(to_boundary_policy(boundary));
    if (plan.boundary_policies.empty() && plan.stage_partition.size() > 1)
        plan.boundary_policies.assign(plan.stage_partition.size() - 1, BoundaryPolicy::GlobalScratch);
    plan.boundary = plan.boundary_policies.empty() ? BoundaryPolicy::None : plan.boundary_policies.front();
    plan.exchange = config.dataflow_layout == DataflowLayout::HermesXor ? ExchangePolicy::SharedMemory : ExchangePolicy::GlobalMemory;
    const auto word_bytes = config.word_bits / 8;
    const auto tile_points = std::uint64_t{1} << plan.subgraph_log_n;
    plan.shared_bytes_per_subgraph = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        std::numeric_limits<std::uint32_t>::max(), tile_points * word_bytes));
    plan.grid_ctas = static_cast<std::uint64_t>(config.batch) << (config.log_n - plan.subgraph_log_n);
    if (config.stage_overlap) {
        if (config.backend != Backend::SharedIterative || plan.stage_partition.size()<2 || !config.batch_tile_count)
            throw std::invalid_argument("NTT batch pipeline requires multiple shared groups and a positive batch tile");
        plan.dispatch=DataflowDispatch::BatchPipeline;
        plan.batch_tile_count=config.batch_tile_count; plan.steady_state_buffers=2;
        plan.grid_ctas=std::min<std::uint64_t>(config.batch,config.batch_tile_count) << (config.log_n-plan.subgraph_log_n);
        plan.boundary=BoundaryPolicy::RingBuffer;
        plan.boundary_policies.assign(plan.stage_partition.size()-1,BoundaryPolicy::RingBuffer);
        plan.inter_stage_global_bytes=config.batch*(1ULL<<config.log_n)*word_bytes*(plan.stage_partition.size()-1);
    }
    project_execution_groups(plan);
    for (std::size_t i = 0; i < plan.execution_groups.size(); ++i) {
        auto& group = plan.execution_groups[i];
        group.shared_layout = dataflow_layout_name(config.dataflow_layout);
        if (i < config.execution_group_mappings.size()) {
            const auto& m = config.execution_group_mappings[i];
            group.core = ntt_subgraph_core_name(m.core);
            group.threads = m.threads_per_block;
            group.units_per_cta = m.units_per_cta;
            group.data_space = m.data_space; group.data_time = m.data_time;
        }
        if (config.backend == Backend::SharedIterative) {
            const auto stages=group.stage_count;
            const auto size=1ULL<<stages;
            group.core="radix2-shoup";
            group.shared_layout=config.dataflow_layout==DataflowLayout::HermesXor ? "writer-aligned" : "linear";
            group.threads=plan.threads_per_block; group.units_per_cta=1;
            group.data_space=std::min<std::uint64_t>(group.threads,size/2);
            group.data_time=(size/2+group.data_space-1)/group.data_space;
            group.live_shared_bytes=2*size*word_bytes;
            group.grid_ctas=(config.stage_overlap ? std::min<std::size_t>(config.batch,config.batch_tile_count) : config.batch)*
                (1ULL<<(config.log_n-stages));
        }
    }
    detail::validate_resident_axis_counts(config.local_stage_partitions,config.exchange_chunks,plan.execution_groups.size());
    for(std::size_t i=0;i<plan.execution_groups.size();++i)
        project_resident_axes(plan.execution_groups[i],config.local_stage_partitions,config.exchange_chunks,i,
                              config.backend==Backend::SharedIterative);
    if (config.backend == Backend::SharedIterative) {
        plan.exchange=ExchangePolicy::SharedMemory;
        plan.data_space=1; plan.processing_unit="radix2-shoup";
        plan.shared_bytes_per_subgraph=0;
        for (const auto& group : plan.execution_groups)
            plan.shared_bytes_per_subgraph=std::max<std::uint32_t>(plan.shared_bytes_per_subgraph,group.live_shared_bytes);
    }
    return plan;
}

PlanResourceEstimate estimate_resources(const MixedDataflowPlan& plan, const HardwareResourceModel& hardware) {
    PlanResourceEstimate result;
    if(plan.backend=="factor-streamed" && !plan.execution_groups.empty()) {
        result.resident_ctas=std::numeric_limits<std::uint32_t>::max();
        result.resident_warps=std::numeric_limits<std::uint32_t>::max();
        for(const auto& group:plan.execution_groups) {
            auto local=plan; local.backend="factor-resource-projection";
            local.threads_per_block=group.threads; local.registers_per_thread=group.compiler_registers_per_thread;
            local.shared_bytes_per_subgraph=group.live_shared_bytes; local.data_space=1;
            local.grid_ctas=group.grid_ctas; local.boundary_policies.clear();
            const auto r=estimate_resources(local,hardware);
            if(!r.feasible) return r;
            result.resident_ctas=std::min(result.resident_ctas,r.resident_ctas);
            result.resident_warps=std::min(result.resident_warps,r.resident_warps);
            result.grid_waves=std::max(result.grid_waves,r.grid_waves);
            result.local_bytes=std::max(result.local_bytes,r.local_bytes);
        }
        // Physical HBM edges also exist inside a logical macro group.
        for(const auto& edge:plan.execution_boundaries) {
            if(edge.bytes>(std::numeric_limits<std::uint64_t>::max()-result.boundary_bytes)/2) {
                result.reason="boundary traffic overflows uint64"; return result;
            }
            result.boundary_bytes+=2*edge.bytes;
        }
        result.feasible=true;
        result.reason="per-factor resource upper bounds satisfied; specialization occupancy remains unverified";
        return result;
    }
    if (plan.graph.log_n == 0 || plan.graph.log_n > 30 || plan.graph.batch == 0 ||
        hardware.sm_count == 0 || hardware.max_threads_per_sm == 0) {
        result.reason = "invalid graph or incomplete hardware limits";
        return result;
    }
    if (plan.threads_per_block == 0 || plan.threads_per_block > hardware.max_threads_per_block) {
        result.reason = "threads_per_block exceeds hardware limit";
        return result;
    }
    if (plan.threads_per_block % 32 != 0) {
        result.reason = "threads_per_block must be a warp multiple";
        return result;
    }
    const auto warps = plan.threads_per_block / 32;
    const auto shared = static_cast<std::uint64_t>(plan.shared_bytes_per_subgraph) * plan.data_space;
    const auto register_bytes = static_cast<std::uint64_t>(plan.registers_per_thread) * plan.threads_per_block * 4ULL;
    if (shared > hardware.shared_bytes_per_sm ||
        (hardware.shared_bytes_per_block && shared > hardware.shared_bytes_per_block)) {
        result.reason = "resident shared state exceeds per-SM shared memory";
        return result;
    }
    if (register_bytes > hardware.registers_per_sm * 4ULL ||
        (hardware.registers_per_block && register_bytes > hardware.registers_per_block * 4ULL)) {
        result.reason = "resident register state exceeds per-SM register file";
        return result;
    }
    const std::uint64_t shared_ctas = hardware.shared_bytes_per_sm && shared
                                          ? hardware.shared_bytes_per_sm / shared
                                          : std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t register_ctas = hardware.registers_per_sm && register_bytes
                                            ? static_cast<std::uint64_t>(hardware.registers_per_sm) * 4ULL / register_bytes
                                            : std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t thread_ctas = hardware.max_threads_per_sm
                                          ? hardware.max_threads_per_sm / plan.threads_per_block
                                          : std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t block_ctas = hardware.max_blocks_per_sm ? hardware.max_blocks_per_sm : thread_ctas;
    result.resident_ctas = static_cast<std::uint32_t>(std::min({shared_ctas, register_ctas, thread_ctas, block_ctas}));
    result.resident_warps = result.resident_ctas * warps;
    result.local_bytes = static_cast<std::uint64_t>(shared);
    const auto capacity = static_cast<std::uint64_t>(result.resident_ctas) * hardware.sm_count;
    if (capacity && plan.grid_ctas)
        result.grid_waves = static_cast<std::uint32_t>(plan.grid_ctas / capacity + (plan.grid_ctas % capacity != 0));
    const auto global_edges = static_cast<std::uint64_t>(std::count_if(plan.boundary_policies.begin(),
        plan.boundary_policies.end(), [](auto edge) { return edge==BoundaryPolicy::GlobalScratch || edge==BoundaryPolicy::RingBuffer; }));
    const auto bytes_per_point = (plan.graph.word_bits / 8) * (plan.graph.complex_values ? 2ULL : 1ULL);
    const auto edge_bytes = (1ULL << plan.graph.log_n) * bytes_per_point * 2ULL * global_edges;
    if (edge_bytes && plan.graph.batch > std::numeric_limits<std::uint64_t>::max() / edge_bytes) {
        result.reason = "boundary traffic overflows uint64";
        return result;
    }
    result.boundary_bytes = edge_bytes * plan.graph.batch;
    const bool cooperative = plan.boundary == BoundaryPolicy::CooperativeGrid ||
                             plan.exchange == ExchangePolicy::CooperativeGrid;
    result.feasible = result.resident_ctas > 0 && result.resident_ctas >= plan.target_resident_ctas &&
                      (!cooperative || (hardware.cooperative_launch && plan.grid_ctas && plan.grid_ctas <= capacity));
    result.reason = result.feasible ? "resource upper bounds satisfied; specialization occupancy remains unverified"
                                   : "resident CTA target or cooperative grid exceeds available resources";
    return result;
}

LoweringStatus check_lowering(const MixedDataflowPlan& plan) {
    if (plan.backend != "baseline" && plan.backend != "tile256" && plan.backend != "hybrid2d" &&
        plan.backend != "compact-stage" && plan.backend != "hierarchical-barrier" && plan.backend != "hierarchical-dataflow" &&
        plan.backend != "hybrid-dataflow" && plan.backend != "temporal-tile" && plan.backend != "hierarchical" && plan.backend != "online-reorder" &&
        plan.backend != "warp-hybrid" && plan.backend != "stage-pipeline" && plan.backend != "cufft" && plan.backend != "shared-iterative" &&
        plan.backend != "factor-streamed")
        return LoweringStatus::MissingBackend;
    if (plan.dispatch == DataflowDispatch::Unresolved) return LoweringStatus::MissingBackend;
    if (plan.graph.log_n == 0 || plan.graph.log_n > 30 ||
        (plan.graph.log_n > 20 && plan.backend != "shared-iterative" && plan.backend != "cufft" && plan.backend != "factor-streamed" &&
         plan.processing_unit != "register-tile")) return LoweringStatus::UnsupportedSize;
    if ((plan.boundary == BoundaryPolicy::RingBuffer && plan.dispatch != DataflowDispatch::BatchPipeline) ||
        plan.boundary == BoundaryPolicy::CooperativeGrid)
        return LoweringStatus::UnsupportedBoundary;
    if (plan.processing_unit == "cufftdx-resident" && plan.graph.log_n != 12 && plan.graph.log_n != 14)
        return LoweringStatus::UnsupportedSize;
    if (plan.exchange == ExchangePolicy::CooperativeGrid && !plan.persistent)
        return LoweringStatus::UnsupportedExchange;
    if (plan.backend == "cufft" && plan.graph.op != DataflowOperator::Fft) return LoweringStatus::UnsupportedOperator;
    if (plan.backend == "factor-streamed" && plan.graph.op != DataflowOperator::Fft) return LoweringStatus::UnsupportedOperator;
    switch (plan.graph.op) {
        case DataflowOperator::Fft:
        case DataflowOperator::Ntt:
        case DataflowOperator::Fwht:
        case DataflowOperator::SubsetZeta:
        case DataflowOperator::SupersetZeta:
        case DataflowOperator::XorZeta:
        case DataflowOperator::Structured2x2:
            // The constructor still checks numeric semantics and compiled codelet availability.
            return LoweringStatus::RequiresBackendValidation;
    }
    return LoweringStatus::UnsupportedOperator;
}

bool validate_plan(MixedDataflowPlan& plan, const HardwareResourceModel& hardware) {
    plan.executable = false;
    plan.rejection_reason.clear();
    if (plan.graph.log_n == 0 || plan.graph.log_n > 30 || plan.graph.batch == 0 || plan.stage_partition.empty() ||
        plan.graph.stage_count != plan.graph.log_n || plan.graph.stage_arities.size() != plan.graph.log_n ||
        plan.data_space == 0 || plan.data_time == 0 || plan.pipeline_buffers == 0) {
        plan.rejection_reason = "graph and stage_partition are required";
        return false;
    }
    std::uint32_t stages = 0;
    for (const auto count : plan.stage_partition) {
        if (count == 0 || count > plan.graph.log_n - stages) {
            plan.rejection_reason = "invalid stage partition";
            return false;
        }
        stages += count;
    }
    if (stages != plan.graph.stage_count || plan.subgraph_log_n > plan.graph.log_n) {
        plan.rejection_reason = "stage partition does not cover operator graph";
        return false;
    }
    if (plan.boundary_policies.size() + 1 != plan.stage_partition.size()) {
        plan.rejection_reason = "one boundary policy is required per logical edge";
        return false;
    }
    if (plan.boundary == BoundaryPolicy::None && plan.cross_subgraph_dependencies) {
        plan.rejection_reason = "cross-subgraph dependencies require an explicit exchange boundary";
        return false;
    }
    const auto lowering = check_lowering(plan);
    if (lowering != LoweringStatus::Supported && lowering != LoweringStatus::RequiresBackendValidation) {
        plan.rejection_reason = lowering_status_name(lowering);
        return false;
    }
    const auto estimate = estimate_resources(plan, hardware);
    if (!estimate.feasible) {
        plan.rejection_reason = estimate.reason;
        return false;
    }
    // Only successful backend construction can certify execution, not this estimate.
    return true;
}

}  // namespace cuntt
