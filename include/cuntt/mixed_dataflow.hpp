#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cuntt {

struct ButterflyConfig;
struct PlanConfig;

// A launch route, not a claim that a particular specialization is compiled.
enum class DataflowDispatch : std::uint8_t {
    Unresolved, TemporalTile, WarpRegister, Hierarchical, OnlineReorder,
    WarpHybrid, StagePipeline, ExternalFft,
    SharedIterative, BatchPipeline,
};

enum class DataflowOperator : std::uint8_t {
    Fft,
    Ntt,
    Fwht,
    SubsetZeta,
    SupersetZeta,
    XorZeta,
    Structured2x2,
};

enum class BoundaryPolicy : std::uint8_t {
    None,
    Register,
    SharedResident,
    ResidentFused,
    GlobalScratch,
    RingBuffer,
    CooperativeGrid,
};

enum class ExchangePolicy : std::uint8_t {
    Register,
    WarpShuffle,
    SharedMemory,
    GlobalMemory,
    CooperativeGrid,
};

struct OperatorGraph {
    DataflowOperator op = DataflowOperator::Fwht;
    std::uint32_t log_n = 0;
    std::uint32_t stage_count = 0;
    std::uint32_t radix = 2;
    std::uint32_t word_bits = 32;
    std::uint32_t accumulator_bits = 32;
    std::size_t batch = 1;
    bool inverse = false;
    bool in_place = false;
    bool normalized_inverse = true;
    std::vector<std::uint32_t> stage_arities;
    std::vector<std::uint32_t> coefficient_stages;
    std::string precision;
    std::size_t element_stride = 1;
    std::size_t batch_stride = 0;
    bool complex_values = false;
};

struct HardwareResourceModel {
    std::string device_name;
    int compute_major = -1;
    int compute_minor = -1;
    std::uint32_t sm_count = 0;
    std::uint32_t max_threads_per_block = 0;
    std::uint32_t max_threads_per_sm = 0;
    std::uint32_t registers_per_sm = 0;
    std::uint32_t shared_bytes_per_sm = 0;
    std::uint64_t memory_bytes = 0;
    double memory_bandwidth_gbps = 0.0;
    double launch_overhead_us = 0.0;
    bool cooperative_launch = false;
    std::uint32_t max_blocks_per_sm = 0;
    std::uint32_t shared_bytes_per_block = 0;
    std::uint32_t registers_per_block = 0;
};

enum class CandidateState : std::uint8_t {
    Abstract, SemanticRejected, ResourceRejected, NeedsLowering,
    NeedsCompilation, Compiled, CorrectnessVerified, Measured,
};

// One physical execution group owns a contiguous interval of logical stages.
// Logical radix stays two; a radix4/radix8 codelet is an execution choice.
struct ExecutionGroup {
    std::uint32_t first_stage = 0, stage_count = 0;
    std::uint32_t stage_space = 1, stage_time = 1;
    std::uint32_t data_space = 1, data_time = 1;
    std::uint32_t batch_space = 1, batch_time = 1;
    std::uint32_t threads = 0, elements_per_thread = 0, units_per_cta = 1;
    std::string core, shared_layout;
    ExchangePolicy exchange = ExchangePolicy::SharedMemory;
    std::uint64_t live_shared_bytes = 0, grid_ctas = 0;
    // These cannot be inferred from logical unfolding factors.
    std::uint32_t compiler_registers_per_thread = 0;
    bool compiler_resources_known = false;
    std::uint64_t compiler_local_bytes_per_thread = 0;
    bool compiler_local_resources_known = false;
    std::uint32_t data_tiles_per_cta = 1, prefetch_depth = 0;
    std::uint32_t logical_macro_group = 0;
    std::uint32_t launch_count = 1;
    bool partial_dependency_ready = false;
    // Codelet and I/O policy are physical lowering identities.  They are
    // separate from the logical operator and therefore part of the execution
    // descriptor used by search, JIT cache keys, and replay validation.
    // Appending these fields preserves existing aggregate initialization.
    std::string codelet = "native";
    std::string io_policy = "dynamic";
    std::vector<std::uint32_t> local_stage_partition;
    std::uint32_t exchange_chunk = 0;
};

struct ExecutionBoundary {
    std::uint32_t producer_group = 0, consumer_group = 0;
    BoundaryPolicy storage = BoundaryPolicy::GlobalScratch;
    std::string producer_layout, consumer_layout;
    std::uint64_t bytes = 0;
    std::uint32_t buffers = 1;
};

// Query the active CUDA device. The function is intentionally kept in the
// common plan layer so every operator uses the same hardware identity and
// resource limits.
HardwareResourceModel query_hardware_resource_model();

struct MixedDataflowPlan {
    OperatorGraph graph;
    std::vector<std::uint32_t> stage_partition;
    std::uint32_t subgraph_log_n = 0;
    std::uint32_t data_space = 1;
    std::uint32_t data_time = 1;
    std::uint32_t target_resident_ctas = 1;
    std::uint32_t pipeline_buffers = 1;
    BoundaryPolicy boundary = BoundaryPolicy::GlobalScratch;
    ExchangePolicy exchange = ExchangePolicy::SharedMemory;
    std::uint32_t threads_per_block = 128;
    std::uint32_t elements_per_thread = 1;
    std::uint32_t shared_bytes_per_subgraph = 0;
    std::uint32_t registers_per_thread = 0;
    bool persistent = false;
    bool fused_twiddle = false;
    bool cross_subgraph_dependencies = false;
    bool executable = false;
    std::string backend;
    std::string rejection_reason;
    DataflowDispatch dispatch = DataflowDispatch::Unresolved;
    std::string processing_unit = "scalar";
    // Keep every logical boundary, including mixed fused/materialized edges.
    std::vector<BoundaryPolicy> boundary_policies;
    std::uint32_t processing_radix = 2;
    std::uint64_t grid_ctas = 0;
    // Batch-level lowering contract.  A value >1 describes how many batch
    // tiles may be resident in one persistent execution group; it is kept
    // separate from stage_partition so the search space does not conflate
    // spatial and temporal unfolding.
    std::uint32_t batch_tile_count = 1;
    std::uint32_t steady_state_buffers = 1;
    std::uint64_t inter_stage_global_bytes = 0;
    std::vector<ExecutionGroup> execution_groups;
    std::vector<ExecutionBoundary> execution_boundaries;
    CandidateState state = CandidateState::Abstract;
};

struct PlanResourceEstimate {
    std::uint32_t resident_ctas = 0;
    std::uint32_t resident_warps = 0;
    std::uint32_t grid_waves = 0;
    std::uint64_t boundary_bytes = 0;
    std::uint64_t local_bytes = 0;
    double predicted_kernel_ms = 0.0;
    bool feasible = false;
    std::string reason;
    // Register allocation and specialization occupancy still need compiler/CUDA evidence.
    bool complete = false;
};

enum class LoweringStatus : std::uint8_t {
    Supported,
    UnsupportedOperator,
    UnsupportedSize,
    UnsupportedBoundary,
    UnsupportedExchange,
    MissingBackend,
    RequiresBackendValidation,
};

const char* dataflow_operator_name(DataflowOperator op) noexcept;
DataflowOperator parse_dataflow_operator(const std::string& name);
const char* boundary_policy_name(BoundaryPolicy policy) noexcept;
const char* exchange_policy_name(ExchangePolicy policy) noexcept;
const char* dataflow_dispatch_name(DataflowDispatch dispatch) noexcept;

OperatorGraph make_operator_graph(DataflowOperator op, std::uint32_t log_n, std::size_t batch = 1);
OperatorGraph make_operator_graph(const ButterflyConfig& config);
OperatorGraph make_operator_graph(const PlanConfig& config);
MixedDataflowPlan make_dataflow_plan(const ButterflyConfig& config);
MixedDataflowPlan make_dataflow_plan(const PlanConfig& config);
PlanResourceEstimate estimate_resources(const MixedDataflowPlan& plan, const HardwareResourceModel& hardware);
LoweringStatus check_lowering(const MixedDataflowPlan& plan);
const char* lowering_status_name(LoweringStatus status) noexcept;
bool validate_plan(MixedDataflowPlan& plan, const HardwareResourceModel& hardware);

}  // namespace cuntt
