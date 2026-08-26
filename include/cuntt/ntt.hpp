#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

namespace cuntt {

constexpr std::uint64_t kDefaultModulus = 1152921504606584833ULL;

enum class Backend {
    Baseline,
    Tile256,
    Hybrid2D,
    CompactStage,
    StagePipeline,
    HybridDataflow,
    HierarchicalBarrier,
    HierarchicalDataflow,
    Merge256 = Hybrid2D,
};

const char* backend_name(Backend backend) noexcept;
Backend     parse_backend(const std::string& name);

enum class StageHandoff {
    Atomic,
    NamedBarrier,
};

const char*  stage_handoff_name(StageHandoff handoff) noexcept;
StageHandoff parse_stage_handoff(const std::string& name);

enum class DataflowLayout {
    HermesXor,
    Linear,
};

const char*    dataflow_layout_name(DataflowLayout layout) noexcept;
DataflowLayout parse_dataflow_layout(const std::string& name);

enum class DataflowStateMode {
    InPlace,
    PingPong,
};

const char*       dataflow_state_mode_name(DataflowStateMode mode) noexcept;
DataflowStateMode parse_dataflow_state_mode(const std::string& name);

enum class OutputOrder {
    Natural,
    BitReversed,
    // APPT static layout for zero-conversion chaining between compatible plans.
    ApptStatic,
};

const char* output_order_name(OutputOrder order) noexcept;
OutputOrder parse_output_order(const std::string& name);

enum class InputOrder {
    Natural,
    ApptStatic,
};

const char* input_order_name(InputOrder order) noexcept;
InputOrder  parse_input_order(const std::string& name);

struct ApptLayoutInfo {
    std::uint32_t              log_n = 0;
    std::vector<std::uint32_t> stage_partition;
    std::uint32_t              fragment_width = 0;
    std::uint32_t              bank_bits = 0;
    bool                       xor_permutation = false;
    std::string                compatibility_id;
};

// Convert a per-transform natural index to/from the APPT static layout.
// Both functions throw when the descriptor is not a supported 7+7+6 layout.
std::uint64_t appt_static_index(std::uint64_t natural_index,
                                const ApptLayoutInfo& layout);
std::uint64_t appt_natural_index(std::uint64_t static_index,
                                 const ApptLayoutInfo& layout);

enum class ComputeUnit {
    Auto,
    Radix2,
    Radix4,
    Radix8,
};

const char* compute_unit_name(ComputeUnit unit) noexcept;
ComputeUnit parse_compute_unit(const std::string& name);

enum class HierarchicalCore {
    DataflowRadix4,
    Hybrid2DRadix4,
};

const char*      hierarchical_core_name(HierarchicalCore core) noexcept;
HierarchicalCore parse_hierarchical_core(const std::string& name);

// A subgraph core is a physical implementation choice. The stage partition and
// mappings below describe the architecture schedule independently of that core.
enum class NttSubgraphCore {
    Radix2,
    Radix4,
    Radix8,
    DataflowRadix4,
    // A generated two-level schedule that reuses one resident radix-4
    // subgraph shape for both roles. Per-role data_time controls traversal of
    // the batch dimension while the spatial subgraph coordinate stays fixed.
    HomogeneousRadix4,
    // A dependency-closed 1024-point subgraph per warp. Eight independent
    // warp-register codelets share a CTA role without CTA-wide barriers.
    HomogeneousWarpRadix2,
    // Four 256-point warp-register codelets form one 1024-point subgraph with
    // a single warp-group exchange; two groups progress independently per CTA.
    HomogeneousWarp256Radix2,
    // The same 256-point warp core with a word-size-dependent row packet and
    // XOR-swizzled shared writer. Each packet fills complete 32-byte sectors
    // at both the producer boundary and the natural-order final output.
    HomogeneousWarp256StaticRadix2,
    // Extends the static writer to the input edge: producer row packets are
    // loaded coalesced, the boundary remains natural, and the consumer applies
    // bit reversal while staging in shared memory.
    HomogeneousWarp256StaticIoRadix2,
    // Warp256 static IO with explicit per-lane coefficient reuse for the
    // register-resident prefix.
    HomogeneousWarp256CoefficientReuseStaticIoRadix2,
    // Smaller register prefixes for the same static-IO schedule. They trade
    // additional shared finishing stages for a smaller per-warp live set.
    HomogeneousWarp128StaticIoRadix2,
    // The same warp128 subgraph with explicit per-lane reuse of coefficients
    // shared by register slots in stages 0 through 5.
    HomogeneousWarp128CoefficientReuseStaticIoRadix2,
    // Four consecutive values per lane fuse the first two stages into a
    // register-local radix-4 unit before the remaining warp-shuffle stages.
    HomogeneousWarp128VectorRadix4StaticIo,
    HomogeneousWarp64StaticIoRadix2,
    // Three prefix warps and one merge warp overlap adjacent 1024-point rows
    // through a double-buffered shared tile.
    HomogeneousWarp128PipelineStaticIoRadix2,
    // Four warps retain continuation ownership: each completes one 256-point
    // pair, two warp pairs merge the 512-point halves, and all warps finish
    // the final stage without a permanently assigned merge role.
    HomogeneousWarp128CooperativeStaticIoRadix2,
    Hybrid2DRadix4,
    // APPT/Hermes mapping: persistent warp stages stream register-resident
    // data tokens through a fixed-size butterfly subgraph.
    ApptPipeline,
    // Generated APPT mapping with persistent CTA roles that publish reordered
    // packet groups directly to the next stage-time fold.
    ApptOnline,
    // Same APPT online schedule with a CTA-resident shared-memory radix-4
    // physical subgraph instead of the warp-register radix-2 core.
    ApptOnlineRadix4,
    // APPT producer plus a dependency-closed resident 7+6 tail. The tail
    // removes the second global intermediate state.
    ApptOnlineFusedTail,
    // Two 32x128 resident tail halves fuse 7+5 stages; only the low half is
    // published before the high half performs the final cross-half stage.
    ApptOnlineSplitTail,
    // One CTA computes both 32x128 tail halves. The first half remains in
    // registers while shared memory is reused for the second half.
    ApptOnlineRegisterTail,
    // The same register-tail graph with warp-register row/column codelets.
    // Shared memory is used only for the row/column exchange and retained
    // low-half state, reducing CTA-wide synchronization inside each subgraph.
    ApptOnlineRegisterTailWarp,
    // Keep the cached CTA radix-4 row core, but execute the 32-point column
    // suffix with warp shuffles to isolate the synchronization/compute tradeoff.
    ApptOnlineRegisterTailColumnWarp,
    // Fuse three shared-memory butterfly stages per round. This preserves the
    // CTA codelet's instruction-level parallelism while reducing barriers.
    ApptOnlineRegisterTailRadix8,
    // Group the producer's a coordinate according to data_space, load that
    // low-bit source dimension cooperatively, and transpose it on chip before
    // executing the same register-tail graph.
    ApptOnlineRegisterTailGrouped,
    // The grouped producer graph with stage 19 moved from the fixed-c tail to
    // the static-to-natural writer, where c-contiguous lanes coalesce the
    // final coefficient requests.
    ApptOnlineRegisterTailGroupedWriterFinal,
    // Keep a fixed spatial subgraph while traversing a data_time-sized group
    // of batch transforms. The physical role mask selects which queues use
    // this order so coefficient reuse can be traded against wavefront delay.
    ApptOnlineRegisterTailGroupedWriterFinalDataTime,
    // Fuse the first two 7-stage dimensions in a fixed-a 128x128 resident 2D
    // subgraph. The stage-13 state is streamed to the 5-stage register tail
    // and existing stage-19 natural-order writer.
    ApptOnlineRegisterTailGroupedWriterFinalResident,
    // The same fixed-a dependency closure, evaluated as four 32-row quarters.
    // A 48 KiB shared tile plus register-retained q0/q1/q2 state targets two
    // resident CTAs per SM on V100.
    ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter,
    // The vector radix-4 prefix with an aligned side table for stage 6. Kept at
    // the end so existing NttSubgraphCore numeric values remain stable.
    HomogeneousWarp128VectorRadix4PackedStage6StaticIo,
    // The same aligned stage-6 side table, distributed as two coefficients per
    // lane so all 32 lanes retain the memory-level parallelism of the d7 core.
    HomogeneousWarp128VectorRadix4PackedStage6DistributedStaticIo,
    // Four rows share one 128-thread radix-4 packet. The group reuses the
    // low-instruction shared core while remaining independently schedulable
    // from the other packet in a 256-thread CTA.
    HomogeneousWarp128PacketSharedRadix4StaticIo,
};

const char*     ntt_subgraph_core_name(NttSubgraphCore core) noexcept;
NttSubgraphCore parse_ntt_subgraph_core(const std::string& name);

enum class BoundaryStorage {
    FullScratch,
    Ring,
    // Keep the adjacent logical subgraphs inside one physical execution
    // group. No global boundary is allocated or published at this edge.
    ResidentFused,
};

const char*     boundary_storage_name(BoundaryStorage storage) noexcept;
BoundaryStorage parse_boundary_storage(const std::string& name);

enum class PacketReadinessMode {
    PerPacket,
    WaveBitmap,
};

const char* packet_readiness_mode_name(PacketReadinessMode mode) noexcept;
PacketReadinessMode parse_packet_readiness_mode(const std::string& name);

enum class PacketComputeLayout {
    InterleavedRows,
    WarpRows,
};

const char* packet_compute_layout_name(PacketComputeLayout layout) noexcept;
PacketComputeLayout parse_packet_compute_layout(const std::string& name);

struct NttSubgraphMapping {
    NttSubgraphCore core               = NttSubgraphCore::Radix2;
    std::uint32_t   threads_per_block  = 256;
    std::uint32_t   units_per_cta      = 1;
    std::uint32_t   data_space         = 1;
    std::uint32_t   data_time          = 1;
    std::uint32_t   role_stages        = 1;
    std::uint32_t   token_interleave   = 1;
    // Low register stages that retain one coefficient pair per lane.
    // Zero disables reuse on ordinary cores and selects the calibrated depth
    // on coefficient-reuse cores.
    std::uint32_t   coefficient_reuse_stages = 0;
    // Zero selects the calibrated service-rate weight for this core/device.
    std::uint32_t   cta_weight         = 0;
};

struct NttBoundaryMapping {
    BoundaryStorage storage = BoundaryStorage::FullScratch;
    std::uint32_t   buffers = 1;
};

// Physical service-rate allocation for the cooperative APPT register-tail
// kernel. It is intentionally independent from logical stage mappings.
struct ApptRoleMapping {
    std::uint32_t producer_weight = 0;
    std::uint32_t tail_weight = 0;
    std::uint32_t writer_weight = 0;
    std::uint32_t fragment_width = 0;
    std::uint32_t writer_tiles_per_cta = 0;
    // Bit 0/1/2 enables data-time traversal in producer/tail/writer.
    std::uint32_t data_time_role_mask = 7;
};

enum class CrossTwiddlePlacement {
    FirstPass,
    SecondPass,
    Fused,
    FusedBarrett,
};

const char*           cross_twiddle_placement_name(CrossTwiddlePlacement placement) noexcept;
CrossTwiddlePlacement parse_cross_twiddle_placement(const std::string& name);

enum class ModularMultiply {
    Shoup,
    Barrett,
};

const char*     modular_multiply_name(ModularMultiply multiply) noexcept;
ModularMultiply parse_modular_multiply(const std::string& name);

struct SelectionInfo {
    bool        automatic = false;
    bool        calibrated = false;
    std::string target;
    std::string implementation;
    std::string confidence;
    std::string reason;
    double      predicted_kernel_ms = 0.0;
};

struct PlanConfig {
    std::uint32_t log_n   = 16;
    std::size_t   batch   = 1;
    std::uint64_t modulus = kDefaultModulus;
    bool          inverse = false;
    Backend       backend = Backend::Tile256;
    // Explicit stage-space unfolding for StagePipeline. Zero selects the default.
    std::uint32_t stage_space   = 0;
    StageHandoff  stage_handoff = StageHandoff::NamedBarrier;
    // HybridDataflow maps stage_space stages onto persistent warp roles and
    // streams 2 * data_space coefficients through the reusable subgraph.
    // Zero values select the generated target default.
    std::uint32_t     flow_tile_log_n  = 0;
    std::uint32_t     data_space       = 0;
    // Consecutive stages fused inside one register-resident warp role.
    std::uint32_t     role_stages      = 0;
    // Compile-time launch-bounds target for generated fused roles.
    std::uint32_t     target_ctas_per_sm = 0;
    // Independent subgraph tokens carried by one channel handoff (data-time unfolding).
    std::uint32_t     data_time        = 0;
    // Independent tokens whose fused stages are instruction-interleaved inside one warp.
    std::uint32_t     token_interleave = 0;
    std::uint32_t     pipeline_buffers = 0;
    DataflowLayout    dataflow_layout  = DataflowLayout::HermesXor;
    DataflowStateMode dataflow_state_mode = DataflowStateMode::InPlace;
    // Hybrid2D mapping. Zero selects a device-specific default.
    // HierarchicalBarrier reuses the same decomposition controls: n1_log is
    // the second resident layer, rows_per_block is data-space unfolding, and
    // data_time is the number of subgraphs serially reused by a persistent CTA.
    std::uint32_t         n1_log                  = 0;
    std::uint32_t         rows_per_block          = 0;
    std::uint32_t         threads_per_block       = 0;
    ComputeUnit           compute_unit            = ComputeUnit::Auto;
    HierarchicalCore      hierarchical_core       = HierarchicalCore::DataflowRadix4;
    // True HierarchicalDataflow schedule. stage_partition entries sum to log_n;
    // each entry is one logical reusable butterfly subgraph. Empty vectors
    // select a generated hardware mapping. There is one boundary per adjacent
    // pair; ResidentFused boundaries lower adjacent logical subgraphs into one
    // physical execution group without materializing a global handoff.
    std::vector<std::uint32_t>       stage_partition;
    std::vector<NttSubgraphMapping>  subgraph_mappings;
    std::vector<NttBoundaryMapping>  boundary_mappings;
    // Optional physical mappings after ResidentFused boundary lowering. The
    // count must equal the resulting execution-group count. Empty selects a
    // mapping derived from the logical subgraphs.
    std::vector<NttSubgraphMapping>  execution_group_mappings;
    // Maximum number of readiness counters inspected before retrying the
    // preferred task. Zero selects the generated scheduling policy.
    std::uint32_t                    ready_window = 0;
    // Publication identity for the packet-streaming 10+10 physical core.
    // WaveBitmap retains one bit per packet while polling four words per wave.
    PacketReadinessMode              packet_readiness_mode =
        PacketReadinessMode::PerPacket;
    // Thread-to-row mapping for online packet stages 0-1.
    PacketComputeLayout              packet_compute_layout =
        PacketComputeLayout::InterleavedRows;
    // Reuse the next wave's acquire barrier as the previous wave's completion
    // barrier, retaining only one final completion barrier.
    bool                             packet_fold_wave_barriers = false;
    // Enable intrusive per-role timing for the generated APPT online kernel.
    // This selects a separate diagnostic kernel and is not a performance mode.
    bool                             profile_appt_roles = false;
    std::uint32_t         word_bits               = 64;
    CrossTwiddlePlacement cross_twiddle_placement = CrossTwiddlePlacement::Fused;
    ModularMultiply       modular_multiply        = ModularMultiply::Shoup;
    InputOrder            input_order             = InputOrder::Natural;
    OutputOrder           output_order            = OutputOrder::Natural;
    ApptRoleMapping       appt_role_mapping;
    // Resolve the backend and physical unit from the calibrated runtime table.
    // Unsupported hardware or semantic contracts fail explicitly.
    bool                  auto_select              = false;
    // Allocate plan-owned scratch storage when the selected mapping requires it.
    // Disable this when the application wants to bind and reuse its own workspace.
    bool                  auto_allocate_workspace = true;
};

struct RunStats {
    double h2d_ms                    = 0.0;
    double kernel_ms                 = 0.0;
    // Host wall time around the event-timed launch region, per invocation.
    // The difference from kernel_ms bounds submission/event/sync overhead.
    double timed_region_wall_ms      = 0.0;
    double host_timing_overhead_ms   = 0.0;
    double d2h_ms                    = 0.0;
    double kernel_ntt_per_second     = 0.0;
    double end_to_end_ntt_per_second = 0.0;
    double kernel_points_per_second  = 0.0;
};

struct PipelineSegmentTrace {
    std::uint64_t start_globaltimer = 0;
    std::uint64_t end_globaltimer   = 0;
};

struct PipelineRoleMetrics {
    std::uint64_t wait_globaltimer     = 0;
    std::uint64_t compute_globaltimer  = 0;
    std::uint64_t boundary_globaltimer = 0;
    std::uint64_t tasks                = 0;
};

struct DeviceInfo {
    int         device_id = 0;
    std::string name;
    int         compute_major       = 0;
    int         compute_minor       = 0;
    std::size_t global_memory_bytes = 0;
    int         multiprocessors     = 0;
};

DeviceInfo current_device_info();

std::uint64_t mod_pow(std::uint64_t base, std::uint64_t exponent, std::uint64_t modulus);
bool          is_prime(std::uint64_t value);
std::uint64_t find_primitive_power_of_two_root(std::uint32_t log_n, std::uint64_t modulus);
void          reference_ntt(std::vector<std::uint64_t>& values, std::uint64_t modulus, bool inverse = false);

class Plan {
  public:
    explicit Plan(PlanConfig config);
    ~Plan();

    Plan(const Plan&)            = delete;
    Plan& operator=(const Plan&) = delete;
    Plan(Plan&&) noexcept;
    Plan& operator=(Plan&&) noexcept;

    const PlanConfig& config() const noexcept;
    const SelectionInfo& selection() const noexcept;
    std::size_t       points_per_transform() const noexcept;
    std::size_t       data_size() const noexcept;
    std::size_t       workspace_size() const noexcept;
    // Returns the static-layout contract for a compatible APPT plan. Throws
    // when this plan has no public static-layout ABI.
    ApptLayoutInfo     appt_layout_info() const;
    // Available after a HierarchicalDataflow execution has completed.
    std::vector<PipelineSegmentTrace> pipeline_trace() const;
    std::vector<PipelineRoleMetrics> pipeline_role_metrics() const;

    // The caller retains ownership of the stream and optional workspace. They must
    // remain valid until all previously submitted work has completed.
    void         set_stream(cudaStream_t stream) noexcept;
    cudaStream_t stream() const noexcept;
    void         set_workspace(void* workspace, std::size_t bytes);
    void*        workspace() const noexcept;

    // Allocation-free, asynchronous device-pointer execution. Input and output
    // must be distinct and match the configured word width.
    void execute_async(const std::uint32_t* input, std::uint32_t* output);
    void execute_async(const std::uint64_t* input, std::uint64_t* output);

    RunStats execute(const std::vector<std::uint64_t>& input, std::vector<std::uint64_t>& output, std::uint32_t warmup = 1, std::uint32_t repeat = 1);

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cuntt
