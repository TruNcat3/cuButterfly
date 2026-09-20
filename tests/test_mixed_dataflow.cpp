#include "cuntt/butterfly.hpp"
#include "cuntt/appt_layout.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename F>
void rejects(F operation) {
    try { operation(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid input was accepted");
}
}  // namespace

int main() try {
    using namespace cuntt;
    bool shared_advertised=false;
    for (const auto& capability : butterfly_capabilities())
        if (capability.backend==ButterflyBackend::SharedIterative) shared_advertised=capability.max_log_n==30;
    require(shared_advertised,"common shared lowering missing from capability inventory");
    ButterflyConfig large_reference;
    large_reference.op=ButterflyOperator::Fft; large_reference.backend=ButterflyBackend::CuFft; large_reference.log_n=24;
    require(check_lowering(make_dataflow_plan(large_reference))==LoweringStatus::RequiresBackendValidation,
            "large cuFFT comparison was limited by an internal core's size cap");
    for (const auto shape : {std::pair<std::uint32_t, std::uint32_t>{64, 64},
                             std::pair<std::uint32_t, std::uint32_t>{64, 512}}) {
        const auto total = shape.first * shape.second;
        constexpr std::uint32_t parallelism = 4;
        for (std::uint32_t n_part : {shape.second, shape.first}) {
            std::vector<bool> seen(total, false);
            for (std::uint32_t logical = 0; logical < total; ++logical) {
                const auto physical = appt_physical_index(logical, total, n_part, parallelism);
                require(physical < total, "APPT physical index escaped tile domain");
                require(!seen[physical], "APPT mapping is not bijective");
                seen[physical] = true;
                require(appt_logical_index(physical, total, n_part, parallelism) == logical,
                        "APPT inverse mapping mismatch");
            }
        }
    }
    HardwareResourceModel hardware;
    hardware.sm_count = 108;
    hardware.max_threads_per_block = 1024;
    hardware.max_threads_per_sm = 2048;
    hardware.registers_per_sm = 65536;
    hardware.shared_bytes_per_sm = 163840;
    hardware.max_blocks_per_sm = 32;
    hardware.shared_bytes_per_block = 163840;
    hardware.registers_per_block = 65536;
    hardware.cooperative_launch = true;

    ButterflyConfig factor;
    factor.op=ButterflyOperator::Fft; factor.precision=ButterflyPrecision::Fp64;
    factor.backend=ButterflyBackend::FactorStreamed; factor.fft_core=FftCore::CufftDxBlock;
    factor.log_n=24; factor.stage_partition={24}; factor.factor_partition={8,8,8};
    factor.shared_layout=SharedLayout::WriterAligned; factor.cross_twiddle=CrossTwiddleMode::Recurrence;
    factor.data_tiles_per_cta=5; factor.prefetch_depth=2;
    auto factor_plan=make_dataflow_plan(factor);
    require(validate_plan(factor_plan,hardware),factor_plan.rejection_reason.c_str());
    const auto factor_estimate=estimate_resources(factor_plan,hardware);
    require(factor_estimate.local_bytes==98304 && factor_estimate.resident_ctas==1 &&
            factor_estimate.boundary_bytes==(1ULL<<24)*16*4 && !factor_estimate.complete,
            "factor resource estimate flattened prefetch memory or physical HBM edges");
    factor_plan.execution_groups.back().compiler_registers_per_thread=255;
    factor_plan.execution_groups.back().threads=1024;
    require(!estimate_resources(factor_plan,hardware).feasible,"compiler allocation on a later factor was ignored");

    ButterflyConfig iterative;
    iterative.op = ButterflyOperator::Fft;
    iterative.backend = ButterflyBackend::SharedIterative;
    iterative.log_n = 12; iterative.batch = 3;
    iterative.tile_threads = 128;
    iterative.stage_partition = {3, 4, 5};
    iterative.shared_layout = SharedLayout::WriterAligned;
    const auto groups = make_dataflow_plan(iterative);
    require(groups.execution_groups.size() == 3 && groups.execution_boundaries.size() == 2,
            "physical groups or materialized edges were flattened");
    require(groups.execution_groups[2].first_stage == 7 && groups.execution_groups[2].stage_count == 5,
            "execution-group logical interval lost");
    require(groups.execution_groups[0].live_shared_bytes == 128 && groups.execution_groups[2].live_shared_bytes == 512,
            "unequal group footprints were flattened");
    require(groups.execution_groups[0].grid_ctas == 1536 && groups.execution_groups[2].grid_ctas == 384,
            "per-group grid parallelism lost");
    require(!groups.execution_groups[0].compiler_resources_known && groups.state == CandidateState::NeedsLowering,
            "abstract group invented compiler or execution evidence");

    PlanConfig ntt_graph;
    ntt_graph.log_n = 12; ntt_graph.compute_unit = ComputeUnit::Radix4;
    ntt_graph.stage_partition = {6, 6}; ntt_graph.data_space = 2;
    const auto ntt_projection = make_dataflow_plan(ntt_graph);
    require(ntt_projection.graph.stage_arities == std::vector<std::uint32_t>(12, 2),
            "NTT codelet radix changed the binary logical graph");
    const auto ntt_estimate = estimate_resources(ntt_projection, hardware);
    require(ntt_estimate.local_bytes == ntt_projection.shared_bytes_per_subgraph * 2ULL,
            "NTT parallel instances were multiplied twice");

    for (const auto op : {ButterflyOperator::Fft, ButterflyOperator::Fwht, ButterflyOperator::SubsetZeta,
                          ButterflyOperator::SupersetZeta, ButterflyOperator::XorZeta, ButterflyOperator::Structured2x2}) {
        ButterflyConfig config;
        config.op = op;
        config.backend = ButterflyBackend::OnlineReorder;
        config.log_n = 20;
        config.local_stages = 10;
        config.batch = 4;
        config.compute_unit = ComputeUnit::Radix4;
        auto plan = make_dataflow_plan(config);
        require(plan.stage_partition == std::vector<std::uint32_t>({10, 10}), "implicit online partition lost");
        require(plan.boundary_policies == std::vector<BoundaryPolicy>({BoundaryPolicy::GlobalScratch}), "online edge lost");
        require(plan.dispatch == DataflowDispatch::OnlineReorder, "wrong online route");
        require(plan.batch_tile_count == 1 && plan.steady_state_buffers == 1,
                "online lowering must expose bulk batch schedule");
        require(plan.inter_stage_global_bytes == (1ULL << 20) * 4 * (op == ButterflyOperator::Fft ? 8ULL : 4ULL),
                "batch boundary traffic is not represented");
        require(plan.graph.stage_arities == std::vector<std::uint32_t>(20, 2), "radix changed logical graph");
        require(plan.processing_radix == 4, "processing radix lost");
        require(plan.data_time == 1 && !plan.persistent, "invented data-time/persistence");
        require(check_lowering(plan) == LoweringStatus::RequiresBackendValidation, "invented compiled support");
        require(validate_plan(plan, hardware), plan.rejection_reason.c_str());
        require(!plan.executable, "resource validation must not certify execution");
        const auto estimate = estimate_resources(plan, hardware);
        const auto element_bytes = op == ButterflyOperator::Fft ? 8ULL : 4ULL;
        require(estimate.boundary_bytes == (1ULL << 20) * 4 * element_bytes * 2, "incorrect edge traffic");
        require(!estimate.complete, "unknown compiler allocation must remain incomplete");
    }

    ButterflyConfig overlap_config;
    overlap_config.op = ButterflyOperator::Fft;
    overlap_config.backend = ButterflyBackend::OnlineReorder;
    overlap_config.fft_core = FftCore::CufftDxBlock;
    overlap_config.log_n = 20;
    overlap_config.local_stages = 8;
    overlap_config.batch = 17;
    overlap_config.stage_overlap = true;
    overlap_config.batch_tile_count = 4;
    const auto overlap_plan = make_dataflow_plan(overlap_config);
    require(overlap_plan.dispatch == DataflowDispatch::BatchPipeline, "pipeline schedule lost");
    require(overlap_plan.batch_tile_count == 4 && overlap_plan.steady_state_buffers == 2,
            "bounded ring schedule lost");
    require(!overlap_plan.persistent && !overlap_plan.executable, "pipeline projection invents execution or persistence");
    require(overlap_plan.stage_partition == std::vector<std::uint32_t>({8, 12}), "pipeline changed partition");
    auto register_config=overlap_config;
    register_config.fft_core=FftCore::RegisterTile;
    register_config.prefix_threads=128; register_config.prefix_ept=16;
    register_config.suffix_threads=512; register_config.suffix_ept=16;
    const auto register_plan=make_dataflow_plan(register_config);
    require(register_plan.execution_groups[0].grid_ctas==2048 && register_plan.execution_groups[1].grid_ctas==512,
            "register FFT grid must account for columns grouped in one CTA");
    require(register_plan.execution_groups[0].live_shared_bytes==16384 && register_plan.execution_groups[1].live_shared_bytes==65536,
            "register FFT group state must include parallel columns");
    require(register_plan.execution_groups[0].data_space==128 && register_plan.execution_groups[0].data_time==8 &&
            register_plan.execution_groups[1].data_space==512 && register_plan.execution_groups[1].data_time==8,
            "grouped FFT data-time must count butterflies in all columns of a CTA");
    require(register_plan.execution_groups[1].core=="cufftdx-block" && register_plan.execution_boundaries[0].storage==BoundaryPolicy::RingBuffer,
            "register FFT suffix core or ring schedule lost");
    iterative.stage_overlap=true; iterative.batch_tile_count=2; iterative.batch=7;
    const auto streamed=make_dataflow_plan(iterative);
    require(streamed.execution_groups[0].grid_ctas==1024 && streamed.execution_groups[0].batch_time==4,
            "pipeline launch size was confused with number of batch tiles");
    require(streamed.execution_boundaries[1].storage==BoundaryPolicy::RingBuffer &&
            streamed.execution_boundaries[1].bytes==2*4096*8 && streamed.execution_boundaries[1].buffers==2,
            "per-edge ring capacity was confused with full-batch traffic");
    require(estimate_resources(streamed,hardware).boundary_bytes==7*4096*8*2*2,
            "streaming incorrectly removed global boundary traffic");
    PlanConfig streamed_ntt;
    streamed_ntt.backend=Backend::SharedIterative; streamed_ntt.log_n=12;
    streamed_ntt.stage_partition={3,4,5}; streamed_ntt.batch=7;
    streamed_ntt.stage_overlap=true; streamed_ntt.batch_tile_count=2;
    const auto ntt_stream=make_dataflow_plan(streamed_ntt);
    require(ntt_stream.dispatch==DataflowDispatch::BatchPipeline && ntt_stream.execution_groups[2].grid_ctas==256,
            "NTT stream mapping lost");
    require(check_lowering(ntt_stream)==LoweringStatus::RequiresBackendValidation,
            "NTT ring schedule did not reach backend validation");

    ButterflyConfig config;
    config.backend = ButterflyBackend::TemporalTile;
    config.log_n = 8;
    config.precision = ButterflyPrecision::Bf16;
    config.accumulation = ButterflyAccumulation::Fp32;
    config.element_stride = 3;
    config.batch_stride = 2048;
    auto plan = make_dataflow_plan(config);
    require(plan.graph.word_bits == 16 && plan.graph.accumulator_bits == 32, "low precision lost");
    require(plan.graph.precision == "bf16", "BF16 confused with FP16");
    require(plan.graph.element_stride == 3 && plan.graph.batch_stride == 2048, "layout lost");
    require(plan.subgraph_log_n == 8 && plan.shared_bytes_per_subgraph == 512, "tile footprint incorrect");
    require(plan.dispatch == DataflowDispatch::TemporalTile, "wrong temporal route");

    config.backend = ButterflyBackend::Hierarchical;
    config.log_n = 12;
    config.local_stages = 8;
    plan = make_dataflow_plan(config);
    require(plan.stage_partition == std::vector<std::uint32_t>({8, 1, 1, 1, 1}), "hierarchical launches misrepresented");
    require(plan.boundary_policies.size() == 4 && !plan.persistent, "hierarchical boundary/persistence incorrect");
    require(plan.dispatch == DataflowDispatch::Hierarchical, "wrong hierarchical route");

    // Generic hierarchical launches have one fixed 256-thread radix-2
    // suffix grid.  The deliberately stale FFT core below must not alter the
    // typed non-FFT route or its physical metadata.
    ButterflyConfig generic_hierarchical;
    generic_hierarchical.op = ButterflyOperator::Fwht;
    generic_hierarchical.backend = ButterflyBackend::Hierarchical;
    generic_hierarchical.log_n = 8;
    generic_hierarchical.batch = 5;
    generic_hierarchical.local_stages = 5;
    generic_hierarchical.tile_threads = 128;
    generic_hierarchical.fft_core = FftCore::CufftDxBlock;
    auto generic_hierarchical_plan = make_dataflow_plan(generic_hierarchical);
    require(generic_hierarchical_plan.dispatch == DataflowDispatch::Hierarchical &&
            generic_hierarchical_plan.processing_unit == "scalar",
            "non-FFT hierarchical route inherited an FFT processing unit");
    require(generic_hierarchical_plan.execution_groups.size() == 4,
            "generic hierarchical physical stages were flattened");
    const auto& generic_prefix = generic_hierarchical_plan.execution_groups.front();
    const auto& generic_tail = generic_hierarchical_plan.execution_groups.back();
    require(generic_prefix.core == "scalar" && generic_prefix.threads == 128 &&
            generic_prefix.grid_ctas == 40 && generic_prefix.live_shared_bytes == 128 &&
            generic_prefix.elements_per_thread == 0 &&
            generic_prefix.exchange == ExchangePolicy::SharedMemory &&
            generic_prefix.data_space == 16 && generic_prefix.data_time == 1,
            "generic hierarchical prefix launch metadata disagrees with the typed kernel");
    require(generic_tail.core == "scalar" && generic_tail.threads == 256 &&
            generic_tail.grid_ctas == 3 && generic_tail.live_shared_bytes == 0 &&
            generic_tail.elements_per_thread == 0 &&
            generic_tail.exchange == ExchangePolicy::GlobalMemory &&
            generic_tail.stage_space == 1 && generic_tail.data_space == 256 &&
            generic_tail.data_time == 1,
            "generic hierarchical suffix did not use ceil grid and fixed radix-2 work");

    ButterflyConfig structured_hierarchical;
    structured_hierarchical.op = ButterflyOperator::Structured2x2;
    structured_hierarchical.backend = ButterflyBackend::Hierarchical;
    structured_hierarchical.log_n = 7;
    structured_hierarchical.batch = 3;
    structured_hierarchical.local_stages = 5;
    structured_hierarchical.tile_threads = 256;
    structured_hierarchical.compute_unit = ComputeUnit::Radix4;
    const auto structured_hierarchical_plan = make_dataflow_plan(structured_hierarchical);
    require(structured_hierarchical_plan.execution_groups.front().grid_ctas == 12 &&
            structured_hierarchical_plan.execution_groups.front().live_shared_bytes == 128,
            "generic structured prefix shape lost its tile launch");
    require(structured_hierarchical_plan.execution_groups.back().grid_ctas == 1 &&
            structured_hierarchical_plan.execution_groups.back().threads == 256 &&
            structured_hierarchical_plan.execution_groups.back().exchange == ExchangePolicy::GlobalMemory &&
            structured_hierarchical_plan.execution_groups.back().data_space == 192,
            "generic structured suffix shape lost its fixed launch");

    ButterflyConfig generic_online;
    generic_online.op = ButterflyOperator::SubsetZeta;
    generic_online.backend = ButterflyBackend::OnlineReorder;
    generic_online.log_n = 10;
    generic_online.batch = 3;
    generic_online.local_stages = 5;
    generic_online.tile_threads = 128;
    generic_online.reorder_columns = 8;
    generic_online.fft_core = FftCore::CufftDxBlock;
    const auto generic_online_plan = make_dataflow_plan(generic_online);
    require(generic_online_plan.dispatch == DataflowDispatch::OnlineReorder &&
            generic_online_plan.processing_unit == "scalar" &&
            generic_online_plan.execution_groups.size() == 2,
            "non-FFT online route inherited an external FFT dispatch");
    require(generic_online_plan.execution_groups[0].core == "scalar" &&
            generic_online_plan.execution_groups[0].threads == 128 &&
            generic_online_plan.execution_groups[0].grid_ctas == 96 &&
            generic_online_plan.execution_groups[0].live_shared_bytes == 128 &&
            generic_online_plan.execution_groups[0].exchange == ExchangePolicy::SharedMemory &&
            generic_online_plan.execution_groups[0].data_space == 16 &&
            generic_online_plan.execution_groups[0].data_time == 1,
            "generic online prefix launch metadata disagrees with the typed kernel");
    require(generic_online_plan.execution_groups[1].core == "scalar" &&
            generic_online_plan.execution_groups[1].threads == 128 &&
            generic_online_plan.execution_groups[1].grid_ctas == 12 &&
            generic_online_plan.execution_groups[1].live_shared_bytes == 1024 &&
            generic_online_plan.execution_groups[1].exchange == ExchangePolicy::SharedMemory &&
            generic_online_plan.execution_groups[1].data_space == 128 &&
            generic_online_plan.execution_groups[1].data_time == 1,
            "generic online suffix column/shared metadata disagrees with the typed kernel");

    ButterflyConfig scalar_fft_online;
    scalar_fft_online.op = ButterflyOperator::Fft;
    scalar_fft_online.backend = ButterflyBackend::OnlineReorder;
    scalar_fft_online.log_n = 9;
    scalar_fft_online.batch = 5;
    scalar_fft_online.local_stages = 5;
    scalar_fft_online.tile_threads = 64;
    scalar_fft_online.reorder_columns = 4;
    const auto scalar_fft_online_plan = make_dataflow_plan(scalar_fft_online);
    require(scalar_fft_online_plan.dispatch == DataflowDispatch::OnlineReorder &&
            scalar_fft_online_plan.execution_groups[0].core == "scalar" &&
            scalar_fft_online_plan.execution_groups[1].core == "scalar" &&
            scalar_fft_online_plan.execution_groups[0].grid_ctas == 80 &&
            scalar_fft_online_plan.execution_groups[1].grid_ctas == 40 &&
            scalar_fft_online_plan.execution_groups[0].live_shared_bytes == 256 &&
            scalar_fft_online_plan.execution_groups[1].live_shared_bytes == 512,
            "scalar FFT online metadata did not preserve complex generic launches");

    config.backend = ButterflyBackend::StagePipeline;
    config.log_n = 8;
    config.tile_threads = 0;
    config.pipeline_warps = 4;
    plan = make_dataflow_plan(config);
    require(plan.threads_per_block == 128 && plan.data_time == 1, "pipeline warp count is not data-time");
    config.backend = ButterflyBackend::WarpHybrid;
    require(make_dataflow_plan(config).threads_per_block == 256, "warp-hybrid thread shape lost");

    config.backend = ButterflyBackend::TemporalTile;
    config.local_exchange = LocalExchange::WarpRegister;
    require(make_dataflow_plan(config).dispatch == DataflowDispatch::WarpRegister, "register route lost");
    config.local_exchange = LocalExchange::SharedMemory;
    config.op = ButterflyOperator::Fft;
    config.fft_core = FftCore::CufftDxBlock;
    config.backend = ButterflyBackend::OnlineReorder;
    config.log_n = 18;
    config.local_stages = 8;
    config.tile_threads = 256;
    plan = make_dataflow_plan(config);
    require(plan.dispatch == DataflowDispatch::ExternalFft, "codelet route lost");
    require(plan.fused_twiddle, "table twiddle can also be fused");
    plan.backend = "invented-backend";
    require(check_lowering(plan) == LoweringStatus::MissingBackend, "unknown backend accepted");
    plan.backend = "hybrid-dataflow";
    require(check_lowering(plan) == LoweringStatus::RequiresBackendValidation, "generic backend route rejected");
    plan = make_dataflow_plan(config);
    plan.processing_unit = "cufftdx-resident";
    require(check_lowering(plan) == LoweringStatus::UnsupportedSize, "unimplemented large resident FFT accepted");

    config.stage_partition = {0, 18};
    rejects([&] { make_dataflow_plan(config); });
    config.stage_partition = {8, 8};
    rejects([&] { make_dataflow_plan(config); });
    config.stage_partition = {8, 6, 4};
    config.boundaries = {{CrossTwiddleMode::Table, DirectBoundary::Strided, FftBoundaryResidency::Fused},
                         {CrossTwiddleMode::Table, DirectBoundary::Strided, FftBoundaryResidency::GlobalScratch}};
    plan = make_dataflow_plan(config);
    require(plan.boundary_policies == std::vector<BoundaryPolicy>({BoundaryPolicy::ResidentFused, BoundaryPolicy::GlobalScratch}),
            "mixed boundaries collapsed to first edge");

    plan.backend = "online-reorder";
    plan.graph = make_operator_graph(DataflowOperator::Ntt, 18);
    require(check_lowering(plan) == LoweringStatus::RequiresBackendValidation, "NTT projection rejected");
    PlanConfig ntt_config;
    ntt_config.log_n = 20;
    ntt_config.batch = 4;
    ntt_config.word_bits = 64;
    ntt_config.backend = Backend::HierarchicalDataflow;
    ntt_config.compute_unit = ComputeUnit::Radix4;
    ntt_config.stage_partition = {8, 6, 6};
    ntt_config.data_space = 4;
    ntt_config.data_time = 2;
    ntt_config.pipeline_buffers = 2;
    ntt_config.boundary_mappings = {{BoundaryStorage::ResidentFused, 0}};
    ntt_config.boundary_mappings.push_back({BoundaryStorage::FullScratch, 2});
    auto ntt_plan = make_dataflow_plan(ntt_config);
    require(ntt_plan.graph.op == DataflowOperator::Ntt && ntt_plan.graph.word_bits == 64, "NTT graph projection lost");
    require(ntt_plan.backend == "hierarchical-dataflow" && ntt_plan.persistent, "NTT backend projection lost");
    require(ntt_plan.boundary_policies == std::vector<BoundaryPolicy>({BoundaryPolicy::ResidentFused, BoundaryPolicy::GlobalScratch}),
            "NTT boundary projection lost");
    require(ntt_plan.data_space == 4 && ntt_plan.data_time == 2 && ntt_plan.pipeline_buffers == 2,
            "NTT unfolding projection lost");
    require(validate_plan(ntt_plan, hardware), ntt_plan.rejection_reason.c_str());
    require(!ntt_plan.executable, "NTT plan validation certified a backend launch");
    rejects([] { make_operator_graph(DataflowOperator::Fft, 31); });
    rejects([] { make_operator_graph(DataflowOperator::Fwht, 8, 0); });

    plan = MixedDataflowPlan{};
    plan.graph = make_operator_graph(DataflowOperator::Fwht, 12);
    plan.threads_per_block = 256;
    plan.shared_bytes_per_subgraph = 32768;
    plan.registers_per_thread = 64;
    plan.grid_ctas = 432;
    auto estimate = estimate_resources(plan, hardware);
    require(estimate.resident_ctas == 4 && estimate.grid_waves == 1, "grid waves must include SM count");
    plan.target_resident_ctas = 5;
    require(!estimate_resources(plan, hardware).feasible, "unattainable CTA target accepted");
    plan.target_resident_ctas = 0;
    hardware.shared_bytes_per_block = 16384;
    require(!estimate_resources(plan, hardware).feasible, "per-block shared limit ignored");
    hardware.shared_bytes_per_block = 163840;
    plan.shared_bytes_per_subgraph = std::numeric_limits<std::uint32_t>::max();
    plan.data_space = 2;
    require(!estimate_resources(plan, hardware).feasible, "shared multiplication overflow accepted");
    plan.graph.log_n = 64;
    require(!estimate_resources(plan, hardware).feasible, "undefined shift accepted");

    std::cout << "PASS mixed-dataflow projection, routes, resource bounds and rejection tests\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
