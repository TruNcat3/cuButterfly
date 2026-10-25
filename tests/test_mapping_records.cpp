#include <cubutterfly/mapping.hpp>
#include <cubutterfly/plan.hpp>
#include <iostream>
#include <stdexcept>
#include <set>
#include <string>
#include <cstdlib>
#include "../src/partition_space.hpp"
#include "../src/fft_factor_plan.hpp"
#include "../src/resident_mapping.hpp"
#include "../src/fwht_contract.hpp"

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() try {
    using namespace cuntt;
    MixedDataflowPlan descriptor;
    ExecutionGroup physical;
    physical.exchange = ExchangePolicy::WarpShuffle;
    physical.grid_ctas = 17;
    physical.live_shared_bytes = 12288;
    physical.static_shared_bytes = 4096;
    physical.dynamic_shared_bytes = 8192;
    physical.shared_components_known = true;
    descriptor.execution_groups.push_back(physical);
    const auto execution_json = cubutterfly::execution_groups_json(descriptor);
    require(execution_json.find("\"exchange\":1") != std::string::npos,
            "execution descriptor omitted physical exchange");
    require(execution_json.find("\"work_blocks\":17") != std::string::npos,
            "execution descriptor omitted physical work blocks");
    require(execution_json.find("\"static_shared_bytes\":4096") != std::string::npos &&
                execution_json.find("\"dynamic_shared_bytes\":8192") != std::string::npos &&
                execution_json.find("\"shared_components_known\":true") != std::string::npos,
            "execution descriptor omitted measured shared-memory components");
    MixedDataflowPlan estimated;
    ExecutionGroup estimated_group;
    estimated_group.live_shared_bytes = 16384;
    estimated.execution_groups.push_back(estimated_group);
    const auto estimated_json = cubutterfly::execution_groups_json(estimated);
    require(estimated_json.find("\"shared_components_known\":false") != std::string::npos &&
                estimated_json.find("\"dynamic_shared_bytes\"") == std::string::npos,
            "logical estimate fabricated dynamic shared-memory components");
    const auto partitions=detail::shared_partitions(6,3,0);
    std::set<std::vector<std::uint32_t>> expected;
    for(unsigned cuts=0;cuts<(1U<<5);++cuts) {
        std::vector<std::uint32_t> p; unsigned stages=1;
        for(unsigned edge=0;edge<5;++edge) {
            if(cuts&(1U<<edge)) { p.push_back(stages); stages=1; } else ++stages;
        }
        p.push_back(stages);
        if(*std::max_element(p.begin(),p.end())<=3) expected.insert(p);
    }
    require(std::set<std::vector<std::uint32_t>>(partitions.begin(),partitions.end())==expected,
            "partition cursor omitted a legal composition");
    const auto bounded=detail::shared_partitions(12,5,10);
    std::set<std::size_t> counts;
    for(const auto& p:bounded) counts.insert(p.size());
    require(counts.size()==10 && *counts.begin()==3 && *counts.rbegin()==12,
            "enumeration budget fixed the number of execution groups");
    const auto resource_partitions=detail::shared_partitions(24,10,4,5,true);
    require(std::find(resource_partitions.begin(),resource_partitions.end(),std::vector<std::uint32_t>{8,8,8})!=resource_partitions.end(),
            "balanced factors absent from bounded resource-constrained enumeration");
    const auto constrained=detail::shared_partitions(6,3,0,2,true);
    require(std::set<std::vector<std::uint32_t>>(constrained.begin(),constrained.end())==
            std::set<std::vector<std::uint32_t>>{{3,3},{2,2,2}},"factor minimum dropped or invented compositions");
    ButterflyConfig fft;
    fft.op = ButterflyOperator::Fft;
    fft.log_n = 20;
    fft.backend = ButterflyBackend::OnlineReorder;
    fft.fft_core = FftCore::RegisterTile;
    fft.stage_partition = {8, 12};
    fft.segment_mappings = {{FftCore::RegisterTile, LocalExchange::SharedMemory, 256, 16},
                            {FftCore::CufftDxBlock, LocalExchange::SharedMemory, 512, 8}};
    fft.execution_group_mappings = fft.segment_mappings;
    fft.prefix_codelet = "cufftdx-thread";
    fft.prefix_shared_layout = "xor";
    fft.segment_mappings[0].codelet = fft.prefix_codelet;
    fft.execution_group_mappings[0].codelet = fft.prefix_codelet;
    fft.boundaries = {{CrossTwiddleMode::Recurrence, DirectBoundary::PrefixTiledTranspose, FftBoundaryResidency::GlobalScratch}};
    const auto mapping = cubutterfly::serialize_mapping(fft);
    ButterflyConfig replay;
    replay.log_n = 20; replay.inverse = true; replay.batch = 7; replay.element_stride = 2;
    cubutterfly::apply_serialized_mapping(mapping, replay);
    require(cubutterfly::serialize_mapping(replay) == mapping, "FFT mapping roundtrip lost execution axes");
    require(replay.inverse && replay.batch == 7 && replay.element_stride == 2, "mapping overwrote mathematical semantics");
    require(replay.execution_group_mappings.back().core == FftCore::CufftDxBlock, "mixed core was flattened");
    require(replay.prefix_codelet == "cufftdx-thread" && replay.prefix_shared_layout == "xor" &&
            replay.execution_group_mappings.front().codelet == "cufftdx-thread",
            "register prefix strategy was lost in mapping replay");

    // Cooperative register prefixes keep the local shared tile fixed while
    // changing only the per-thread EPT and physical prefix launch width.
    auto cooperative = fft;
    cooperative.prefix_codelet_lanes = 2;
    cooperative.prefix_threads = 512; // R=16, G=2, Columns=16
    cooperative.prefix_ept = 8;
    cooperative.prefix_units_per_cta = 16;
    cooperative.segment_mappings[0].threads = cooperative.prefix_threads;
    cooperative.segment_mappings[0].ept = cooperative.prefix_ept;
    cooperative.execution_group_mappings[0].threads = cooperative.prefix_threads;
    cooperative.execution_group_mappings[0].ept = cooperative.prefix_ept;
    const auto cooperative_mapping = cubutterfly::serialize_mapping(cooperative);
    require(cooperative_mapping.find("\"prefix_codelet_lanes\":2") != std::string::npos,
            "cooperative prefix lane axis was omitted from mapping JSON");
    ButterflyConfig cooperative_replay;
    cooperative_replay.prefix_codelet_lanes = 7;
    cubutterfly::apply_serialized_mapping(cooperative_mapping, cooperative_replay);
    require(cooperative_replay.prefix_codelet_lanes == 2 && cooperative_replay.prefix_ept == 8 &&
                cooperative_replay.prefix_threads == 512,
            "cooperative prefix geometry was lost in mapping replay");
    ButterflyConfig legacy_mapping;
    legacy_mapping.prefix_codelet_lanes = 8;
    cubutterfly::apply_serialized_mapping("{\"kind\":\"butterfly\"}", legacy_mapping);
    require(legacy_mapping.prefix_codelet_lanes == 1,
            "legacy mapping without prefix_codelet_lanes did not default to one");

    ButterflyConfig register_shape;
    register_shape.op = ButterflyOperator::Fwht;
    register_shape.local_exchange = LocalExchange::WarpRegister;
    register_shape.register_threads = 64;
    register_shape.tile_threads = 64;
    auto register_replay = register_shape;
    register_replay.register_threads = 256;
    cubutterfly::apply_serialized_mapping(cubutterfly::serialize_mapping(register_shape), register_replay);
    require(register_replay.register_threads == 64 && register_replay.tile_threads == 64,
            "register resource request was lost in public mapping replay");
    cubutterfly::apply_serialized_mapping("{}", register_replay);
    require(register_replay.register_threads == 0, "legacy replay retained explicit register resources");
    for (const auto* invalid : {"{\"register_threads\":-1}", "{\"register_threads\":4294967360}",
                                "{\"register_threads\":64.5}"}) {
        bool failed = false;
        try { cubutterfly::apply_serialized_mapping(invalid, register_replay); }
        catch (const std::exception&) { failed = true; }
        require(failed, "invalid register resource encoding was accepted");
    }

    ButterflyConfig factor;
    factor.op=ButterflyOperator::Fft; factor.precision=ButterflyPrecision::Fp64;
    factor.backend=ButterflyBackend::FactorStreamed; factor.fft_core=FftCore::CufftDxBlock;
    factor.log_n=24; factor.stage_partition={24}; factor.factor_partition={8,8,8};
    factor.shared_layout=SharedLayout::WriterAligned; factor.cross_twiddle=CrossTwiddleMode::Recurrence;
    factor.data_tiles_per_cta=5; factor.prefetch_depth=2;
    factor.factor_io_policies={"static-unrolled", "dynamic", "static-unrolled"};
    detail::normalize_factor_mapping(factor);
    const auto factor_json=cubutterfly::serialize_mapping(factor);
    ButterflyConfig factor_replay=factor; factor_replay.batch=3; factor_replay.inverse=true;
    cubutterfly::apply_serialized_mapping(factor_json,factor_replay);
    detail::normalize_factor_mapping(factor_replay);
    require(cubutterfly::serialize_mapping(factor_replay)==factor_json,"factor replay changed independent axes");
    const auto ir=make_dataflow_plan(factor);
    require(ir.stage_partition.size()==1 && ir.execution_groups.size()==3 && ir.execution_boundaries.size()==2,
            "macro group incorrectly implies a single physical launch");
    require(ir.execution_groups[0].grid_ctas==1639 && ir.execution_groups[0].live_shared_bytes==131072,
            "data folding and prefetch capacity projection disagree");
    require(ir.execution_groups[0].io_policy == "static-unrolled" &&
            ir.execution_groups[1].io_policy == "dynamic" &&
            ir.execution_groups[2].io_policy == "static-unrolled",
            "factor I/O policy was lost from the physical execution descriptor");
    auto regrouped=factor; regrouped.stage_partition={8,16}; regrouped.boundaries.clear();
    const auto grouped_ir=make_dataflow_plan(regrouped);
    require(grouped_ir.execution_groups[1].logical_macro_group==1 && grouped_ir.execution_groups[2].logical_macro_group==1,
            "macro labels lost physical factor ownership");
    bool rejected=false;
    try { auto bad=factor; bad.stage_partition={10,14}; detail::normalize_factor_mapping(bad); }
    catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"factor spanning macro boundary was silently accepted");
    auto native=factor; native.fft_core=FftCore::RegisterTile; native.execution_group_mappings.clear();
    detail::normalize_factor_mapping(native);
    const auto native_ir=make_dataflow_plan(native);
    require(native_ir.execution_groups[1].core=="register-tile" && native_ir.execution_boundaries.size()==2,
            "local core substitution changed factor boundaries");
    auto sliced=factor; sliced.factor_slices=4; sliced.factor_overlap=true;
    detail::normalize_factor_mapping(sliced);
    auto sliced_replay=factor;
    cubutterfly::apply_serialized_mapping(cubutterfly::serialize_mapping(sliced),sliced_replay);
    require(sliced_replay.factor_slices==4 && sliced_replay.factor_overlap,"factor schedule lost in replay");
    const auto sliced_ir=make_dataflow_plan(sliced);
    require(sliced_ir.execution_groups[0].launch_count==4 && sliced_ir.execution_groups[1].partial_dependency_ready &&
            !sliced_ir.execution_groups[2].partial_dependency_ready && sliced_ir.execution_groups[2].launch_count==1,
            "partial readiness or final fan-in is misreported");
    const auto r0=detail::factor_tile_range(sliced,0,1),r1=detail::factor_tile_range(sliced,1,1);
    require(r0.first==8 && r0.span==8 && r0.period==32 && r0.count==2048 &&
            r1.first==2048 && r1.span==2048 && r1.period==8192 && r1.count==2048,
            "slice digit did not move above processed digits");

    PlanConfig ntt;
    ntt.stage_partition = {7, 7, 6};
    ntt.data_space = 4; ntt.data_time = 3;
    ntt.stage_overlap = true; ntt.batch_tile_count = 7;
    ntt.subgraph_mappings.resize(3);
    ntt.subgraph_mappings[1].data_space = 8;
    ntt.subgraph_mappings[2].cta_weight = 5;
    ntt.execution_group_mappings = ntt.subgraph_mappings;
    ntt.appt_role_mapping.fragment_width = 8;
    ntt.boundary_mappings = {{BoundaryStorage::FullScratch, 2}, {BoundaryStorage::FullScratch, 1}};
    PlanConfig replay_ntt;
    replay_ntt.modulus = 998244353; replay_ntt.inverse = true;
    const auto ntt_mapping = cubutterfly::serialize_mapping(ntt);
    cubutterfly::apply_serialized_mapping(ntt_mapping, replay_ntt);
    require(cubutterfly::serialize_mapping(replay_ntt) == ntt_mapping, "NTT mapping roundtrip lost execution axes");
    require(replay_ntt.inverse && replay_ntt.modulus == 998244353, "NTT mapping overwrote semantics");
    for (const auto& invalid : {"{\"schema_version\":99}", "{\"tile_threads\":-1}", "{\"kind\":\"ntt\"}"}) {
        bool rejected = false;
        try { cubutterfly::apply_serialized_mapping(invalid, replay); } catch (const std::exception&) { rejected = true; }
        require(rejected, "malformed mapping accepted");
    }
    ButterflyConfig resident;
    resident.op=ButterflyOperator::Fwht; resident.backend=ButterflyBackend::SharedIterative;
    resident.log_n=11; resident.stage_partition={6,5}; resident.tile_threads=64;
    resident.local_stage_partitions={{2,4},{3,2}}; resident.exchange_chunks={0,0};
    detail::validate_shared_resident_axes(resident.local_stage_partitions,resident.exchange_chunks,resident.stage_partition);
    auto resident_copy=resident;
    cubutterfly::apply_serialized_mapping(cubutterfly::serialize_mapping(resident),resident_copy);
    require(resident_copy.local_stage_partitions==resident.local_stage_partitions,"resident partitions lost in replay");
    const auto resident_ir=make_dataflow_plan(resident);
    require(resident_ir.execution_groups[1].local_stage_partition==std::vector<std::uint32_t>({3,2}) &&
            resident_ir.execution_groups[0].codelet=="native-register-subgraph" &&
            resident_ir.execution_groups[0].elements_per_thread==16,"resident execution descriptor is incomplete");
    cubutterfly::apply_serialized_mapping("{}",resident_copy);
    require(resident_copy.local_stage_partitions.empty() && resident_copy.exchange_chunks.empty(),"legacy replay retained new axes");
    ntt.local_stage_partitions={{3,4},{2,5},{1,5}}; ntt.exchange_chunks={0,0,0};
    cubutterfly::apply_serialized_mapping(cubutterfly::serialize_mapping(ntt),replay_ntt);
    require(replay_ntt.local_stage_partitions==ntt.local_stage_partitions,"NTT resident axes lost in replay");
    for(const auto& invalid:{"{\"local_stage_partitions\":[[-1,2]]}","{\"local_stage_partitions\":[3]}","{\"exchange_chunks\":[-1]}"}) {
        bool rejected=false;
        try { cubutterfly::apply_serialized_mapping(invalid,resident_copy); } catch(const std::exception&) { rejected=true; }
        require(rejected,"invalid resident axis accepted");
    }
    bool rejected_chunk=false;
    try {detail::validate_shared_resident_axes({{3,3}}, {4}, {6});} catch(const std::invalid_argument&) {rejected_chunk=true;}
    require(rejected_chunk,"ignored shared output chunk accepted");
    std::cout << "Complete mapping roundtrip and semantic isolation passed\n";
    ButterflyConfig epilogue;
    epilogue.op = ButterflyOperator::Fwht;
    epilogue.precision = ButterflyPrecision::Fp16;
    epilogue.accumulation = ButterflyAccumulation::Fp32;
    epilogue.final_storage_rounding = true;
    epilogue.output_scale = 0.125;
    for (auto log_n : {5U, 6U}) {
        auto domain = epilogue;
        domain.log_n = log_n;
        domain.placement = ButterflyPlacement::OutOfPlace;
        require(detail::short_fp16_fwht_supported(domain), "final-storage short candidate absent");
        auto stage = domain;
        stage.final_storage_rounding = false;
        stage.output_scale = 1.0;
        require(!detail::short_fp16_fwht_supported(stage), "stage-storage admitted final-only short candidate");
        domain.inverse = true;
        require(!detail::short_fp16_fwht_supported(domain), "inverse short candidate admitted");
        domain.inverse = false;
        domain.element_stride = 2;
        require(!detail::short_fp16_fwht_supported(domain), "strided short candidate admitted");
        domain.element_stride = 1;
        domain.accumulation = ButterflyAccumulation::Native;
        require(!detail::short_fp16_fwht_supported(domain), "native accumulation short candidate admitted");
    }
    auto epilogue_copy = epilogue;
    const auto epilogue_mapping = cubutterfly::serialize_mapping(epilogue);
    cubutterfly::apply_serialized_mapping(epilogue_mapping, epilogue_copy);
    require(cubutterfly::serialize_mapping(epilogue_copy) == epilogue_mapping, "output epilogue mapping roundtrip");
    require(epilogue_mapping.find("promoted-full-fp32-final-storage") != std::string::npos,
            "promoted fallback metadata missing");
    epilogue_copy.output_scale = 0.25;
    bool rejected_epilogue = false;
    try { cubutterfly::apply_serialized_mapping(epilogue_mapping, epilogue_copy); }
    catch (const std::invalid_argument&) { rejected_epilogue = true; }
    require(rejected_epilogue, "mapping accepted conflicting output scale");

    // Constrained research re-search must classify mechanisms from the
    // serialized mapping, rather than relying on a hand-written candidate id.
    auto overlap = epilogue;
    overlap.stage_overlap = true;
    require(detail::mapping_uses_disabled_mechanism(
                cubutterfly::serialize_mapping(overlap), {"stage-overlap"}),
            "stage-overlap restriction did not match the mapping");
    auto fused = resident;
    fused.boundaries = {{CrossTwiddleMode::Table, DirectBoundary::Strided,
                         FftBoundaryResidency::Fused}};
    require(detail::mapping_uses_disabled_mechanism(
                cubutterfly::serialize_mapping(fused), {"resident-fusion"}),
            "resident-fusion restriction did not match the mapping");
    auto successor = resident;
    successor.shared_layout = SharedLayout::WriterAligned;
    require(detail::mapping_uses_disabled_mechanism(
                cubutterfly::serialize_mapping(successor), {"successor-layout"}),
            "successor-layout restriction did not match the mapping");
    successor.shared_layout = SharedLayout::XorSwizzle;
    require(detail::mapping_uses_disabled_mechanism(
                cubutterfly::serialize_mapping(successor), {"successor-layout"}),
            "xor successor-layout restriction did not match the mapping");
    unsetenv("CUBUTTERFLY_SEARCH_DISABLE");
    require(detail::search_disabled_mechanisms().empty(), "unset search restriction was not empty");
    setenv("CUBUTTERFLY_SEARCH_DISABLE", "stage-overlap, successor-layout,stage-overlap", 1);
    const auto restrictions = detail::search_disabled_mechanisms();
    require(restrictions == std::vector<std::string>({"stage-overlap", "successor-layout"}),
            "search restriction parsing did not preserve unique order");
    setenv("CUBUTTERFLY_SEARCH_DISABLE", "unknown-mechanism", 1);
    bool rejected_restriction = false;
    try { (void)detail::search_disabled_mechanisms(); }
    catch (const std::invalid_argument&) { rejected_restriction = true; }
    unsetenv("CUBUTTERFLY_SEARCH_DISABLE");
    require(rejected_restriction, "unknown search restriction was accepted");
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
