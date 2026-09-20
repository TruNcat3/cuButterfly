#include "fft_factor_plan.hpp"
#if __has_include("jit_config.hpp")
#include "jit_config.hpp"
#define CUBUTTERFLY_HAS_JIT_CONFIG 1
#endif
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace cuntt::detail {
namespace {
bool factor_jit_enabled() {
    const auto* requested = std::getenv("CUBUTTERFLY_COMPILE_MODE");
    const std::string mode = requested ? requested :
#ifdef CUBUTTERFLY_HAS_JIT_CONFIG
        CUBUTTERFLY_DEFAULT_CODEGEN_MODE;
#else
        "on-demand";
#endif
    return mode == "research" || mode == "on-demand" || mode == "auto";
}
}

FactorTileRange factor_tile_range(const ButterflyConfig& c,unsigned group,unsigned slice) {
    if(group>=c.factor_partition.size() || slice>=c.factor_slices)
        throw std::invalid_argument("factor range index out of bounds");
    const auto total=(std::uint64_t(c.batch)<<(c.log_n-c.factor_partition[group]))/c.factor_columns;
    if(group+1==c.factor_partition.size() || c.factor_slices==1) return {0,total,total,total};
    // The unprocessed tail digit is s % K in j=s*P+p, with P=2^done.
    // Its address position shifts above P after each completed factor.
    unsigned done=0;
    for(unsigned g=0;g<group;++g) done+=c.factor_partition[g];
    const auto period=(1ULL<<(c.factor_partition.back()+done))/c.factor_columns;
    const auto span=period/c.factor_slices;
    return {span*slice,total/c.factor_slices,period,span};
}
void normalize_factor_mapping(ButterflyConfig& c) {
    if (c.prefix_codelet != "native" || c.prefix_shared_layout != "linear")
        throw std::invalid_argument("prefix codelet/layout applies only to online register-tile FFTs");
    if(c.log_n<1 || c.log_n>30 || !c.batch || c.op!=ButterflyOperator::Fft ||
       (c.precision!=ButterflyPrecision::Fp32 && c.precision!=ButterflyPrecision::Fp64) ||
       (c.fft_core!=FftCore::CufftDxBlock && c.fft_core!=FftCore::RegisterTile) || c.complex_multiply!=ComplexMultiply::FourMul ||
       c.accumulation!=ButterflyAccumulation::Native || c.local_exchange!=LocalExchange::SharedMemory ||
       c.shared_layout!=SharedLayout::WriterAligned || c.cross_twiddle!=CrossTwiddleMode::Recurrence ||
       c.direct_boundary!=DirectBoundary::Strided || c.stage_overlap ||
       (c.compute_unit!=ComputeUnit::Auto && c.compute_unit!=ComputeUnit::Radix2))
        throw std::invalid_argument("factor-streamed requires FP32/64 native FFT, cuFFTDx block or register-tile, writer-aligned shared, recurrence, direct boundary; batch stage-overlap is separate");
    if(!c.data_tiles_per_cta || c.prefetch_depth>4 ||
       !c.factor_ept || (c.factor_ept&(c.factor_ept-1)) || c.factor_ept>32 ||
       !c.factor_columns || (c.factor_columns&(c.factor_columns-1)) ||
       c.factor_columns>(c.precision==ButterflyPrecision::Fp64 ? 8U : 16U))
        throw std::invalid_argument("invalid factor EPT, columns, data tile count or prefetch depth");
    if(c.data_tiles_per_cta>std::numeric_limits<std::uint32_t>::max()/std::max(1U,c.factor_ept/2))
        throw std::invalid_argument("factor data-time fold overflows its IR field");
    if(c.stage_partition.empty()) c.stage_partition={c.log_n};
    if(c.factor_partition.empty()) c.factor_partition=c.stage_partition;
    if(c.factor_io_policies.empty()) c.factor_io_policies.assign(c.factor_partition.size(), "dynamic");
    if(c.factor_io_policies.size()!=c.factor_partition.size())
        throw std::invalid_argument("factor_io_policies must contain one policy per physical factor");
    for(const auto& policy:c.factor_io_policies) {
        if(policy!="dynamic" && policy!="static-unrolled")
            throw std::invalid_argument("factor_io_policies must be dynamic or static-unrolled");
        if(policy=="static-unrolled" && !factor_jit_enabled())
            throw std::invalid_argument("static-unrolled factor I/O requires CUBUTTERFLY_COMPILE_MODE=research or auto");
    }
    if(!c.factor_slices || (c.factor_slices&(c.factor_slices-1)))
        throw std::invalid_argument("factor_slices must be a positive power of two");
    if(c.factor_overlap && c.factor_slices==1)
        throw std::invalid_argument("factor_overlap requires multiple dependency slices");
    auto endpoints=[&](const auto& p,unsigned limit) {
        std::set<unsigned> ends; unsigned sum=0;
        for(auto s:p) {
            if(!s || s>limit || sum>c.log_n || s>c.log_n-sum)
                throw std::invalid_argument("factor and macro partitions must cover log_n with positive stages");
            sum+=s; ends.insert(sum);
        }
        if(sum!=c.log_n) throw std::invalid_argument("incomplete factor or macro partition");
        return ends;
    };
    const auto macros=endpoints(c.stage_partition,30), factors=endpoints(c.factor_partition,14);
    if(c.factor_slices>1 && (c.factor_partition.size()<2 ||
       c.factor_slices>(1U<<c.factor_partition.back())/c.factor_columns))
        throw std::invalid_argument("factor slices must fit the last digit and complete first-factor column tiles");
    for(auto e:macros) if(!factors.count(e))
        throw std::invalid_argument("a factor cannot cross a logical macro-stage boundary");
    if(!c.segment_mappings.empty()) throw std::invalid_argument("factor mappings belong to physical execution_group_mappings");
    if(c.boundaries.empty()) c.boundaries.assign(c.stage_partition.size()-1,
        {CrossTwiddleMode::Recurrence,DirectBoundary::Strided,FftBoundaryResidency::GlobalScratch});
    if(c.boundaries.size()+1!=c.stage_partition.size()) throw std::invalid_argument("macro boundary count mismatch");
    for(const auto& b:c.boundaries) if(b.residency!=FftBoundaryResidency::GlobalScratch ||
        b.cross_twiddle!=CrossTwiddleMode::Recurrence || b.layout!=DirectBoundary::Strided)
        throw std::invalid_argument("factor macro boundaries are explicit global digit rotations");
    std::vector<FftSegmentMapping> groups; unsigned done=0;
    for(auto s:c.factor_partition) {
        if(c.fft_core==FftCore::RegisterTile && (s%2 || c.factor_ept!=(1U<<(s/2))))
            throw std::invalid_argument("native factor core requires even-log factors with EPT=sqrt(factor size)");
        const auto n=1U<<s;
        const auto independent=1ULL<<(c.log_n-s);
        if(n<c.factor_ept || independent<c.factor_columns || (done && (1ULL<<done)<c.factor_columns))
            throw std::invalid_argument("factor shape cannot supply the selected EPT/column grouping");
        const auto threads=(n/c.factor_ept)*c.factor_columns;
        if(threads<32 || threads>1024 || threads%32)
            throw std::invalid_argument("factor thread count must be a warp multiple in [32,1024]");
        FftSegmentMapping mapping{c.fft_core,LocalExchange::SharedMemory,threads,c.factor_ept};
        mapping.io_policy = c.factor_io_policies[groups.size()];
        groups.push_back(std::move(mapping));
        done+=s;
    }
    if(!c.execution_group_mappings.empty()) {
        if(c.execution_group_mappings.size()!=groups.size()) throw std::invalid_argument("factor execution group count mismatch");
        for(unsigned i=0;i<groups.size();++i) {
            const auto& a=c.execution_group_mappings[i]; const auto& b=groups[i];
            if(a.core!=b.core || a.exchange!=b.exchange || a.threads!=b.threads || a.ept!=b.ept ||
               a.codelet!=b.codelet || a.io_policy!=b.io_policy)
                throw std::invalid_argument("factor execution mapping contradicts core configuration");
        }
    }
    c.execution_group_mappings=std::move(groups);
    c.tile_threads=c.execution_group_mappings.front().threads;
    c.local_stages=c.factor_partition.front();
}

MixedDataflowPlan make_factor_dataflow_plan(const ButterflyConfig& input) {
    auto c=input; normalize_factor_mapping(c);
    MixedDataflowPlan p; p.graph=make_operator_graph(c); p.stage_partition=c.stage_partition;
    p.backend="factor-streamed"; p.processing_unit=fft_core_name(c.fft_core);
    p.dispatch=DataflowDispatch::ExternalFft; p.boundary=BoundaryPolicy::GlobalScratch;
    p.exchange=ExchangePolicy::SharedMemory; p.threads_per_block=c.tile_threads;
    p.subgraph_log_n=*std::max_element(c.factor_partition.begin(),c.factor_partition.end());
    p.persistent=c.data_tiles_per_cta>1; p.pipeline_buffers=std::max(1U,c.prefetch_depth);
    p.cross_subgraph_dependencies=c.factor_partition.size()>1;
    p.boundary_policies.assign(c.stage_partition.size()-1,BoundaryPolicy::GlobalScratch);
    const unsigned width=c.precision==ButterflyPrecision::Fp64 ? 16 : 8;
    unsigned done=0,macro=0,macro_end=c.stage_partition.front();
    for(unsigned i=0;i<c.factor_partition.size();++i) {
        while(done>=macro_end) macro_end+=c.stage_partition[++macro];
        const auto s=c.factor_partition[i]; const auto& m=c.execution_group_mappings[i];
        ExecutionGroup g; g.first_stage=done; g.stage_count=s; g.stage_time=s;
        g.logical_macro_group=macro; g.threads=m.threads; g.elements_per_thread=m.ept;
        g.units_per_cta=c.factor_columns; g.core=fft_core_name(c.fft_core); g.shared_layout="writer-aligned";
        g.io_policy=c.factor_io_policies[i];
        g.data_tiles_per_cta=c.data_tiles_per_cta; g.prefetch_depth=c.prefetch_depth;
        g.data_space=m.threads; g.data_time=((1U<<s)*c.factor_columns/2/m.threads)*c.data_tiles_per_cta;
        const auto tiles=(std::uint64_t(c.batch)<<(c.log_n-s))/c.factor_columns;
        g.launch_count=i+1<c.factor_partition.size() ? c.factor_slices : 1;
        g.partial_dependency_ready=c.factor_overlap && i && i+1<c.factor_partition.size();
        g.grid_ctas=(tiles/g.launch_count+c.data_tiles_per_cta-1)/c.data_tiles_per_cta;
        // Input ring and reusable FFT/output work area. The compiler replaces
        // this conservative estimate with the imported core's actual footprint.
        g.live_shared_bytes=(1ULL<<s)*c.factor_columns*width*(c.prefetch_depth+1);
        p.grid_ctas=std::max(p.grid_ctas,g.grid_ctas); p.execution_groups.push_back(g);
        if(i) {
            const auto bytes=(std::uint64_t(c.batch)<<c.log_n)*width;
            p.execution_boundaries.push_back({i-1,i,BoundaryPolicy::GlobalScratch,"processed-digits-low","processed-digits-low",bytes,1});
            p.inter_stage_global_bytes+=bytes;
        }
        done+=s;
    }
    p.state=CandidateState::NeedsCompilation;
    return p;
}
}
